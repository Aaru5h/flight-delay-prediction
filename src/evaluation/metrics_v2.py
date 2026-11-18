"""Unrounded v2 metrics and paired, day-clustered uncertainty estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def validate_arrays(y, probability):
    y, p = np.asarray(y), np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Labels and probabilities must be nonempty aligned vectors")
    if not np.isin(y, [0, 1]).all() or not (np.isfinite(p) & (p >= 0) & (p <= 1)).all():
        raise ValueError("Expected binary labels and finite probabilities in [0, 1]")
    return y.astype(int), p


def reliability_bins(y, probability, n_bins=10):
    """Equal-mass bins; stable sorting makes ties deterministic."""
    y, p = validate_arrays(y, probability)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    return [
        {"count": len(rows), "mean_probability": float(p[rows].mean()),
         "observed_fraction": float(y[rows].mean())}
        for rows in np.array_split(np.argsort(p, kind="stable"), min(n_bins, len(y)))
    ]


def alert_budget(y, probability, ids, fraction=0.1):
    """Flag exactly floor(fraction*n) rows, with flight ID breaking ties."""
    y, p = validate_arrays(y, probability)
    ids = np.asarray(ids, dtype=str)
    if ids.shape != y.shape or len(set(ids)) != len(ids) or not 0 < fraction <= 1:
        raise ValueError("Alert cohorts require unique flight IDs and a valid budget")
    count = int(np.floor(fraction * len(y)))
    order = np.lexsort((ids, -p))
    selected = order[:count]
    tp = int(y[selected].sum())
    mask = np.zeros(len(y), dtype=bool)
    mask[selected] = True
    return {
        "fraction": fraction, "alerts": count, "true_positives": tp,
        "false_positives": count - tp, "false_negatives": int(y.sum()) - tp,
        "precision": tp / count if count else None,
        "recall": tp / int(y.sum()) if y.sum() else None,
        "boundary_probability": float(p[selected[-1]]) if count else None,
        "tie_rule": "probability descending, flight_row_id ascending",
    }, mask


def metrics(y, probability, ids, reference_prevalence, *, fraction=0.1, n_bins=10):
    y, p = validate_arrays(y, probability)
    reference_values = np.asarray(reference_prevalence, dtype=float)
    if not (np.isfinite(reference_values) & (reference_values >= 0) & (reference_values <= 1)).all():
        raise ValueError("Invalid training reference prevalence")
    bins = reliability_bins(y, p, n_bins)
    brier = float(np.mean((y - p) ** 2))
    reference = float(np.mean((y - reference_prevalence) ** 2))
    budget, _ = alert_budget(y, p, ids, fraction)
    return {
        "rows": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier_score": brier, "reference_prevalence": float(reference_values.mean()),
        "reference_brier_score": reference,
        "brier_skill_score": 1 - brier / reference if reference else None,
        "equal_mass_ece": sum(b["count"] * abs(b["mean_probability"] - b["observed_fraction"]) for b in bins) / len(y),
        "reliability_bins": bins, "alert_budget": budget,
    }


def _ranking_plan(y, probability):
    order = np.argsort(-probability, kind="stable")
    sorted_p = probability[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    return order, starts, y[order]


def _weighted_ranking(plan, weights):
    order, starts, labels = plan
    w = weights[order]
    positives = np.add.reduceat(w * labels, starts)
    negatives = np.add.reduceat(w * (1 - labels), starts)
    tp, fp = np.cumsum(positives), np.cumsum(negatives)
    total_p, total_n = tp[-1], fp[-1]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    ap = float(np.sum(positives * precision) / total_p) if total_p else np.nan
    auc = float(np.sum(positives * (total_n - fp + 0.5 * negatives)) / (total_p * total_n)) if total_p and total_n else np.nan
    return ap, auc


def paired_day_bootstrap(y, probabilities, dates, reference_prevalences, *, resamples=1000, seed=42, confidence=0.95):
    """Use identical resampled service days for every matched model.

    Sorting each model once and weighting score-tie groups is exactly equivalent
    to duplicating sampled days, including ties in average precision and ROC AUC.
    """
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    first = next(iter(probabilities.values()))
    y, _ = validate_arrays(y, first)
    days, codes = np.unique(np.asarray(dates, dtype=str), return_inverse=True)
    if len(codes) != len(y) or len(days) < 2:
        raise ValueError("Day bootstrap needs aligned rows from at least two days")
    plans, errors, references = {}, {}, {}
    for name, p in probabilities.items():
        _, p = validate_arrays(y, p)
        plans[name] = _ranking_plan(y, p)
        errors[name] = (y - p) ** 2
        references[name] = (y - reference_prevalences[name]) ** 2
    samples = {name: np.full((resamples, 4), np.nan) for name in plans}
    rng = np.random.default_rng(seed)
    for i in range(resamples):
        counts = np.bincount(rng.integers(len(days), size=len(days)), minlength=len(days))
        weights = counts[codes].astype(float)
        for name in plans:
            ap, auc = _weighted_ranking(plans[name], weights)
            brier = float(np.dot(weights, errors[name]) / weights.sum())
            ref = float(np.dot(weights, references[name]) / weights.sum())
            samples[name][i] = ap, auc, brier, 1 - brier / ref if ref else np.nan
    alpha = (1 - confidence) / 2

    def interval(values):
        valid = values[np.isfinite(values)]
        if not len(valid):
            return {"lower": None, "upper": None, "valid_resamples": 0}
        lo, hi = np.quantile(valid, [alpha, 1 - alpha])
        return {"lower": float(lo), "upper": float(hi), "valid_resamples": len(valid)}

    names = ("average_precision", "roc_auc", "brier_score", "brier_skill_score")
    output = {name: {metric: interval(values[:, i]) for i, metric in enumerate(names)} for name, values in samples.items()}
    deltas = {}
    if "multi_lead/xgboost" in samples:
        base = samples["multi_lead/xgboost"][:, 0]
        deltas = {name: interval(values[:, 0] - base) for name, values in samples.items() if name != "multi_lead/xgboost"}
    return {
        "unit": "local scheduled service date", "days": len(days), "resamples": resamples,
        "seed": seed, "confidence_level": confidence, "paired_across_models": True,
        "metrics": output, "average_precision_difference_vs_xgboost": deltas,
    }
