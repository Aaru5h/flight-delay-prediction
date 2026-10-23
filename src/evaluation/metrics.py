"""Common ranking, threshold, and calibration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def validate_probability_inputs(y_true, probability) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and probabilities must be non-empty aligned vectors")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be binary 0/1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and within [0, 1]")
    return y, p


def select_max_f1_threshold(y_validation, probability_validation) -> dict:
    """Select a threshold using validation evidence only."""
    y, p = validate_probability_inputs(y_validation, probability_validation)
    precision, recall, thresholds = precision_recall_curve(y, p)
    if thresholds.size == 0:
        raise ValueError("threshold selection requires both label classes")
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = int(np.nanargmax(f1))
    return {
        "rule": "maximum validation F1; ties resolved by first sklearn threshold order",
        "threshold": float(thresholds[best]),
        "validation_precision": float(precision[best]),
        "validation_recall": float(recall[best]),
        "validation_f1": float(f1[best]),
    }


def recall_at_precision_floor(y_validation, probability_validation, precision_floor: float) -> dict:
    """Validation-only recall at a stated precision floor, with explicit failure."""
    if not 0 < precision_floor <= 1:
        raise ValueError("precision_floor must be in (0, 1]")
    y, p = validate_probability_inputs(y_validation, probability_validation)
    precision, recall, thresholds = precision_recall_curve(y, p)
    eligible = np.flatnonzero(precision[:-1] >= precision_floor)
    if eligible.size == 0:
        return {"precision_floor": precision_floor, "achievable": False, "threshold": None, "recall": 0.0}
    best = eligible[np.argmax(recall[eligible])]
    return {
        "precision_floor": precision_floor,
        "achievable": True,
        "threshold": float(thresholds[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def calibration_bins(y_true, probability, n_bins: int = 10) -> pd.DataFrame:
    y, p = validate_probability_inputs(y_true, probability)
    edges = np.linspace(0, 1, n_bins + 1)
    index = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = index == bin_id
        if mask.any():
            rows.append({
                "bin": bin_id,
                "lower": float(edges[bin_id]),
                "upper": float(edges[bin_id + 1]),
                "count": int(mask.sum()),
                "mean_probability": float(p[mask].mean()),
                "positive_rate": float(y[mask].mean()),
            })
    return pd.DataFrame(rows)


def expected_calibration_error(bins: pd.DataFrame) -> float:
    total = bins["count"].sum()
    return float(((bins["count"] / total) * (bins["mean_probability"] - bins["positive_rate"]).abs()).sum())


def compute_metrics(y_true, probability, threshold: float) -> dict:
    y, p = validate_probability_inputs(y_true, probability)
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    bins = calibration_bins(y, p)
    return {
        "rows": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "positive_prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10_equal_width": expected_calibration_error(bins),
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
