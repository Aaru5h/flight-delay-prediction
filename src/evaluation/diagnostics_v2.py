"""Validation-only feature-group refit ablations and permutation diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from src.evaluation.artifacts_v2 import CONTRACT, read_json, write_json
from src.models.classical.step5_data import FeatureSet, inner_split, make_folds, read_v2_features, ID, LEAD
from src.models.classical.step5_models import fit_candidate, prepare_fold


def run_diagnostics(dataset_dir, step5_metadata, settings):
    protocol = yaml.safe_load(CONTRACT.read_text())
    features, population, _, _ = read_v2_features(Path(dataset_dir), protocol)
    folds, _ = make_folds(population, protocol)
    fold = next(f for f in folds if f.fold_id == settings["diagnostic_fold"])
    model_metadata = step5_metadata["feature_sets"]["multi_lead"]["models"]["xgboost"]
    training_settings = step5_metadata["model_config"]
    params = model_metadata["selected_params"]
    seed = protocol["model_selection"]["seed"]
    inner, holdout = inner_split(population, fold, training_settings["inner_holdout_days"])
    train = features.frame.loc[features.frame[ID].isin(fold.train_ids)]
    validation = features.frame.loc[features.frame[ID].isin(fold.evaluation_ids)].copy()

    def fitted(allowlist):
        prepared = prepare_fold("xgboost", allowlist, [c for c in features.categorical if c in allowlist],
                                train, train.loc[train[ID].isin(inner)], train.loc[train[ID].isin(holdout)], validation)
        return fit_candidate("xgboost", prepared, training_settings["xgboost"], params, seed)[0]

    def scores(model, frame, allowlist):
        probability = model.predict_proba(frame[allowlist])[:, 1]
        return {str(lead): float(average_precision_score(rows.substantial_delay, probability[frame[LEAD].eq(lead).to_numpy()]))
                for lead, rows in frame.groupby(LEAD, sort=True)}

    manifest = read_json(Path(dataset_dir) / "feature_manifest.json")
    groups = {}
    for feature in manifest["features"]:
        rule = feature["availability_rule"]
        group = "weather" if "weather" in rule else "history" if "arrival outcomes" in rule else "same_day_departures" if "prior origin departures" in rule else "schedule"
        groups.setdefault(group, []).append(feature["name"])
    base_model = fitted(features.allowlist)
    baseline = scores(base_model, validation, features.allowlist)
    rng = np.random.default_rng(seed)
    permutation, ablations = {}, {}
    for name, columns in groups.items():
        drops = []
        # Shuffle the group jointly within lead, preserving its internal relationships.
        for _ in range(settings["permutation_repeats"]):
            altered = validation.copy()
            for _, rows in validation.groupby(LEAD):
                order = rng.permutation(len(rows))
                altered.loc[rows.index, columns] = rows[columns].iloc[order].to_numpy()
            score = scores(base_model, altered, features.allowlist)
            drops.append({lead: baseline[lead] - score[lead] for lead in baseline})
        permutation[name] = {
            "features": columns, "repeats": drops,
            "mean_ap_drop_by_lead": {lead: float(np.mean([d[lead] for d in drops])) for lead in baseline},
        }
        if name != "schedule":
            allowlist = [c for c in features.allowlist if c not in columns]
            score = scores(fitted(allowlist), validation, allowlist)
            ablations[f"without_{name}"] = {
                "removed_features": columns, "average_precision_by_lead": score,
                "ap_change_vs_full_by_lead": {lead: score[lead] - baseline[lead] for lead in baseline},
            }
    return {
        "status": "completed", "fold": fold.fold_id, "model": "xgboost",
        "population": "October out-of-fold validation; no retrospective labels",
        "caveat": "Selected configuration is refit with each group omitted; ablations are not separately retuned. Hyperparameters used the same development folds, so results are diagnostic.",
        "baseline_average_precision_by_lead": baseline, "permutation_importance": permutation,
        "refit_ablations": ablations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = read_json(parser.parse_args().request)
    write_json(args["output_path"], run_diagnostics(args["dataset_dir"], read_json(args["step5_metadata"]), args["settings"]))


if __name__ == "__main__":
    main()
