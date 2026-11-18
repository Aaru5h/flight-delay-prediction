"""Validate the complete Step 5-7 evaluation handoff without importing models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import config
from src.data.loading import file_fingerprint
from src.models.classical.step5_data import ID, LEAD, LABEL, AVAILABLE, make_folds, read_v2_features

KEY = [ID, LEAD, "fold_id"]
CONTRACT = config.CONFIG_DIR / "project_contract_v2.yaml"
SETTINGS = config.CONFIG_DIR / "step8_evaluation.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass
class ModelArtifact:
    name: str
    feature_set: str
    directory: Path
    run_id: str
    run_metadata: dict
    feature_metadata: dict
    metadata: dict
    predictions: pd.DataFrame

    @property
    def key(self):
        return f"{self.feature_set}/{self.name}"


def validate_predictions(frame, expected, run_id, model_name):
    required = {*KEY, "split", "y_true", "y_probability", "model_name", "run_id"}
    if not required.issubset(frame) or frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Malformed or duplicate validation prediction keys")
    if set(frame["split"]) != {"validation"} or set(frame["run_id"]) != {run_id} or set(frame["model_name"]) != {model_name}:
        raise ValueError("Validation prediction split, run or model mismatch")
    if not frame.y_true.isin([0, 1]).all() or not (np.isfinite(frame.y_probability) & frame.y_probability.between(0, 1)).all():
        raise ValueError("Invalid validation labels or probabilities")
    if len(frame) != len(expected):
        raise ValueError("Validation prediction population mismatch")
    joined = expected.merge(frame, on=KEY, validate="one_to_one", how="left", suffixes=("_expected", ""))
    if joined.y_probability.isna().any() or not joined.y_true.eq(joined.y_true_expected).all():
        raise ValueError("Validation prediction IDs or labels disagree with the dataset")
    return joined.drop(columns="y_true_expected")


def load_handoff(dataset_dir, step5_dir, step6_dir, step7_dir=None, *, allow_reduced_budget=False):
    protocol = yaml.safe_load(CONTRACT.read_text())
    features, population, sources, _ = read_v2_features(Path(dataset_dir), protocol)
    folds, final = make_folds(population, protocol)
    expected = pd.concat([
        features.frame.loc[features.frame[ID].isin(f.evaluation_ids), [ID, LEAD, LABEL, AVAILABLE]]
        .rename(columns={LABEL: "y_true"}).assign(fold_id=f.fold_id)
        for f in folds
    ], ignore_index=True)
    expected["sched_date"] = expected[ID].map(population.sched_date)
    models, omissions, canonical_folds, budgets = [], [], None, set()
    for directory in [step5_dir, step6_dir, step7_dir]:
        if directory is None:
            omissions.append("sequence_model: no Step 7 handoff supplied")
            continue
        directory = Path(directory)
        report = read_json(directory / "run_metadata.json")
        if report.get("protocol_sha256") != file_fingerprint(CONTRACT) or report.get("raw_data_fingerprints") != sources:
            raise ValueError(f"Protocol or source fingerprint mismatch: {directory}")
        for filename, expected_hash in report.get("code_sha256", {}).items():
            candidates = list(config.PROJECT_ROOT.joinpath("src").rglob(filename))
            if len(candidates) != 1 or file_fingerprint(candidates[0]) != expected_hash:
                raise ValueError(f"Training/inference source changed since the model run: {filename}")
        if report.get("gate", {}).get("decision") == "no-go":
            omissions.append("sequence_model: gate no-go; " + json.dumps(report["gate"]["conditions"]))
            continue
        budgets.add(report["trials_per_model"])
        if report["protocol_budget_completed"] != (report["trials_per_model"] == protocol["model_selection"]["trials_per_model_per_feature_set"]):
            raise ValueError("Inconsistent search-budget completion metadata")
        if not report.get("protocol_budget_completed") and not allow_reduced_budget:
            raise ValueError("Full 40-configuration searches are required; use --allow-reduced-budget for development only")
        fold_path = directory / report["fold_manifest"]
        if file_fingerprint(fold_path) != report["fold_manifest_sha256"]:
            raise ValueError("Fold manifest was modified")
        frame = pd.read_csv(fold_path, dtype={ID: str})
        if frame.duplicated([ID, "fold_id"]).any():
            raise ValueError("Duplicate fold assignments")
        assignments = frame.sort_values(["fold_id", ID]).reset_index(drop=True)
        dates = pd.to_datetime(frame["sched_date"], utc=True).dt.tz_localize(None).dt.normalize()
        if not dates.eq(frame[ID].map(population.sched_date)).all():
            raise ValueError("Fold manifest dates disagree with source schedule")
        if canonical_folds is None:
            canonical_folds = assignments
            wanted = pd.concat([
                pd.DataFrame({ID: f.train_ids, "fold_id": f.fold_id, "split": "train"})
                for f in [*folds, final]
            ] + [
                pd.DataFrame({ID: f.evaluation_ids, "fold_id": f.fold_id, "split": "retrospective" if f is final else "validation"})
                for f in [*folds, final]
            ]).sort_values(["fold_id", ID]).reset_index(drop=True)
            if not assignments[[ID, "fold_id", "split"]].astype(str).equals(wanted[[ID, "fold_id", "split"]].astype(str)):
                raise ValueError("Fold manifest disagrees with recomputed availability-safe folds")
        elif not canonical_folds.equals(assignments):
            raise ValueError("Models have different fold assignments")
        for feature_set, item in report["feature_sets"].items():
            if feature_set != "legacy_aml" and item["feature_file_sha256"] != features.file_fingerprints:
                raise ValueError("Model was trained on different Step 4 feature files")
            for name, model in item["models"].items():
                if name == "sequence_model":
                    sequence_hashes = item.get("sequence_file_sha256", {})
                    if len(sequence_hashes) != len(protocol["prediction_time"]["lead_time_minutes"]):
                        raise ValueError("Sequence-model handoff lacks sequence-file fingerprints")
                    for filename, expected_hash in sequence_hashes.items():
                        if file_fingerprint(Path(dataset_dir) / filename) != expected_hash:
                            raise ValueError("Sequence inputs changed since model training")
                if len(model["trials"]) != report["trials_per_model"]:
                    raise ValueError("Model trial count disagrees with its search budget")
                if model["feature_allowlist"] != item["feature_allowlist"]:
                    raise ValueError("Model and feature-set input columns disagree")
                if model["fit_cutoff_utc"] != final.fit_cutoff_utc.isoformat():
                    raise ValueError("Final model fit boundary mismatch")
                if not (directory / model["model_file"]).is_file():
                    raise FileNotFoundError(directory / model["model_file"])
                prediction = pd.read_csv(directory / model["validation_predictions"], dtype={ID: str})
                wanted = expected.loc[expected[LEAD].eq(60)] if feature_set == "legacy_aml" else expected
                prediction = validate_predictions(prediction, wanted, item["run_id"], name)
                models.append(ModelArtifact(name, feature_set, directory, item["run_id"], report, item, model, prediction))
    required = {"schedule_only/schedule_only", "multi_lead/logistic_regression", "multi_lead/xgboost", "multi_lead/embedding_mlp",
                "legacy_aml/logistic_regression", "legacy_aml/xgboost", "legacy_aml/embedding_mlp"}
    keys = [m.key for m in models]
    if len(keys) != len(set(keys)) or not required.issubset(keys):
        raise ValueError(f"Missing or duplicate required model handoffs: {sorted(required - set(keys))}")
    if len(budgets) != 1:
        raise ValueError("Models must use the same search budget")
    return protocol, features, population, folds, final, models, omissions
