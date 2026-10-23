"""Train Phase 3 baselines and export validation-only predictions.

The held-out test labels are assigned to the split manifest but never passed
to fitting, tuning, or Phase 3 metrics. Phase 5 owns final test evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.data import config as data_config
from src.evaluation.split import assign_chronological_split
from src.models.classical.pipelines import build_logistic_pipeline, build_xgboost_pipeline

ROOT = data_config.PROJECT_ROOT
DEFAULT_CONFIG = ROOT / "configs" / "classical_models.json"
FEATURE_MANIFEST = data_config.PROCESSED_DATA_DIR / "phase2_feature_manifest.json"
FEATURE_TABLE = data_config.PROCESSED_DATA_DIR / "phase2_features.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


# Backward-compatible name; Contributor 5's shared split module is canonical.
assign_split = assign_chronological_split


def load_phase2(feature_path: Path, manifest_path: Path) -> tuple[pd.DataFrame, dict, list[str], list[str]]:
    manifest = json.loads(manifest_path.read_text())
    frame = pd.read_csv(feature_path, low_memory=False)
    allowlist = manifest["allowlist"]
    missing = sorted(set(allowlist + ["flight_row_id", "substantial_delay", "sched_date"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Phase 2 feature table is missing columns: {missing}")
    if set(manifest["denylist"]) & set(allowlist):
        raise ValueError("feature manifest allowlist contains denied fields")
    categorical = [f["name"] for f in manifest["features"] if f["type"] == "categorical"]
    numeric = [name for name in allowlist if name not in categorical]
    frame["split"] = assign_split(frame["sched_date"])
    if frame["flight_row_id"].duplicated().any() or frame["substantial_delay"].isna().any():
        raise ValueError("feature table IDs must be unique and all targets known")
    return frame, manifest, categorical, numeric


def validation_metrics(y_true: pd.Series, probability: np.ndarray) -> dict:
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("model produced invalid probabilities")
    return {
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
    }


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical: list[str],
    numeric: list[str],
    candidates: list[dict],
    seed: int,
) -> tuple[object, dict, list[dict]]:
    """Bounded validation-only search; no test frame is accepted by design."""
    trials = []
    best_pipeline = None
    best_settings = None
    best_score = -np.inf
    for index, settings in enumerate(candidates, start=1):
        pipeline = build_xgboost_pipeline(categorical, numeric, settings, seed)
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_validation)[:, 1]
        metrics = validation_metrics(y_validation, probability)
        trials.append({"candidate": index, "settings": settings, "validation_metrics": metrics})
        if metrics["pr_auc"] > best_score:
            best_score = metrics["pr_auc"]
            best_pipeline, best_settings = pipeline, settings
    return best_pipeline, best_settings, trials


def _prediction_frame(ids, y, probability, model_name: str, run_id: str) -> pd.DataFrame:
    result = pd.DataFrame({
        "flight_row_id": ids.to_numpy(),
        "y_true": y.astype(int).to_numpy(),
        "y_probability": probability,
        "split": "validation",
        "model_name": model_name,
        "run_id": run_id,
    })
    if result["flight_row_id"].duplicated().any() or not np.isfinite(result["y_probability"]).all():
        raise AssertionError("invalid prediction output")
    return result


def run_phase3(
    feature_path: Path = FEATURE_TABLE,
    manifest_path: Path = FEATURE_MANIFEST,
    settings_path: Path = DEFAULT_CONFIG,
    artifacts_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict:
    artifacts_dir = Path(artifacts_dir or ROOT / "artifacts" / "classical")
    reports_dir = Path(reports_dir or data_config.REPORTS_DIR)
    settings = json.loads(Path(settings_path).read_text())
    frame, feature_manifest, categorical, numeric = load_phase2(Path(feature_path), Path(manifest_path))
    allowlist = feature_manifest["allowlist"]
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]
    X_train, y_train = train[allowlist], train["substantial_delay"].astype(int)
    X_validation, y_validation = validation[allowlist], validation["substantial_delay"].astype(int)

    run_id = "phase3-" + hashlib.sha256(
        (_sha256(Path(feature_path)) + json.dumps(settings, sort_keys=True)).encode()
    ).hexdigest()[:12]
    logistic = build_logistic_pipeline(categorical, numeric, settings["logistic_regression"])
    logistic.fit(X_train, y_train)
    logistic_probability = logistic.predict_proba(X_validation)[:, 1]
    logistic_metrics = validation_metrics(y_validation, logistic_probability)

    xgb, xgb_settings, trials = tune_xgboost(
        X_train, y_train, X_validation, y_validation, categorical, numeric,
        settings["xgboost_candidates"], settings["random_seed"],
    )
    xgb_probability = xgb.predict_proba(X_validation)[:, 1]
    xgb_metrics = validation_metrics(y_validation, xgb_probability)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic, artifacts_dir / "logistic_regression.joblib")
    joblib.dump(xgb, artifacts_dir / "xgboost.joblib")
    logistic_predictions = _prediction_frame(validation["flight_row_id"], y_validation, logistic_probability, "logistic_regression", run_id)
    xgb_predictions = _prediction_frame(validation["flight_row_id"], y_validation, xgb_probability, "xgboost", run_id)
    logistic_predictions.to_csv(artifacts_dir / "logistic_validation_predictions.csv", index=False)
    xgb_predictions.to_csv(artifacts_dir / "xgboost_validation_predictions.csv", index=False)

    split_manifest = frame[["flight_row_id", "sched_date", "split"]].copy()
    split_manifest.to_csv(artifacts_dir / "split_manifest.csv", index=False)
    contract = yaml.safe_load((ROOT / "configs" / "project_contract.yaml").read_text())
    metadata = {
        "phase": 3,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_policy": "test labels not used for fitting, tuning, or Phase 3 metrics; Phase 5 owns final test evaluation",
        "rows": {"train": len(train), "validation": len(validation), "test_sealed": len(test)},
        "target_prevalence": {"train": float(y_train.mean()), "validation": float(y_validation.mean())},
        "features": {"count": len(allowlist), "allowlist": allowlist, "categorical": categorical, "numeric": numeric},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "random_seed": settings["random_seed"],
        "feature_manifest_version": feature_manifest["manifest_version"],
        "feature_manifest_sha256": feature_manifest["manifest_sha256"],
        "feature_table_sha256": _sha256(Path(feature_path)),
        "split_manifest_version": settings["split_manifest_version"],
        "split_contract": contract["split"],
        "code_commit": _commit(),
        "models": {
            "logistic_regression": {"settings": settings["logistic_regression"], "validation_metrics": logistic_metrics},
            "xgboost": {"selected_settings": xgb_settings, "selection_metric": "validation PR AUC", "validation_metrics": xgb_metrics, "trials": trials},
        },
    }
    (artifacts_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (reports_dir / "phase3_classical_baselines.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 3 classical baselines")
    parser.add_argument("--feature-path", type=Path, default=FEATURE_TABLE)
    parser.add_argument("--manifest-path", type=Path, default=FEATURE_MANIFEST)
    parser.add_argument("--settings-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    args = parser.parse_args()
    metadata = run_phase3(args.feature_path, args.manifest_path, args.settings_path, args.artifacts_dir, args.reports_dir)
    print(json.dumps({"run_id": metadata["run_id"], "rows": metadata["rows"], "models": metadata["models"]}, indent=2))


if __name__ == "__main__":
    main()
