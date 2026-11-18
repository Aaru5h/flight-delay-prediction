"""Train the v2 classical baselines and export out-of-fold predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import sklearn
import xgboost
import yaml

from src.data import config
from src.data.loading import file_fingerprint
from src.models.classical.step5_data import (
    DATE, FOLD_MANIFEST_VERSION, ID, LABEL, LEAD,
    FeatureSet, Fold, inner_split, make_folds, read_legacy_features, read_v2_features,
)
from src.models.classical.step5_models import fit_candidate, prepare_fold, sample_candidates

LOGGER = logging.getLogger(__name__)
CONTRACT_PATH = config.CONFIG_DIR / "project_contract_v2.yaml"
SETTINGS_PATH = config.CONFIG_DIR / "step5_classical.json"
PREDICTION_COLUMNS = [ID, LEAD, "fold_id", "split", "y_true", "y_probability", "model_name", "run_id"]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _code_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT, capture_output=True, text=True,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _write_fold_manifest(path: Path, metadata: pd.DataFrame, folds: list[Fold], final: Fold) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        for fold in [*folds, final]:
            evaluation_split = "retrospective" if fold is final else "validation"
            rows = pd.concat([
                metadata.loc[fold.train_ids, [DATE]].assign(split="train"),
                metadata.loc[fold.evaluation_ids, [DATE]].assign(split=evaluation_split),
            ]).assign(fold_id=fold.fold_id)
            rows.reset_index()[[ID, "fold_id", DATE, "split"]].to_csv(
                stream, index=False, header=fold is folds[0], date_format="%Y-%m-%d"
            )


def _fold_data(
    features: FeatureSet, metadata: pd.DataFrame, fold: Fold, family: str,
    inner_holdout_days: int, *, validation: bool,
):
    inner_ids, holdout_ids = inner_split(metadata, fold, inner_holdout_days)
    frame = features.frame
    columns = list(dict.fromkeys([ID, LEAD, LABEL, *features.allowlist]))
    train = frame.loc[frame[ID].isin(fold.train_ids), columns]
    inner_train = train.loc[train[ID].isin(inner_ids)]
    holdout = train.loc[train[ID].isin(holdout_ids)]
    evaluation = frame.loc[frame[ID].isin(fold.evaluation_ids), columns] if validation else None
    prepared = prepare_fold(
        family, features.allowlist, features.categorical, train, inner_train, holdout, evaluation
    )
    return prepared, evaluation


def _probabilities(model, prepared, evaluation: pd.DataFrame) -> np.ndarray:
    scores = model.named_steps["model"].predict_proba(prepared.x_evaluation)[:, 1]
    if scores.shape != (len(evaluation),) or not (np.isfinite(scores) & (scores >= 0) & (scores <= 1)).all():
        raise ValueError("Model produced invalid validation probabilities")
    return scores


def _average_precision_by_lead(evaluation: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    metrics = {}
    leads = evaluation[LEAD].to_numpy()
    targets = evaluation[LABEL].to_numpy()
    for lead in np.unique(leads):
        mask = leads == lead
        if len(np.unique(targets[mask])) != 2:
            raise ValueError(f"Validation requires both classes at {lead} minutes")
        metrics[str(lead)] = float(average_precision_score(targets[mask], scores[mask]))
    if not metrics:
        raise ValueError("Empty validation population")
    return metrics


def _train_benchmark(
    stage: Path,
    features: FeatureSet,
    metadata: pd.DataFrame,
    folds: list[Fold],
    final: Fold,
    model_name: str,
    settings: dict,
    budget: int,
    seed: int,
    run_id: str,
) -> dict:
    family = "logistic_regression" if model_name == "schedule_only" else model_name
    model_settings = settings[family]
    candidates = sample_candidates(family, model_settings, budget, seed)
    trials = [
        {"trial": index, "params": candidate, "folds": {}}
        for index, candidate in enumerate(candidates, start=1)
    ]
    for fold in folds:
        prepared, evaluation = _fold_data(
            features, metadata, fold, family, settings["inner_holdout_days"], validation=True,
        )
        for trial in trials:
            model, diagnostics = fit_candidate(family, prepared, model_settings, trial["params"], seed)
            metrics = _average_precision_by_lead(evaluation, _probabilities(model, prepared, evaluation))
            trial["folds"][fold.fold_id] = {"average_precision_by_lead": metrics, **diagnostics}
        LOGGER.info("%s / %s / %s finished %d trials", features.name, model_name, fold.fold_id, budget)

    for trial in trials:
        scores = [
            score for fold in trial["folds"].values() for score in fold["average_precision_by_lead"].values()
        ]
        trial["mean_validation_average_precision"] = float(np.mean(scores))
    best = max(trials, key=lambda trial: trial["mean_validation_average_precision"])

    stem = f"{features.name}_{model_name}"
    prediction_file = f"{stem}_validation_predictions.csv"
    with (stage / prediction_file).open("w", encoding="utf-8", newline="") as stream:
        for fold in folds:
            prepared, evaluation = _fold_data(
                features, metadata, fold, family, settings["inner_holdout_days"], validation=True,
            )
            model, _ = fit_candidate(family, prepared, model_settings, best["params"], seed)
            prediction = evaluation[[ID, LEAD, LABEL]].rename(columns={LABEL: "y_true"}).assign(
                fold_id=fold.fold_id, split="validation", model_name=model_name, run_id=run_id,
                y_probability=_probabilities(model, prepared, evaluation),
            )
            prediction[PREDICTION_COLUMNS].to_csv(stream, index=False, header=fold is folds[0])

    prepared, _ = _fold_data(
        features, metadata, final, family, settings["inner_holdout_days"], validation=False,
    )
    model, diagnostics = fit_candidate(family, prepared, model_settings, best["params"], seed)
    model_file = f"{stem}.joblib"
    joblib.dump(model, stage / model_file, compress=3)
    return {
        "model_name": model_name,
        "feature_set": features.name,
        "feature_allowlist": features.allowlist,
        "selected_trial": best["trial"],
        "selected_params": best["params"],
        "mean_validation_average_precision": best["mean_validation_average_precision"],
        "fit_cutoff_utc": final.fit_cutoff_utc.isoformat(),
        "final_fit_training_prevalence": float(prepared.y_train.mean()),
        "final_fit_diagnostics": diagnostics,
        "validation_predictions": prediction_file,
        "model_file": model_file,
        "trials": trials,
    }


def run_step5(
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    settings_path: Path | None = None,
    *,
    legacy_features: Path | None = None,
    legacy_manifest: Path | None = None,
    trial_limit: int | None = None,
) -> dict:
    """Run the ordered folds, then publish selected models and validation scores."""
    dataset = Path(dataset_dir or config.PROCESSED_DATA_DIR / "step4")
    output = Path(output_dir or config.PROJECT_ROOT / "artifacts" / "step5")
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose a new output directory.")
    if (legacy_features is None) != (legacy_manifest is None):
        raise ValueError("Supply both legacy feature and manifest paths for the legacy comparison")
    protocol = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    settings = json.loads(Path(settings_path or SETTINGS_PATH).read_text(encoding="utf-8"))
    selection = protocol["model_selection"]
    full_budget = selection["trials_per_model_per_feature_set"]
    budget = trial_limit if trial_limit is not None else full_budget
    if not 1 <= budget <= full_budget:
        raise ValueError(f"Trial limit must be between 1 and {full_budget}")
    logistic = settings["logistic_regression"]
    booster = settings["xgboost"]
    if min(
        settings["inner_holdout_days"], logistic["max_iter"], logistic["iterations_per_check"],
        logistic["patience_checks"], booster["max_estimators"], booster["early_stopping_rounds"],
    ) < 1:
        raise ValueError("Training and holdout limits must be positive")
    seed = selection["seed"]
    features, metadata, source_fingerprints, schedule_allowlist = read_v2_features(dataset, protocol)
    folds, final = make_folds(metadata, protocol)
    schedule_features = FeatureSet(
        "schedule_only", features.frame, features.manifest_version, schedule_allowlist,
        [column for column in features.categorical if column in schedule_allowlist],
        features.file_fingerprints,
    )
    benchmarks = [
        (schedule_features, ("schedule_only",)),
        (features, ("logistic_regression", "xgboost")),
    ]
    if legacy_features is not None:
        legacy = read_legacy_features(Path(legacy_features), Path(legacy_manifest), features, protocol)
        benchmarks.append((legacy, ("logistic_regression", "xgboost")))
    if any("sched_dep_hour" not in item.allowlist for item, _ in benchmarks):
        raise ValueError("The hour spline requires sched_dep_hour in every benchmark feature set")

    packages = {
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
    }
    code_files = (Path(__file__), Path(__file__).with_name("step5_data.py"), Path(__file__).with_name("step5_models.py"))
    code_hashes = {path.name: file_fingerprint(path) for path in code_files}
    provenance = {
        "protocol_version": protocol["contract_version"],
        "protocol_sha256": file_fingerprint(CONTRACT_PATH),
        "raw_data_fingerprints": source_fingerprints,
        "fold_manifest_version": FOLD_MANIFEST_VERSION,
        "model_config": settings,
        "seed": seed,
        "package_versions": packages,
        "code_sha256": code_hashes,
        "trials_per_model": budget,
        "full_protocol_budget": full_budget,
        "protocol_budget_completed": budget == full_budget,
        "selection_metric": selection["primary_score"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".step5-", dir=output.parent) as work:
        stage = Path(work) / "baselines"
        stage.mkdir()
        fold_file = stage / "fold_manifest.csv"
        _write_fold_manifest(fold_file, metadata, folds, final)
        provenance["fold_manifest_sha256"] = file_fingerprint(fold_file)
        report = {
            **provenance,
            "code_commit": _code_commit(),
            "fold_manifest": fold_file.name,
            "folds": [
                {"fold_id": fold.fold_id, "fit_cutoff_utc": fold.fit_cutoff_utc.isoformat(),
                 "train_flights": len(fold.train_ids), "evaluation_flights": len(fold.evaluation_ids)}
                for fold in [*folds, final]
            ],
            "feature_sets": {},
        }
        for item, model_names in benchmarks:
            manifest = {
                "feature_set": item.name,
                "feature_manifest_version": item.manifest_version,
                "feature_file_sha256": item.file_fingerprints,
                "feature_allowlist": item.allowlist,
            }
            run_hash = hashlib.sha256(json.dumps({**provenance, **manifest}, sort_keys=True).encode()).hexdigest()
            run_id = f"step5-{item.name}-{run_hash[:12]}"
            report["feature_sets"][item.name] = {
                **manifest,
                "run_id": run_id,
                "models": {
                    name: _train_benchmark(stage, item, metadata, folds, final, name, settings, budget, seed, run_id)
                    for name in model_names
                },
            }
        _write_json(stage / "run_metadata.json", report)
        stage.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Step 5 classical walk-forward baselines")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--settings-path", type=Path)
    parser.add_argument("--legacy-features", type=Path)
    parser.add_argument("--legacy-manifest", type=Path)
    parser.add_argument("--trial-limit", type=int, help="Reduced development run for smoke testing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_step5(**vars(args))
    output = args.output_dir or config.PROJECT_ROOT / "artifacts" / "step5"
    print(f"Wrote {len(result['feature_sets'])} feature sets to {output}")


if __name__ == "__main__":
    main()
