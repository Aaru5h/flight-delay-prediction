"""Step 6: train the embedding MLP on the protocol v2 walk-forward folds.

The MLP uses the same Step 4 features, fold assignments, availability rules,
inner early-stopping holdout, trial budget and selection score as the Step 5
baselines, so its validation predictions can be compared row for row.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import logging
from pathlib import Path
import platform
import time
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import config
from src.data.loading import file_fingerprint
from src.models.classical.step5_data import (
    FOLD_MANIFEST_VERSION, ID, LABEL, LEAD,
    FeatureSet, Fold, inner_split, make_folds, read_legacy_features, read_v2_features,
)
from src.models.neural.common import (
    PREDICTION_COLUMNS, average_precision_by_lead, code_commit, write_fold_manifest, write_json,
)
from src.models.neural.fit import (
    Arrays, NeuralScorer, build_network, predict_proba, save_checkpoint, take, train_network,
)
from src.models.neural.preprocess import TabularPreprocessor

LOGGER = logging.getLogger(__name__)
CONTRACT_PATH = config.CONFIG_DIR / "project_contract_v2.yaml"
SETTINGS_PATH = config.CONFIG_DIR / "step6_neural.json"
STEP5_SETTINGS_PATH = config.CONFIG_DIR / "step5_classical.json"


def sample_candidates(settings: dict, budget: int, seed: int, *, sequence: bool = False) -> list[dict]:
    """Draw the same deterministic configurations for every feature set."""
    rng = np.random.default_rng(seed)
    space = settings["search"]
    candidates = []
    for _ in range(budget):
        candidate = {
            "hidden_layers": list(space["hidden_layers"][int(rng.integers(len(space["hidden_layers"])))]),
            "dropout": float(rng.uniform(*space["dropout"])),
            "learning_rate": float(10 ** rng.uniform(*space["learning_rate_log10_range"])),
            "weight_decay": float(10 ** rng.uniform(*space["weight_decay_log10_range"])),
            "batch_size": int(space["batch_size"][int(rng.integers(len(space["batch_size"])))]),
            "numeric_encoding": space["numeric_encoding"][int(rng.integers(len(space["numeric_encoding"])))],
            "periodic_frequencies": int(
                space["periodic_frequencies"][int(rng.integers(len(space["periodic_frequencies"])))]
            ),
            "periodic_sigma": float(10 ** rng.uniform(*space["periodic_sigma_log10_range"])),
            "positive_weighting": space["positive_weighting"][int(rng.integers(len(space["positive_weighting"])))],
        }
        if sequence:
            hidden = settings["sequence"]["gru_hidden"]
            candidate["gru_hidden"] = int(hidden[int(rng.integers(len(hidden)))])
        candidates.append(candidate)
    return candidates


@dataclass
class NeuralFold:
    preprocessor: TabularPreprocessor
    train: Arrays
    y_train: np.ndarray
    inner_preprocessor: TabularPreprocessor
    inner_train: Arrays
    y_inner_train: np.ndarray
    holdout: Arrays
    y_holdout: np.ndarray
    evaluation: Arrays | None
    evaluation_frame: pd.DataFrame | None
    sequence_state: dict | None = None
    inner_sequence_dims: dict | None = None
    sequence_dims: dict | None = None


def _positions(frame: pd.DataFrame, ids: pd.Index) -> np.ndarray:
    return np.flatnonzero(frame[ID].isin(ids).to_numpy())


def prepare_fold(
    features: FeatureSet, metadata: pd.DataFrame, fold: Fold, settings: dict,
    *, validation: bool, sequences=None,
) -> NeuralFold:
    """Fit preprocessing inside the outer and inner training windows only."""
    inner_ids, holdout_ids = inner_split(metadata, fold, settings["inner_holdout_days"])
    frame = features.frame
    groups = {
        "train": _positions(frame, fold.train_ids),
        "inner": _positions(frame, inner_ids),
        "holdout": _positions(frame, holdout_ids),
    }
    if validation:
        groups["evaluation"] = _positions(frame, fold.evaluation_ids)
    labels = frame[LABEL].to_numpy()
    for name in ("train", "inner", "holdout"):
        if len(np.unique(labels[groups[name]])) != 2:
            raise ValueError(f"Fold {fold.fold_id} needs both classes in its {name} window")
    if not set(groups["holdout"]).issubset(groups["train"]) or set(groups["inner"]) & set(groups["holdout"]):
        raise AssertionError("Inner fit and holdout must partition part of the training window")

    params = settings["preprocessing"]
    def fitted(rows: np.ndarray) -> TabularPreprocessor:
        return TabularPreprocessor(
            features.allowlist, features.categorical,
            min_category_count=params["min_category_count"], numeric_clip=params["numeric_clip"],
        ).fit(frame.iloc[rows])

    def encode(preprocessor: TabularPreprocessor, rows: np.ndarray) -> Arrays:
        codes, numeric = preprocessor.transform(frame.iloc[rows])
        return {"cat": codes, "num": numeric}

    outer, inner = fitted(groups["train"]), fitted(groups["inner"])
    prepared = NeuralFold(
        preprocessor=outer,
        train=encode(outer, groups["train"]),
        y_train=labels[groups["train"]].astype(np.float32),
        inner_preprocessor=inner,
        inner_train=encode(inner, groups["inner"]),
        y_inner_train=labels[groups["inner"]].astype(np.float32),
        holdout=encode(inner, groups["holdout"]),
        y_holdout=labels[groups["holdout"]].astype(np.float32),
        evaluation=encode(outer, groups["evaluation"]) if validation else None,
        evaluation_frame=frame.iloc[groups["evaluation"]][[ID, LEAD, LABEL]] if validation else None,
    )
    if sequences is not None:
        outer_sequences = sequences.fit(groups["train"], outer)
        inner_sequences = sequences.fit(groups["inner"], inner)
        prepared.train.update(sequences.arrays(outer_sequences, groups["train"]))
        prepared.inner_train.update(sequences.arrays(inner_sequences, groups["inner"]))
        prepared.holdout.update(sequences.arrays(inner_sequences, groups["holdout"]))
        if validation:
            prepared.evaluation.update(sequences.arrays(outer_sequences, groups["evaluation"]))
        prepared.sequence_state = outer_sequences.state()
        prepared.sequence_dims = sequences.dims(outer_sequences)
        prepared.inner_sequence_dims = sequences.dims(inner_sequences)
    return prepared


def fit_candidate(
    data: NeuralFold, settings: dict, candidate: dict, seed: int, device: str = "cpu",
):
    """Choose the epoch count on the inner holdout, then refit on the full window."""
    training = settings["training"]
    gru = {"gru_hidden": candidate["gru_hidden"]} if data.sequence_dims else {}
    inner_model, _ = build_network(
        data.inner_preprocessor, candidate, settings, float(data.y_inner_train.mean()),
        {**(data.inner_sequence_dims or {}), **gru} or None, seed=seed,
    )
    inner = train_network(
        inner_model, data.inner_train, data.y_inner_train, candidate, training, seed,
        holdout=(data.holdout, data.y_holdout), device=device,
    )
    model, kwargs = build_network(
        data.preprocessor, candidate, settings, float(data.y_train.mean()),
        {**(data.sequence_dims or {}), **gru} or None, seed=seed,
    )
    outer = train_network(
        model, data.train, data.y_train, candidate, training, seed, epochs=inner.best_epoch, device=device,
    )
    diagnostics = {
        "selected_epochs": inner.best_epoch,
        "inner_epochs_run": inner.epochs_run,
        "best_holdout_log_loss": float(min(inner.holdout_losses)),
        "refit_epochs": outer.epochs_run,
        "inner_training_loss": inner.train_losses,
        "inner_holdout_loss": inner.holdout_losses,
        "refit_training_loss": outer.train_losses,
    }
    return outer.model, kwargs, diagnostics


def train_neural_benchmark(
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
    *,
    device: str = "cpu",
    sequences=None,
) -> dict:
    candidates = sample_candidates(settings, budget, seed, sequence=sequences is not None)
    trials = [{"trial": index, "params": c, "folds": {}} for index, c in enumerate(candidates, start=1)]
    stored: dict[int, dict[str, np.ndarray]] = {trial["trial"]: {} for trial in trials}
    evaluation_frames: dict[str, pd.DataFrame] = {}
    batch = settings["training"]["eval_batch_size"]
    for fold in folds:
        data = prepare_fold(features, metadata, fold, settings, validation=True, sequences=sequences)
        evaluation_frames[fold.fold_id] = data.evaluation_frame
        for trial in trials:
            started = time.monotonic()
            model, _, diagnostics = fit_candidate(data, settings, trial["params"], seed, device)
            scores = predict_proba(model, data.evaluation, batch, device)
            metrics = average_precision_by_lead(data.evaluation_frame, scores)
            trial["folds"][fold.fold_id] = {"average_precision_by_lead": metrics, **diagnostics}
            stored[trial["trial"]][fold.fold_id] = scores.astype(np.float32)
            LOGGER.info("%s / %s / %s trial %d/%d: mean AP %.4f, %d refit epochs (%.1fs)",
                        features.name, model_name, fold.fold_id, trial["trial"], budget,
                        float(np.mean(list(metrics.values()))), diagnostics["refit_epochs"], time.monotonic() - started)
        LOGGER.info("%s / %s / %s finished %d trials", features.name, model_name, fold.fold_id, budget)
        # Release this fold's arrays before the next, larger window is built.
        data = None
        gc.collect()

    for trial in trials:
        values = [s for f in trial["folds"].values() for s in f["average_precision_by_lead"].values()]
        trial["mean_validation_average_precision"] = float(np.mean(values))
    best = max(trials, key=lambda trial: trial["mean_validation_average_precision"])

    stem = f"{features.name}_{model_name}"
    prediction_file = f"{stem}_validation_predictions.csv"
    with (stage / prediction_file).open("w", encoding="utf-8", newline="") as stream:
        for fold in folds:
            evaluation = evaluation_frames[fold.fold_id]
            prediction = evaluation.rename(columns={LABEL: "y_true"}).assign(
                fold_id=fold.fold_id, split="validation", model_name=model_name, run_id=run_id,
                y_probability=stored[best["trial"]][fold.fold_id].astype(np.float64),
            )
            prediction[PREDICTION_COLUMNS].to_csv(stream, index=False, header=fold is folds[0])

    data = prepare_fold(features, metadata, final, settings, validation=False, sequences=sequences)
    model, kwargs, diagnostics = fit_candidate(data, settings, best["params"], seed, device)
    model_file = f"{stem}.pt"
    save_checkpoint(
        stage / model_file, model.cpu(), kwargs, data.preprocessor, sequence_state=data.sequence_state,
        metadata={"run_id": run_id, "model_name": model_name, "feature_set": features.name,
                  "fit_cutoff_utc": final.fit_cutoff_utc.isoformat(), "params": best["params"]},
    )
    reloaded = NeuralScorer(stage / model_file)
    if reloaded.allowlist != features.allowlist:
        raise AssertionError("Saved checkpoint does not keep the feature order")
    check = predict_proba(model, take(data.train, np.arange(min(256, len(data.y_train)))), batch)
    reload_check = predict_proba(reloaded.model, take(data.train, np.arange(min(256, len(data.y_train)))), batch)
    if not np.allclose(check, reload_check, atol=1e-6, rtol=0):
        raise AssertionError("Reloaded checkpoint does not reproduce the trained probabilities")
    return {
        "model_name": model_name,
        "feature_set": features.name,
        "feature_allowlist": features.allowlist,
        "selected_trial": best["trial"],
        "selected_params": best["params"],
        "mean_validation_average_precision": best["mean_validation_average_precision"],
        "fit_cutoff_utc": final.fit_cutoff_utc.isoformat(),
        "final_fit_training_prevalence": float(data.y_train.mean()),
        "final_fit_diagnostics": diagnostics,
        "validation_predictions": prediction_file,
        "model_file": model_file,
        "reload_check_max_abs_diff": float(np.max(np.abs(check - reload_check))),
        "trials": trials,
    }


def load_protocol_and_settings(settings_path: Path | None, trial_limit: int | None):
    protocol = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    settings = json.loads(Path(settings_path or SETTINGS_PATH).read_text(encoding="utf-8"))
    baseline = json.loads(STEP5_SETTINGS_PATH.read_text(encoding="utf-8"))
    if settings["inner_holdout_days"] != baseline["inner_holdout_days"]:
        raise ValueError("The neural early-stopping holdout must match the Step 5 baselines")
    selection = protocol["model_selection"]
    full_budget = selection["trials_per_model_per_feature_set"]
    budget = trial_limit if trial_limit is not None else full_budget
    if not 1 <= budget <= full_budget:
        raise ValueError(f"Trial limit must be between 1 and {full_budget}")
    training = settings["training"]
    if min(training["max_epochs"], training["patience_epochs"], training["eval_batch_size"]) < 1:
        raise ValueError("Training limits must be positive")
    if training.get("threads"):
        torch.set_num_threads(int(training["threads"]))
    return protocol, settings, budget, full_budget


def provenance(protocol: dict, settings: dict, source_fingerprints: dict, budget: int, full_budget: int,
               code_files: tuple[Path, ...], device: str) -> dict:
    return {
        "protocol_version": protocol["contract_version"],
        "protocol_sha256": file_fingerprint(CONTRACT_PATH),
        "raw_data_fingerprints": source_fingerprints,
        "fold_manifest_version": FOLD_MANIFEST_VERSION,
        "model_config": settings,
        "seed": protocol["model_selection"]["seed"],
        "device": device,
        "package_versions": {
            "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "code_sha256": {path.name: file_fingerprint(path) for path in code_files},
        "trials_per_model": budget,
        "full_protocol_budget": full_budget,
        "protocol_budget_completed": budget == full_budget,
        "selection_metric": protocol["model_selection"]["primary_score"],
    }


NEURAL_CODE = tuple(
    Path(__file__).with_name(name) for name in ("train_step6.py", "fit.py", "networks.py", "preprocess.py")
)


def run_step6(
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    settings_path: Path | None = None,
    *,
    legacy_features: Path | None = None,
    legacy_manifest: Path | None = None,
    trial_limit: int | None = None,
    device: str = "cpu",
) -> dict:
    """Run the walk-forward folds and publish the selected MLPs and predictions."""
    dataset = Path(dataset_dir or config.PROCESSED_DATA_DIR / "step4")
    output = Path(output_dir or config.PROJECT_ROOT / "artifacts" / "step6")
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose a new output directory.")
    if (legacy_features is None) != (legacy_manifest is None):
        raise ValueError("Supply both legacy feature and manifest paths for the legacy comparison")
    protocol, settings, budget, full_budget = load_protocol_and_settings(settings_path, trial_limit)
    seed = protocol["model_selection"]["seed"]
    features, metadata, source_fingerprints, _ = read_v2_features(dataset, protocol)
    folds, final = make_folds(metadata, protocol)
    benchmarks = [features]
    if legacy_features is not None:
        benchmarks.append(read_legacy_features(Path(legacy_features), Path(legacy_manifest), features, protocol))

    base = provenance(protocol, settings, source_fingerprints, budget, full_budget, NEURAL_CODE, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".step6-", dir=output.parent) as work:
        stage = Path(work) / "neural"
        stage.mkdir()
        fold_file = stage / "fold_manifest.csv"
        write_fold_manifest(fold_file, metadata, folds, final)
        base["fold_manifest_sha256"] = file_fingerprint(fold_file)
        report = {
            **base,
            "code_commit": code_commit(),
            "fold_manifest": fold_file.name,
            "folds": [
                {"fold_id": fold.fold_id, "fit_cutoff_utc": fold.fit_cutoff_utc.isoformat(),
                 "train_flights": len(fold.train_ids), "evaluation_flights": len(fold.evaluation_ids)}
                for fold in [*folds, final]
            ],
            "feature_sets": {},
        }
        for item in benchmarks:
            manifest = {
                "feature_set": item.name,
                "feature_manifest_version": item.manifest_version,
                "feature_file_sha256": item.file_fingerprints,
                "feature_allowlist": item.allowlist,
            }
            run_hash = hashlib.sha256(json.dumps({**base, **manifest}, sort_keys=True).encode()).hexdigest()
            run_id = f"step6-{item.name}-{run_hash[:12]}"
            report["feature_sets"][item.name] = {
                **manifest,
                "run_id": run_id,
                "models": {
                    "embedding_mlp": train_neural_benchmark(
                        stage, item, metadata, folds, final, "embedding_mlp", settings, budget, seed, run_id,
                        device=device,
                    )
                },
            }
        write_json(stage / "run_metadata.json", report)
        stage.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Step 6 embedding MLP on the walk-forward folds")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--settings-path", type=Path)
    parser.add_argument("--legacy-features", type=Path)
    parser.add_argument("--legacy-manifest", type=Path)
    parser.add_argument("--trial-limit", type=int, help="Reduced development run for smoke testing")
    parser.add_argument("--device", default="cpu", help="cpu (default, reproducible), mps or cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_step6(**vars(args))
    output = args.output_dir or config.PROJECT_ROOT / "artifacts" / "step6"
    print(f"Wrote {len(result['feature_sets'])} feature sets to {output}")


if __name__ == "__main__":
    main()
