"""Step 8: freeze per-lead calibration and a deterministic alert-budget policy."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.data import config
from src.data.loading import file_fingerprint
from src.evaluation.artifacts_v2 import CONTRACT, SETTINGS, load_handoff, read_json, write_json
from src.evaluation.metrics_v2 import alert_budget

LOGGER = logging.getLogger(__name__)


def fit_calibrator(frame, method, minimum_rows):
    if len(frame) < minimum_rows or frame.y_true.nunique() < 2:
        raise ValueError("Insufficient known, available calibration labels from both classes")
    if method == "identity":
        return None
    if method != "isotonic":
        raise ValueError("Calibration method must be predeclared as isotonic or identity")
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(frame.y_probability, frame.y_true)


def apply_calibrator(calibrator, probabilities):
    p = np.asarray(probabilities, dtype=float)
    return p.copy() if calibrator is None else calibrator.predict(p)


def run_step8(dataset_dir=None, step5_dir=None, step6_dir=None, step7_dir=None, output_dir=None,
              settings_path=None, *, allow_reduced_budget=False, archive_path=None):
    root = config.PROJECT_ROOT / "artifacts"
    dataset = Path(dataset_dir or config.PROCESSED_DATA_DIR / "step4")
    output = Path(output_dir or root / "step8")
    if output.exists():
        raise FileExistsError(f"{output} already exists; frozen decisions are immutable")
    settings_file = Path(settings_path or SETTINGS)
    settings = read_json(settings_file)
    import_audit = None
    if archive_path is not None:
        if not allow_reduced_budget:
            raise ValueError("Imported prediction runs require explicit reduced-budget evaluation")
        from src.evaluation.archive_handoff import load_archive_handoff
        protocol, features, population, folds, final, models, omissions, import_audit = load_archive_handoff(
            archive_path, dataset, output.parent / "supplied_predictions")
    else:
        protocol, features, population, folds, final, models, omissions = load_handoff(
            dataset, step5_dir or root / "step5", step6_dir or root / "step6", step7_dir,
            allow_reduced_budget=allow_reduced_budget,
        )
    if settings["alert_budget_fraction"] != protocol["evaluation"]["alert_budget_fraction"]:
        raise ValueError("Alert budget does not match the frozen protocol")
    if settings["calibration_month"] != protocol["validation"]["validation_months"][-1]:
        raise ValueError("Calibration must use the most recent out-of-fold month")
    if min(settings["minimum_calibration_rows"], settings["ece_bins"], settings["bootstrap_resamples"], settings["permutation_repeats"]) < 1:
        raise ValueError("Evaluation counts must be positive")
    bootstrap = protocol["evaluation"]["bootstrap"]
    changed_evaluation = (settings["bootstrap_resamples"] != bootstrap["resamples"]
                          or settings["confidence_level"] != bootstrap["confidence_level"]
                          or settings["bootstrap_seed"] != bootstrap["seed"]
                          or settings["ece_bins"] != protocol["evaluation"]["ece_bins"])
    if changed_evaluation and not allow_reduced_budget:
        raise ValueError("Evaluation settings must match the protocol; reduced settings are development only")
    audit_tests = ["tests/test_step3_asof.py", "tests/test_step4_multi_lead.py", "tests/test_step7_sequence.py"]
    audit = subprocess.run([sys.executable, "-m", "pytest", "-q", *audit_tests], cwd=config.PROJECT_ROOT,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                           capture_output=True, text=True)
    if audit.returncode:
        raise RuntimeError("Cutoff and sequence tests failed before freezing:\n" + audit.stdout[-3000:] + audit.stderr[-1000:])
    frozen = {
        "step": 8, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": protocol["contract_version"], "settings": settings,
        "raw_data_fingerprints": models[0].run_metadata["raw_data_fingerprints"],
        "fold_manifest_version": models[0].run_metadata["fold_manifest_version"],
        "dataset_dir": str(dataset.resolve()), "models": {}, "omissions": omissions,
        "fit_cutoff_utc": final.fit_cutoff_utc.isoformat(),
        "holdout_status": "Nov-Dec 2013 was previously viewed; retrospective evidence only",
        "development_run": changed_evaluation or any(not m.run_metadata["protocol_budget_completed"] for m in models),
        "calibration_policy": "Fixed method; fit October OOF labels available by final fit cutoff. No retrospective outcomes used.",
        "validation_policy": "Raw OOF ranking; calibration diagnostics use earlier-month fits only. OOF predictions also selected hyperparameters, so these are development estimates, not nested validation.",
        "operating_policy": "For each reported lead-time cohort rank calibrated probabilities and flag floor(0.10*n); tie-break by flight_row_id. Monthly/subgroup tables recompute a local budget and are not one global dispatch schedule.",
        "leakage_tests": {"passed": True, "tests": audit_tests, "result": audit.stdout.strip()},
        "files": {str(CONTRACT.resolve()): file_fingerprint(CONTRACT),
                  str(settings_file.resolve()): file_fingerprint(settings_file)},
    }
    if import_audit is not None:
        frozen["import_audit"] = import_audit
        frozen["files"][str(Path(archive_path).resolve())] = import_audit["archive_sha256"]
    for path in config.PROJECT_ROOT.joinpath("src").rglob("*.py"):
        frozen["files"][str(path.resolve())] = file_fingerprint(path)
    for name, digest in features.file_fingerprints.items():
        frozen["files"][str((dataset / name).resolve())] = digest
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".step8-", dir=output.parent) as work:
        stage = Path(work) / "decisions"
        stage.mkdir()
        calibrators = {}
        for model in models:
            prediction = model.predictions
            calibrators[model.key], decisions = {}, {}
            prequential = prediction.copy()
            labelled = features.frame.loc[features.frame.lead_time_minutes.eq(60)].set_index("flight_row_id").substantial_delay
            priors = {f.fold_id: float(labelled.reindex(f.train_ids).mean()) for f in folds}
            prequential["reference_prevalence"] = prequential.fold_id.map(priors)
            prequential["calibrated_probability"] = prequential.y_probability
            prequential["calibration_source"] = "identity_no_previous_month"
            for lead in sorted(prediction.lead_time_minutes.unique()):
                rows = prediction.loc[prediction.lead_time_minutes.eq(lead)]
                fit = rows.loc[rows.fold_id.eq(settings["calibration_month"]) & rows.outcome_available_utc.le(final.fit_cutoff_utc)]
                calibrator = fit_calibrator(fit, settings["calibration_method"], settings["minimum_calibration_rows"])
                calibrators[model.key][int(lead)] = calibrator
                budget, _ = alert_budget(fit.y_true, apply_calibrator(calibrator, fit.y_probability), fit.flight_row_id, settings["alert_budget_fraction"])
                decisions[str(lead)] = {
                    "method": settings["calibration_method"], "fit_rows": len(fit),
                    "fit_month": settings["calibration_month"], "fit_cutoff_utc": final.fit_cutoff_utc.isoformat(),
                    "latest_label_available_utc": fit.outcome_available_utc.max().isoformat(),
                    "fit_population_sha256": hashlib.sha256("\n".join(sorted(fit.flight_row_id)).encode()).hexdigest(),
                    "calibration_fit_budget_reference": budget,
                    "reference_caveat": "In-sample descriptive boundary, not an independently validated fixed threshold. Evaluation uses the predeclared ranking budget.",
                }
                for fold in folds:
                    target = rows.loc[rows.fold_id.eq(fold.fold_id)]
                    earlier = rows.loc[rows.fold_id.lt(fold.fold_id) & rows.outcome_available_utc.le(fold.fit_cutoff_utc)]
                    if earlier.empty:
                        continue
                    previous_month = earlier.fold_id.max()
                    earlier = earlier.loc[earlier.fold_id.eq(previous_month)]
                    if len(earlier) < settings["minimum_calibration_rows"] or earlier.y_true.nunique() < 2:
                        continue
                    historical = fit_calibrator(earlier, settings["calibration_method"], settings["minimum_calibration_rows"])
                    prequential.loc[target.index, "calibrated_probability"] = apply_calibrator(historical, target.y_probability)
                    prequential.loc[target.index, "calibration_source"] = previous_month
            stem = model.key.replace("/", "_")
            validation_file = f"{stem}_validation.csv"
            prequential.to_csv(stage / validation_file, index=False)
            frozen["models"][model.key] = {
                "name": model.name, "feature_set": model.feature_set, "run_id": model.run_id,
                "model_path": str((model.directory / model.metadata["model_file"]).resolve()),
                "feature_allowlist": model.metadata["feature_allowlist"],
                "feature_manifest_version": model.feature_metadata["feature_manifest_version"],
                "selected_params": model.metadata["selected_params"],
                "package_versions": model.run_metadata["package_versions"],
                "reference_prevalence": model.metadata["final_fit_training_prevalence"],
                "mean_validation_average_precision": model.metadata["mean_validation_average_precision"],
                "feature_file_sha256": model.feature_metadata["feature_file_sha256"],
                "validation_file": validation_file, "calibration": decisions,
                "trials_per_model": model.run_metadata["trials_per_model"],
                "protocol_budget_completed": model.run_metadata["protocol_budget_completed"],
            }
            if "retrospective_predictions_path" in model.metadata:
                retrospective = model.metadata["retrospective_predictions_path"]
                frozen["models"][model.key]["retrospective_predictions_path"] = retrospective
                frozen["files"][retrospective] = file_fingerprint(retrospective)
            for filename in ["run_metadata.json", model.run_metadata["fold_manifest"], model.metadata["model_file"], model.metadata["validation_predictions"]]:
                path = model.directory / filename
                frozen["files"][str(path.resolve())] = file_fingerprint(path)
            for filename, expected_hash in model.feature_metadata.get("sequence_file_sha256", {}).items():
                frozen["files"][str((dataset / filename).resolve())] = expected_hash
            frozen["models"][model.key]["validation_sha256"] = file_fingerprint(stage / validation_file)
            LOGGER.info("Froze %s calibration for %d leads", model.key, len(decisions))
        joblib.dump(calibrators, stage / "calibrators.joblib")
        frozen["calibrators_sha256"] = file_fingerprint(stage / "calibrators.joblib")
        write_json(stage / "frozen_decisions.json", frozen)
        (stage / "README.md").write_text(
            "# Step 8 frozen decisions\n\nCalibration and the 10% ranking budget were frozen before retrospective scoring. "
            "See `frozen_decisions.json` for exact populations, availability checks, artifacts and settings. "
            "Validation CSVs contain strictly earlier-month calibration diagnostics; the final October calibrators are in `calibrators.joblib`.\n"
        )
        stage.rename(output)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-dir", "step5-dir", "step6-dir", "step7-dir", "output-dir", "settings-path", "archive-path"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--allow-reduced-budget", action="store_true", help="Development only; marked in every output")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_step8(**vars(parser.parse_args()))
    print(f"Froze calibration and operating policy for {len(result['models'])} models")


if __name__ == "__main__":
    main()
