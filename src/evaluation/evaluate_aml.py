"""Reproducible Phase 5 AML evaluation of frozen Phase 3 pipelines.

Validation is divided chronologically: Sep 1–24 for Phase 3 XGBoost tuning,
Sep 25–Oct 12 for calibrator fitting, and Oct 13–31 for calibration assessment
and operating-threshold selection. Test labels never enter those decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.data import config as data_config
from src.evaluation.calibration import fit_probability_calibrator, plot_pr_and_roc_curves, plot_reliability_diagrams
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.threshold import select_operating_thresholds
from src.models.classical.train import assign_split

ROOT = data_config.PROJECT_ROOT
FEATURE_TABLE = data_config.PROCESSED_DATA_DIR / "phase2_features.csv"
FEATURE_MANIFEST = data_config.PROCESSED_DATA_DIR / "phase2_feature_manifest.json"
CLASSICAL_ARTIFACTS = ROOT / "artifacts/classical"
EVALUATION_ARTIFACTS = ROOT / "artifacts/predictions"
REPORTS_DIR = data_config.REPORTS_DIR
FIGURES_DIR = REPORTS_DIR / "figures"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_inputs(feature_path: Path, manifest_path: Path, artifacts_dir: Path):
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads((artifacts_dir / "run_metadata.json").read_text())
    if _sha256(feature_path) != metadata["feature_table_sha256"]:
        raise ValueError("Phase 3 model artifact does not match this feature table")
    if manifest["manifest_sha256"] != metadata["feature_manifest_sha256"]:
        raise ValueError("Phase 3 model artifact does not match this feature manifest")
    df = pd.read_csv(feature_path, low_memory=False)
    split = pd.read_csv(artifacts_dir / "split_manifest.csv", usecols=["flight_row_id", "sched_date", "split"])
    if df.flight_row_id.duplicated().any() or split.flight_row_id.duplicated().any():
        raise ValueError("duplicate flight IDs")
    if set(df.flight_row_id) != set(split.flight_row_id):
        raise ValueError("split manifest and feature table contain different flights")
    df = df.merge(split, on="flight_row_id", how="left", validate="one_to_one", suffixes=("", "_split"))
    if not (df.sched_date == df.sched_date_split).all() or not (df.split == assign_split(df.sched_date)).all():
        raise ValueError("split manifest date or assignment mismatch")
    if df.substantial_delay.isna().any() or df["split"].isna().any():
        raise ValueError("unknown labels or split assignments")
    return df, manifest, metadata


def _prediction_frame(ids, labels, probabilities, model_name, run_id, threshold):
    out = pd.DataFrame({
        "flight_row_id": ids.to_numpy(), "y_true": labels,
        "y_probability": probabilities, "y_pred": (probabilities >= threshold).astype(int),
        "split": "test", "model_name": model_name, "run_id": run_id,
    })
    if out.flight_row_id.duplicated().any() or not np.isfinite(out.y_probability).all() or not out.y_probability.between(0, 1).all():
        raise ValueError("invalid test prediction artifact")
    return out


def _verify_saved_validation(path: Path, val: pd.DataFrame, probability: np.ndarray, run_id: str) -> None:
    saved = pd.read_csv(path)
    expected = pd.DataFrame({"flight_row_id": val.flight_row_id.to_numpy(), "y_true": val.substantial_delay.to_numpy(dtype=int), "probability": probability})
    if saved.flight_row_id.duplicated().any() or len(saved) != len(expected) or set(saved.run_id) != {run_id} or set(saved.split) != {"validation"}:
        raise ValueError(f"stale validation prediction artifact: {path}")
    joined = expected.merge(saved, on="flight_row_id", validate="one_to_one")
    if len(joined) != len(expected) or not np.array_equal(joined.y_true_x, joined.y_true_y) or not np.allclose(joined.probability, joined.y_probability, atol=1e-6, rtol=0):
        raise ValueError(f"saved validation predictions do not match model: {path}")


def run_aml_evaluation(
    feature_path: Path = FEATURE_TABLE,
    manifest_path: Path = FEATURE_MANIFEST,
    artifacts_dir: Path = CLASSICAL_ARTIFACTS,
    eval_artifacts_dir: Path = EVALUATION_ARTIFACTS,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    precision_floor: float = 0.35,
) -> dict:
    """Fit validation calibrators, freeze decisions, and score a common test set."""
    feature_path, manifest_path, artifacts_dir = map(Path, (feature_path, manifest_path, artifacts_dir))
    eval_artifacts_dir, reports_dir, figures_dir = map(Path, (eval_artifacts_dir, reports_dir, figures_dir))
    df, manifest, metadata = _check_inputs(feature_path, manifest_path, artifacts_dir)
    val = df.loc[df.split.eq("validation")].copy()
    test = df.loc[df.split.eq("test")].copy()
    val_dates = pd.to_datetime(val.sched_date, utc=True)
    cal = val_dates.ge(pd.Timestamp("2013-09-25", tz="UTC")) & val_dates.lt(pd.Timestamp("2013-10-13", tz="UTC"))
    assess = val_dates.ge(pd.Timestamp("2013-10-13", tz="UTC"))
    if not cal.any() or not assess.any() or metadata.get("xgboost_tuning_window", {}).get("end") != "2013-09-24":
        raise ValueError("Phase 3 model was not selected on the reserved tuning window")
    allowlist = manifest["allowlist"]
    models = {
        "Logistic Regression": joblib.load(artifacts_dir / "logistic_regression.joblib"),
        "XGBoost": joblib.load(artifacts_dir / "xgboost.joblib"),
    }
    y_val = val.substantial_delay.to_numpy(dtype=int)
    val_results, decisions, frozen = {}, {}, {}
    selected_val = {}
    for name, model in models.items():
        raw = model.predict_proba(val[allowlist])[:, 1]
        slug = name.lower().replace(" ", "_")
        saved_name = "logistic_validation_predictions.csv" if name == "Logistic Regression" else "xgboost_validation_predictions.csv"
        _verify_saved_validation(artifacts_dir / saved_name, val, raw, metadata["run_id"])
        calibrator = fit_probability_calibrator(y_val[cal], raw[cal])
        calibrated = calibrator.predict(raw[assess])
        raw_brier = float(brier_score_loss(y_val[assess], raw[assess]))
        calibrated_brier = float(brier_score_loss(y_val[assess], calibrated))
        # A meaningful improvement on a separate validation slice is required.
        use_calibrated = calibrated_brier < raw_brier - 0.001
        chosen = calibrated if use_calibrated else raw[assess]
        decision = select_operating_thresholds(y_val[assess], chosen, precision_floor=precision_floor)
        selected_val[name] = (y_val[assess], chosen)
        decisions[name] = decision
        val_results[name] = {
            "full_validation_raw": evaluate_predictions(y_val, raw),
            "assessment_selected": evaluate_predictions(y_val[assess], chosen, decision["best_f1_requirement"]["threshold"]),
            "calibration": {
                "fit_rows": int(cal.sum()), "assessment_rows": int(assess.sum()),
                "assessment_brier_raw": raw_brier, "assessment_brier_isotonic": calibrated_brier,
                "selected": "isotonic" if use_calibrated else "raw",
            },
        }
        frozen[name] = {"model": model, "calibrator": calibrator if use_calibrated else None, "thresholds": decision}

    # All fitting and selection above is complete before test outcomes are read.
    y_test = test.substantial_delay.to_numpy(dtype=int)
    test_results, selected_test = {}, {}
    eval_artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, bundle in frozen.items():
        raw = bundle["model"].predict_proba(test[allowlist])[:, 1]
        probs = bundle["calibrator"].predict(raw) if bundle["calibrator"] is not None else raw
        decision = bundle["thresholds"]
        f1_threshold = decision["best_f1_requirement"]["threshold"]
        floor_threshold = decision["precision_requirement"]["threshold"]
        test_results[name] = {
            "at_best_f1": evaluate_predictions(y_test, probs, f1_threshold),
            "at_precision_floor": evaluate_predictions(y_test, probs, floor_threshold) if floor_threshold is not None else None,
        }
        selected_test[name] = (y_test, probs)
        slug = name.lower().replace(" ", "_")
        _prediction_frame(test.flight_row_id, y_test, probs, slug, metadata["run_id"], f1_threshold).to_csv(eval_artifacts_dir / f"{slug}_test_predictions.csv", index=False)
        joblib.dump({"model": bundle["model"], "calibrator": bundle["calibrator"], "thresholds": decision}, eval_artifacts_dir / f"{slug}_scorer.joblib")

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagrams(selected_val, "Validation assessment (Oct 13–31)", figures_dir / "validation_reliability_diagram.png")
    plot_pr_and_roc_curves(selected_val, "Validation assessment (Oct 13–31)", figures_dir / "validation_pr_roc_curves.png")
    plot_reliability_diagrams(selected_test, "Test (Nov–Dec)", figures_dir / "test_reliability_diagram.png")
    plot_pr_and_roc_curves(selected_test, "Test (Nov–Dec)", figures_dir / "test_pr_roc_curves.png")
    report = {
        "evaluation_phase": 5, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase3_run_id": metadata["run_id"], "feature_table_sha256": _sha256(feature_path),
        "model_sha256": {"Logistic Regression": _sha256(artifacts_dir / "logistic_regression.joblib"), "XGBoost": _sha256(artifacts_dir / "xgboost.joblib")},
        "evaluation_code_sha256": _sha256(Path(__file__)),
        "feature_manifest_sha256": manifest["manifest_sha256"],
        "precision_floor": precision_floor, "floor_status": "exploratory; not stakeholder approved",
        "holdout_status": "previous repository test results were viewed; this rerun is retrospective, not a pristine first test",
        "split_summary": {
            "train_rows": int(df.split.eq("train").sum()), "tuning_rows": int((~cal & ~assess).sum()),
            "calibration_rows": int(cal.sum()), "assessment_rows": int(assess.sum()),
            "validation_rows": len(val), "test_rows": len(test),
            "train_prevalence": float(df.loc[df.split.eq("train"), "substantial_delay"].mean()),
            "validation_prevalence": float(np.mean(y_val)), "test_prevalence": float(np.mean(y_test)),
        },
        "validation_decisions": decisions, "validation_metrics": val_results, "test_metrics": test_results,
        "figures": {"validation_reliability": "reports/figures/validation_reliability_diagram.png", "validation_curves": "reports/figures/validation_pr_roc_curves.png", "test_reliability": "reports/figures/test_reliability_diagram.png", "test_curves": "reports/figures/test_pr_roc_curves.png"},
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "phase5_aml_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    (reports_dir / "phase5_aml_evaluation.md").write_text(_evaluation_markdown(report))
    (reports_dir / "identify_the_gap.md").write_text(_gap_markdown(report))
    return report


def _evaluation_markdown(report: dict) -> str:
    lines = ["# Phase 5 — AML evaluation", "", f"Run: `{report['phase3_run_id']}`; feature SHA-256: `{report['feature_table_sha256']}`.", "", f"The {report['precision_floor']:.0%} precision floor is exploratory, with at least 5% validation recall and 100 alerts required. The test period was already viewed in earlier repository work, so this is a retrospective evaluation, not a pristine first holdout.", "", "Validation uses Sep 1–24 for XGBoost tuning, Sep 25–Oct 12 to fit calibration, and Oct 13–31 to assess calibration and choose thresholds. Test outcomes do not affect these decisions.", "", "| Model | Full val PR-AUC (raw) | Full val ROC-AUC (raw) | Assessment Brier raw | Assessment Brier isotonic | Selected probabilities | Validation precision | Validation recall | Test PR-AUC | Test ROC-AUC | Test Brier | Test precision | Test recall |", "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("Logistic Regression", "XGBoost"):
        val, decision, test = report["validation_metrics"][name], report["validation_decisions"][name], report["test_metrics"][name]
        c, f = val["calibration"], decision["precision_requirement"]
        t = test["at_precision_floor"]
        fmt = lambda x: f"{x:.4f}" if x is not None else "infeasible"
        lines.append(f"| {name} | {val['full_validation_raw']['pr_auc']:.4f} | {val['full_validation_raw']['roc_auc']:.4f} | {c['assessment_brier_raw']:.4f} | {c['assessment_brier_isotonic']:.4f} | {c['selected']} | {fmt(f['validation_precision'])} | {fmt(f['validation_recall'])} | {test['at_best_f1']['pr_auc']:.4f} | {test['at_best_f1']['roc_auc']:.4f} | {test['at_best_f1']['brier_score']:.4f} | {fmt(t['precision'] if t else None)} | {fmt(t['recall'] if t else None)} |")
    lines += ["", "## Validation-selected F1 reference", "", "| Model | Validation precision | Validation recall | Test precision | Test recall | Test F1 | Test false positives | Test false negatives |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("Logistic Regression", "XGBoost"):
        v = report["validation_metrics"][name]["assessment_selected"]
        t = report["test_metrics"][name]["at_best_f1"]
        lines.append(f"| {name} | {v['precision']:.4f} | {v['recall']:.4f} | {t['precision']:.4f} | {t['recall']:.4f} | {t['f1']:.4f} | {t['confusion_matrix']['fp']:,} | {t['confusion_matrix']['fn']:,} |")
    lines += ["", f"The test precision and recall columns use the validation-selected {report['precision_floor']:.0%} floor threshold where feasible. Infeasible means no threshold met the floor plus the minimum recall and alert-count guards. See the JSON for F1 reference thresholds, confusion matrices, and exact decisions.", "", "Figures: [validation reliability](figures/validation_reliability_diagram.png), [validation PR/ROC](figures/validation_pr_roc_curves.png), [test reliability](figures/test_reliability_diagram.png), [test PR/ROC](figures/test_pr_roc_curves.png).", ""]
    return "\n".join(lines)


def _gap_markdown(report: dict) -> str:
    lr = report["test_metrics"]["Logistic Regression"]["at_best_f1"]
    xg = report["test_metrics"]["XGBoost"]["at_best_f1"]
    floor_feasible = {name: report["validation_decisions"][name]["precision_requirement"]["floor_achieved"] for name in ("Logistic Regression", "XGBoost")}
    floor_description = "infeasible for both models" if not any(floor_feasible.values()) else ", ".join(f"{name}: {'feasible' if floor_feasible[name] else 'infeasible'}" for name in floor_feasible)
    lr_val = report["validation_metrics"]["Logistic Regression"]["assessment_selected"]
    xg_val = report["validation_metrics"]["XGBoost"]["assessment_selected"]
    lines = ["# Identify the Gap — AML baselines", "", "This analysis uses the frozen Phase 3 models and the Phase 5 metrics from the same feature-table hash. The existing repository had already exposed test results before this rerun; treat these as retrospective evidence, not a new pristine holdout.", "", f"On the Nov–Dec test period, delay prevalence is {report['split_summary']['test_prevalence']:.1%}. Logistic Regression PR-AUC is {lr['pr_auc']:.4f} and XGBoost PR-AUC is {xg['pr_auc']:.4f}; ROC-AUC is {lr['roc_auc']:.4f} and {xg['roc_auc']:.4f}, respectively. These results show limited discrimination, but one year and two models do not establish a performance ceiling.", "", f"On the threshold-selection slice, F1 reference precision/recall is {lr_val['precision']:.1%}/{lr_val['recall']:.1%} for Logistic Regression and {xg_val['precision']:.1%}/{xg_val['recall']:.1%} for XGBoost. On test, the frozen F1 points detect {lr['recall']:.1%} and {xg['recall']:.1%} of delays, with {lr['confusion_matrix']['fp']:,} and {xg['confusion_matrix']['fp']:,} false alerts. The {report['precision_floor']:.0%} precision floor is exploratory; {floor_description} under the minimum recall and alert-count guards. See the [evaluation report](phase5_aml_evaluation.md) for each model's feasibility and test outcomes; an infeasible floor is never presented as achieved.", "", "Calibration is chosen using a later validation slice than the fitting slice, then applied consistently to test probabilities when selected. Seasonal prevalence shifts from validation to test, so probability reliability and alert behavior may change. The supplied weather is retrospective observation data, not archived forecast vintages. Historical outcomes use a conservative availability bound rather than actual release timestamps. These data limits are plausible reasons for errors, not proof that a neural model will improve them.", "", "Phase 4 remains the deep learning phase; rescheduling is outside this AML analysis.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen AML baselines")
    parser.add_argument("--precision-floor", type=float, default=0.35)
    args = parser.parse_args()
    result = run_aml_evaluation(precision_floor=args.precision_floor)
    print(json.dumps({"run_id": result["phase3_run_id"], "test_metrics": result["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
