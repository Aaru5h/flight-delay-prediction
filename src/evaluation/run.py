"""Contributor 5 final comparison and report generation.

Validation decisions are frozen to disk before held-out test scoring starts.
No test label is passed to model selection or threshold selection.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/flight-delay-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data.config import PROCESSED_DATA_DIR, PROJECT_ROOT, REPORTS_DIR
from src.evaluation.metrics import calibration_bins, compute_metrics, select_max_f1_threshold
from src.evaluation.split import build_split_manifest, save_split_artifacts
from src.evaluation.validation import validate_predictions

MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "xgboost": "xgboost.joblib",
}
VALIDATION_FILES = {
    "logistic_regression": "logistic_validation_predictions.csv",
    "xgboost": "xgboost_validation_predictions.csv",
}


def _json_dump(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _prediction_frame(ids, labels, probability, model_name: str, run_id: str, split: str) -> pd.DataFrame:
    return pd.DataFrame({
        "flight_row_id": ids.to_numpy(),
        "y_true": labels.astype(int).to_numpy(),
        "y_probability": probability,
        "split": split,
        "model_name": model_name,
        "run_id": run_id,
    })


def _plot_curves(predictions: dict[str, pd.DataFrame], output_dir: Path, split: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, frame in predictions.items():
        y, p = frame["y_true"], frame["y_probability"]
        precision, recall, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(recall, precision, label=name)
        axes[1].plot(fpr, tpr, label=name)
    axes[0].axhline(next(iter(predictions.values()))["y_true"].mean(), color="grey", linestyle="--", label="prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", title=f"Precision–Recall — {split}")
    axes[1].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"ROC — {split}")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{split}_ranking_curves.png", dpi=160)
    plt.close(fig)


def _plot_reliability(predictions: dict[str, pd.DataFrame], output_dir: Path, split: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="perfect calibration")
    for name, frame in predictions.items():
        bins = calibration_bins(frame["y_true"], frame["y_probability"])
        ax.plot(bins["mean_probability"], bins["positive_rate"], marker="o", label=name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive rate", title=f"Reliability — {split}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{split}_reliability.png", dpi=160)
    plt.close(fig)


def _plot_confusion(metrics_by_model: dict, output_dir: Path, split: str) -> None:
    names = list(metrics_by_model)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        c = metrics_by_model[name]["confusion_matrix"]
        matrix = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        image = ax.imshow(matrix, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:,}", ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual", title=name)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(f"Confusion matrices — {split} (validation-selected thresholds)")
    fig.tight_layout()
    fig.savefig(output_dir / f"{split}_confusion_matrices.png", dpi=160)
    plt.close(fig)


def _subgroup_analysis(frame: pd.DataFrame, threshold: float, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = frame[column].fillna("missing")
        for value, indices in values.groupby(values).groups.items():
            group = frame.loc[indices]
            y = group["y_true"].astype(int).to_numpy()
            pred = (group["y_probability"].to_numpy() >= threshold).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            rows.append({
                "slice": column,
                "group": str(value),
                "rows": len(group),
                "positive_prevalence": float(y.mean()),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "false_positives": fp,
                "false_negatives": fn,
            })
    return pd.DataFrame(rows).sort_values(["slice", "rows"], ascending=[True, False])


def _render_report(result: dict) -> str:
    split_rows = result["split_profile"]
    validation = result["validation_metrics"]
    test = result["test_metrics"]
    lines = [
        "# Contributor 5 — Final Evaluation, Integration and Gap Analysis", "",
        "**Evaluation status:** final held-out comparison completed once under the frozen validation plan.", "",
        "## Shared chronological split", "",
        "| Split | Dates | Rows | Positive | Negative | Prevalence |", "|---|---|---:|---:|---:|---:|",
    ]
    for row in split_rows:
        lines.append(f"| {row['split']} | {row['min_date']} to {row['max_date']} | {row['rows']:,} | {row['positive']:,} | {row['negative']:,} | {row['positive_prevalence']:.4f} |")
    lines += ["", "The validation prevalence is materially lower than train and test; metrics are therefore interpreted as a chronological distribution shift, not balanced by changing dates.", "", "## Frozen validation decisions", ""]
    lines.append(f"- Primary model: **{result['selection']['primary_model']}**, selected by validation PR AUC.")
    lines.append("- Operating threshold: maximum F1 on validation for each model; no stakeholder-approved precision floor exists.")
    lines.append("- Calibration: no post-hoc calibrator fitted because no independent calibration fold was reserved. Raw probability calibration is reported with Brier score, ECE, and reliability plots.")
    lines += ["", "## Validation comparison", "", "| Model | PR AUC | ROC AUC | Brier | ECE | Threshold | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in validation.items():
        lines.append(f"| {name} | {m['pr_auc']:.6f} | {m['roc_auc']:.6f} | {m['brier_score']:.6f} | {m['ece_10_equal_width']:.6f} | {m['threshold']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} |")
    lines += ["", "## Final held-out test comparison", "", "| Model | PR AUC | ROC AUC | Brier | ECE | Frozen threshold | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in test.items():
        lines.append(f"| {name} | {m['pr_auc']:.6f} | {m['roc_auc']:.6f} | {m['brier_score']:.6f} | {m['ece_10_equal_width']:.6f} | {m['threshold']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} |")
    lines += [
        "", "XGBoost ranks better on the held-out test period even though Logistic Regression was selected on validation. The primary selection is not changed after observing test results; this reversal is evidence of temporal instability.",
        "", "## Identify the Gap", "",
        "1. **Performance gap:** ranking is useful but moderate; neither model is strong enough to support automated operational decisions without further validation.",
        "2. **Seasonal gap:** validation and test prevalence differ sharply, exposing temporal instability that a single-year dataset cannot resolve.",
        "3. **Feature gap:** weather is retrospective observation data rather than archived forecasts, and aircraft/rotation, gate, crew, network, and live congestion information are absent.",
        "4. **Target gap:** missing arrival delays are unlabeled, and the source has no reliable cancellation/diversion target.",
        "5. **Calibration gap:** probability quality varies by period. A future study should reserve a separate calibration window or use nested time-series validation before fitting a calibrator.",
        "6. **Model gap:** the comparison covers the two required tabular baselines. More complex temporal or neural models require a separate contributor-owned experiment and cannot be claimed here.",
        "7. **Generalization gap:** results cover only 2013 NYC-origin flights and do not establish performance for later years, other airports, or deployment conditions.",
        "", "## Integration and handoff", "",
        "The split, prediction schemas, model artifacts, metrics, figures, and report are reproducible by code. Test labels were not used for model or threshold selection. Subgroup results include denominators in `reports/evaluation/subgroup_analysis.csv`; they are diagnostic and not causal or fairness claims.",
    ]
    return "\n".join(lines) + "\n"


def run_evaluation(
    feature_path: Path = PROCESSED_DATA_DIR / "phase2_features.csv",
    manifest_path: Path = PROCESSED_DATA_DIR / "phase2_feature_manifest.json",
    model_dir: Path = PROJECT_ROOT / "artifacts" / "classical",
    output_dir: Path = REPORTS_DIR / "evaluation",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest = json.loads(manifest_path.read_text())
    frame = pd.read_csv(feature_path, low_memory=False)
    split_manifest, split_summary = build_split_manifest(frame)
    save_split_artifacts(frame, output_dir)
    labels = frame[["flight_row_id", "substantial_delay"]]

    validation_predictions = {}
    thresholds = {}
    validation_metrics = {}
    run_metadata = json.loads((model_dir / "run_metadata.json").read_text())
    for model_name, filename in VALIDATION_FILES.items():
        raw = pd.read_csv(model_dir / filename)
        checked = validate_predictions(raw, split_manifest, labels, "validation")
        decision = select_max_f1_threshold(checked["y_true"], checked["y_probability"])
        thresholds[model_name] = decision
        validation_metrics[model_name] = compute_metrics(
            checked["y_true"], checked["y_probability"], decision["threshold"]
        )
        validation_predictions[model_name] = checked

    primary_model = max(validation_metrics, key=lambda name: validation_metrics[name]["pr_auc"])
    evaluation_plan = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_selection_rule": "highest validation PR AUC on identical validation IDs",
        "primary_model": primary_model,
        "threshold_rule": "maximum validation F1 per model",
        "precision_floor": {"status": "not stakeholder approved", "value": None},
        "calibration": {"method": "none", "reason": "no independent calibration fold reserved; report raw calibration diagnostics"},
        "thresholds": thresholds,
        "feature_manifest_version": feature_manifest["manifest_version"],
        "split_manifest_version": run_metadata["split_manifest_version"],
    }
    _json_dump(output_dir / "evaluation_plan_frozen_before_test.json", evaluation_plan)

    test_rows = split_manifest.loc[split_manifest["split"].eq("test"), ["flight_row_id"]].merge(
        frame, on="flight_row_id", validate="one_to_one"
    )
    test_predictions = {}
    test_metrics = {}
    for model_name, model_filename in MODEL_FILES.items():
        model = joblib.load(model_dir / model_filename)
        probability = model.predict_proba(test_rows[feature_manifest["allowlist"]])[:, 1]
        predictions = _prediction_frame(
            test_rows["flight_row_id"], test_rows["substantial_delay"], probability,
            model_name, run_metadata["run_id"], "test",
        )
        checked = validate_predictions(predictions, split_manifest, labels, "test")
        checked.to_csv(output_dir / f"{model_name}_test_predictions.csv", index=False)
        test_predictions[model_name] = checked
        test_metrics[model_name] = compute_metrics(
            checked["y_true"], checked["y_probability"], thresholds[model_name]["threshold"]
        )

    _plot_curves(validation_predictions, figures_dir, "validation")
    _plot_curves(test_predictions, figures_dir, "test")
    _plot_reliability(validation_predictions, figures_dir, "validation")
    _plot_reliability(test_predictions, figures_dir, "test")
    _plot_confusion(test_metrics, figures_dir, "test")

    primary = test_predictions[primary_model].merge(
        frame[["flight_row_id", "carrier", "origin", "sched_dep_hour", "weather_stale_or_missing"]],
        on="flight_row_id", validate="one_to_one",
    )
    subgroup = _subgroup_analysis(
        primary, thresholds[primary_model]["threshold"],
        ["carrier", "origin", "sched_dep_hour", "weather_stale_or_missing"],
    )
    subgroup.to_csv(output_dir / "subgroup_analysis.csv", index=False)

    split_records = []
    for row in split_summary.to_dict("records"):
        row["min_date"] = row["min_date"].strftime("%Y-%m-%d")
        row["max_date"] = row["max_date"].strftime("%Y-%m-%d")
        split_records.append(row)
    result = {
        "phase": 5,
        "owner": "Contributor 5",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_profile": split_records,
        "selection": evaluation_plan,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "population_check": "both models evaluated on identical IDs in each split",
        "limitations": [
            "one year of NYC-origin flights", "retrospective weather observations",
            "incomplete destination references", "no approved operational precision floor",
        ],
    }
    _json_dump(output_dir / "evaluation_results.json", result)
    (REPORTS_DIR / "contributor5_final_evaluation.md").write_text(_render_report(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Contributor 5 final evaluation")
    parser.add_argument("--feature-path", type=Path, default=PROCESSED_DATA_DIR / "phase2_features.csv")
    parser.add_argument("--manifest-path", type=Path, default=PROCESSED_DATA_DIR / "phase2_feature_manifest.json")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "classical")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "evaluation")
    args = parser.parse_args()
    result = run_evaluation(args.feature_path, args.manifest_path, args.model_dir, args.output_dir)
    print(json.dumps({"primary_model": result["selection"]["primary_model"], "validation": result["validation_metrics"], "test": result["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
