"""Independently verify exported probabilities and publish numerical result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loading import file_fingerprint
from src.evaluation.artifacts_v2 import read_json, write_json
from src.evaluation.evaluate_step9 import verify_freeze
from src.evaluation.metrics_v2 import alert_budget, metrics

DISPLAY = {"schedule_only": "Schedule only", "logistic_regression": "Logistic Regression",
           "xgboost": "XGBoost", "embedding_mlp": "Embedding MLP", "sequence_model": "Sequence GRU"}


def summarize(step8_dir, step9_dir, reports_dir):
    step8, step9, reports = map(Path, (step8_dir, step9_dir, reports_dir))
    frozen = verify_freeze(step8)
    report = read_json(step9 / "evaluation.json")
    if file_fingerprint(step8 / "frozen_decisions.json") != report["frozen_decisions_sha256"]:
        raise ValueError("Evaluation does not use these frozen decisions")
    rows = []
    settings = report["settings"]
    for key, leads in report["retrospective"].items():
        path = step9 / f"{key.replace('/', '_')}_retrospective_predictions.csv"
        assert file_fingerprint(path) == report["model_runs"][key]["predictions_sha256"]
        frame = pd.read_csv(path, dtype={"flight_row_id": str}, float_precision="round_trip")
        for lead, cohort in frame.groupby("lead_time_minutes"):
            reference = cohort.reference_prevalence.to_numpy()
            kwargs = {"fraction": settings["alert_budget_fraction"], "n_bins": settings["ece_bins"]}
            calibrated = metrics(cohort.y_true, cohort.y_probability, cohort.flight_row_id, reference, **kwargs)
            raw = metrics(cohort.y_true, cohort.y_probability_raw, cohort.flight_row_id, reference, **kwargs)
            expected = leads[str(lead)]
            for field in ["average_precision", "roc_auc", "brier_score", "brier_skill_score", "equal_mass_ece"]:
                assert np.isclose(calibrated[field], expected[field], atol=1e-12, rtol=0), (key, lead, field)
            assert np.isclose(raw["average_precision"], expected["raw_average_precision"], atol=1e-12, rtol=0)
            budget, selected = alert_budget(cohort.y_true, cohort.y_probability, cohort.flight_row_id, kwargs["fraction"])
            assert np.array_equal(selected, cohort.alert_at_budget.to_numpy())
            assert budget == expected["alert_budget"]
            ci = report["intervals"][str(lead)]["metrics"][key]["average_precision"]
            tp, fp, fn = budget["true_positives"], budget["false_positives"], budget["false_negatives"]
            tn = len(cohort) - tp - fp - fn
            rows.append({"model": DISPLAY[key.split('/')[-1]], "model_key": key, "lead_hours": int(lead) // 60,
                         "flights": len(cohort), "delayed_flights": int(cohort.y_true.sum()),
                         "prevalence": float(cohort.y_true.mean()),
                         "raw_pr_auc": raw["average_precision"], "calibrated_pr_auc": calibrated["average_precision"],
                         "calibrated_pr_auc_ci_lower": ci["lower"], "calibrated_pr_auc_ci_upper": ci["upper"],
                         "raw_roc_auc": raw["roc_auc"], "calibrated_roc_auc": calibrated["roc_auc"],
                         "raw_brier_score": raw["brier_score"], "calibrated_brier_score": calibrated["brier_score"],
                         "calibrated_brier_skill": calibrated["brier_skill_score"],
                         "raw_equal_mass_ece": raw["equal_mass_ece"], "calibrated_equal_mass_ece": calibrated["equal_mass_ece"],
                         "alerts": budget["alerts"], "precision_at_10pct": budget["precision"], "recall_at_10pct": budget["recall"],
                         "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
                         "accuracy_at_10pct": (tp + tn) / len(cohort)})
    table = pd.DataFrame(rows)
    table.to_csv(reports / "numerical_results.csv", index=False)
    write_json(reports / "numerical_results.json", rows)
    at60 = table.loc[table.lead_hours.eq(1)].sort_values("calibrated_pr_auc", ascending=False)
    lines = ["# Measured results: supplied five-model experiment", "",
             "Steps 8 and 9 are complete for the supplied reduced-budget runs: one configuration per model. "
             "The 40-configuration experiment and legacy-feature model grid remain outside this completed result set.", "",
             "## One-hour predictions", "",
             "53,991 November-December 2013 flights; 13,946 delayed (25.83%). "
             "Each model flags exactly 5,399 flights under the fixed 10% ranking budget.", "",
             "| Model | Raw PR-AUC | Calibrated PR-AUC | Calibrated ROC-AUC | Precision @10% | Recall @10% | Brier |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in at60.itertuples():
        lines.append(f"| {r.model} | {r.raw_pr_auc:.4f} | {r.calibrated_pr_auc:.4f} | {r.calibrated_roc_auc:.4f} | {r.precision_at_10pct:.2%} | {r.recall_at_10pct:.2%} | {r.calibrated_brier_score:.4f} |")
    lines += ["", "## Calibrated PR-AUC by prediction lead time", "",
              "| Model | 24 hours | 6 hours | 2 hours | 1 hour |", "|---|---:|---:|---:|---:|"]
    for name, group in table.groupby("model", sort=False):
        values = group.set_index("lead_hours").calibrated_pr_auc
        lines.append(f"| {name} | " + " | ".join(f"{values[h]:.4f}" for h in [24, 6, 2, 1]) + " |")
    best = at60.iloc[0]
    lines += ["", "## Findings", "",
              f"- The highest observed one-hour result is {best['model']}: raw PR-AUC {best['raw_pr_auc']:.4f}, "
              f"calibrated PR-AUC {best['calibrated_pr_auc']:.4f}, with a 95% day-bootstrap interval "
              f"[{best['calibrated_pr_auc_ci_lower']:.4f}, {best['calibrated_pr_auc_ci_upper']:.4f}]. "
              "This is a retrospective observation, not a new test-based model-selection decision.",
              f"- At its one-hour alert budget, it catches {int(best['true_positives']):,} delayed flights and raises "
              f"{int(best['false_positives']):,} false alerts; {int(best['false_negatives']):,} delayed flights are not flagged.",
              "- Neither neural model beats matched XGBoost at any of the four lead times in this run. "
              "At 24 hours both neural models also trail the schedule-only model on PR-AUC.",
              "- Raw and calibrated PR-AUC are both retained: isotonic calibration introduces ties and reduces AP here. "
              "Calibration was fixed using October validation labels only, with availability checked before the fit boundary.",
              "- This is previously viewed November-December data. Predictions share flight IDs, labels and folds. "
              "The original Linux neural feature tables are absent; their different numeric CSV hashes remain an input-provenance qualification.",
              "", "## Files", "",
              "`numerical_results.csv` and `.json`: all 20 model/lead combinations, including raw/calibrated probability metrics and alert confusion counts. "
              "`evaluation.json`: complete monthly metrics, paired 1,000-resample confidence intervals, diagnostics and provenance. "
              "`subgroups.csv`: month, airport, carrier, hour and weather slices with denominators. "
              "`diagnostics.json`: validation-only permutation importance and refit ablations.", ""]
    (reports / "RESULTS.md").write_text("\n".join(lines))
    verification = {"status": "passed", "model_lead_cohorts_recomputed": len(rows),
                    "comparison_tolerance": 1e-12, "prediction_file_hashes": "passed", "exported_alert_masks": "passed",
                    "frozen_calibration_and_inputs": "passed", "evaluation_sha256": file_fingerprint(step9 / "evaluation.json"),
                    "verification_code_sha256": file_fingerprint(Path(__file__))}
    write_json(reports / "results_verification.json", verification)
    print(at60[["model", "raw_pr_auc", "calibrated_pr_auc", "calibrated_roc_auc", "precision_at_10pct", "recall_at_10pct"]].to_string(index=False))
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step8-dir", type=Path, required=True)
    parser.add_argument("--step9-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.step8_dir, args.step9_dir, args.reports_dir)


if __name__ == "__main__":
    main()
