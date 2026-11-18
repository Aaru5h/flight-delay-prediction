"""Step 9: evaluate frozen models, matched populations, uncertainty and errors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data import config
from src.data.loading import file_fingerprint
from src.evaluation.artifacts_v2 import CONTRACT, read_json, write_json
from src.evaluation.calibrate_step8 import apply_calibrator
from src.evaluation.metrics_v2 import alert_budget, metrics, paired_day_bootstrap

LOGGER = logging.getLogger(__name__)
KEY = ["flight_row_id", "lead_time_minutes", "fold_id"]


def verify_freeze(step8_dir):
    frozen = read_json(step8_dir / "frozen_decisions.json")
    for name, digest in frozen["files"].items():
        path = Path(name)
        if not path.is_file() or file_fingerprint(path) != digest:
            raise ValueError(f"Frozen artifact has changed or is missing: {path}")
    if file_fingerprint(step8_dir / "calibrators.joblib") != frozen["calibrators_sha256"]:
        raise ValueError("Frozen calibrators were modified")
    for model in frozen["models"].values():
        if file_fingerprint(step8_dir / model["validation_file"]) != model["validation_sha256"]:
            raise ValueError("Frozen validation predictions were modified")
    return frozen


def isolated_worker(module, request, stage, name):
    request_path = stage / f"request_{name}.json"
    write_json(request_path, request)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
           "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/private/tmp/flight-delay-matplotlib")}
    with (stage / f"{name}.log").open("w") as stream:
        result = subprocess.run([sys.executable, "-m", module, "--request", str(request_path)],
                                cwd=config.PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    request_path.unlink()
    if result.returncode:
        detail = (stage / f"{name}.log").read_text()[-4000:]
        raise RuntimeError(f"{name} failed in isolated worker:\n{detail}")


def matched_rows(reference, other):
    if reference.duplicated(KEY).any() or other.duplicated(KEY).any() or len(reference) != len(other):
        raise ValueError("Duplicate or different model evaluation populations")
    joined = reference[KEY + ["y_true", "sched_date"]].merge(other, on=KEY, how="left", validate="one_to_one", suffixes=("_expected", ""))
    if (joined.y_probability_raw.isna().any() or not joined.y_true.eq(joined.y_true_expected).all()
            or not joined.sched_date.eq(joined.sched_date_expected).all()):
        raise ValueError("Model evaluation keys, labels or dates disagree")
    return joined.drop(columns=["y_true_expected", "sched_date_expected"])


def cohort_metrics(frame, *, validation=False, fraction=0.1, n_bins=10):
    probability = "calibrated_probability" if validation else "y_probability"
    reference = frame.reference_prevalence.to_numpy()
    result = metrics(frame.y_true, frame[probability], frame.flight_row_id, reference, fraction=fraction, n_bins=n_bins)
    result["raw_average_precision"] = float(average_precision_score(frame.y_true, frame.y_probability_raw)) if frame.y_true.sum() else None
    return result


def success_gates(results, intervals, development, diagnostics_complete, sequence_status):
    output = {}
    baseline = "multi_lead/xgboost"
    schedule = "schedule_only/schedule_only"
    for model in ("multi_lead/embedding_mlp", "multi_lead/sequence_model"):
        if model not in results:
            output[model] = {"status": "not_evaluated", "reason": sequence_status}
            continue
        rows = results[model]
        checks = {}
        for lead, item in rows.items():
            own = item["average_precision"]
            other = results[baseline][lead]["average_precision"]
            ci, xci = intervals[lead]["metrics"][model]["average_precision"], intervals[lead]["metrics"][baseline]["average_precision"]
            checks[lead] = {
                "exceeds_schedule_only": own > results[schedule][lead]["average_precision"],
                "positive_brier_skill": item["brier_skill_score"] > 0,
                "exceeds_matched_xgboost": own > other,
                "nonoverlapping_ap_intervals": ci["lower"] is not None and xci["upper"] is not None and ci["lower"] > xci["upper"],
                "paired_ap_difference_interval": intervals[lead]["average_precision_difference_vs_xgboost"].get(model),
            }
        at60 = rows["60"]["average_precision"]
        output[model] = {
            "by_lead": checks, "exceeds_legacy_aml_0_4846_at_60m": at60 > 0.4846,
            "measured_success_gates_pass": at60 > 0.4846 and all(
                c["exceeds_schedule_only"] and c["positive_brier_skill"] and c["exceeds_matched_xgboost"] for c in checks.values()),
            "superiority_claim_allowed": not development and diagnostics_complete and all(c["nonoverlapping_ap_intervals"] for c in checks.values()),
        }
    return output


def _markdown(report):
    status = "DEVELOPMENT ONLY: reduced search budget" if report["development_run"] else "Full configured search budget"
    lines = ["# Step 9 model evaluation", "", f"**{status}.**", "", report["holdout_status"], "",
             "Ranking below uses calibrated probabilities; raw average precision is also retained in JSON to expose isotonic ties. "
             "Brier skill uses the corresponding training prevalence, not the evaluation prevalence. "
             "Alerts are the predeclared top 10% of each lead-time cohort, with deterministic ties.", "",
             "| Feature set / model | Lead (min) | Flights | AP | Raw AP | ROC AUC | Brier skill | Precision @10% | Recall @10% |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "undefined" if value is None else f"{value:.4f}"
    for model, leads in report["retrospective"].items():
        for lead, item in leads.items():
            b = item["alert_budget"]
            lines.append(f"| {model} | {lead} | {item['rows']:,} | {fmt(item['average_precision'])} | {fmt(item['raw_average_precision'])} | {fmt(item['roc_auc'])} | {fmt(item['brier_skill_score'])} | {fmt(b['precision'])} | {fmt(b['recall'])} |")
    lines += ["", "## Uncertainty and success gates", "",
              "Paired bootstrap resamples whole local service days. JSON includes 95% intervals for AP, ROC AUC, Brier score and Brier skill, plus paired AP differences against matched XGBoost.", ""]
    for model, item in report["success_gates"].items():
        if item.get("status") == "not_evaluated":
            lines.append(f"- {model}: not evaluated; {item['reason']}.")
        else:
            lines.append(f"- {model}: measured performance gates {'passed' if item['measured_success_gates_pass'] else 'not all passed'}; superiority claim {'supported' if item['superiority_claim_allowed'] else 'not supported'}.")
    lines += ["", "## Diagnostics and interpretation", "",
              "`subgroups.csv` reports month, origin, carrier, scheduled-hour and weather-availability slices, including denominators. "
              "Their 10% budgets are local to each subgroup. `diagnostics.json` contains validation-only feature-group permutation importance and true refit ablations. "
              "Ablations reuse selected hyperparameters and are diagnostic, not separately tuned comparisons.", "",
              "Validation calibration is prequential: each month uses available labels from an earlier OOF month, with identity fallback where history is insufficient. "
              "The final calibrators use only eligible October OOF labels. Model selection and OOF diagnostics reuse development folds; they are not nested unbiased estimates.", "",
              "Observed weather is a persistence proxy, not archived forecasts. Reconstructed event availability uses an assumed buffer. "
              "The data covers one year of NYC departures; same-day departure inputs are omitted unless approved. "
              "No real cancellation/diversion target or causal rescheduling claim is made. Step 10 report/slides are outside this deliverable.", ""]
    if report.get("import_audit"):
        lines += ["## Imported-run scope", "",
                  "These numbers use the supplied five-model archive (one configuration per model). "
                  "No 40-configuration completion or legacy-feature model comparison is claimed. "
                  "Neural numeric-table hashes differ from the Mac tables; the supplier attributes this to cross-platform rounding. "
                  "The original Linux tables are absent, so that explanation has not been independently verified. "
                  "Raw sources, feature definitions, fold assignments, prediction IDs and target labels agree. "
                  "Original checkpoint inference is not required to evaluate the supplied saved probabilities.", ""]
    return "\n".join(lines)


def _plots(report, predictions, stage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, leads in report["retrospective"].items():
        if model.startswith("legacy_aml/"):
            continue
        times = sorted(map(int, leads))
        means = [leads[str(t)]["average_precision"] for t in times]
        bounds = [report["intervals"][str(t)]["metrics"][model]["average_precision"] for t in times]
        ax.errorbar(np.array(times) / 60, means,
                    yerr=[[max(0, m - b["lower"]) for m, b in zip(means, bounds)],
                          [max(0, b["upper"] - m) for m, b in zip(means, bounds)]], marker="o", capsize=3, label=model)
    ax.set(xlabel="Hours before departure", ylabel="Average precision", title="Retrospective lead-time comparison (95% day-bootstrap intervals)")
    ax.set_xticks([1, 2, 6, 24]); ax.invert_xaxis(); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(stage / "lead_time_curve.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, lead in zip(axes.flat, (1440, 360, 120, 60)):
        ax.plot([0, 1], [0, 1], "--", color="gray")
        for model, leads in report["retrospective"].items():
            if model.startswith("legacy_aml/"):
                continue
            bins = leads[str(lead)]["reliability_bins"]
            ax.plot([b["mean_probability"] for b in bins], [b["observed_fraction"] for b in bins], marker=".", label=model)
        ax.set(title=f"T-{lead // 60} h", xlabel="Mean predicted probability", ylabel="Observed delay fraction")
    axes[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(stage / "reliability.png", dpi=160); plt.close(fig)
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    for row, lead in enumerate((1440, 360, 120, 60)):
        for model, frame in predictions.items():
            if model.startswith("legacy_aml/"):
                continue
            cohort = frame.loc[frame.lead_time_minutes.eq(lead)]
            precision, recall, _ = precision_recall_curve(cohort.y_true, cohort.y_probability)
            fpr, tpr, _ = roc_curve(cohort.y_true, cohort.y_probability)
            axes[row, 0].plot(recall, precision, label=model)
            axes[row, 1].plot(fpr, tpr, label=model)
        axes[row, 0].set(title=f"T-{lead // 60} h: precision-recall", xlabel="Recall", ylabel="Precision")
        axes[row, 1].set(title=f"T-{lead // 60} h: ROC", xlabel="False positive rate", ylabel="True positive rate")
    axes[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(stage / "pr_roc_curves.png", dpi=160); plt.close(fig)
    diagnostics = report["diagnostics"]
    if diagnostics.get("status") == "completed":
        groups = diagnostics["permutation_importance"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(list(groups), [np.mean(list(item["mean_ap_drop_by_lead"].values())) for item in groups.values()])
        ax.set(xlabel="Mean validation AP drop after group permutation", title="October feature-group diagnostics (XGBoost)")
        fig.tight_layout(); fig.savefig(stage / "feature_importance.png", dpi=160); plt.close(fig)


def run_step9(step8_dir=None, output_dir=None, reports_dir=None, *, legacy_features=None, skip_diagnostics=False):
    step8 = Path(step8_dir or config.PROJECT_ROOT / "artifacts/step8")
    output = Path(output_dir or config.PROJECT_ROOT / "artifacts/step9")
    reports = Path(reports_dir or config.REPORTS_DIR / "step9")
    if output.exists() or reports.exists():
        raise FileExistsError("Choose new Step 9 output and report directories; results are immutable")
    frozen = verify_freeze(step8)
    settings = frozen["settings"]
    if any(m["feature_set"] == "legacy_aml" for m in frozen["models"].values()) and legacy_features is None:
        raise ValueError("Supply --legacy-features for the required old/new comparison grid")
    if legacy_features is not None:
        legacy_features = Path(legacy_features).resolve()
        for item in frozen["models"].values():
            if item["feature_set"] == "legacy_aml" and file_fingerprint(legacy_features) != item["feature_file_sha256"].get(legacy_features.name):
                raise ValueError("Legacy feature table changed since training")
    calibrators = joblib.load(step8 / "calibrators.joblib")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".step9-", dir=output.parent) as work:
        stage = Path(work) / "evaluation"
        stage.mkdir()
        report = {"step": 9, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "protocol_version": frozen["protocol_version"],
                  "raw_data_fingerprints": frozen["raw_data_fingerprints"],
                  "fold_manifest_version": frozen["fold_manifest_version"],
                  "frozen_decisions_sha256": file_fingerprint(step8 / "frozen_decisions.json"),
                  "evaluation_code_sha256": file_fingerprint(Path(__file__)),
                  "development_run": frozen["development_run"], "holdout_status": frozen["holdout_status"],
                  "settings": settings, "retrospective": {}, "validation": {}, "intervals": {},
                  "monthly": {}, "model_runs": {}, "omissions": frozen["omissions"]}
        report["leakage_tests"] = frozen["leakage_tests"]
        if "import_audit" in frozen:
            report["import_audit"] = frozen["import_audit"]
        predictions, reference, subgroups = {}, None, []
        # Diagnostics do not alter the frozen model or calibrators and use no retrospective labels.
        if not skip_diagnostics:
            classical = next(Path(p) for p in frozen["files"] if p.endswith("run_metadata.json") and "xgboost" in read_json(p).get("feature_sets", {}).get("multi_lead", {}).get("models", {}))
            isolated_worker("src.evaluation.diagnostics_v2", {"dataset_dir": frozen["dataset_dir"],
                            "step5_metadata": str(classical), "settings": settings,
                            "output_path": str(stage / "diagnostics.json")}, stage, "diagnostics")
        else:
            write_json(stage / "diagnostics.json", {"status": "skipped", "reason": "Explicit development option"})
        for key, model in frozen["models"].items():
            stem = key.replace("/", "_")
            path = stage / f"{stem}_retrospective_predictions.csv"
            if "retrospective_predictions_path" in model:
                from src.evaluation.archive_handoff import read_archived_retrospective
                frame = read_archived_retrospective(model, frozen["dataset_dir"])
            else:
                isolated_worker("src.evaluation.score_v2", {"dataset_dir": frozen["dataset_dir"], "model": model,
                                "legacy_features": str(legacy_features) if legacy_features else None,
                                "output_path": str(path)}, stage, stem)
                frame = pd.read_csv(path, dtype={"flight_row_id": str})
            if reference is None:
                if model["feature_set"] == "legacy_aml":
                    raise ValueError("A multi-lead reference must be scored before legacy models")
                reference = frame.copy()
            wanted = reference.loc[reference.lead_time_minutes.eq(60)] if model["feature_set"] == "legacy_aml" else reference
            frame = matched_rows(wanted, frame)
            frame["y_probability"] = np.nan
            frame["reference_prevalence"] = model["reference_prevalence"]
            frame["alert_at_budget"] = False
            for lead, rows in frame.groupby("lead_time_minutes"):
                probability = apply_calibrator(calibrators[key][int(lead)], rows.y_probability_raw)
                frame.loc[rows.index, "y_probability"] = probability
                _, selected = alert_budget(rows.y_true, probability, rows.flight_row_id, settings["alert_budget_fraction"])
                frame.loc[rows.index, "alert_at_budget"] = selected
            frame.to_csv(path, index=False)
            predictions[key] = frame
            report["model_runs"][key] = {"run_id": model["run_id"], "trials_per_model": model["trials_per_model"],
                                         "feature_manifest_version": model["feature_manifest_version"],
                                         "selected_params": model["selected_params"], "package_versions": model["package_versions"],
                                         "calibration": model["calibration"],
                                         "model_sha256": frozen["files"][model["model_path"]],
                                         "mean_fold_validation_average_precision": model["mean_validation_average_precision"],
                                         "predictions_sha256": file_fingerprint(path)}
            report["retrospective"][key] = {str(lead): cohort_metrics(rows, fraction=settings["alert_budget_fraction"], n_bins=settings["ece_bins"])
                                                   for lead, rows in frame.groupby("lead_time_minutes")}
            validation = pd.read_csv(step8 / model["validation_file"], dtype={"flight_row_id": str})
            validation["y_probability_raw"] = validation.y_probability
            report["validation"][key] = {str(lead): cohort_metrics(rows, validation=True, fraction=settings["alert_budget_fraction"], n_bins=settings["ece_bins"])
                                                for lead, rows in validation.groupby("lead_time_minutes")}
            report["monthly"][key] = {}
            for split, population in (("validation", validation), ("retrospective", frame)):
                for (lead, month), rows in population.groupby(["lead_time_minutes", population.sched_date.str[:7]]):
                    report["monthly"][key][f"{lead}/{month}"] = cohort_metrics(rows, validation=split == "validation", fraction=settings["alert_budget_fraction"], n_bins=settings["ece_bins"])
            LOGGER.info("Scored and reconciled %s (%s rows)", key, f"{len(frame):,}")
        for lead in sorted(reference.lead_time_minutes.unique()):
            matched = {key: rows.loc[rows.lead_time_minutes.eq(lead)].sort_values(KEY).reset_index(drop=True)
                       for key, rows in predictions.items() if lead in set(rows.lead_time_minutes)}
            sample = next(iter(matched.values()))
            report["intervals"][str(lead)] = paired_day_bootstrap(
                sample.y_true, {key: rows.y_probability.to_numpy() for key, rows in matched.items()}, sample.sched_date,
                {key: rows.reference_prevalence.iloc[0] for key, rows in matched.items()},
                resamples=settings["bootstrap_resamples"], seed=settings["bootstrap_seed"], confidence=settings["confidence_level"])
            LOGGER.info("Completed paired day bootstrap at %s minutes", lead)
        # Static subgroup fields come from the same fingerprinted Step 4 rows.
        manifest = read_json(Path(frozen["dataset_dir"]) / "feature_manifest.json")
        for part in manifest["partitions"]:
            columns = ["flight_row_id", "lead_time_minutes", "origin", "carrier", "sched_dep_hour", "weather_stale_or_missing"]
            available = set(pd.read_csv(Path(frozen["dataset_dir"]) / part["features"], nrows=0).columns)
            attrs = pd.read_csv(Path(frozen["dataset_dir"]) / part["features"], usecols=[c for c in columns if c in available])
            for key, frame in predictions.items():
                rows = frame.loc[frame.lead_time_minutes.eq(part["lead_time_minutes"])].merge(attrs, on=["flight_row_id", "lead_time_minutes"], validate="one_to_one")
                rows["month"] = rows.sched_date.str[:7]
                for dimension in ["month", *[c for c in columns[2:] if c in rows]]:
                    for group, cohort in rows.groupby(dimension, dropna=False):
                        item = cohort_metrics(cohort, fraction=settings["alert_budget_fraction"], n_bins=settings["ece_bins"])
                        subgroups.append({"model": key, "lead_time_minutes": part["lead_time_minutes"], "dimension": dimension,
                                          "group": str(group), **{k: v for k, v in item.items() if not isinstance(v, (list, dict))},
                                          "precision_at_budget": item["alert_budget"]["precision"], "recall_at_budget": item["alert_budget"]["recall"],
                                          "alerts": item["alert_budget"]["alerts"], "false_positives": item["alert_budget"]["false_positives"],
                                          "false_negatives": item["alert_budget"]["false_negatives"], "fewer_than_100_flights": len(cohort) < 100})
        pd.DataFrame(subgroups).to_csv(stage / "subgroups.csv", index=False)
        report["success_gates"] = success_gates(report["retrospective"], report["intervals"], frozen["development_run"], not skip_diagnostics, "; ".join(frozen["omissions"]))
        report["diagnostics"] = read_json(stage / "diagnostics.json")
        write_json(stage / "evaluation.json", report)
        (stage / "evaluation.md").write_text(_markdown(report))
        _plots(report, predictions, stage)
        stage.rename(output)
    reports.mkdir(parents=True)
    for filename in ("evaluation.json", "evaluation.md", "diagnostics.json", "subgroups.csv", "lead_time_curve.png", "reliability.png", "pr_roc_curves.png", "feature_importance.png"):
        if not (output / filename).exists():
            continue
        shutil.copy2(output / filename, reports / filename)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("step8-dir", "output-dir", "reports-dir", "legacy-features"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--skip-diagnostics", action="store_true", help="Development only; does not complete Step 9")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_step9(**vars(parser.parse_args()))
    print(f"Evaluated {len(result['retrospective'])} models; development run: {result['development_run']}")


if __name__ == "__main__":
    main()
