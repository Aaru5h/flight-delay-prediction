"""Phase 1 pipeline entry point: load, profile, clean, label, report.

Run from the repository root:

    python -m src.data.prepare

Outputs (deterministic; reruns produce meaningful diffs):
    data/processed/labelled_flights.csv   one row per source flight, with
                                          flight_row_id, parsed schedule
                                          fields, the label and label_status
                                          (arr_delay kept for audit only)
    data/processed/phase1_run_manifest.json
    reports/phase1_data_quality.json
    reports/phase1_data_quality.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.data import config
from src.data.labeling import add_delay_label, label_summary
from src.data.loading import load_all_sources
from src.data.profiling import (
    column_missingness,
    count_full_duplicates,
    date_coverage,
    duplicate_key_counts,
    join_coverage,
    monthly_label_distribution,
)
from src.data.report import (
    build_report,
    compare_with_reference,
    load_reference_profile,
    write_outputs,
)
from src.data.schedule import add_flight_row_id, add_schedule_fields

LABELLED_COLUMNS = [
    "flight_row_id", "year", "month", "day", "sched_date",
    "sched_dep_time", "sched_arr_time", "sched_dep_minute", "sched_arr_minute",
    "sched_dep_time_invalid", "sched_arr_time_invalid",
    "sched_dep_ts", "sched_arr_ts",
    "carrier", "flight", "tailnum", "origin", "dest", "distance", "time_hour",
    "arr_delay", "substantial_delay", "label_status",
]


def _code_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return None


def run_phase1(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict:
    """Execute the full Phase 1 pipeline and return the quality report dict."""
    raw_dir = Path(raw_dir or config.RAW_DATA_DIR)
    processed_dir = Path(processed_dir or config.PROCESSED_DATA_DIR)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)

    # Step 1-2: load and schema-check the five sources.
    sources = load_all_sources(raw_dir)

    flights = sources["flights"].df

    # Step 3: stable identifier, created before any exclusion or sorting.
    flights = add_flight_row_id(flights, sources["flights"].fingerprint)

    # Step 4: schedule parsing.
    flights = add_schedule_fields(flights)
    invalid_schedule_counts = {
        "sched_dep_time_invalid": int(flights["sched_dep_time_invalid"].sum()),
        "sched_arr_time_invalid": int(flights["sched_arr_time_invalid"].sum()),
        "sched_dep_ts_missing": int(flights["sched_dep_ts"].isna().sum()),
    }

    # Step 5: target construction.
    flights = add_delay_label(flights)
    summary = label_summary(flights)
    if summary["known_labels"] + summary["unknown_labels"] != summary["total_rows"]:
        raise AssertionError("label counts do not reconcile with total rows")

    # Step 6: profile and report.
    missingness = column_missingness(flights)
    full_duplicates = count_full_duplicates(flights)
    duplicate_keys = duplicate_key_counts(sources)
    coverage = join_coverage(flights, sources)
    dates = date_coverage(flights)
    monthly = monthly_label_distribution(flights)
    reference = load_reference_profile(config.CONTRACT_PATH)
    reference_checks = compare_with_reference(summary, reference)

    report = build_report(
        sources=sources,
        flights=flights,
        label_summary=summary,
        missingness=missingness,
        full_duplicates=full_duplicates,
        duplicate_keys=duplicate_keys,
        coverage=coverage,
        date_coverage=dates,
        monthly=monthly,
        invalid_schedule_counts=invalid_schedule_counts,
        reference_checks=reference_checks,
    )

    # Step 7: deterministic outputs.
    processed_dir.mkdir(parents=True, exist_ok=True)
    labelled_path = processed_dir / "labelled_flights.csv"
    flights[LABELLED_COLUMNS].to_csv(
        labelled_path, index=False, date_format="%Y-%m-%dT%H:%M:%S"
    )
    run_manifest = {
        "phase": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m src.data.prepare",
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "code_commit": _code_commit(),
        "raw_dir": str(raw_dir),
        "source_fingerprints": {
            name: s.fingerprint for name, s in sources.items()
        },
        "outputs": {
            "labelled_flights": str(labelled_path),
            "labelled_rows": int(len(flights)),
            "labelled_columns": len(LABELLED_COLUMNS),
        },
    }
    (processed_dir / "phase1_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )
    json_path, md_path = write_outputs(report, reports_dir)

    print(f"Labelled dataset: {labelled_path} ({len(flights)} rows)")
    print(f"Data-quality report: {json_path}")
    print(f"Data-quality report: {md_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 pipeline")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    args = parser.parse_args()
    run_phase1(raw_dir=args.raw_dir, processed_dir=args.processed_dir,
               reports_dir=args.reports_dir)


if __name__ == "__main__":
    main()
