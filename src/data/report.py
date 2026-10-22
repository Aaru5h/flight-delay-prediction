"""Assembly and rendering of the Phase 1 data-quality report (Step 6).

One nested dictionary feeds both outputs: a machine-readable JSON summary and
a readable Markdown report, both written deterministically so reruns produce
meaningful diffs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def load_reference_profile(contract_path: Path) -> dict:
    """Validation expectations for the supplied files, from the contract."""
    with open(contract_path) as fh:
        contract = yaml.safe_load(fh)
    return contract.get("data", {}).get("reference_profile", {})


def compare_with_reference(summary: dict, reference: dict) -> dict:
    """Measured label counts vs contract expectations for the supplied files."""
    checks = {}
    for key in ("total_rows", "known_labels", "positive", "negative", "unknown_labels"):
        expected = reference.get(key)
        measured = summary.get(key)
        checks[key] = {
            "expected": expected,
            "measured": measured,
            "matches": (expected == measured) if expected is not None else None,
        }
    return checks


def build_report(
    sources: dict,
    flights: pd.DataFrame,
    label_summary: dict,
    missingness: dict,
    full_duplicates: int,
    duplicate_keys: dict,
    coverage: dict,
    date_coverage: dict,
    monthly: list,
    invalid_schedule_counts: dict,
    reference_checks: dict,
) -> dict:
    """Assemble the complete quality report as one nested dictionary."""
    return {
        "phase": 1,
        "schema_validation": {
            name: {"passed": True, "rows": int(len(s.df)), "columns": int(s.df.shape[1])}
            for name, s in sources.items()
        },
        "source_files": {
            name: {
                "file": s.path.name,
                "rows": int(len(s.df)),
                "columns": int(s.df.shape[1]),
                "size_bytes": s.size_bytes,
                "sha256": s.fingerprint,
            }
            for name, s in sources.items()
        },
        "flights_missingness": missingness,
        "duplicates": {
            "full_duplicate_rows_in_flights": full_duplicates,
            "lookup_key_duplicates": duplicate_keys,
        },
        "schedule_parsing": invalid_schedule_counts,
        "date_coverage": date_coverage,
        "label_summary": label_summary,
        "reference_profile_checks": reference_checks,
        "join_coverage": coverage,
        "monthly_label_distribution": monthly,
        "notes": [
            "Missing arr_delay is an unknown label, not a negative and not a "
            "confirmed cancellation (no cancellation/diversion flag exists).",
            "arr_delay is an audit column in the labelled dataset only; it "
            "must never enter a model's feature matrix.",
            "Weather coverage here is a same-hour profile; the T-60 as-of "
            "join is built in Phase 2.",
            "Unmatched destination airports (e.g. BQN, PSE, SJU, STT) and "
            "incomplete plane-lookup coverage are preserved and reported, "
            "never dropped.",
        ],
    }


def render_markdown(report: dict) -> str:
    """Render the report dictionary as a readable Markdown document."""
    lines = ["# Phase 1 — Data Quality and Label Report", ""]

    lines += ["## 1. Source files", "",
              "| File | Rows | Columns | Size (bytes) | SHA-256 (first 8) |",
              "|---|---:|---:|---:|---|"]
    for name, s in report["source_files"].items():
        lines.append(f"| {s['file']} | {s['rows']} | {s['columns']} | "
                     f"{s['size_bytes']} | `{s['sha256'][:8]}` |")

    lines += ["", "## 2. Schema validation", ""]
    for name, check in report["schema_validation"].items():
        lines.append(f"- `{name}`: {'passed' if check['passed'] else 'FAILED'} "
                     f"({check['rows']} rows, {check['columns']} columns)")

    lines += ["", "## 3. Flights missingness (columns with gaps)", "",
              "| Column | Missing | % |", "|---|---:|---:|"]
    for col, m in report["flights_missingness"].items():
        lines.append(f"| {col} | {m['missing']} | {m['pct']} |")

    dup = report["duplicates"]
    lines += ["", "## 4. Duplicates and lookup keys", "",
              f"- Full duplicate flight rows: **{dup['full_duplicate_rows_in_flights']}**",
              "", "| Lookup | Key | Duplicate key rows |", "|---|---|---:|"]
    for name, d in dup["lookup_key_duplicates"].items():
        lines.append(f"| {name} | {', '.join(d['key_columns'])} | "
                     f"{d['duplicate_key_rows']} |")

    sp = report["schedule_parsing"]
    lines += ["", "## 5. Schedule parsing", "",
              f"- Invalid `sched_dep_time` values: **{sp['sched_dep_time_invalid']}**",
              f"- Invalid `sched_arr_time` values: **{sp['sched_arr_time_invalid']}**",
              f"- Rows with unusable scheduled departure timestamp: "
              f"**{sp['sched_dep_ts_missing']}**"]
    dc = report["date_coverage"]
    lines += ["", f"- Scheduled dates: {dc['min']} to {dc['max']} "
              f"({dc['distinct_days']} distinct days)"]

    ls = report["label_summary"]
    lines += ["", "## 6. Label summary and reconciliation", "",
              f"- Total flight rows: **{ls['total_rows']}**",
              f"- Known labels: **{ls['known_labels']}** "
              f"(positive {ls['positive']}, negative {ls['negative']}, "
              f"prevalence {ls['positive_prevalence']})",
              f"- Unknown labels: **{ls['unknown_labels']}** "
              f"(also missing dep_time: "
              f"{ls['unknown_label_detail']['also_missing_dep_time']}; "
              f"dep_time present: "
              f"{ls['unknown_label_detail']['dep_time_present']})",
              "",
              f"Reconciliation: known + unknown = "
              f"{ls['known_labels'] + ls['unknown_labels']} = total rows "
              f"({ls['total_rows']}): "
              f"**{'OK' if ls['known_labels'] + ls['unknown_labels'] == ls['total_rows'] else 'MISMATCH'}**"]

    checks = report["reference_profile_checks"]
    lines += ["", "### Comparison with the contract reference profile", "",
              "| Count | Expected | Measured | Match |", "|---|---:|---:|---|"]
    for key, c in checks.items():
        match = "yes" if c["matches"] else ("no" if c["matches"] is False else "n/a")
        lines.append(f"| {key} | {c['expected']} | {c['measured']} | {match} |")

    cov = report["join_coverage"]
    lines += ["", "## 7. Join-key coverage (profile only; joins are Phase 2)", ""]
    udest = cov["airports_destination"]["unmatched_airports"]
    lines.append(f"- Unmatched destination airports: "
                 f"{udest['rows']} rows over {udest['airport_codes']}")
    uorigin = cov["airports_origin"]["unmatched_airports"]
    lines.append(f"- Unmatched origin airports: {uorigin['rows']} rows")
    ucar = cov["airlines"]["unmatched_carriers"]
    lines.append(f"- Unmatched carriers: {ucar['rows']} rows")
    pl = cov["planes"]
    lines.append(f"- Plane lookup: {pl['matched_rows']} of "
                 f"{pl['non_missing_tailnum_rows']} non-missing tailnum rows "
                 f"matched ({pl['unmatched_rows']} unmatched)")
    w = cov["weather"]
    lines.append(f"- Same-hour weather observations available for "
                 f"{w['rows_with_same_hour_observation']} of "
                 f"{w['rows_with_same_hour_observation'] + w['rows_without_same_hour_observation']} "
                 f"rows (profile only)")

    lines += ["", "## 8. Monthly label distribution", "",
              "| Month | Labelled | Positive | Negative | Prevalence |",
              "|---:|---:|---:|---:|---:|"]
    for row in report["monthly_label_distribution"]:
        lines.append(f"| {row['month']} | {row['labelled']} | {row['positive']} | "
                     f"{row['negative']} | {row['prevalence']} |")

    lines += ["", "## 9. Notes", ""]
    lines += [f"- {note}" for note in report["notes"]]
    lines.append("")
    return "\n".join(lines)


def write_outputs(report: dict, reports_dir: Path) -> tuple[Path, Path]:
    """Write the JSON and Markdown reports deterministically."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "phase1_data_quality.json"
    md_path = reports_dir / "phase1_data_quality.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path
