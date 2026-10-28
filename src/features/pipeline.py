"""Phase 2 pipeline: build and save the leakage-safe feature matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data import config
from src.data.loading import load_all_sources
from src.features.history import add_historical_features
from src.features.manifest import ALLOWLIST, CATEGORICAL_FEATURES, DENYLIST
from src.features.schedule import add_schedule_features, join_reference_features
from src.features.weather import WEATHER_INPUTS, join_weather_asof


def _feature_manifest() -> dict:
    categorical = set(CATEGORICAL_FEATURES)
    weather = set(WEATHER_INPUTS) | {"weather_stale_or_missing"}
    history = {name for name in ALLOWLIST if name.startswith(("carrier_", "route_"))} - {"carrier", "route"}
    entries = []
    for name in ALLOWLIST:
        if name in weather:
            source, rule, missing = "weather", "latest origin observation at/before T-60, maximum age 180 minutes", "numeric missing plus stale/missing indicator"
        elif name in history:
            source, rule, missing = "label history", "training-date outcomes only; earlier calendar dates and conservative availability: scheduled NYC departure + 24 hours + positive arrival delay <= T-60", "missing indicator; imputation in training-only model pipeline"
        elif "airport_" in name:
            source, rule, missing = "airports reference", "static reference joined by scheduled origin/destination", "preserve missing; impute in training-only model pipeline"
        elif name == "scheduled_duration_minutes":
            source, rule, missing = "published schedule + destination airport timezone", "convert scheduled New York departure and destination-local scheduled arrival to UTC; known by T-60", "unmatched destination timezone stays missing; impute in training-only model pipeline"
        else:
            source, rule, missing = "published flight schedule", "known by T-60", "unknown category or training-only imputation"
        entries.append({"name": name, "type": "categorical" if name in categorical else "numeric", "source": source, "availability_rule": rule, "missing_value_policy": missing, "status": "approved"})
    payload = {"manifest_version": "2.2.0", "prediction_time": "T-60", "allowlist": ALLOWLIST, "denylist": DENYLIST, "features": entries, "audit_only_columns": ["prediction_cutoff_utc", "weather_observed_utc", "weather_age_minutes"], "limitations": ["Weather values are retrospective observations, not archived forecast vintages.", "Historical rates are frozen to training-date outcomes after 2013-08-31 so validation/test labels never become features.", "Historical arrival availability is conservatively estimated from scheduled NYC departure plus 24 hours and positive arrival delay; no release timestamp is supplied.", "Aircraft/tail-number features are excluded because assignment availability at T-60 is not established."]}
    encoded = json.dumps(payload, sort_keys=True).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def build_feature_table(labelled: pd.DataFrame, sources: dict) -> tuple[pd.DataFrame, dict]:
    labelled = labelled.copy()
    labelled["sched_date"] = pd.to_datetime(labelled["sched_date"])
    labelled["sched_dep_ts"] = pd.to_datetime(labelled["sched_dep_ts"])
    # Scheduled volume must include flights with unknown eventual arrival labels.
    scheduled = add_schedule_features(labelled)
    df = scheduled.loc[scheduled["label_status"].eq("known")].copy()
    df, coverage = join_reference_features(df, sources["airlines"].df, sources["airports"].df)
    df = join_weather_asof(df, sources["weather"].df)
    df = add_historical_features(df, history_training_end="2013-08-31")
    if len(df) != labelled["label_status"].eq("known").sum() or not df["flight_row_id"].is_unique:
        raise AssertionError("Phase 2 changed the eligible labelled-flight grain")
    if set(DENYLIST) & set(ALLOWLIST):
        raise AssertionError("denylisted feature appears in estimator allowlist")
    return df, coverage


def run_phase2(raw_dir: Path | None = None, processed_dir: Path | None = None, reports_dir: Path | None = None) -> dict:
    raw_dir = Path(raw_dir or config.RAW_DATA_DIR)
    processed_dir = Path(processed_dir or config.PROCESSED_DATA_DIR)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    labelled_path = processed_dir / "labelled_flights.csv"
    if not labelled_path.is_file():
        raise FileNotFoundError(f"Phase 1 artifact missing: {labelled_path}; run python -m src.data.prepare first")
    labelled = pd.read_csv(labelled_path, low_memory=False)
    sources = load_all_sources(raw_dir)
    table, coverage = build_feature_table(labelled, sources)
    audit = ["flight_row_id", "substantial_delay", "sched_date", "prediction_cutoff_utc", "weather_observed_utc", "weather_age_minutes"]
    output = table[[*audit, *ALLOWLIST]].copy()
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "phase2_features.csv"
    output.to_csv(output_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    manifest = _feature_manifest()
    (processed_dir / "phase2_feature_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {"phase": 2, "rows": len(output), "features": len(ALLOWLIST), "feature_manifest_sha256": manifest["manifest_sha256"], "reference_coverage": coverage, "weather": {"matched_observation_rows": int(output["weather_observed_utc"].notna().sum()), "fresh_usable_rows": int((~output["weather_stale_or_missing"]).sum()), "stale_or_missing_rows": int(output["weather_stale_or_missing"].sum())}, "history": {"carrier_missing_rows": int(output["carrier_history_missing"].sum()), "route_missing_rows": int(output["route_history_missing"].sum())}, "output": str(output_path)}
    (reports_dir / "phase2_feature_quality.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 2 feature pipeline")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    args = parser.parse_args()
    summary = run_phase2(args.raw_dir, args.processed_dir, args.reports_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
