"""Step 4 export of the protocol v2 feature datasets and quality report."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import timedelta
import gzip
import hashlib
import io
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml

from src.data import config
from src.data.labeling import LABEL_KNOWN, add_delay_label
from src.data.loading import SourceData, file_fingerprint, load_all_sources
from src.data.schedule import add_flight_row_id
from src.features.asof import AsOfFeatureBuilder, EventSequence

LOGGER = logging.getLogger(__name__)
LABEL = "substantial_delay"
KEY_COLUMNS = ("flight_row_id", "lead_time_minutes")
AUDIT_COLUMNS = (
    "sched_date", "scheduled_departure_utc", "prediction_cutoff_utc",
    "outcome_available_utc", "weather_observed_utc",
)
SCHEDULE_COLUMNS = ["sched_date", "scheduled_departure_utc"]
MANIFEST_FILENAME = "feature_manifest.json"
QUALITY_FILENAME = "data_quality.json"
CONTRACT_PATH = config.CONFIG_DIR / "project_contract_v2.yaml"
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _labelled_population(
    builder: AsOfFeatureBuilder, source: SourceData, labelled_path: Path
) -> tuple[pd.DataFrame, dict]:
    if not labelled_path.is_file():
        raise FileNotFoundError(f"Missing {labelled_path}. Run python -m src.data.prepare first.")
    labels = pd.read_csv(
        labelled_path,
        usecols=["flight_row_id", LABEL, "label_status"],
        dtype={"flight_row_id": "string", LABEL: "Int64", "label_status": "string"},
    ).set_index("flight_row_id")
    if labels.index.hasnans or not labels.index.is_unique:
        raise ValueError("Phase 1 flight_row_id values must be present and unique")
    flight_ids = builder.flights.index
    if len(labels) != len(flight_ids) or not flight_ids.isin(labels.index).all():
        raise ValueError("Phase 1 flight IDs do not match the raw source. Regenerate Phase 1 labels.")

    expected = add_flight_row_id(source.df[["arr_delay"]], source.fingerprint)
    expected = add_delay_label(expected).set_index("flight_row_id").reindex(flight_ids)
    labels = labels.reindex(flight_ids)
    if not (
        labels[LABEL].equals(expected[LABEL])
        and labels["label_status"].equals(expected["label_status"].astype("string"))
    ):
        raise ValueError("Phase 1 labels disagree with the source data and delay threshold")

    valid_departures = builder.flights["scheduled_departure_utc"].notna()
    known_labels = labels["label_status"].eq(LABEL_KNOWN)
    eligible = known_labels & valid_departures
    flights = builder.flights.loc[eligible, SCHEDULE_COLUMNS].join(
        labels[[LABEL]], validate="one_to_one"
    ).join(builder.outcome_available_utc, validate="one_to_one")
    return flights, {
        "source_flights": len(labels),
        "known_label_flights": int(known_labels.sum()),
        "unknown_label_flights": int((~known_labels).sum()),
        "invalid_departure_flights": int((~valid_departures).sum()),
        "excluded_known_label_invalid_departure_flights": int((known_labels & ~valid_departures).sum()),
        "eligible_flights": len(flights),
        "eligible_flights_missing_outcome_availability": int(flights["outcome_available_utc"].isna().sum()),
    }


def _json_value(value):
    if isinstance(value, pd.Timestamp):
        return value.strftime(UTC_FORMAT)
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _sequence_payload(sequence: EventSequence | None, cutoff: pd.Timestamp) -> dict | None:
    if sequence is None:
        return None
    rows = []
    for row, active in zip(sequence.rows, sequence.mask, strict=True):
        if active != (row is not None):
            raise ValueError("Sequence mask does not match its padded records")
        if row is not None and max(row["event_utc"], row.get("available_utc", row["event_utc"])) > cutoff:
            raise ValueError("Sequence contains an event unavailable at the prediction cutoff")
        rows.append(None if row is None else {name: _json_value(value) for name, value in row.items()})
    return {"rows": rows, "mask": sequence.mask}


def _write_partition(
    stage: Path,
    builder: AsOfFeatureBuilder,
    flights: pd.DataFrame,
    lead: int,
    allowlist: list[str],
    columns: list[str],
    include_sequences: bool,
    batch_size: int,
) -> tuple[dict, dict]:
    feature_file = f"features_{lead}m.csv"
    sequence_file = f"sequences_{lead}m.jsonl.gz" if include_sequences else None
    feature_names = set(allowlist)
    missing_features = pd.Series(0, index=allowlist, dtype="int64")
    weather_no_observation = weather_stale = weather_sequence_rows = departure_sequence_rows = rows_written = 0
    offset = timedelta(minutes=lead)

    with ExitStack() as stack:
        csv_file = stack.enter_context((stage / feature_file).open("w", encoding="utf-8", newline=""))
        csv_file.write(",".join(columns) + "\n")
        sequences = None
        if sequence_file:
            compressed = gzip.GzipFile(stage / sequence_file, "wb", compresslevel=3, mtime=0)
            sequences = stack.enter_context(io.TextIOWrapper(compressed, encoding="utf-8"))
        for start in range(0, len(flights), batch_size):
            records = []
            for flight in flights.iloc[start:start + batch_size].itertuples():
                cutoff = flight.scheduled_departure_utc - offset
                result = builder.build(flight.Index, cutoff, include_sequences=include_sequences)
                if result.features.keys() != feature_names or result.features["lead_time_minutes"] != lead:
                    raise ValueError(f"Step 3 feature schema or lead mismatch for {flight.Index}")
                observed = result.weather_observed_utc
                if observed is not None and observed > cutoff:
                    raise ValueError(f"Future weather reading for {flight.Index} at {cutoff}")
                records.append({
                    "flight_row_id": flight.Index,
                    "sched_date": flight.sched_date.date(),
                    "scheduled_departure_utc": flight.scheduled_departure_utc,
                    "prediction_cutoff_utc": cutoff,
                    "outcome_available_utc": flight.outcome_available_utc,
                    "weather_observed_utc": observed,
                    LABEL: flight.substantial_delay,
                    **result.features,
                })
                if sequences is not None:
                    weather = _sequence_payload(result.weather_sequence, cutoff)
                    departures = _sequence_payload(result.departure_sequence, cutoff)
                    sequences.write(json.dumps({
                        "flight_row_id": flight.Index,
                        "lead_time_minutes": lead,
                        "weather_sequence": weather,
                        "departure_sequence": departures,
                    }, allow_nan=False, separators=(",", ":")) + "\n")
                    weather_sequence_rows += int(weather is not None and any(weather["mask"]))
                    departure_sequence_rows += int(departures is not None and any(departures["mask"]))

            batch = pd.DataFrame.from_records(records, columns=columns)
            batch.to_csv(csv_file, index=False, header=False, date_format=UTC_FORMAT)
            missing_features += batch[allowlist].isna().sum()
            observed = batch["weather_observed_utc"].notna()
            weather_no_observation += int((~observed).sum())
            weather_stale += int((observed & batch["weather_stale_or_missing"]).sum())
            rows_written += len(records)

    LOGGER.info("Wrote %s with %s rows", feature_file, f"{rows_written:,}")
    return (
        {"lead_time_minutes": lead, "features": feature_file, "sequences": sequence_file, "rows": rows_written},
        {
            "lead_time_minutes": lead,
            "rows": rows_written,
            "positive_labels": int(flights[LABEL].sum()),
            "negative_labels": int(flights[LABEL].eq(0).sum()),
            "no_prior_weather_rows": weather_no_observation,
            "stale_weather_rows": weather_stale,
            "fresh_weather_rows": rows_written - weather_no_observation - weather_stale,
            "rows_with_weather_sequence": weather_sequence_rows if include_sequences else None,
            "rows_with_departure_sequence": departure_sequence_rows if include_sequences else None,
            "missing_feature_values": missing_features.to_dict(),
        },
    )


def run_step4(
    raw_dir: Path | None = None,
    labelled_path: Path | None = None,
    output_dir: Path | None = None,
    *,
    include_sequences: bool = False,
    allow_same_day_departures: bool = False,
    batch_size: int = 4096,
) -> dict:
    """Build and publish the four partitions. Return the data-quality report."""
    raw = Path(raw_dir or config.RAW_DATA_DIR)
    labels = Path(labelled_path or config.PROCESSED_DATA_DIR / "labelled_flights.csv")
    output = Path(output_dir or config.PROCESSED_DATA_DIR / "step4")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"{output} exists and is not empty. Choose a new output directory.")

    protocol = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    lead_times = protocol["prediction_time"]["lead_time_minutes"]
    sources = load_all_sources(raw)
    builder = AsOfFeatureBuilder(sources, allow_same_day_departures=allow_same_day_departures)
    manifest = builder.feature_manifest()
    allowlist = manifest["allowlist"]
    reserved = {"flight_row_id", LABEL, *AUDIT_COLUMNS, *protocol["prediction_time"]["prohibited_target_flight_inputs"]}
    if manifest["lead_time_minutes"] != lead_times:
        raise ValueError("Step 3 and protocol v2 disagree on the lead times")
    if len(set(allowlist)) != len(allowlist) or reserved.intersection(allowlist):
        raise ValueError("Step 3 estimator allowlist contains duplicate or prohibited fields")
    columns = [*KEY_COLUMNS, *AUDIT_COLUMNS, LABEL, *(name for name in allowlist if name != "lead_time_minutes")]
    flights, coverage = _labelled_population(builder, sources["flights"], labels)
    manifest.update({
        "protocol_version": protocol["contract_version"],
        "grain": list(KEY_COLUMNS),
        "columns": columns,
        "audit_columns": list(AUDIT_COLUMNS),
        "label_column": LABEL,
        "source_fingerprints": {name: source.fingerprint for name, source in sources.items()},
        "labelled_file_sha256": file_fingerprint(labels),
        "quality_report": QUALITY_FILENAME,
    })
    report = {"protocol_version": protocol["contract_version"], **coverage}

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".step4-", dir=output.parent) as work:
        stage = Path(work) / "dataset"
        stage.mkdir()
        partitions = []
        lead_quality = []
        for lead in lead_times:
            partition, quality = _write_partition(
                stage, builder, flights, lead, allowlist, columns, include_sequences, batch_size
            )
            partitions.append(partition)
            lead_quality.append(quality)
        manifest["partitions"] = partitions
        fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
        manifest["manifest_sha256"] = fingerprint
        report["feature_manifest_sha256"] = fingerprint
        report["lead_times"] = lead_quality
        report["dataset_rows"] = sum(partition["rows"] for partition in partitions)
        for filename, payload in ((MANIFEST_FILENAME, manifest), (QUALITY_FILENAME, report)):
            (stage / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the four Step 4 feature datasets")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--labelled-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-sequences", action="store_true", help="Export masked sequences alongside the CSVs")
    parser.add_argument(
        "--allow-same-day-departures", action="store_true", help="Enable instructor-approved departure inputs"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_step4(**vars(args))
    output = args.output_dir or config.PROCESSED_DATA_DIR / "step4"
    print(f"Published {report['dataset_rows']:,} rows and {QUALITY_FILENAME} to {output}")


if __name__ == "__main__":
    main()
