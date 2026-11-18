"""Synthetic Step 4 exports for the Step 6 and Step 7 tests."""

from __future__ import annotations

from datetime import timedelta
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.weather import WEATHER_INPUTS
from src.models.neural.train_step6 import CONTRACT_PATH, SETTINGS_PATH

UTC = "%Y-%m-%dT%H:%M:%SZ"
NAMES = ["carrier", "sched_dep_hour", "distance", "lead_time_minutes", "temp"]


def quick_settings(tmp_path: Path) -> Path:
    settings = json.loads(SETTINGS_PATH.read_text())
    settings["training"].update(max_epochs=3, patience_epochs=1, warmup_epochs=1, eval_batch_size=64)
    settings["preprocessing"].update(min_category_count=1)
    settings["search"].update(hidden_layers=[[8]], batch_size=[32])
    settings["sequence"].update(gru_hidden=[4])
    path = tmp_path / "quick_neural.json"
    path.write_text(json.dumps(settings))
    return path


def _weather_events(cutoff: pd.Timestamp, i: int) -> list[dict]:
    if i == 3:
        return []
    events = []
    for minutes in (150, 90, 30):
        event = {"event_utc": (cutoff - timedelta(minutes=minutes)).strftime(UTC)}
        event.update({name: float(i) + minutes / 100 for name in WEATHER_INPUTS})
        event["wind_gust"] = None
        events.append(event)
    return events


def _departure_events(cutoff: pd.Timestamp, i: int) -> list[dict]:
    events = []
    for minutes, delay in ((120, 5.0 * i), (45, 20.0 * (i % 2))):
        event_time = cutoff - timedelta(minutes=minutes)
        events.append({
            "event_utc": event_time.strftime(UTC),
            "carrier": "AA" if minutes == 120 else "UA",
            "dep_delay": delay,
            "available_utc": (event_time + timedelta(minutes=15)).strftime(UTC),
            "minutes_before_cutoff": float(minutes),
        })
    return events


def _padded(events: list[dict], length: int) -> dict:
    padding = length - len(events)
    return {"rows": [None] * padding + events, "mask": [False] * padding + [True] * len(events)}


def write_step4_fixture(tmp_path: Path, *, sequences: bool = False, departures: bool = False):
    """Return (dataset_dir, protocol, manifest) for a tiny four-lead export."""
    protocol = yaml.safe_load(CONTRACT_PATH.read_text())
    leads = protocol["prediction_time"]["lead_time_minutes"]
    root = tmp_path / "step4"
    root.mkdir()
    partitions = []
    for lead in leads:
        rows, records = [], []
        for month in range(1, 13):
            for day in (2, 18, 27):
                for i in range(4):
                    local_date = pd.Timestamp(year=2013, month=month, day=day)
                    scheduled = local_date.tz_localize("UTC") + timedelta(hours=14 + i)
                    cutoff = scheduled - timedelta(minutes=lead)
                    flight = f"flight-{month:02d}-{day:02d}-{i}"
                    rows.append({
                        "flight_row_id": flight,
                        "lead_time_minutes": lead,
                        "sched_date": local_date.date().isoformat(),
                        "prediction_cutoff_utc": cutoff,
                        "outcome_available_utc": scheduled + timedelta(hours=5) if i != 1 else pd.NaT,
                        "substantial_delay": i % 2 if month < 11 else "sealed",
                        "carrier": "AA" if i < 3 else "UA",
                        "sched_dep_hour": 9 + i,
                        "distance": i * 100,
                        "temp": np.nan if i == 3 else float(i % 2) + 0.01 * lead,
                    })
                    records.append({
                        "flight_row_id": flight,
                        "lead_time_minutes": lead,
                        "weather_sequence": _padded(_weather_events(cutoff, i), 12),
                        "departure_sequence": _padded(_departure_events(cutoff, i), 32) if departures else None,
                    })
        features = f"features_{lead}m.csv"
        pd.DataFrame(rows).to_csv(root / features, index=False)
        sequence_file = None
        if sequences:
            sequence_file = f"sequences_{lead}m.jsonl.gz"
            with gzip.open(root / sequence_file, "wt", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
        partitions.append({
            "lead_time_minutes": lead, "features": features, "sequences": sequence_file, "rows": len(rows),
        })
    manifest = {
        "protocol_version": protocol["contract_version"],
        "manifest_version": "asof-fixture",
        "label_column": "substantial_delay",
        "same_day_departures_enabled": departures,
        "allowlist": NAMES,
        "features": [
            {"name": name,
             "type": "categorical" if name in {"carrier", "lead_time_minutes"} else "numeric",
             "availability_rule": (
                 "origin weather valid by cutoff" if name == "temp" else "published schedule and fixed calendar"
             )}
            for name in NAMES
        ],
        "partitions": partitions,
        "source_fingerprints": {"flights": "80ac"},
    }
    (root / "feature_manifest.json").write_text(json.dumps(manifest))
    return root, protocol, manifest


def rewrite_sequence_record(root: Path, lead: int, flight: str, change) -> None:
    """Apply ``change(record)`` to one record of a fixture sequence file."""
    path = root / f"sequences_{lead}m.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    for record in records:
        if record["flight_row_id"] == flight:
            change(record)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
