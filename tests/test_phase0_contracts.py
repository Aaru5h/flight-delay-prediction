"""Phase 0 contract tests.

These verify that the raw inputs are unchanged, that the frozen contract file
is coherent, and that the chronological split profile reconciles with the
supplied flights CSV. Reference counts are validation expectations for the
supplied data version (see configs/project_contract.yaml), not runtime
constants for pipeline code.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CONTRACT_PATH = ROOT / "configs" / "project_contract.yaml"
MANIFEST_PATH = ROOT / "configs" / "data_manifest.json"
SPLIT_PROFILE_PATH = ROOT / "reports" / "phase0_split_profile.csv"

EXPECTED_FLIGHTS_HEADER = [
    "year", "month", "day", "dep_time", "sched_dep_time", "dep_delay",
    "arr_time", "sched_arr_time", "arr_delay", "carrier", "flight",
    "tailnum", "origin", "dest", "air_time", "distance", "hour",
    "minute", "time_hour",
]


def load_contract():
    with open(CONTRACT_PATH) as fh:
        return yaml.safe_load(fh)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_flights():
    with open(RAW / "nyc_flights.csv", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        for row in reader:
            yield idx, row


def test_raw_files_match_data_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert {f["file"] for f in manifest["files"]} == {
        "nyc_flights.csv", "nyc_airlines.csv", "nyc_airports.csv",
        "nyc_planes.csv", "nyc_weather.csv",
    }
    for entry in manifest["files"]:
        path = RAW / entry["file"]
        assert path.exists(), f"missing raw input: {entry['file']}"
        assert sha256_of(path) == entry["sha256"], (
            f"{entry['file']} changed since the manifest was written"
        )


def test_flights_header_matches_documented_schema():
    with open(RAW / "nyc_flights.csv", newline="") as fh:
        header = next(csv.reader(fh))
    assert header == EXPECTED_FLIGHTS_HEADER


def test_contract_freezes_expected_values():
    contract = load_contract()
    assert contract["target"]["threshold_minutes"] == 15
    assert contract["target"]["missing_policy"] == "unknown_label_excluded"
    assert contract["prediction_time"]["cutoff_minutes_before_scheduled_departure"] == 60
    assert contract["prediction_time"]["weather"]["max_age_minutes"] == 180
    assert set(contract["prediction_time"]["prohibited_features"]) == {
        "dep_time", "dep_delay", "arr_time", "arr_delay", "air_time",
    }
    assert contract["prediction_time"]["conditional_features"]["tailnum"]["status"] == "excluded"
    assert "strictly earlier" in contract["prediction_time"]["historical_features"]["window"]
    assert contract["split"]["status"] == "frozen_for_model_comparison"
    assert contract["metrics"]["operational"]["threshold_selection_data"] == "validation_only"


def test_split_profile_reconciles_with_flights_csv():
    def split_for(m, d):
        if (m, d) <= (8, 31):
            return "train"
        if (m, d) <= (10, 31):
            return "validation"
        return "test"

    agg = defaultdict(lambda: [0, 0, 0, 0])  # rows, labelled, positive, unknown
    for idx, row in iter_flights():
        m, d = int(row[idx["month"]]), int(row[idx["day"]])
        cell = agg[(m, split_for(m, d))]
        cell[0] += 1
        ad = row[idx["arr_delay"]].strip()
        if ad in ("", "NA"):
            cell[3] += 1
        else:
            cell[1] += 1
            if float(ad) >= 15:
                cell[2] += 1

    with open(SPLIT_PROFILE_PATH, newline="") as fh:
        saved = {(int(r["month"]), r["split"]): r for r in csv.DictReader(fh)}

    assert set(saved) == set(agg), "split profile months/splits diverged from the data"
    for key, (rows, labelled, positive, unknown) in agg.items():
        s = saved[key]
        assert int(s["rows"]) == rows, key
        assert int(s["labelled"]) == labelled, key
        assert int(s["positive"]) == positive, key
        assert int(s["unknown_label"]) == unknown, key
        assert int(s["negative"]) == labelled - positive, key

    total_rows = sum(v[0] for v in agg.values())
    total_labelled = sum(v[1] for v in agg.values())
    total_positive = sum(v[2] for v in agg.values())
    total_unknown = sum(v[3] for v in agg.values())
    assert total_rows == 336_776
    assert (total_labelled, total_positive, total_unknown) == (327_346, 80_100, 9_430)
    assert total_labelled + total_unknown == total_rows


def test_reference_profile_matches_supplied_data():
    contract = load_contract()
    ref = contract["data"]["reference_profile"]

    counts = defaultdict(int)
    for idx, row in iter_flights():
        ad = row[idx["arr_delay"]].strip()
        if ad in ("", "NA"):
            counts["unknown_labels"] += 1
        else:
            counts["known_labels"] += 1
            if float(ad) >= 15:
                counts["positive"] += 1
            else:
                counts["negative"] += 1

    assert counts["known_labels"] == ref["known_labels"]
    assert counts["positive"] == ref["positive"]
    assert counts["negative"] == ref["negative"]
    assert counts["unknown_labels"] == ref["unknown_labels"]
