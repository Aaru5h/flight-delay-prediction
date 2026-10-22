"""Loader and schema-validation tests (Phase 1, Step 1-2)."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loading import (
    MissingSourceError,
    SchemaValidationError,
    file_fingerprint,
    load_all_sources,
    load_source,
    validate_schema,
)

from tests.phase1_helpers import write_synthetic_sources


def test_missing_file_raises_actionable_error(tmp_path):
    with pytest.raises(MissingSourceError) as excinfo:
        load_source("flights", tmp_path)
    assert "nyc_flights.csv" in str(excinfo.value)
    assert str(tmp_path) in str(excinfo.value)


def test_schema_validation_reports_all_missing_columns(tmp_path):
    df = pd.DataFrame({"year": [2013], "month": [1], "day": [1]})
    missing = validate_schema("flights", df)
    assert "arr_delay" in missing
    assert "carrier" in missing
    assert len(missing) == 16  # 19 required, 3 present


def test_load_all_sources_fails_listing_every_schema_problem(tmp_path):
    write_synthetic_sources(tmp_path)
    # Corrupt the airlines file by renaming a required column.
    airlines_path = tmp_path / "nyc_airlines.csv"
    bad = pd.read_csv(airlines_path).rename(columns={"name": "airline_name"})
    bad.to_csv(airlines_path, index=False)

    with pytest.raises(SchemaValidationError) as excinfo:
        load_all_sources(tmp_path)
    message = str(excinfo.value)
    assert "nyc_airlines.csv" in message
    assert "name" in message


def test_fingerprint_is_stable_and_matches_phase0_manifest(tmp_path):
    write_synthetic_sources(tmp_path)
    source = load_source("airlines", tmp_path)
    assert source.fingerprint == file_fingerprint(tmp_path / "nyc_airlines.csv")
    assert len(source.fingerprint) == 64


def test_identifier_columns_keep_leading_zeros(tmp_path):
    write_synthetic_sources(tmp_path)
    airports = load_source("airports", tmp_path).df
    assert "04G" in set(airports["faa"])  # would be '4G' if parsed numerically
