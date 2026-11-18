"""Step 4 integration tests using the real Phase 1 and Step 3 entry points."""

import gzip
import json

import pandas as pd
import pytest

from src.data.prepare import run_phase1
from src.features.asof import AsOfFeatureBuilder, LEAD_TIMES_MINUTES
from src.features.build_multi_lead import run_step4
from tests.phase1_helpers import write_synthetic_sources


@pytest.fixture
def labelled_sources(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    write_synthetic_sources(raw)
    flights_path = raw / "nyc_flights.csv"
    flights = pd.read_csv(flights_path)
    flights.loc[0, ["dep_time", "dep_delay", "arr_time"]] = [515, 0, 830]
    flights.loc[1, ["origin", "sched_dep_time", "dep_time", "dep_delay", "arr_time"]] = [
        "EWR", 729, 729, 0, 850,
    ]
    flights.loc[2, "sched_dep_time"] = 2400
    flights.to_csv(flights_path, index=False)
    weather_path = raw / "nyc_weather.csv"
    weather = pd.read_csv(weather_path)
    weather.loc[0, "time_hour"] = "2013-01-01T09:00:00Z"
    weather.to_csv(weather_path, index=False)
    run_phase1(raw_dir=raw, processed_dir=processed, reports_dir=tmp_path / "reports")
    return raw, processed / "labelled_flights.csv"


def test_step4_exports_all_leads_and_reconciles_quality_by_flight(tmp_path, labelled_sources):
    raw, labels = labelled_sources
    labelled = pd.read_csv(labels)
    # File order is deliberately different so the join must use the stable ID.
    labelled.iloc[::-1].to_csv(labels, index=False)
    output = tmp_path / "step4"
    report = run_step4(raw, labels, output, batch_size=2)
    assert report == json.loads((output / "data_quality.json").read_text())
    manifest = json.loads((output / "feature_manifest.json").read_text())
    assert manifest["grain"] == ["flight_row_id", "lead_time_minutes"]
    assert set(manifest["audit_columns"]).isdisjoint(manifest["allowlist"])
    assert "substantial_delay" not in manifest["allowlist"]
    assert "lead_time_minutes" in manifest["allowlist"]
    assert report["feature_manifest_sha256"] == manifest["manifest_sha256"]
    assert report["source_flights"] == 5
    assert report["known_label_flights"] == 4
    assert report["unknown_label_flights"] == 1
    assert report["invalid_departure_flights"] == 2
    assert report["excluded_known_label_invalid_departure_flights"] == 1
    assert report["eligible_flights"] == 3
    assert report["eligible_flights_missing_outcome_availability"] == 1
    assert report["dataset_rows"] == 12

    expected = labelled.iloc[[0, 1, 4]].set_index("flight_row_id")["substantial_delay"]
    observed_leads = []
    for partition, quality in zip(manifest["partitions"], report["lead_times"], strict=True):
        frame = pd.read_csv(output / partition["features"]).set_index("flight_row_id")
        lead = partition["lead_time_minutes"]
        observed_leads.append(lead)
        assert partition["sequences"] is None
        assert quality["rows"] == len(frame) == 3
        assert quality["positive_labels"] == 1
        assert quality["negative_labels"] == 2
        assert quality["missing_feature_values"]["temp"] == int(frame["temp"].isna().sum())
        assert not frame["lead_time_minutes"].ne(lead).any()
        assert frame["sched_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
        pd.testing.assert_series_equal(frame["substantial_delay"], expected.astype(int))
        departure = pd.to_datetime(frame["scheduled_departure_utc"], utc=True)
        cutoff = pd.to_datetime(frame["prediction_cutoff_utc"], utc=True)
        assert (departure - cutoff).dt.total_seconds().eq(lead * 60).all()
    assert tuple(observed_leads) == LEAD_TIMES_MINUTES


def test_step4_exports_keyed_sequences_with_utc_audit_and_null_padding(tmp_path, labelled_sources):
    raw, labels = labelled_sources
    output = tmp_path / "step4"
    report = run_step4(
        raw, labels, output, include_sequences=True, allow_same_day_departures=True
    )
    manifest = json.loads((output / "feature_manifest.json").read_text())
    partition = manifest["partitions"][-1]
    quality = report["lead_times"][-1]
    features = pd.read_csv(output / partition["features"])
    with gzip.open(output / partition["sequences"], "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == quality["rows"] == 3
    assert [row["flight_row_id"] for row in records] == features["flight_row_id"].tolist()
    assert all(row["lead_time_minutes"] == 60 for row in records)
    assert quality["rows_with_weather_sequence"] == 2
    assert quality["rows_with_departure_sequence"] == 2
    departure = records[1]["departure_sequence"]
    assert sum(departure["mask"]) == 1
    assert departure["rows"][0] is None
    cutoff = features.loc[1, "prediction_cutoff_utc"]
    assert departure["rows"][-1]["available_utc"] <= cutoff
    assert records[1]["weather_sequence"]["rows"][-1]["event_utc"] <= cutoff


def test_step4_rejects_mislabeled_input_and_publishes_no_partial_bundle(tmp_path, labelled_sources):
    raw, labels = labelled_sources
    output = tmp_path / "step4"
    labelled = pd.read_csv(labels)
    labelled.loc[0, "substantial_delay"] = 1
    labelled.to_csv(labels, index=False)
    with pytest.raises(ValueError, match="labels disagree"):
        run_step4(raw, labels, output)
    assert not output.exists()


def test_step4_mid_run_failure_leaves_output_available_for_retry(
    tmp_path, labelled_sources, monkeypatch
):
    raw, labels = labelled_sources
    output = tmp_path / "step4"
    original = AsOfFeatureBuilder.build

    def fail_second_partition(self, flight_row_id, cutoff_utc, **kwargs):
        result = original(self, flight_row_id, cutoff_utc, **kwargs)
        if result.features["lead_time_minutes"] == 360:
            raise RuntimeError("simulated write interruption")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(AsOfFeatureBuilder, "build", fail_second_partition)
        with pytest.raises(RuntimeError, match="write interruption"):
            run_step4(raw, labels, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".step4-*"))
    assert run_step4(raw, labels, output)["dataset_rows"] == 12
