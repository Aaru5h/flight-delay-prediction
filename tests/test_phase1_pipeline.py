"""End-to-end Phase 1 pipeline test on synthetic raw files."""

import json

import pandas as pd

from src.data.prepare import run_phase1

from tests.phase1_helpers import write_synthetic_sources


def test_pipeline_produces_labelled_dataset_and_reports(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    write_synthetic_sources(raw)

    report = run_phase1(raw_dir=raw, processed_dir=processed, reports_dir=reports)

    # Labelled dataset exists, has the stable ID and reconciles.
    labelled = pd.read_csv(processed / "labelled_flights.csv")
    assert len(labelled) == 5
    assert labelled["flight_row_id"].is_unique
    assert (labelled["label_status"] == "known").sum() == 4
    assert (labelled["substantial_delay"] == 1).sum() == 2
    assert (labelled["substantial_delay"] == 0).sum() == 2
    assert labelled["substantial_delay"].isna().sum() == 1

    # Reports exist and agree with the measured data.
    json_report = json.loads((reports / "phase1_data_quality.json").read_text())
    assert json_report["label_summary"]["total_rows"] == 5
    assert json_report["schedule_parsing"]["sched_dep_time_invalid"] == 1
    checks = json_report["reference_profile_checks"]
    # Synthetic data intentionally differs from the contract reference profile.
    assert checks["total_rows"]["matches"] is False

    md_text = (reports / "phase1_data_quality.md").read_text()
    assert "# Phase 1 — Data Quality and Label Report" in md_text
    assert "Reconciliation" in md_text

    # Run manifest records fingerprints for every source file.
    manifest = json.loads((processed / "phase1_run_manifest.json").read_text())
    assert set(manifest["source_fingerprints"]) == {
        "flights", "airlines", "airports", "planes", "weather"
    }


def test_pipeline_is_deterministic(tmp_path):
    raw = tmp_path / "raw"
    write_synthetic_sources(raw)

    first = run_phase1(raw_dir=raw, processed_dir=tmp_path / "a", reports_dir=tmp_path / "r1")
    second = run_phase1(raw_dir=raw, processed_dir=tmp_path / "b", reports_dir=tmp_path / "r2")

    # Same inputs -> same report content and same row IDs.
    first.pop("reference_profile_checks"), second.pop("reference_profile_checks")
    assert first["label_summary"] == second["label_summary"]
    assert first["source_files"] == second["source_files"]

    ids_a = pd.read_csv(tmp_path / "a" / "labelled_flights.csv", usecols=["flight_row_id"])
    ids_b = pd.read_csv(tmp_path / "b" / "labelled_flights.csv", usecols=["flight_row_id"])
    assert ids_a["flight_row_id"].tolist() == ids_b["flight_row_id"].tolist()
