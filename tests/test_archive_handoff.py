"""Archive ingestion must reject traversal and corrupted retrospective labels."""

from zipfile import ZipFile
import json

import pandas as pd
import pytest

from src.evaluation.archive_handoff import extract_verified, read_archived_retrospective


def test_archive_cannot_escape_destination(tmp_path):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("../escaped.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_verified(archive, tmp_path / "import")
    assert not (tmp_path / "escaped.txt").exists()


def test_retrospective_import_checks_labels_and_complete_leads(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    flights = pd.DataFrame({"flight_row_id": ["a", "b"], "sched_date": ["2013-11-01", "2013-12-02"],
                            "prediction_cutoff_utc": ["2013-10-31T12:00:00Z", "2013-12-01T12:00:00Z"],
                            "outcome_available_utc": ["2013-11-01T17:00:00Z", "2013-12-02T17:00:00Z"],
                            "substantial_delay": [0, 1]})
    flights.to_csv(data / "features_1440m.csv", index=False)
    pred = pd.concat([pd.DataFrame({"flight_row_id": ["a", "b"], "lead_time_minutes": lead,
                                    "fold_id": "retrospective", "split": "retrospective", "y_true": [0, 1],
                                    "y_probability": [0.2, 0.8], "model_name": "mlp", "run_id": "run"})
                      for lead in [60, 120, 360, 1440]], ignore_index=True)
    path = tmp_path / "predictions.csv"
    pred.to_csv(path, index=False)
    model = {"retrospective_predictions_path": str(path), "run_id": "run", "name": "mlp"}
    result = read_archived_retrospective(model, data)
    assert len(result) == 8 and "y_probability_raw" in result
    pred.loc[0, "y_true"] = 1
    pred.to_csv(path, index=False)
    with pytest.raises(ValueError, match="labels"):
        read_archived_retrospective(model, data)
