"""Delay-label tests (Phase 1, Step 5): threshold boundary and unknown labels."""

import pandas as pd

from src.data.labeling import add_delay_label, label_summary, labelled_subset

from tests.phase1_helpers import flights_rows


def _labelled(rows):
    return add_delay_label(flights_rows(rows))


def test_threshold_boundary_is_exactly_15_minutes():
    out = _labelled([
        dict(arr_delay=14.0),
        dict(arr_delay=15.0),
        dict(arr_delay=-30.0),   # early arrival
        dict(arr_delay=0.0),     # exactly on time
    ])
    assert out["substantial_delay"].tolist() == [0, 1, 0, 0]


def test_missing_arr_delay_is_unknown_never_zero():
    out = _labelled([
        dict(arr_delay=pd.NA),
        dict(arr_delay=5.0),
    ])
    assert out["substantial_delay"].tolist() == [pd.NA, 0]
    assert out["label_status"].tolist() == ["unknown", "known"]


def test_label_summary_reconciles_and_reports_missingness_pattern():
    out = _labelled([
        dict(arr_delay=20.0, dep_time=517.0),
        dict(arr_delay=5.0, dep_time=533.0),
        dict(arr_delay=pd.NA, dep_time=pd.NA),      # likely cancelled/diverted/lost
        dict(arr_delay=pd.NA, dep_time=810.0),      # departed but no arrival record
    ])
    summary = label_summary(out)
    assert summary["total_rows"] == 4
    assert summary["known_labels"] == 2
    assert summary["positive"] == 1
    assert summary["negative"] == 1
    assert summary["unknown_labels"] == 2
    assert summary["positive_prevalence"] == 0.5
    assert summary["unknown_label_detail"] == {
        "also_missing_dep_time": 1,
        "dep_time_present": 1,
    }


def test_labelled_subset_keeps_ids_and_excludes_unknown():
    from src.data.schedule import add_flight_row_id

    out = add_flight_row_id(_labelled([
        dict(arr_delay=20.0, carrier="UA"),
        dict(arr_delay=pd.NA, carrier="UA"),
    ]), "a" * 64)
    subset = labelled_subset(out)
    assert len(subset) == 1
    assert "flight_row_id" in subset.columns
    assert subset["flight_row_id"].nunique() == 1


def test_arr_delay_audit_column_is_untouched():
    out = _labelled([dict(arr_delay=15.0), dict(arr_delay=pd.NA)])
    # The audit column must survive so exclusions can be traced back to arr_delay.
    assert out["arr_delay"].tolist()[0] == 15.0
    assert pd.isna(out["arr_delay"].tolist()[1])
