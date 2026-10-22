"""Profiling tests (Phase 1, Step 6): missingness, duplicates, join coverage."""

import pandas as pd

from src.data.loading import load_all_sources
from src.data.profiling import (
    column_missingness,
    count_full_duplicates,
    date_coverage,
    duplicate_key_counts,
    join_coverage,
    monthly_label_distribution,
)
from src.data.schedule import add_flight_row_id, add_schedule_fields

from tests.phase1_helpers import flights_rows, write_synthetic_sources


def test_missingness_counts_and_skips_complete_columns():
    df = flights_rows([
        dict(carrier="UA", arr_delay=1.0, tailnum="N1"),
        dict(carrier="UA", arr_delay=pd.NA, tailnum="N2"),
        dict(carrier="AA", arr_delay=pd.NA, tailnum=pd.NA),
    ])
    miss = column_missingness(df)
    assert miss["arr_delay"]["missing"] == 2
    assert miss["tailnum"]["missing"] == 1
    assert "carrier" not in miss  # carrier filled for every row


def test_full_duplicate_detection_ignores_synthetic_ids():
    df = flights_rows([
        dict(carrier="UA", flight=1, arr_delay=5.0),
        dict(carrier="UA", flight=1, arr_delay=5.0),  # exact duplicate
        dict(carrier="UA", flight=2, arr_delay=5.0),
    ])
    df = add_flight_row_id(df, "a" * 64)
    assert count_full_duplicates(df) == 2  # both members of the duplicate pair


def test_duplicate_key_detection_for_lookups(tmp_path):
    write_synthetic_sources(tmp_path)
    sources = load_all_sources(tmp_path)
    dup = duplicate_key_counts(sources)
    assert all(v["duplicate_key_rows"] == 0 for v in dup.values())

    # Add a duplicated carrier key and confirm it is caught.
    sources["airlines"].df = pd.concat([
        sources["airlines"].df,
        pd.DataFrame({"carrier": ["UA"], "name": ["United duplicate"]}),
    ], ignore_index=True)
    dup = duplicate_key_counts(sources)
    assert dup["airlines"]["duplicate_key_rows"] == 2


def test_join_coverage_reports_unmatched_destination_codes(tmp_path):
    write_synthetic_sources(tmp_path)
    sources = load_all_sources(tmp_path)
    flights = add_schedule_fields(add_flight_row_id(sources["flights"].df, sources["flights"].fingerprint))
    coverage = join_coverage(flights, sources)

    # Synthetic flights use BQN and STT, which are absent from the airport lookup.
    dest = coverage["airports_destination"]["unmatched_airports"]
    assert set(dest["airport_codes"]) == {"BQN", "STT"}
    assert dest["rows"] == 2
    # Origins all match.
    assert coverage["airports_origin"]["unmatched_airports"]["rows"] == 0
    # Every carrier matches the airline lookup.
    assert coverage["airlines"]["unmatched_carriers"]["rows"] == 0
    # tailnum N1AA appears twice and matches; the missing tailnum is excluded
    # from the non-missing count.
    assert coverage["planes"]["non_missing_tailnum_rows"] == 4
    assert coverage["planes"]["unmatched_rows"] == 0
    # Same-hour weather observations exist for all five synthetic flights.
    assert coverage["weather"]["rows_with_same_hour_observation"] == 5


def test_date_and_monthly_coverage(tmp_path):
    from src.data.labeling import add_delay_label

    write_synthetic_sources(tmp_path)
    sources = load_all_sources(tmp_path)
    flights = add_delay_label(add_schedule_fields(sources["flights"].df))
    dates = date_coverage(flights)
    assert dates["min"] == "2013-01-01"
    assert dates["max"] == "2013-03-03"
    assert dates["distinct_days"] == 3

    monthly = monthly_label_distribution(flights)
    by_month = {r["month"]: r for r in monthly}
    assert by_month[1]["labelled"] == 2 and by_month[1]["positive"] == 1
    assert by_month[2]["positive"] == 1
    # March has one known negative, one unknown (excluded from prevalence).
    assert by_month[3]["labelled"] == 1 and by_month[3]["negative"] == 1
