"""flight_row_id and schedule-parsing tests (Phase 1, Steps 3-4)."""

import pandas as pd

from src.data.schedule import add_flight_row_id, add_schedule_fields, parse_hhmm

from tests.phase1_helpers import flights_rows


def test_flight_row_id_is_unique_and_position_based():
    df = flights_rows([
        dict(year=2013, month=1, day=1, carrier="UA", arr_delay=5.0),
        dict(year=2013, month=1, day=2, carrier="UA", arr_delay=5.0),
    ])
    out = add_flight_row_id(df, "a" * 64)
    assert out["flight_row_id"].tolist() == [
        f"flights-aaaaaaaa-00000{i}" for i in (1, 2)
    ]


def test_flight_row_id_survives_filtering_and_differs_for_identical_rows():
    df = flights_rows([
        dict(year=2013, month=1, day=1, carrier="UA", arr_delay=5.0),
        dict(year=2013, month=1, day=2, carrier="UA", arr_delay=5.0),
    ])
    out = add_flight_row_id(df, "a" * 64)
    subset = out[out["arr_delay"] == 5.0]
    assert subset["flight_row_id"].nunique() == 2  # identical rows, distinct IDs
    assert not out["flight_row_id"].isna().any()


def test_flight_row_id_refuses_to_overwrite():
    df = flights_rows([dict(year=2013, month=1, day=1)])
    df["flight_row_id"] = "existing"
    try:
        add_flight_row_id(df, "a" * 64)
    except ValueError as exc:
        assert "refusing" in str(exc)
    else:
        raise AssertionError("expected ValueError for existing flight_row_id")


def test_parse_hhmm_valid_values_and_boundaries():
    minutes, invalid = parse_hhmm(pd.Series([0, 5, 515, 2359, 130], dtype="Int64"))
    assert minutes.tolist() == [0, 5, 315, 1439, 90]
    assert not invalid.any()


def test_parse_hhmm_flags_impossible_values():
    minutes, invalid = parse_hhmm(pd.Series([2400, 775, -15, 9999]))
    assert minutes.isna().all()  # never silently normalized
    assert invalid.tolist() == [True, True, True, True]  # 775 -> 7:75, bad minutes


def test_parse_hhmm_missing_values_are_not_invalid():
    minutes, invalid = parse_hhmm(pd.Series([pd.NA, None, 515], dtype="Int64"))
    assert minutes.tolist()[:2] == [pd.NA, pd.NA]
    assert minutes.tolist()[2] == 315
    assert invalid.tolist()[:2] == [False, False]
    assert not invalid.iloc[2]


def test_schedule_fields_build_local_timestamps():
    df = flights_rows([
        dict(year=2013, month=1, day=1, sched_dep_time=515, sched_arr_time=819),
    ])
    out = add_schedule_fields(add_flight_row_id(df, "a" * 64))
    row = out.iloc[0]
    assert row["sched_date"] == pd.Timestamp("2013-01-01")
    assert row["sched_dep_minute"] == 315
    assert row["sched_arr_minute"] == 499
    assert row["sched_dep_ts"] == pd.Timestamp("2013-01-01 05:15:00")
    assert row["sched_arr_ts"] == pd.Timestamp("2013-01-01 08:19:00")
    assert not row["sched_dep_time_invalid"]
    assert not row["sched_arr_time_invalid"]


def test_overnight_arrival_moves_to_next_day():
    df = flights_rows([
        # departs 22:30, scheduled to arrive 00:45 the following day
        dict(year=2013, month=7, day=4, sched_dep_time=2230, sched_arr_time=45),
    ])
    out = add_schedule_fields(df)
    row = out.iloc[0]
    assert row["sched_dep_ts"] == pd.Timestamp("2013-07-04 22:30:00")
    assert row["sched_arr_ts"] == pd.Timestamp("2013-07-05 00:45:00")


def test_invalid_or_missing_schedule_times_stay_flagged_not_filled():
    df = flights_rows([
        dict(year=2013, month=3, day=3, sched_dep_time=2400, sched_arr_time=pd.NA),
    ])
    out = add_schedule_fields(df)
    row = out.iloc[0]
    assert row["sched_dep_time_invalid"]
    assert pd.isna(row["sched_dep_ts"])
    assert pd.isna(row["sched_dep_minute"])
    assert pd.isna(row["sched_arr_minute"])
    assert pd.isna(row["sched_arr_ts"])
    assert not row["sched_arr_time_invalid"]  # missing is not invalid
