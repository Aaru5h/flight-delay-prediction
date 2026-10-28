"""Phase 2 leakage, weather, history and join-integrity tests."""

import pandas as pd
import pytest

from src.features.history import add_historical_features
from src.features.manifest import ALLOWLIST, DENYLIST
from src.features.pipeline import build_feature_table
from src.features.schedule import join_reference_features
from src.features.weather import join_weather_asof


def _flight(row_id, date, local_time, carrier="AA", origin="EWR", dest="LAX", label=0):
    departure = pd.Timestamp(f"{date} {local_time}")
    return {
        "flight_row_id": row_id,
        "sched_date": pd.Timestamp(date),
        "sched_dep_ts": departure,
        "actual_arr_utc": departure.tz_localize("America/New_York").tz_convert("UTC") + pd.Timedelta(hours=2),
        "carrier": carrier,
        "origin": origin,
        "dest": dest,
        "substantial_delay": label,
    }


def test_denylist_and_allowlist_are_disjoint():
    assert not set(DENYLIST) & set(ALLOWLIST)
    assert {"dep_time", "dep_delay", "arr_time", "arr_delay", "air_time", "tailnum"} <= set(DENYLIST)


def test_weather_uses_latest_past_observation_and_rejects_future_and_stale():
    flights = pd.DataFrame([
        _flight("fresh", "2013-01-01", "10:00"),  # cutoff 14:00Z
        _flight("stale", "2013-01-02", "10:00"),  # cutoff 14:00Z
    ])
    weather = pd.DataFrame([
        {"origin": "EWR", "time_hour": "2013-01-01T13:00:00Z", "temp": 10},
        {"origin": "EWR", "time_hour": "2013-01-01T15:00:00Z", "temp": 99},  # future
        {"origin": "EWR", "time_hour": "2013-01-02T09:00:00Z", "temp": 20},  # 5h stale
    ])
    for col in ["dewp", "humid", "wind_dir", "wind_speed", "wind_gust", "precip", "pressure", "visib"]:
        weather[col] = 0.0
    out = join_weather_asof(flights, weather).set_index("flight_row_id")
    assert out.loc["fresh", "temp"] == 10
    assert out.loc["fresh", "weather_age_minutes"] == 60
    assert not bool(out.loc["fresh", "weather_stale_or_missing"])
    assert pd.isna(out.loc["stale", "temp"])
    assert bool(out.loc["stale", "weather_stale_or_missing"])
    assert out.loc["stale", "weather_age_minutes"] == 300
    observed = out["weather_observed_utc"].dropna()
    assert (observed <= out.loc[observed.index, "prediction_cutoff_utc"]).all()


def test_history_uses_strictly_earlier_dates_only():
    original = pd.DataFrame([
        _flight("a", "2013-01-01", "10:00", label=1),
        _flight("b", "2013-01-02", "10:00", label=0),
        _flight("c", "2013-01-02", "12:00", label=1),
        _flight("d", "2013-01-03", "10:00", label=0),
    ])
    first = add_historical_features(original).set_index("flight_row_id")
    changed = original.copy()
    changed.loc[changed["sched_date"] >= pd.Timestamp("2013-01-02"), "substantial_delay"] = 1
    second = add_historical_features(changed).set_index("flight_row_id")
    # Jan 2 uses Jan 1 only, so changing Jan 2 and later has no effect.
    for row_id in ["b", "c"]:
        assert first.loc[row_id, "carrier_delay_rate"] == second.loc[row_id, "carrier_delay_rate"]
        assert first.loc[row_id, "route_delay_rate"] == second.loc[row_id, "route_delay_rate"]
    assert first.loc["b", "carrier_prior_count"] == 1
    assert first.loc["c", "carrier_prior_count"] == 1  # not 2: same-day b excluded
    assert bool(first.loc["a", "carrier_history_missing"])


def test_reference_joins_preserve_unmatched_rows_and_block_duplicate_keys():
    flights = pd.DataFrame([
        {"flight_row_id": "a", "carrier": "AA", "origin": "EWR", "dest": "XXX"},
        {"flight_row_id": "b", "carrier": "ZZ", "origin": "EWR", "dest": "LAX"},
    ])
    airlines = pd.DataFrame({"carrier": ["AA"], "name": ["American"]})
    airports = pd.DataFrame({
        "faa": ["EWR", "LAX"], "name": ["Newark", "Los Angeles"],
        "lat": [1.0, 2.0], "lon": [3.0, 4.0], "alt": [5, 6], "tz": [-5, -8],
    })
    out, coverage = join_reference_features(flights, airlines, airports)
    assert out["flight_row_id"].tolist() == ["a", "b"]
    assert coverage["unmatched_destination_airport_rows"] == 1
    assert coverage["unmatched_airline_rows"] == 1
    with pytest.raises(ValueError, match="not unique"):
        join_reference_features(flights, pd.concat([airlines, airlines]), airports)


def test_historical_timing_excludes_arrivals_after_cutoff():
    """Verify that prior-date flights landing AFTER target T-60 cutoff are excluded."""
    # Flight 1: Jan 1 departing 10:00 local, scheduled to arrive 12:00 local, 0 delay (actual arr Jan 1 12:00)
    # Flight 2: Jan 1 departing 23:00 local, scheduled to arrive Jan 2 02:00, 240m delay -> lands Jan 2 06:00!
    # Target 3: Jan 2 departing 05:00 local -> cutoff is Jan 2 04:00 local!
    # At Jan 2 04:00, Flight 1 had arrived (12:00 Jan 1 <= 04:00 Jan 2) -> INCLUDED
    # Flight 2 does NOT arrive until 06:00 Jan 2 (> 04:00 Jan 2) -> EXCLUDED!
    flights = pd.DataFrame([
        {
            "flight_row_id": "f1",
            "sched_date": pd.Timestamp("2013-01-01"),
            "sched_dep_ts": pd.Timestamp("2013-01-01 10:00:00"),
            "sched_arr_ts": pd.Timestamp("2013-01-01 12:00:00"),
            "arr_delay": 0.0,
            "actual_arr_utc": pd.Timestamp("2013-01-01 12:00", tz="America/New_York").tz_convert("UTC"),
            "carrier": "AA",
            "origin": "EWR",
            "dest": "ORD",
            "substantial_delay": 0,
        },
        {
            "flight_row_id": "f2",
            "sched_date": pd.Timestamp("2013-01-01"),
            "sched_dep_ts": pd.Timestamp("2013-01-01 23:00:00"),
            "sched_arr_ts": pd.Timestamp("2013-01-02 02:00:00"),
            "arr_delay": 240.0,  # Lands Jan 2 at 06:00!
            "actual_arr_utc": pd.Timestamp("2013-01-02 06:00", tz="America/New_York").tz_convert("UTC"),
            "carrier": "AA",
            "origin": "EWR",
            "dest": "SFO",
            "substantial_delay": 1,
        },
        {
            "flight_row_id": "target",
            "sched_date": pd.Timestamp("2013-01-02"),
            "sched_dep_ts": pd.Timestamp("2013-01-02 05:00:00"),  # Cutoff is 04:00
            "sched_arr_ts": pd.Timestamp("2013-01-02 08:00:00"),
            "arr_delay": 0.0,
            "actual_arr_utc": pd.Timestamp("2013-01-02 08:00", tz="America/New_York").tz_convert("UTC"),
            "carrier": "AA",
            "origin": "EWR",
            "dest": "ORD",
            "substantial_delay": 0,
        },
    ])
    out = add_historical_features(flights).set_index("flight_row_id")
    # For target flight, carrier prior count should only be 1 (Flight 1), NOT 2!
    assert out.loc["target", "carrier_prior_count"] == 1
    assert out.loc["target", "carrier_delay_rate"] is not pd.NA


def test_unavailable_other_carrier_label_cannot_change_smoothed_rate():
    flights = pd.DataFrame([
        _flight("known", "2013-01-01", "10:00", carrier="AA", label=0),
        _flight("late", "2013-01-01", "23:00", carrier="BB", label=1),
        _flight("target", "2013-01-02", "05:00", carrier="AA", label=0),
    ])
    flights.loc[flights.flight_row_id.eq("late"), "actual_arr_utc"] = pd.Timestamp("2013-01-02 11:00", tz="UTC")
    first = add_historical_features(flights).set_index("flight_row_id")
    changed = flights.copy()
    changed.loc[changed.flight_row_id.eq("late"), "substantial_delay"] = 0
    second = add_historical_features(changed).set_index("flight_row_id")
    assert first.loc["target", "carrier_delay_rate"] == second.loc["target", "carrier_delay_rate"]


def test_source_arrival_bound_withholds_previous_day_outcome():
    flights = pd.DataFrame([
        {**_flight("prior", "2013-01-01", "23:00", label=1), "arr_delay": 240.0},
        {**_flight("target", "2013-01-02", "05:00", label=0), "arr_delay": 0.0},
    ]).drop(columns="actual_arr_utc")
    out = add_historical_features(flights).set_index("flight_row_id")
    assert out.loc["target", "carrier_prior_count"] == 0


def test_validation_labels_do_not_feed_later_validation_features():
    flights = pd.DataFrame([
        _flight("train", "2013-08-30", "10:00", label=0),
        _flight("validation_early", "2013-09-02", "10:00", label=1),
        _flight("validation_late", "2013-09-05", "10:00", label=0),
    ])
    first = add_historical_features(flights, history_training_end="2013-08-31").set_index("flight_row_id")
    changed = flights.copy()
    changed.loc[changed.flight_row_id.eq("validation_early"), "substantial_delay"] = 0
    second = add_historical_features(changed, history_training_end="2013-08-31").set_index("flight_row_id")
    assert first.loc["validation_late", "carrier_prior_count"] == 1
    assert first.loc["validation_late", "carrier_delay_rate"] == second.loc["validation_late", "carrier_delay_rate"]


def test_scheduled_volume_includes_unknown_label_flights():
    from types import SimpleNamespace
    flights = pd.DataFrame([
        dict(flight_row_id="known", sched_date="2013-01-01", sched_dep_ts="2013-01-01 10:00", sched_arr_ts="2013-01-01 12:00", sched_dep_minute=600, sched_arr_minute=720, carrier="AA", origin="EWR", dest="LAX", distance=2400, arr_delay=0, substantial_delay=0, label_status="known"),
        dict(flight_row_id="unknown", sched_date="2013-01-01", sched_dep_ts="2013-01-01 10:30", sched_arr_ts="2013-01-01 12:30", sched_dep_minute=630, sched_arr_minute=750, carrier="AA", origin="EWR", dest="LAX", distance=2400, arr_delay=None, substantial_delay=None, label_status="unknown"),
    ])
    wx = pd.DataFrame([dict(origin="EWR", time_hour="2013-01-01T14:00:00Z", **{name: 0 for name in ["temp", "dewp", "humid", "wind_dir", "wind_speed", "wind_gust", "precip", "pressure", "visib"]})])
    airports = pd.DataFrame({"faa": ["EWR", "LAX"], "name": ["Newark", "Los Angeles"], "lat": [40.7, 34.0], "lon": [-74.2, -118.4], "alt": [18, 125], "tz": [-5, -8]})
    sources = {"airlines": SimpleNamespace(df=pd.DataFrame({"carrier": ["AA"], "name": ["American"]})), "airports": SimpleNamespace(df=airports), "weather": SimpleNamespace(df=wx)}
    out, _ = build_feature_table(flights, sources)
    assert len(out) == 1
    assert out.iloc[0].origin_scheduled_hour_volume == 2


def test_scheduled_duration_uses_destination_timezone():
    from src.features.schedule import add_schedule_features
    flights = pd.DataFrame([dict(flight_row_id="a", sched_date=pd.Timestamp("2013-01-01"), sched_dep_ts=pd.Timestamp("2013-01-01 05:15"), sched_dep_minute=315, sched_arr_minute=499, carrier="UA", origin="EWR", dest="IAH")])
    airports = pd.DataFrame({"faa": ["EWR", "IAH"], "name": ["Newark", "Houston"], "lat": [40.7, 30.0], "lon": [-74.2, -95.3], "alt": [18, 97], "tz": [-5, -6], "tzone": ["America/New_York", "America/Chicago"]})
    airlines = pd.DataFrame({"carrier": ["UA"], "name": ["United"]})
    out, _ = join_reference_features(add_schedule_features(flights), airlines, airports)
    assert out.iloc[0].scheduled_duration_minutes == 244
    overnight = flights.copy()
    overnight["sched_dep_ts"] = pd.Timestamp("2013-01-01 23:00")
    overnight["sched_dep_minute"] = 1380
    overnight["sched_arr_minute"] = 30
    next_day, _ = join_reference_features(add_schedule_features(overnight), airlines, airports)
    assert next_day.iloc[0].scheduled_duration_minutes == 150
