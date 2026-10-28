"""Phase 2 leakage, weather, history and join-integrity tests."""

import pandas as pd
import pytest

from src.features.history import add_historical_features
from src.features.manifest import ALLOWLIST, DENYLIST
from src.features.schedule import join_reference_features
from src.features.weather import join_weather_asof


def _flight(row_id, date, local_time, carrier="AA", origin="EWR", dest="LAX", label=0):
    return {
        "flight_row_id": row_id,
        "sched_date": pd.Timestamp(date),
        "sched_dep_ts": pd.Timestamp(f"{date} {local_time}"),
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