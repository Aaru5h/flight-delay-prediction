"""Point-in-time boundaries and handoff rules for the v2 feature builder."""

from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from src.features.asof import AsOfFeatureBuilder, LEAD_TIMES_MINUTES
from src.features.weather import WEATHER_INPUTS

TARGET_ID = "flights-12345678-000001"
CUTOFF = pd.Timestamp("2013-11-05T18:00:00Z")


def _flight(day, **overrides):
    date = pd.Timestamp(day)
    return {
        "year": date.year, "month": date.month, "day": date.day,
        "origin": "JFK", "dest": "ORD", "carrier": "AA", "distance": 740,
        "sched_dep_time": 1000, "dep_time": 1000, "dep_delay": 0,
        "sched_arr_time": 1130, "arr_time": 1130, "arr_delay": 0,
        **overrides,
    }


def _weather(hour, temp, origin="JFK"):
    return {
        **dict.fromkeys(WEATHER_INPUTS, 1.0),
        "time_hour": f"2013-11-05T{hour:02d}:00:00Z", "origin": origin, "temp": float(temp),
    }


def _frame_sources(flights, weather):
    airports = pd.DataFrame({
        "faa": ["JFK", "LGA", "ORD"],
        "name": ["Kennedy", "LaGuardia", "Chicago"],
        "tzone": ["America/New_York", "America/New_York", "America/Chicago"],
        "lat": [40.0, 40.0, 41.0], "lon": [-74.0, -74.0, -88.0],
        "alt": [10, 10, 200], "tz": [-5, -5, -6],
    })
    return {
        "flights": SimpleNamespace(df=flights, fingerprint="12345678"),
        "weather": SimpleNamespace(df=weather),
        "airlines": SimpleNamespace(df=pd.DataFrame({"carrier": ["AA"], "name": ["American"]})),
        "airports": SimpleNamespace(df=airports),
    }


@pytest.fixture
def inputs():
    flights = pd.DataFrame([
        _flight("2013-11-05", sched_dep_time=1400, dep_time=1400, sched_arr_time=1530, arr_time=1530),
        _flight("2013-11-05", arr_time=1145, arr_delay=15),  # available exactly at 18:00Z
        _flight("2013-11-05", arr_time=1150, arr_delay=20),  # available after cutoff
        _flight("2013-11-05", sched_dep_time=800, dep_time=800),
        _flight("2013-10-25", arr_time=1150, arr_delay=20),
        _flight("2013-09-12"),
        _flight("2013-11-05", origin="LGA"),
        _flight("2013-11-05", sched_dep_time=1300, dep_time=1300,  # departed, buffer pending
                sched_arr_time=1430, arr_time=1430),
        _flight("2013-11-05", dest="UNK", arr_time=1130),
    ])
    weather = pd.DataFrame([
        _weather(12, 40), _weather(15, 43), _weather(18, 52), _weather(19, 99),
        _weather(17, 90, origin="LGA"),
    ])
    return flights, weather


def test_rolling_windows_and_weather_change_by_lead_time(inputs):
    builder = AsOfFeatureBuilder(_frame_sources(*inputs))
    result = builder.build(TARGET_ID, CUTOFF, include_sequences=True)
    features = result.features
    assert features["route"] == "JFK-ORD"
    assert features["sched_dep_hour"] == 14
    assert features["lead_time_minutes"] == 60
    assert features["route_prior_count_7d"] == 2
    assert features["route_prior_count_28d"] == 3
    assert features["carrier_prior_count_7d"] == 3
    assert features["route_delay_rate_7d"] == pytest.approx((1 + 20 / 3) / 22)
    assert builder.outcome_available_utc.iloc[1] == CUTOFF
    assert builder.outcome_available_utc.iloc[2] > CUTOFF
    assert pd.isna(builder.outcome_available_utc.iloc[-1])
    assert features["temp"] == 52
    assert features["weather_temp_trend_3h"] == pytest.approx(3)
    assert features["weather_temp_trend_6h"] == pytest.approx(2)
    assert result.weather_observed_utc == CUTOFF
    assert result.departure_sequence is None
    assert result.weather_sequence.mask == (False,) * 9 + (True,) * 3
    assert all(row["event_utc"] <= CUTOFF for row in result.weather_sequence.rows if row)
    assert set(features) == set(builder.feature_manifest()["allowlist"])
    assert not {"sched_month", "sched_day_of_year", "arr_delay", "substantial_delay", "dep_delay"} & features.keys()

    departure = builder.flights.loc[TARGET_ID, "scheduled_departure_utc"]
    queries = {lead: builder.build(TARGET_ID, departure - timedelta(minutes=lead)) for lead in LEAD_TIMES_MINUTES}
    assert [result.features["lead_time_minutes"] for result in queries.values()] == list(LEAD_TIMES_MINUTES)
    assert queries[1440].features["route_prior_count_7d"] == 0
    assert queries[1440].features["weather_stale_or_missing"]


def test_departure_inputs_require_opt_in_and_exclude_target_flight(inputs):
    flights, weather = inputs
    early_target = flights.copy()
    early_target.loc[0, ["dep_time", "dep_delay", "arr_time", "arr_delay"]] = [800, -360, 930, -360]
    sources = _frame_sources(early_target, weather)
    default = AsOfFeatureBuilder(sources).build(TARGET_ID, CUTOFF, include_sequences=True)
    assert "origin_departures_so_far" not in default.features
    assert default.departure_sequence is None

    builder = AsOfFeatureBuilder(sources, allow_same_day_departures=True)
    assert builder.outcome_available_utc.loc[TARGET_ID] < CUTOFF
    result = builder.build(TARGET_ID, CUTOFF, include_sequences=True)
    assert result.features["route_prior_count_7d"] == 2
    assert result.features["origin_departures_so_far"] == 4
    assert result.features["origin_mean_dep_delay_so_far"] == 0
    sequence = result.departure_sequence
    actual = [row for row in sequence.rows if row]
    assert sum(sequence.mask) == len(actual) == 6
    assert len(sequence.mask) == 32
    assert all(row["available_utc"] <= CUTOFF for row in actual)
    assert all(row["minutes_before_cutoff"] >= 15 for row in actual)
    assert {row["dep_delay"] for row in actual} == {0}
    assert set(result.features) == set(builder.feature_manifest()["allowlist"])


def test_future_perturbations_leave_inputs_and_fixed_predictions_unchanged(inputs):
    flights, weather = inputs
    changed_flights = flights.copy()
    changed_weather = weather.copy()
    changed_flights.loc[2, ["arr_time", "arr_delay"]] = [1315, 105]
    changed_flights.loc[7, ["dep_time", "dep_delay", "arr_time", "arr_delay"]] = [1330, 30, 1500, 30]
    changed_weather.loc[3, "temp"] = -500

    def query(flights, weather):
        builder = AsOfFeatureBuilder(_frame_sources(flights, weather), allow_same_day_departures=True)
        return builder.build(TARGET_ID, CUTOFF, include_sequences=True)

    original = query(flights, weather)
    changed = query(changed_flights, changed_weather)
    pd.testing.assert_series_equal(pd.Series(original.features), pd.Series(changed.features))
    assert original.weather_sequence == changed.weather_sequence
    assert original.departure_sequence == changed.departure_sequence

    def predict(features):
        return 0.4 * features["route_delay_rate_7d"] + 0.01 * features["temp"]

    assert predict(original.features) == predict(changed.features)


def test_stale_weather_and_invalid_cutoffs(inputs):
    flights, weather = inputs
    builder = AsOfFeatureBuilder(_frame_sources(flights, weather.iloc[[0]]))
    result = builder.build(TARGET_ID, CUTOFF, include_sequences=True)
    assert result.features["weather_stale_or_missing"]
    assert pd.isna(result.features["temp"])
    assert pd.isna(result.features["weather_temp_trend_6h"])
    assert result.weather_sequence.mask == (False,) * 12
    assert result.weather_observed_utc == pd.Timestamp("2013-11-05T12:00:00Z")
    with pytest.raises(ValueError, match="timezone-aware"):
        builder.build(TARGET_ID, "2013-11-05T18:00:00")
    with pytest.raises(ValueError, match="valid scheduled departure"):
        builder.build(TARGET_ID, CUTOFF - timedelta(minutes=1))


def test_arrival_clocks_are_verified_across_the_date_boundary(inputs):
    flights, weather = inputs
    overnight = _flight(
        "2013-11-04", sched_dep_time=2300, dep_time=2400, dep_delay=60,
        sched_arr_time=30, arr_time=45, arr_delay=15,
    )
    wrong_clock = {**overnight, "arr_time": 1345}
    source = pd.concat([flights, pd.DataFrame([overnight, wrong_clock])], ignore_index=True)
    builder = AsOfFeatureBuilder(_frame_sources(source, weather))
    assert builder.outcome_available_utc.iloc[-2] == pd.Timestamp("2013-11-05T07:00:00Z")
    assert pd.isna(builder.outcome_available_utc.iloc[-1])
