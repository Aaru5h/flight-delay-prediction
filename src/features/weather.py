"""Origin-weather as-of join at the T-60 prediction cutoff."""

from __future__ import annotations

import datetime as dt

import pandas as pd

WEATHER_INPUTS = [
    "temp", "dewp", "humid", "wind_dir", "wind_speed", "wind_gust",
    "precip", "pressure", "visib",
]


def add_prediction_cutoff(flights: pd.DataFrame) -> pd.DataFrame:
    """Add UTC cutoff from the contracted local America/New_York schedule."""
    df = flights.copy()
    local = pd.to_datetime(df["sched_dep_ts"], errors="coerce").dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    )
    df["prediction_cutoff_utc"] = local.dt.tz_convert("UTC") - dt.timedelta(minutes=60)
    return df


def join_weather_asof(
    flights: pd.DataFrame, weather: pd.DataFrame, max_age_minutes: int = 180
) -> pd.DataFrame:
    """Join latest origin observation no later than cutoff; reject stale data."""
    df = add_prediction_cutoff(flights)
    wx = weather[["origin", "time_hour", *WEATHER_INPUTS]].copy()
    df["origin"] = df["origin"].astype("string")
    wx["origin"] = wx["origin"].astype("string")
    wx["weather_observed_utc"] = pd.to_datetime(wx.pop("time_hour"), utc=True, errors="coerce")
    if wx.duplicated(["origin", "weather_observed_utc"]).any():
        raise ValueError("weather origin/timestamp lookup is not unique")

    # pandas merge_asof requires global ordering by the as-of key.
    left = df.sort_values(["prediction_cutoff_utc", "origin"]).copy()
    right = wx.sort_values(["weather_observed_utc", "origin"]).copy()
    joined = pd.merge_asof(
        left, right,
        left_on="prediction_cutoff_utc", right_on="weather_observed_utc",
        by="origin", direction="backward", allow_exact_matches=True,
    )
    age = (joined["prediction_cutoff_utc"] - joined["weather_observed_utc"]).dt.total_seconds() / 60
    joined["weather_age_minutes"] = age
    stale = age.gt(max_age_minutes)
    joined["weather_stale_or_missing"] = age.isna() | stale
    # Preserve timestamp and age for audit, but withhold stale measurements.
    joined.loc[stale, WEATHER_INPUTS] = pd.NA
    joined = joined.sort_index()
    if (joined["weather_observed_utc"] > joined["prediction_cutoff_utc"]).fillna(False).any():
        raise AssertionError("future weather observation reached a flight")
    return joined
