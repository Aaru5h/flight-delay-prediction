"""Origin-weather as-of join at the T-60 prediction cutoff."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.data.schedule import NYC_TIMEZONE, local_schedule_to_utc

WEATHER_INPUTS = [
    "temp", "dewp", "humid", "wind_dir", "wind_speed", "wind_gust",
    "precip", "pressure", "visib",
]


def add_prediction_cutoff(flights: pd.DataFrame) -> pd.DataFrame:
    """Add UTC cutoff from the contracted local America/New_York schedule."""
    df = flights.copy()
    departure_utc = local_schedule_to_utc(df["sched_dep_ts"], NYC_TIMEZONE)
    df["prediction_cutoff_utc"] = departure_utc - dt.timedelta(minutes=60)
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
    left = df.sort_values(["prediction_cutoff_utc", "origin"]).reset_index(drop=True)
    right = wx.dropna(subset=["weather_observed_utc", "origin"]).sort_values(
        ["weather_observed_utc", "origin"]
    )
    # Keep rows without an unambiguous cutoff, but never match weather to them.
    keys = left[["prediction_cutoff_utc", "origin"]].dropna()
    matched = pd.merge_asof(
        keys, right,
        left_on="prediction_cutoff_utc", right_on="weather_observed_utc",
        by="origin", direction="backward", allow_exact_matches=True,
    )
    matched.index = keys.index
    joined = left.join(matched[["weather_observed_utc", *WEATHER_INPUTS]])
    age = (joined["prediction_cutoff_utc"] - joined["weather_observed_utc"]).dt.total_seconds() / 60
    joined["weather_age_minutes"] = age
    stale = age.gt(max_age_minutes)
    joined["weather_stale_or_missing"] = age.isna() | stale
    # Preserve timestamp and age for audit, but withhold stale measurements.
    joined.loc[stale, WEATHER_INPUTS] = pd.NA
    if (joined["weather_observed_utc"] > joined["prediction_cutoff_utc"]).fillna(False).any():
        raise AssertionError("future weather observation reached a flight")
    return joined
