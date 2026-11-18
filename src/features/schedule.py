"""Schedule, calendar and reference features available at T-60."""

from __future__ import annotations

import logging
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from src.data.schedule import NYC_TIMEZONE, local_schedule_to_utc

LOGGER = logging.getLogger(__name__)
MAX_DURATION_LOCALIZATION_FAILURE_RATE = 0.001


def route_key(origin: pd.Series, destination: pd.Series) -> pd.Series:
    """Return the canonical origin-destination key, preserving missing codes."""
    return origin.astype("string") + "-" + destination.astype("string")


def add_schedule_features(flights: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic features using scheduled information only."""
    df = flights.copy()
    ts = pd.to_datetime(df["sched_dep_ts"], errors="coerce")
    df["sched_dep_hour"] = ts.dt.hour.astype("Int8")
    df["sched_day_of_week"] = ts.dt.dayofweek.astype("Int8")
    df["sched_month"] = ts.dt.month.astype("Int8")
    df["sched_day_of_year"] = ts.dt.dayofyear.astype("Int16")
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    angle = 2 * np.pi * minute_of_day / (24 * 60)
    df["sched_hour_sin"] = np.sin(angle)
    df["sched_hour_cos"] = np.cos(angle)
    df["sched_dow_sin"] = np.sin(2 * np.pi * df["sched_day_of_week"] / 7)
    df["sched_dow_cos"] = np.cos(2 * np.pi * df["sched_day_of_week"] / 7)
    df["route"] = route_key(df["origin"], df["dest"])

    # Published schedules are available before T-60, so same-hour scheduled
    # volume is safe; no realized departure or arrival values are used.
    keys = [df["sched_date"], df["origin"], df["sched_dep_hour"]]
    df["origin_scheduled_hour_volume"] = df.groupby(keys, dropna=False)[
        "flight_row_id"
    ].transform("size").astype("Int32")
    return df


def _assert_unique_lookup(reference: pd.DataFrame, key: str, name: str) -> None:
    duplicates = reference.loc[reference[key].notna() & reference[key].duplicated(False), key]
    if not duplicates.empty:
        values = sorted(duplicates.astype(str).unique().tolist())[:5]
        raise ValueError(f"{name} lookup key {key!r} is not unique: {values}")


def _scheduled_duration(flights: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Return the UTC duration and report unresolvable destination schedules."""
    duration = pd.Series(np.nan, index=flights.index, dtype=float)
    departure_utc = local_schedule_to_utc(flights["sched_dep_ts"], NYC_TIMEZONE)
    arrival = pd.to_datetime(flights["sched_date"], errors="coerce").dt.normalize()
    arrival += pd.to_timedelta(pd.to_numeric(flights["sched_arr_minute"], errors="coerce"), unit="m")
    zones = flights.get("destination_airport_tzone", pd.Series(pd.NA, index=flights.index))
    zones = zones.astype("string").str.strip().replace("", pd.NA)
    eligible = arrival.notna() & departure_utc.notna() & zones.notna()

    failed_rows = 0
    failed_zones = []
    dst_rows = 0
    eligible_zones = zones.loc[eligible]
    for zone, indices in eligible_zones.groupby(eligible_zones).groups.items():
        try:
            timezone = ZoneInfo(zone)
            local_arrival = arrival.loc[indices]
            departure = departure_utc.loc[indices]
            arrival_utc = local_schedule_to_utc(local_arrival, timezone)
            overnight = arrival_utc.le(departure)
            if overnight.any():
                next_local = local_arrival.loc[overnight] + np.timedelta64(1, "D")
                next_utc = local_schedule_to_utc(next_local, timezone)
                arrival_utc = arrival_utc.where(~overnight, next_utc)
        except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
            failed_rows += len(indices)
            failed_zones.append(zone)
            LOGGER.warning("Scheduled-duration localization failed for %s on %d rows: %s", zone, len(indices), exc)
            continue
        dst_rows += int(arrival_utc.isna().sum())
        duration.loc[indices] = (arrival_utc - departure).dt.total_seconds() / 60

    eligible_rows = int(eligible.sum())
    if failed_rows > eligible_rows * MAX_DURATION_LOCALIZATION_FAILURE_RATE:
        raise ValueError(
            f"Scheduled-duration localization failed on {failed_rows}/{eligible_rows} eligible rows "
            f"(limit {MAX_DURATION_LOCALIZATION_FAILURE_RATE:.1%}; zones: {', '.join(failed_zones)})"
        )
    return duration, {
        "eligible_rows": eligible_rows,
        "missing_destination_timezone_rows": int(zones.isna().sum()),
        "localization_failed_rows": failed_rows,
        "localization_failed_zones": failed_zones,
        "destination_dst_ambiguous_or_nonexistent_rows": dst_rows,
    }


def join_reference_features(
    flights: pd.DataFrame, airlines: pd.DataFrame, airports: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Left join airline/airport reference values without changing row grain."""
    _assert_unique_lookup(airlines, "carrier", "airlines")
    _assert_unique_lookup(airports, "faa", "airports")
    before = len(flights)
    df = flights.merge(
        airlines[["carrier", "name"]].rename(columns={"name": "airline_name"}),
        on="carrier", how="left", validate="many_to_one",
    )
    airport_cols = ["faa", "name", "lat", "lon", "alt", "tz"]
    if "tzone" in airports.columns:
        airport_cols.append("tzone")
    for side, key in (("origin", "origin"), ("destination", "dest")):
        renamed = {
            col: (key if col == "faa" else f"{side}_airport_{col}")
            for col in airport_cols
        }
        df = df.merge(
            airports[airport_cols].rename(columns=renamed),
            on=key, how="left", validate="many_to_one",
        )
    df["scheduled_duration_minutes"] = float("nan")
    if len(df) != before or not df["flight_row_id"].is_unique:
        raise AssertionError("reference joins changed flight row grain")
    coverage = {
        "rows": before,
        "unmatched_airline_rows": int(df["airline_name"].isna().sum()),
        "unmatched_origin_airport_rows": int(df["origin_airport_name"].isna().sum()),
        "unmatched_destination_airport_rows": int(
            df["destination_airport_name"].isna().sum()
        ),
    }
    if {"sched_dep_ts", "sched_date", "sched_arr_minute"} <= set(df.columns):
        df["scheduled_duration_minutes"], coverage["schedule_duration"] = _scheduled_duration(df)
    return df, coverage
