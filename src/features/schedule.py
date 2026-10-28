"""Schedule, calendar and reference features available at T-60."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_schedule_features(flights: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic features using scheduled information only."""
    df = flights.copy()
    ts = pd.to_datetime(df["sched_dep_ts"], errors="coerce")
    df["sched_dep_hour"] = ts.dt.hour.astype("Int8")
    df["sched_day_of_week"] = ts.dt.dayofweek.astype("Int8")
    df["sched_month"] = ts.dt.month.astype("Int8")
    df["sched_day_of_year"] = ts.dt.dayofyear.astype("Int16")
    df["sched_hour_sin"] = np.sin(2 * np.pi * df["sched_dep_hour"] / 24)
    df["sched_hour_cos"] = np.cos(2 * np.pi * df["sched_dep_hour"] / 24)
    df["sched_dow_sin"] = np.sin(2 * np.pi * df["sched_day_of_week"] / 7)
    df["sched_dow_cos"] = np.cos(2 * np.pi * df["sched_day_of_week"] / 7)
    df["route"] = df["origin"].astype("string") + "-" + df["dest"].astype("string")

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
    # Arrival clocks are destination-local. Convert both ends to UTC before
    # calculating scheduled duration; unmatched destination zones stay missing.
    df["scheduled_duration_minutes"] = float("nan")
    if {"destination_airport_tzone", "sched_dep_ts", "sched_date", "sched_arr_minute"} <= set(df.columns):
        dep_utc = pd.to_datetime(df["sched_dep_ts"], errors="coerce").dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT"
        ).dt.tz_convert("UTC")
        base_arrival = pd.to_datetime(df["sched_date"], errors="coerce").dt.normalize() + pd.to_timedelta(
            pd.to_numeric(df["sched_arr_minute"], errors="coerce"), unit="m"
        )
        for zone, indices in df.groupby("destination_airport_tzone").groups.items():
            if not isinstance(zone, str) or not zone:
                continue
            try:
                local_arrival = base_arrival.loc[indices].dt.tz_localize(
                    zone, ambiguous="NaT", nonexistent="NaT"
                ).dt.tz_convert("UTC")
                next_day = local_arrival.le(dep_utc.loc[indices])
                if next_day.any():
                    next_local = (base_arrival.loc[indices[next_day]] + pd.Timedelta(days=1)).dt.tz_localize(
                        zone, ambiguous="NaT", nonexistent="NaT"
                    ).dt.tz_convert("UTC")
                    local_arrival = local_arrival.where(~next_day, next_local.reindex(local_arrival.index))
                minutes = (local_arrival - dep_utc.loc[indices]).dt.total_seconds() / 60
                df.loc[indices, "scheduled_duration_minutes"] = minutes.to_numpy()
            except (ValueError, TypeError):
                continue
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
    return df, coverage
