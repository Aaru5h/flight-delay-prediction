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
    dep = pd.to_numeric(df["sched_dep_minute"], errors="coerce")
    arr = pd.to_numeric(df["sched_arr_minute"], errors="coerce")
    df["scheduled_duration_minutes"] = (arr - dep) % (24 * 60)
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
    for side, key in (("origin", "origin"), ("destination", "dest")):
        renamed = {
            col: (key if col == "faa" else f"{side}_airport_{col}")
            for col in airport_cols
        }
        df = df.merge(
            airports[airport_cols].rename(columns=renamed),
            on=key, how="left", validate="many_to_one",
        )
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
