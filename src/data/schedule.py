"""Stable flight-row identifiers and scheduled-time parsing (Phase 1, Steps 3-4).

The identifier is built from the source-file fingerprint and the original row
position, so it is stable for a given file version and survives filtering.
Scheduled HHMM-style times are parsed into minutes-since-midnight and local
timestamps; values that cannot be HHMM times are flagged, never silently
normalized.
"""

from __future__ import annotations

import pandas as pd


def add_flight_row_id(flights: pd.DataFrame, fingerprint: str) -> pd.DataFrame:
    """Add `flight_row_id` based on file fingerprint + original row position.

    The frame must still be in source-file order (it is, immediately after
    loading). The identifier is created before any exclusion, sorting or
    filtering so every source row keeps a stable identity.
    """
    if "flight_row_id" in flights.columns:
        raise ValueError("flight_row_id already exists; refusing to overwrite")
    df = flights.copy()
    df["flight_row_id"] = [
        f"flights-{fingerprint[:8]}-{position:06d}"
        for position in range(1, len(df) + 1)
    ]
    if df["flight_row_id"].duplicated().any():  # defensive; positions are unique
        raise ValueError("generated flight_row_id values are not unique")
    return df


def parse_hhmm(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse HHMM-style integer times.

    Returns (minutes_since_midnight as nullable Int32, is_invalid as bool).
    A value counts as invalid when it is present but not a plausible HHMM
    time (negative, above 2359, or minute part above 59). Missing values give
    NA minutes and are not flagged invalid.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    present = numeric.notna()
    plausible = (
        present & (numeric >= 0) & (numeric <= 2359) & (numeric % 100 <= 59)
    )
    minutes = ((numeric // 100) * 60 + numeric % 100).where(plausible)
    minutes = minutes.astype("Int32")
    invalid = present & ~plausible
    return minutes, invalid


def add_schedule_fields(flights: pd.DataFrame) -> pd.DataFrame:
    """Add parsed schedule fields used by the split contract and Phase 2.

    Adds: sched_date (local scheduled date), sched_dep_minute / sched_arr_minute
    (minutes since midnight, NA when unknown or invalid),
    sched_dep_time_invalid / sched_arr_time_invalid flags, and
    sched_dep_ts / sched_arr_ts (naive local America/New_York timestamps,
    per the Phase 0 contract). A scheduled arrival earlier in the day than its
    departure is treated as an overnight arrival and moved to the next day.
    """
    df = flights.copy()

    dep_minute, dep_invalid = parse_hhmm(df["sched_dep_time"])
    arr_minute, arr_invalid = parse_hhmm(df["sched_arr_time"])

    sched_date = pd.to_datetime(df[["year", "month", "day"]])
    dep_ts = sched_date + pd.to_timedelta(dep_minute, unit="m")
    overnight = (arr_minute < dep_minute).fillna(False)
    arr_ts = (
        sched_date
        + pd.to_timedelta(arr_minute, unit="m")
        + pd.to_timedelta(overnight.astype(int), unit="D")
    )

    df["sched_date"] = sched_date
    df["sched_dep_minute"] = dep_minute
    df["sched_arr_minute"] = arr_minute
    df["sched_dep_time_invalid"] = dep_invalid
    df["sched_arr_time_invalid"] = arr_invalid
    df["sched_dep_ts"] = dep_ts
    df["sched_arr_ts"] = arr_ts
    return df
