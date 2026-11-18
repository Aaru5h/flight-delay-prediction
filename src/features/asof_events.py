"""Reconstruct auditable event times from the NYC flight records."""

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data.schedule import NYC_TIMEZONE, local_schedule_to_utc, parse_hhmm

EVENT_AVAILABILITY_BUFFER = timedelta(minutes=15)


def _elapsed_delay(values: pd.Series) -> pd.Series:
    minutes = pd.to_numeric(values, errors="coerce")
    return pd.to_timedelta(minutes.where(np.isfinite(minutes)), unit="m")


def _matches_recorded_clock(
    event_utc: pd.Series, recorded_hhmm: pd.Series, timezone: str | ZoneInfo
) -> pd.Series:
    local = event_utc.dt.tz_convert(timezone)
    # The raw event clocks can use 2400 for midnight. The UTC instant already
    # accounts for the date shift because it was computed from elapsed delay.
    midnight = recorded_hhmm.eq(2400).fillna(False)
    recorded_minutes, _ = parse_hhmm(recorded_hhmm.mask(midnight, 0))
    return (local.dt.hour * 60 + local.dt.minute).eq(recorded_minutes).fillna(False)


def reconstruct_events(flights: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return actual departure and arrival instants when both clocks agree.

    Uses elapsed delays against the UTC schedule and validates the result
    against the recorded event clock. Unknown destination zones, inconsistent
    clocks and arrivals that precede departure yield missing timestamps.
    """
    scheduled_departure = local_schedule_to_utc(flights["sched_dep_ts"], NYC_TIMEZONE)
    departure = scheduled_departure + _elapsed_delay(flights["dep_delay"])
    departure = departure.where(_matches_recorded_clock(departure, flights["dep_time"], NYC_TIMEZONE))

    scheduled_arrival = scheduled_departure + pd.to_timedelta(
        flights["scheduled_duration_minutes"], unit="m"
    )
    arrival = scheduled_arrival + _elapsed_delay(flights["arr_delay"])
    arrival = arrival.where(arrival.gt(departure))
    candidates = flights.loc[arrival.notna(), "destination_airport_tzone"]
    matched = pd.Series(False, index=flights.index)
    for zone, indices in candidates.groupby(candidates).groups.items():
        matched.loc[indices] = _matches_recorded_clock(
            arrival.loc[indices], flights.loc[indices, "arr_time"], ZoneInfo(zone.strip())
        )
    return departure, arrival.where(matched)
