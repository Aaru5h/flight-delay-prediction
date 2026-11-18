"""Step 3 point-in-time feature queries for the four-lead-time protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.data.labeling import THRESHOLD_MINUTES
from src.data.loading import SourceData
from src.data.schedule import (
    NYC_TIMEZONE,
    add_flight_row_id,
    add_schedule_fields,
    local_schedule_to_utc,
)
from src.features.asof_events import EVENT_AVAILABILITY_BUFFER, reconstruct_events
from src.features.schedule import add_schedule_features, join_reference_features
from src.features.weather import WEATHER_INPUTS

LEAD_TIMES_MINUTES = (1440, 360, 120, 60)
HISTORY_WINDOWS_DAYS = (7, 28)
HISTORY_GROUPS = ("carrier", "route", "origin")
HISTORY_PRIOR_STRENGTH = 20
WEATHER_MAX_AGE = timedelta(hours=3)
WEATHER_TREND_WINDOWS_HOURS = (3, 6)
WEATHER_TRENDS = ("temp", "humid", "wind_speed", "pressure", "visib")
DEPARTURE_SEQUENCE_LENGTH = 32
WEATHER_SEQUENCE_LENGTH = 12
NANOSECONDS_PER_HOUR = 3_600_000_000_000

CATEGORICAL_FEATURES = ("carrier", "origin", "dest", "route", "sched_dep_hour", "lead_time_minutes")
SCHEDULE_NUMERIC_FEATURES = (
    "distance", "sched_hour_sin", "sched_hour_cos", "sched_dow_sin", "sched_dow_cos",
    "days_to_nearest_holiday", "scheduled_duration_minutes", "origin_scheduled_hour_volume",
    "origin_airport_lat", "origin_airport_lon", "origin_airport_alt",
    "destination_airport_lat", "destination_airport_lon", "destination_airport_alt",
)
SAME_DAY_FEATURES = ("origin_departures_so_far", "origin_mean_dep_delay_so_far")


@dataclass(frozen=True)
class EventSequence:
    """Oldest to newest, with empty slots on the left and True for real events."""

    rows: tuple[dict[str, Any] | None, ...]
    mask: tuple[bool, ...]


@dataclass(frozen=True)
class AsOfFeatures:
    features: dict[str, Any]
    weather_observed_utc: pd.Timestamp | None
    weather_sequence: EventSequence | None = None
    departure_sequence: EventSequence | None = None


def _timestamp_ns(values: pd.Series) -> np.ndarray:
    return values.to_numpy(dtype="datetime64[ns]").astype(np.int64)


class _Timeline:
    def __init__(self, rows: pd.DataFrame):
        self.rows = rows.sort_values("event_utc", kind="stable").reset_index(drop=True)
        self.times = _timestamp_ns(self.rows["event_utc"])

    def bounds(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
        return (
            int(self.times.searchsorted(start.value)),
            int(self.times.searchsorted(end.value, side="right")),
        )


class _ArrivalHistory:
    def __init__(self, rows: pd.DataFrame):
        timeline = _Timeline(rows)
        self.times = timeline.times
        self.delayed_prefix = np.r_[0, timeline.rows["delayed"].to_numpy(dtype=np.int64).cumsum()]

    def counts(self, start_ns: int, end_ns: int) -> tuple[int, int]:
        start = self.times.searchsorted(start_ns)
        end = self.times.searchsorted(end_ns, side="right")
        return int(end - start), int(self.delayed_prefix[end] - self.delayed_prefix[start])


class _WeatherTimeline(_Timeline):
    def __init__(self, rows: pd.DataFrame):
        super().__init__(rows)
        self.values = {
            name: self.rows[name].to_numpy(dtype=float, na_value=np.nan) for name in WEATHER_INPUTS
        }


def _holiday_distance(service_dates: pd.Series) -> pd.Series:
    dates = pd.DatetimeIndex(service_dates.dropna().unique())
    if dates.empty:
        return pd.Series(pd.NA, index=service_dates.index, dtype="Int16")
    holidays = USFederalHolidayCalendar().holidays(
        dates.min() - timedelta(days=366), dates.max() + timedelta(days=366)
    )
    nearest = {date: min(abs((holidays - date).days)) for date in dates}
    return service_dates.map(nearest).astype("Int16")


def _weather_timelines(weather: pd.DataFrame) -> dict[str, _WeatherTimeline]:
    rows = weather[["origin", "time_hour", *WEATHER_INPUTS]].copy()
    rows["event_utc"] = pd.to_datetime(rows.pop("time_hour"), errors="coerce", utc=True)
    rows = rows.dropna(subset=["origin", "event_utc"])
    if rows.duplicated(["origin", "event_utc"]).any():
        raise ValueError("Weather observations must be unique by origin and hour")
    return {origin: _WeatherTimeline(group) for origin, group in rows.groupby("origin")}


def _padded_sequence(records: list[dict[str, Any]], length: int) -> EventSequence:
    padding = length - len(records)
    return EventSequence((None,) * padding + tuple(records), (False,) * padding + (True,) * len(records))


class AsOfFeatureBuilder:
    """Index the source records once, then query one flight at one UTC cutoff.

    Construct from the mapping returned by load_all_sources. The public
    flights frame contains the schedule and IDs needed to enumerate queries.
    outcome_available_utc is separate metadata for fold eligibility.
    """

    def __init__(self, sources: Mapping[str, SourceData], *, allow_same_day_departures: bool = False):
        source = sources["flights"]
        flights = add_flight_row_id(source.df, source.fingerprint)
        flights = add_schedule_features(add_schedule_fields(flights))
        flights, _ = join_reference_features(flights, sources["airlines"].df, sources["airports"].df)
        departure, arrival = reconstruct_events(flights)
        self._allow_same_day_departures = allow_same_day_departures

        flights["scheduled_departure_utc"] = local_schedule_to_utc(flights["sched_dep_ts"], NYC_TIMEZONE)
        flights["days_to_nearest_holiday"] = _holiday_distance(flights["sched_date"])
        schedule_columns = (*CATEGORICAL_FEATURES[:-1], *SCHEDULE_NUMERIC_FEATURES)
        self.flights = flights.set_index("flight_row_id")[[
            "sched_date", "scheduled_departure_utc", *schedule_columns,
        ]].copy()
        self.outcome_available_utc = (arrival + EVENT_AVAILABILITY_BUFFER).set_axis(self.flights.index)
        self.outcome_available_utc.name = "outcome_available_utc"

        arrivals = flights[["flight_row_id", *HISTORY_GROUPS]].assign(
            event_utc=arrival, delayed=flights["arr_delay"].ge(THRESHOLD_MINUTES)
        ).dropna(subset=["event_utc"])
        self._own_arrivals = arrivals.set_index("flight_row_id")[["event_utc", "delayed"]]
        self._all_arrivals = _ArrivalHistory(arrivals)
        self._arrival_groups = {
            group: {key: _ArrivalHistory(rows) for key, rows in arrivals.groupby(group)}
            for group in HISTORY_GROUPS
        }
        self._weather = _weather_timelines(sources["weather"].df)
        self._departures = {}
        if allow_same_day_departures:
            departures = flights[["flight_row_id", "origin", "carrier", "dep_delay"]].assign(
                event_utc=departure
            ).dropna(subset=["event_utc", "origin"])
            self._departures = {
                origin: _Timeline(group) for origin, group in departures.groupby("origin")
            }

    def feature_manifest(self) -> dict:
        """Return the ordered estimator fields and the availability rules."""
        history = [
            f"{group}_{stat}_{days}d"
            for days in HISTORY_WINDOWS_DAYS
            for group in HISTORY_GROUPS
            for stat in ("prior_count", "delay_rate")
        ]
        trends = [
            f"weather_{name}_trend_{hours}h"
            for hours in WEATHER_TREND_WINDOWS_HOURS
            for name in WEATHER_TRENDS
        ]
        fields = {
            "published schedule and fixed calendar": [*CATEGORICAL_FEATURES, *SCHEDULE_NUMERIC_FEATURES],
            "arrival outcomes available by cutoff": history,
            "origin weather valid by cutoff": [*WEATHER_INPUTS, "weather_stale_or_missing", *trends],
        }
        if self._allow_same_day_departures:
            fields["prior origin departures available by cutoff"] = SAME_DAY_FEATURES
        features = [
            {
                "name": name,
                "type": "categorical" if name in CATEGORICAL_FEATURES else "numeric",
                "availability_rule": availability,
            }
            for availability, names in fields.items()
            for name in names
        ]
        return {
            "manifest_version": "asof-1.0.0",
            "lead_time_minutes": list(LEAD_TIMES_MINUTES),
            "event_availability_buffer_minutes": EVENT_AVAILABILITY_BUFFER // timedelta(minutes=1),
            "same_day_departures_enabled": self._allow_same_day_departures,
            "allowlist": [feature["name"] for feature in features],
            "features": features,
        }

    def _history(self, flight_row_id: str, flight: pd.Series, cutoff: pd.Timestamp) -> dict:
        features = {}
        last_event = (cutoff - EVENT_AVAILABILITY_BUFFER).value
        own = self._own_arrivals.loc[flight_row_id] if flight_row_id in self._own_arrivals.index else None
        for days in HISTORY_WINDOWS_DAYS:
            first_event = (cutoff - timedelta(days=days)).value
            own_in_window = own is not None and first_event <= own["event_utc"].value <= last_event
            own_delay = int(own_in_window and own["delayed"])
            global_count, global_delayed = self._all_arrivals.counts(first_event, last_event)
            global_count -= int(own_in_window)
            global_delayed -= own_delay
            prior_rate = global_delayed / global_count if global_count else np.nan
            for group, histories in self._arrival_groups.items():
                timeline = histories.get(flight[group])
                count, delayed = timeline.counts(first_event, last_event) if timeline is not None else (0, 0)
                if timeline is not None:
                    count -= int(own_in_window)
                    delayed -= own_delay
                rate = (
                    (delayed + HISTORY_PRIOR_STRENGTH * prior_rate) / (count + HISTORY_PRIOR_STRENGTH)
                    if count
                    else np.nan
                )
                features[f"{group}_prior_count_{days}d"] = count
                features[f"{group}_delay_rate_{days}d"] = rate
        return features

    def _weather_features(
        self, origin: str, cutoff: pd.Timestamp, include_sequence: bool
    ) -> tuple[dict, pd.Timestamp | None, EventSequence | None]:
        features = dict.fromkeys(WEATHER_INPUTS, np.nan)
        features["weather_stale_or_missing"] = True
        timeline = self._weather.get(origin)
        observed = None
        sequence_records = []
        for hours in WEATHER_TREND_WINDOWS_HOURS:
            features.update(dict.fromkeys((f"weather_{name}_trend_{hours}h" for name in WEATHER_TRENDS), np.nan))
        if timeline is not None:
            end = int(timeline.times.searchsorted(cutoff.value, side="right"))
            if end:
                last = end - 1
                observed = pd.Timestamp(timeline.times[last], unit="ns", tz="UTC")
                if cutoff - observed <= WEATHER_MAX_AGE:
                    features.update({name: values[last] for name, values in timeline.values.items()})
                    features["weather_stale_or_missing"] = False
                    for hours in WEATHER_TREND_WINDOWS_HOURS:
                        earliest = (cutoff - timedelta(hours=hours)).value
                        start = timeline.times.searchsorted(earliest)
                        for name in WEATHER_TRENDS:
                            values = timeline.values[name]
                            previous = np.flatnonzero(np.isfinite(values[start:last]))
                            if np.isfinite(values[last]) and previous.size:
                                first = start + previous[0]
                                span = timeline.times[last] - timeline.times[first]
                                span_hours = span / NANOSECONDS_PER_HOUR
                                features[f"weather_{name}_trend_{hours}h"] = (
                                    values[last] - values[first]
                                ) / span_hours
                    if include_sequence:
                        start = max(0, end - WEATHER_SEQUENCE_LENGTH)
                        sequence_records = timeline.rows.iloc[start:end][
                            ["event_utc", *WEATHER_INPUTS]
                        ].to_dict("records")
        sequence = _padded_sequence(sequence_records, WEATHER_SEQUENCE_LENGTH) if include_sequence else None
        return features, observed, sequence

    def _departure_features(
        self, flight_row_id: str, flight: pd.Series, cutoff: pd.Timestamp, include_sequence: bool
    ) -> tuple[dict, EventSequence | None]:
        counts = {"origin_departures_so_far": 0, "origin_mean_dep_delay_so_far": np.nan}
        timeline = self._departures.get(flight["origin"])
        sequence_records = []
        if timeline is not None:
            first = flight["sched_date"].tz_localize(NYC_TIMEZONE).tz_convert("UTC")
            start, end = timeline.bounds(first, cutoff - EVENT_AVAILABILITY_BUFFER)
            same_day = timeline.rows.iloc[start:end]
            same_day = same_day.loc[same_day["flight_row_id"].ne(flight_row_id)]
            counts = {
                "origin_departures_so_far": len(same_day),
                "origin_mean_dep_delay_so_far": same_day["dep_delay"].mean(),
            }
            if include_sequence:
                start = max(0, end - DEPARTURE_SEQUENCE_LENGTH - 1)
                recent = timeline.rows.iloc[start:end]
                recent = recent.loc[recent["flight_row_id"].ne(flight_row_id)].tail(DEPARTURE_SEQUENCE_LENGTH)
                sequence = recent[["event_utc", "carrier", "dep_delay"]].assign(
                    available_utc=recent["event_utc"] + EVENT_AVAILABILITY_BUFFER,
                    minutes_before_cutoff=(cutoff - recent["event_utc"]).dt.total_seconds() / 60,
                )
                sequence_records = sequence.to_dict("records")
        sequence = _padded_sequence(sequence_records, DEPARTURE_SEQUENCE_LENGTH) if include_sequence else None
        return counts, sequence

    def build(
        self, flight_row_id: str, cutoff_utc: datetime | pd.Timestamp | str, *, include_sequences: bool = False
    ) -> AsOfFeatures:
        """Build features at one contracted cutoff for the given flight ID."""
        flight = self.flights.loc[flight_row_id]
        cutoff = pd.Timestamp(cutoff_utc)
        if pd.isna(cutoff) or cutoff.tzinfo is None:
            raise ValueError("Use a timezone-aware cutoff")
        cutoff = cutoff.tz_convert("UTC")
        departure = flight["scheduled_departure_utc"]
        if pd.isna(departure):
            raise ValueError(f"Flight {flight_row_id} has no unambiguous scheduled departure")
        elapsed = departure - cutoff
        lead_time = elapsed / timedelta(minutes=1)
        if lead_time not in LEAD_TIMES_MINUTES:
            raise ValueError("Cutoff must be 24 h, 6 h, 2 h or 1 h before a valid scheduled departure")
        features = flight[list(CATEGORICAL_FEATURES[:-1] + SCHEDULE_NUMERIC_FEATURES)].to_dict()
        features["lead_time_minutes"] = int(lead_time)
        features.update(self._history(flight_row_id, flight, cutoff))
        weather, observed, weather_sequence = self._weather_features(flight["origin"], cutoff, include_sequences)
        features.update(weather)
        departure_sequence = None
        if self._allow_same_day_departures:
            departures, departure_sequence = self._departure_features(flight_row_id, flight, cutoff, include_sequences)
            features.update(departures)
        return AsOfFeatures(features, observed, weather_sequence, departure_sequence)
