"""Step 7 sequence inputs: load, verify, gate and preprocess.

The Step 4 sequence files hold, for each (flight_row_id, lead_time_minutes),
left-padded weather readings and, when same-day departures are approved,
earlier origin departures, each with a UTC event time and a boolean mask.
Loading moves real events to the front (oldest first) so a packed GRU never
reads a padded slot, and every check needed by the protocol's sequence gate is
recomputed here independently of the builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.weather import WEATHER_INPUTS
from src.models.classical.step5_data import CUTOFF, ID, LEAD
from src.models.neural.preprocess import MISSING_INDEX, UNKNOWN_INDEX

WEATHER_LENGTH = 12
DEPARTURE_LENGTH = 32
NS_PER_MINUTE = 60_000_000_000
NO_EVENT = np.iinfo(np.int64).min


def _utc_ns(text: str) -> int:
    return int(np.datetime64(text.rstrip("Z"), "ns").astype(np.int64))


@dataclass
class RawSequences:
    """Sequence values aligned to the rows of a feature frame (right-padded)."""

    wx_values: np.ndarray
    wx_hours: np.ndarray
    wx_len: np.ndarray
    dep_minutes: np.ndarray | None
    dep_delay: np.ndarray | None
    dep_carrier: np.ndarray | None
    dep_len: np.ndarray | None
    carriers: list[str]

    @property
    def departures_enabled(self) -> bool:
        return self.dep_len is not None

    def take(self, rows: np.ndarray) -> dict:
        raw = {"wx_values": self.wx_values[rows], "wx_hours": self.wx_hours[rows], "wx_len": self.wx_len[rows]}
        if self.departures_enabled:
            codes = self.dep_carrier[rows]
            names = np.asarray([*self.carriers, None], dtype=object)
            raw.update(
                dep_minutes=self.dep_minutes[rows], dep_delay=self.dep_delay[rows],
                dep_carrier=names[np.where(codes < 0, len(self.carriers), codes)],
                dep_len=self.dep_len[rows],
            )
        return raw


@dataclass
class SequenceChecks:
    records: int = 0
    development_records: int = 0
    missing_development_records: int = 0
    structure_errors: int = 0
    order_errors: int = 0
    cutoff_violations: int = 0
    availability_violations: int = 0
    offset_mismatches: int = 0
    partitions_without_sequences: list[int] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    weather_coverage_by_lead: dict[str, float] = field(default_factory=dict)
    departure_coverage_by_lead: dict[str, float] = field(default_factory=dict)

    def note(self, message: str) -> None:
        if len(self.examples) < 10:
            self.examples.append(message)


def _structure(sequence: dict | None, length: int, checks: SequenceChecks, key) -> list[dict]:
    if sequence is None:
        return []
    rows, mask = sequence.get("rows"), sequence.get("mask")
    if not isinstance(rows, list) or not isinstance(mask, list) or len(rows) != length or len(mask) != length:
        checks.structure_errors += 1
        checks.note(f"{key}: sequence is not {length} slots with a matching mask")
        return []
    if any(bool(active) != (row is not None) for row, active in zip(rows, mask)):
        checks.structure_errors += 1
        checks.note(f"{key}: mask does not match padded records")
        return []
    if any(earlier and not later for earlier, later in zip(mask[:-1], mask[1:])):
        checks.structure_errors += 1
        checks.note(f"{key}: padding is not on the left")
        return []
    return [row for row in rows if row is not None]


def load_sequences(dataset_dir: Path, manifest: dict, frame: pd.DataFrame) -> tuple[RawSequences, SequenceChecks]:
    """Read every sequence record and align development rows to ``frame``."""
    n = len(frame)
    position = {
        (flight, int(lead)): row
        for row, (flight, lead) in enumerate(zip(frame[ID].astype(str), frame[LEAD].astype(int)))
    }
    if len(position) != n:
        raise ValueError("Feature rows must be unique by flight and lead time")
    departures_enabled = bool(manifest.get("same_day_departures_enabled"))
    checks = SequenceChecks()
    wx_values = np.full((n, WEATHER_LENGTH, len(WEATHER_INPUTS)), np.nan, dtype=np.float32)
    wx_events = np.full((n, WEATHER_LENGTH), NO_EVENT, dtype=np.int64)
    wx_len = np.zeros(n, dtype=np.int16)
    if departures_enabled:
        dep_events = np.full((n, DEPARTURE_LENGTH), NO_EVENT, dtype=np.int64)
        dep_available = np.full((n, DEPARTURE_LENGTH), NO_EVENT, dtype=np.int64)
        dep_minutes = np.full((n, DEPARTURE_LENGTH), np.nan, dtype=np.float32)
        dep_delay = np.full((n, DEPARTURE_LENGTH), np.nan, dtype=np.float32)
        dep_carrier = np.full((n, DEPARTURE_LENGTH), -1, dtype=np.int16)
        dep_len = np.zeros(n, dtype=np.int16)
    carriers: dict[str, int] = {}
    seen = np.zeros(n, dtype=bool)

    for partition in manifest["partitions"]:
        lead = int(partition[LEAD])
        name = partition.get("sequences")
        if not name or not (Path(dataset_dir) / name).is_file():
            checks.partitions_without_sequences.append(lead)
            continue
        with gzip.open(Path(dataset_dir) / name, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                key = (str(record[ID]), int(record[LEAD]))
                checks.records += 1
                if key[1] != lead:
                    checks.structure_errors += 1
                    checks.note(f"{key}: record stored in the {lead}-minute file")
                    continue
                weather = _structure(record.get("weather_sequence"), WEATHER_LENGTH, checks, key)
                departures = _structure(record.get("departure_sequence"), DEPARTURE_LENGTH, checks, key)
                if departures and not departures_enabled:
                    checks.structure_errors += 1
                    checks.note(f"{key}: departure sequence present although departures are not approved")
                row = position.get(key)
                if row is None:
                    continue
                checks.development_records += 1
                seen[row] = True
                wx_len[row] = len(weather)
                for slot, event in enumerate(weather):
                    wx_events[row, slot] = _utc_ns(event["event_utc"])
                    wx_values[row, slot] = [
                        np.nan if event.get(item) is None else float(event[item]) for item in WEATHER_INPUTS
                    ]
                if departures_enabled:
                    dep_len[row] = len(departures)
                    for slot, event in enumerate(departures):
                        dep_events[row, slot] = _utc_ns(event["event_utc"])
                        dep_available[row, slot] = _utc_ns(event.get("available_utc") or event["event_utc"])
                        dep_minutes[row, slot] = float(event["minutes_before_cutoff"])
                        delay = event.get("dep_delay")
                        dep_delay[row, slot] = np.nan if delay is None else float(delay)
                        carrier = event.get("carrier")
                        if carrier is not None:
                            dep_carrier[row, slot] = carriers.setdefault(str(carrier), len(carriers))

    checks.missing_development_records = int((~seen).sum())
    cutoff = (
        pd.to_datetime(frame[CUTOFF], utc=True).dt.tz_convert(None)
        .to_numpy(dtype="datetime64[ns]").astype(np.int64)
    )

    def verify(events: np.ndarray, lengths: np.ndarray, available: np.ndarray | None = None) -> np.ndarray:
        active = np.arange(events.shape[1]) < lengths[:, None]
        late = active & (events > cutoff[:, None])
        checks.cutoff_violations += int(late.sum())
        if available is not None:
            checks.availability_violations += int((active & (available > cutoff[:, None])).sum())
        both = active[:, 1:] & active[:, :-1]
        checks.order_errors += int((both & (events[:, 1:] <= events[:, :-1])).sum())
        safe = np.where(active, events, cutoff[:, None])
        hours = np.where(active, (cutoff[:, None] - safe) / (60 * NS_PER_MINUTE), np.nan)
        return hours.astype(np.float32)

    wx_hours = verify(wx_events, wx_len)
    leads = frame[LEAD].astype(int).to_numpy()
    for lead in np.unique(leads):
        checks.weather_coverage_by_lead[str(lead)] = float((wx_len[leads == lead] > 0).mean())
    if departures_enabled:
        stated = dep_minutes
        actual = verify(dep_events, dep_len, dep_available) * 60
        mismatch = np.isfinite(actual) & (np.abs(stated - actual) > 1e-3)
        checks.offset_mismatches += int(mismatch.sum())
        for lead in np.unique(leads):
            checks.departure_coverage_by_lead[str(lead)] = float((dep_len[leads == lead] > 0).mean())
    raw = RawSequences(
        wx_values, wx_hours, wx_len,
        dep_minutes if departures_enabled else None,
        dep_delay if departures_enabled else None,
        dep_carrier if departures_enabled else None,
        dep_len if departures_enabled else None,
        list(carriers),
    )
    return raw, checks


def gate_decision(checks: SequenceChecks, *, mlp_check: dict, leakage_tests_passed: bool | None,
                  departures_enabled: bool) -> dict:
    """Apply the protocol's sequence_model_gate and explain each condition."""
    conditions = {
        "reproducible_mlp": {
            "passed": bool(mlp_check.get("passed")),
            "detail": mlp_check.get("detail", ""),
        },
        "sequences_exported_for_every_lead": {
            "passed": not checks.partitions_without_sequences and checks.missing_development_records == 0,
            "detail": (
                f"partitions without sequences: {checks.partitions_without_sequences}; "
                f"development rows without a record: {checks.missing_development_records}"
            ),
        },
        "timestamps_available_by_cutoff": {
            "passed": checks.cutoff_violations == 0 and checks.availability_violations == 0
            and checks.offset_mismatches == 0,
            "detail": (
                f"events after cutoff: {checks.cutoff_violations}; inputs available after cutoff: "
                f"{checks.availability_violations}; offset mismatches: {checks.offset_mismatches}"
            ),
        },
        "masks_and_padding": {
            "passed": checks.structure_errors == 0,
            "detail": f"structural errors: {checks.structure_errors}",
        },
        "events_in_time_order": {
            "passed": checks.order_errors == 0,
            "detail": f"out-of-order event pairs: {checks.order_errors}",
        },
        "leakage_tests_pass": {
            "passed": leakage_tests_passed is True,
            "detail": "not run" if leakage_tests_passed is None else ("passed" if leakage_tests_passed else "failed"),
        },
    }
    go = all(item["passed"] for item in conditions.values())
    return {
        "decision": "go" if go else "no-go",
        "conditions": conditions,
        "inputs": "weather and same-day departures" if departures_enabled else (
            "weather only (same-day departures not approved; protocol fallback)"
        ),
        "records_checked": checks.records,
        "development_records": checks.development_records,
        "weather_coverage_by_lead": checks.weather_coverage_by_lead,
        "departure_coverage_by_lead": checks.departure_coverage_by_lead,
        "examples": checks.examples,
    }


class SequencePreprocessor:
    """Training-only scaling of sequence values; padded slots stay zero."""

    def __init__(self, settings: dict, carrier_lookup: dict[str, int] | None, departures_enabled: bool,
                 carrier_index: int | None = None):
        self.settings = settings
        self.carrier_lookup = carrier_lookup or {}
        self.departures_enabled = departures_enabled
        self.carrier_index = carrier_index
        self.weather_means: list[float] = []
        self.weather_scales: list[float] = []

    def fit(self, raw: dict) -> "SequencePreprocessor":
        active = np.arange(WEATHER_LENGTH) < raw["wx_len"][:, None]
        values = raw["wx_values"][active].astype(np.float64)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(values, axis=0) if len(values) else np.zeros(len(WEATHER_INPUTS))
            scales = np.nanstd(values, axis=0) if len(values) else np.ones(len(WEATHER_INPUTS))
        self.weather_means = np.where(np.isfinite(means), means, 0.0).tolist()
        self.weather_scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0).tolist()
        return self

    @property
    def weather_dim(self) -> int:
        return len(WEATHER_INPUTS) + 2

    def transform(self, raw: dict) -> dict:
        n = len(raw["wx_len"])
        active = np.arange(WEATHER_LENGTH) < raw["wx_len"][:, None]
        means = np.asarray(self.weather_means, dtype=np.float32)
        scales = np.asarray(self.weather_scales, dtype=np.float32)
        values = (raw["wx_values"].astype(np.float32, copy=False) - means) / scales
        missing = np.isnan(values).any(axis=2, keepdims=True) & active[..., None]
        np.nan_to_num(values, copy=False, nan=0.0)
        np.clip(values, -8, 8, out=values)
        hours = np.nan_to_num(raw["wx_hours"], nan=0.0).astype(np.float32)[..., None]
        hours /= np.float32(self.settings["weather_hours_scale"])
        wx = np.concatenate([values, missing.astype(np.float32), hours], axis=2)
        wx *= active[..., None]
        # Standardised values fit comfortably in float16, which halves memory on full folds.
        arrays = {"wx": wx.astype(np.float16), "wx_len": raw["wx_len"].astype(np.int64)}
        del wx, values
        if self.departures_enabled:
            low, high = self.settings["departure_delay_clip_minutes"]
            dep_active = np.arange(DEPARTURE_LENGTH) < raw["dep_len"][:, None]
            minutes = np.nan_to_num(raw["dep_minutes"], nan=0.0) / self.settings["departure_minutes_scale"]
            delay = np.clip(np.nan_to_num(raw["dep_delay"], nan=0.0), low, high) / 60.0
            dep = np.stack([minutes, delay], axis=2).astype(np.float32) * dep_active[..., None]
            carriers = pd.Series(np.asarray(raw["dep_carrier"], dtype=object).ravel())
            codes = carriers.map(self.carrier_lookup).where(carriers.notna(), MISSING_INDEX)
            codes = codes.fillna(UNKNOWN_INDEX).to_numpy(dtype=np.int64).reshape(n, DEPARTURE_LENGTH)
            arrays.update(dep=dep.astype(np.float16), dep_carrier=codes * dep_active,
                          dep_len=raw["dep_len"].astype(np.int64))
        return arrays

    def state(self) -> dict:
        return {"settings": self.settings, "carrier_lookup": self.carrier_lookup,
                "departures_enabled": self.departures_enabled, "carrier_index": self.carrier_index,
                "weather_means": self.weather_means, "weather_scales": self.weather_scales}

    @classmethod
    def from_state(cls, state: dict) -> "SequencePreprocessor":
        item = cls(state["settings"], state["carrier_lookup"], state["departures_enabled"], state["carrier_index"])
        item.weather_means = list(state["weather_means"])
        item.weather_scales = list(state["weather_scales"])
        return item


class SequenceSource:
    """Adapter used by the Step 6 fold preparation to add sequence inputs."""

    def __init__(self, raw: RawSequences, settings: dict):
        self.raw = raw
        self.settings = settings["sequence"]

    def fit(self, rows: np.ndarray, tabular) -> SequencePreprocessor:
        has_carrier = "carrier" in tabular.categorical
        if self.raw.departures_enabled and not has_carrier:
            raise ValueError("Departure sequences need the carrier category in the feature set")
        return SequencePreprocessor(
            self.settings,
            tabular.category_lookup("carrier") if has_carrier else None,
            self.raw.departures_enabled,
            tabular.categorical.index("carrier") if has_carrier else None,
        ).fit(self.raw.take(rows))

    def arrays(self, preprocessor: SequencePreprocessor, rows: np.ndarray) -> dict:
        return preprocessor.transform(self.raw.take(rows))

    def dims(self, preprocessor: SequencePreprocessor) -> dict:
        dims = {"weather_dim": preprocessor.weather_dim}
        if self.raw.departures_enabled:
            dims.update(departure_dim=2, departure_carrier_index=preprocessor.carrier_index)
        return dims
