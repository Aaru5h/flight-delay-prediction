"""Read feature artifacts and assign protocol v2 baseline folds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loading import file_fingerprint

ID = "flight_row_id"
LEAD = "lead_time_minutes"
DATE = "sched_date"
CUTOFF = "prediction_cutoff_utc"
AVAILABLE = "outcome_available_utc"
LABEL = "substantial_delay"
FOLD_MANIFEST_VERSION = "v2-walk-forward-1"


@dataclass
class FeatureSet:
    name: str
    frame: pd.DataFrame
    manifest_version: str
    allowlist: list[str]
    categorical: list[str]
    file_fingerprints: dict[str, str]


@dataclass
class Fold:
    fold_id: str
    fit_cutoff_utc: pd.Timestamp
    train_ids: pd.Index
    evaluation_ids: pd.Index


def _feature_columns(manifest: dict, protocol: dict, *, multi_lead: bool) -> tuple[list[str], list[str]]:
    allowlist = manifest["allowlist"]
    denied = {
        ID, DATE, CUTOFF, AVAILABLE, LABEL, "label_status", "scheduled_departure_utc",
        "weather_observed_utc", "weather_age_minutes", "split", "fold_id",
        *manifest.get("audit_columns", []), *manifest.get("denylist", []),
        *protocol["prediction_time"]["prohibited_target_flight_inputs"],
    }
    if multi_lead:
        denied.update(("sched_month", "sched_day_of_year"))
    feature_types = {item["name"]: item["type"] for item in manifest["features"]}
    if (
        not allowlist or len(set(allowlist)) != len(allowlist) or denied.intersection(allowlist)
        or set(feature_types) != set(allowlist)
        or any(kind not in {"categorical", "numeric"} for kind in feature_types.values())
    ):
        raise ValueError("Invalid feature manifest or leakage-prone allowlist")
    return allowlist, [name for name in allowlist if feature_types[name] == "categorical"]


def _dates(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame[DATE], utc=True).dt.tz_localize(None).dt.normalize()


def _label_values(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values)
    if not values.isin([0, 1]).all():
        raise ValueError("Development labels must be known binary values")
    return values.astype("int8")


def _compact_numeric(frame: pd.DataFrame, allowlist: list[str], categorical: list[str]) -> None:
    numeric = [name for name in allowlist if name not in categorical]
    frame[numeric] = frame[numeric].astype("float32")


def _window(protocol: dict) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    window = protocol["validation"]
    retrospective = window["retrospective_period"]
    return (
        pd.Timestamp(window["train_start"]),
        pd.Timestamp(retrospective["start"]),
        pd.Timestamp(retrospective["end"]),
    )


def _require_unique_ids(index: pd.Index) -> None:
    if index.hasnans or not index.is_unique:
        raise ValueError("Feature partitions require one unique, non-null flight ID per lead")


def read_v2_features(
    dataset_dir: Path, protocol: dict
) -> tuple[FeatureSet, pd.DataFrame, dict[str, str], list[str]]:
    """Keep development features and retrospective schedule metadata only."""
    manifest_path = dataset_dir / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lead_times = protocol["prediction_time"]["lead_time_minutes"]
    partitions = {part[LEAD]: part for part in manifest["partitions"]}
    if (
        list(partitions) != lead_times
        or len(manifest["partitions"]) != len(lead_times)
        or manifest["protocol_version"] != protocol["contract_version"]
        or manifest["label_column"] != LABEL
    ):
        raise ValueError("Step 4 partitions or version do not match protocol v2")
    allowlist, categorical = _feature_columns(manifest, protocol, multi_lead=True)
    schedule_only = [
        item["name"] for item in manifest["features"]
        if item["availability_rule"] == "published schedule and fixed calendar"
    ]
    if LEAD not in allowlist or LEAD not in schedule_only:
        raise ValueError("The full and schedule-only v2 feature sets require lead_time_minutes")

    start, retro_start, retro_end = _window(protocol)
    columns = set(allowlist) | {ID, LEAD, DATE, CUTOFF, AVAILABLE, LABEL}
    fingerprints = {manifest_path.name: file_fingerprint(manifest_path)}
    frames = []
    metadata = None
    reference_labels = None
    largest_lead = max(lead_times)
    for lead in sorted(lead_times, reverse=True):
        path = dataset_dir / partitions[lead]["features"]
        fingerprints[path.name] = file_fingerprint(path)
        development, audit = [], []
        for chunk in pd.read_csv(path, usecols=columns, dtype={ID: "string"}, chunksize=50_000):
            chunk[DATE] = _dates(chunk)
            if not chunk[LEAD].eq(lead).all() or not chunk[DATE].between(start, retro_end).all():
                raise ValueError(f"Unexpected lead time or service date in {path.name}")
            chunk[CUTOFF] = pd.to_datetime(chunk[CUTOFF], utc=True)
            if chunk[CUTOFF].isna().any():
                raise ValueError(f"Missing prediction cutoff in {path.name}")
            audit.append(chunk[[ID, DATE, CUTOFF]].copy())
            dev = chunk.loc[chunk[DATE].lt(retro_start), [ID, DATE, CUTOFF, AVAILABLE, LABEL, *allowlist]].copy()
            dev[LABEL] = _label_values(dev[LABEL])
            dev[AVAILABLE] = pd.to_datetime(dev[AVAILABLE], utc=True)
            _compact_numeric(dev, allowlist, categorical)
            development.append(dev)
        partition = pd.concat(development, ignore_index=True)
        audit = pd.concat(audit, ignore_index=True).set_index(ID)
        _require_unique_ids(audit.index)
        if len(audit) != partitions[lead]["rows"]:
            raise ValueError(f"Row count mismatch in {path.name}")
        labels = partition.set_index(ID)[[LABEL, AVAILABLE]]
        if metadata is None:
            metadata = audit.rename(columns={CUTOFF: "first_cutoff_utc"})
            metadata = metadata.join(labels[[AVAILABLE]], validate="one_to_one")
            reference_labels = labels
        else:
            if not audit.index.sort_values().equals(metadata.index.sort_values()):
                raise ValueError("Step 4 feature partitions disagree on flight IDs")
            aligned = audit.reindex(metadata.index)
            flight_cutoffs = metadata["first_cutoff_utc"] + pd.to_timedelta(largest_lead - lead, unit="m")
            if (
                not aligned[DATE].equals(metadata[DATE])
                or not aligned[CUTOFF].equals(flight_cutoffs)
                or not labels.reindex(reference_labels.index).equals(reference_labels)
            ):
                raise ValueError("Step 4 partitions disagree on labels, dates or cutoffs")
        frames.append(partition)
    return (
        FeatureSet(
            "multi_lead", pd.concat(frames, ignore_index=True), manifest["manifest_version"],
            allowlist, categorical, fingerprints,
        ),
        metadata,
        manifest["source_fingerprints"],
        schedule_only,
    )


def read_legacy_features(
    path: Path, manifest_path: Path, v2: FeatureSet, protocol: dict
) -> FeatureSet:
    """Match the old T−60 feature representation to the same v2 flights."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowlist, categorical = _feature_columns(manifest, protocol, multi_lead=False)
    if LEAD in allowlist:
        raise ValueError("The legacy feature table must contain only the T−60 representation")
    start, retro_start, _ = _window(protocol)
    columns = set(allowlist) | {ID, DATE, LABEL, CUTOFF}
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, dtype={ID: "string"}, chunksize=50_000):
        chunk[DATE] = _dates(chunk)
        dev = chunk.loc[chunk[DATE].ge(start) & chunk[DATE].lt(retro_start)].copy()
        dev[LABEL] = _label_values(dev[LABEL])
        dev[CUTOFF] = pd.to_datetime(dev[CUTOFF], utc=True)
        _compact_numeric(dev, allowlist, categorical)
        parts.append(dev)
    legacy = pd.concat(parts, ignore_index=True).set_index(ID)
    reference = v2.frame.loc[v2.frame[LEAD].eq(60), [ID, DATE, CUTOFF, LABEL, AVAILABLE]].set_index(ID)
    _require_unique_ids(legacy.index)
    if not legacy.index.sort_values().equals(reference.index.sort_values()):
        raise ValueError("Legacy and v2 T−60 development flight populations differ")
    legacy = legacy.reindex(reference.index)
    if not legacy[[DATE, CUTOFF, LABEL]].equals(reference[[DATE, CUTOFF, LABEL]]):
        raise ValueError("Legacy labels, dates or cutoffs disagree with Step 4")
    legacy[AVAILABLE] = reference[AVAILABLE]
    legacy[LEAD] = 60
    return FeatureSet(
        "legacy_aml", legacy.reset_index(), manifest["manifest_version"], allowlist, categorical,
        {path.name: file_fingerprint(path), manifest_path.name: file_fingerprint(manifest_path)},
    )


def make_folds(metadata: pd.DataFrame, protocol: dict) -> tuple[list[Fold], Fold]:
    """Use the earliest available prediction cutoff across all four lead times."""
    start, retro_start, retro_end = _window(protocol)
    windows = [
        (month, pd.Period(month).start_time, pd.Period(month).end_time.normalize())
        for month in protocol["validation"]["validation_months"]
    ]
    folds = []
    for fold_id, evaluation_start, evaluation_end in [*windows, ("retrospective", retro_start, retro_end)]:
        evaluation = metadata[DATE].between(evaluation_start, evaluation_end)
        if not evaluation.any():
            raise ValueError(f"No flights for the {fold_id} evaluation window")
        cutoff = metadata.loc[evaluation, "first_cutoff_utc"].min()
        train = metadata[DATE].ge(start) & metadata[DATE].lt(evaluation_start) & metadata[AVAILABLE].le(cutoff)
        if not train.any():
            raise ValueError(f"No cutoff-eligible training flights for {fold_id}")
        folds.append(Fold(fold_id, cutoff, metadata.index[train], metadata.index[evaluation]))
    return folds[:-1], folds[-1]


def inner_split(metadata: pd.DataFrame, fold: Fold, days: int) -> tuple[pd.Index, pd.Index]:
    """Hold out the last training dates and apply the earlier inner cutoff."""
    train = metadata.loc[fold.train_ids]
    first_holdout_date = train[DATE].max() - timedelta(days=days - 1)
    holdout = train[DATE].ge(first_holdout_date)
    cutoff = train.loc[holdout, "first_cutoff_utc"].min()
    eligible = ~holdout & train[AVAILABLE].le(cutoff)
    if not eligible.any() or not holdout.any():
        raise ValueError(f"Fold {fold.fold_id} has too few training dates for early stopping")
    return train.index[eligible], train.index[holdout]
