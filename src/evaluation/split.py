"""Canonical chronological split creation and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def assign_chronological_split(dates: pd.Series) -> pd.Series:
    """Apply the frozen date boundaries without separating a service date."""
    normalized = pd.to_datetime(dates, errors="raise", utc=True).dt.tz_localize(None).dt.normalize()
    values = np.select(
        [normalized <= pd.Timestamp("2013-08-31"), normalized <= pd.Timestamp("2013-10-31")],
        ["train", "validation"], default="test",
    )
    return pd.Series(values, index=dates.index, dtype="string")


def build_split_manifest(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"flight_row_id", "sched_date", "substantial_delay"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"cannot build split manifest; missing columns: {missing}")
    if frame["flight_row_id"].isna().any() or frame["flight_row_id"].duplicated().any():
        raise ValueError("flight_row_id must be non-null and unique")
    manifest = frame[["flight_row_id", "sched_date"]].copy()
    manifest["sched_date"] = pd.to_datetime(manifest["sched_date"], utc=True).dt.tz_localize(None).dt.normalize()
    manifest["split"] = assign_chronological_split(manifest["sched_date"])
    manifest = manifest.sort_values(["sched_date", "flight_row_id"]).reset_index(drop=True)
    profiled = manifest.merge(
        frame[["flight_row_id", "substantial_delay"]], on="flight_row_id", validate="one_to_one"
    )
    summary = profiled.groupby("split", sort=False).agg(
        rows=("flight_row_id", "size"),
        positive=("substantial_delay", "sum"),
        min_date=("sched_date", "min"),
        max_date=("sched_date", "max"),
        unique_dates=("sched_date", "nunique"),
    ).reset_index()
    summary["positive"] = summary["positive"].astype(int)
    summary["negative"] = summary["rows"] - summary["positive"]
    summary["positive_prevalence"] = summary["positive"] / summary["rows"]
    validate_split_manifest(manifest)
    return manifest, summary


def validate_split_manifest(manifest: pd.DataFrame) -> None:
    required = {"flight_row_id", "sched_date", "split"}
    if not required <= set(manifest.columns):
        raise ValueError(f"split manifest requires {sorted(required)}")
    if manifest["flight_row_id"].isna().any() or manifest["flight_row_id"].duplicated().any():
        raise ValueError("split manifest contains null or duplicate IDs")
    if set(manifest["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("split manifest must contain train, validation, and test")
    dates_by_split = {
        name: set(pd.to_datetime(group["sched_date"]).dt.normalize())
        for name, group in manifest.groupby("split")
    }
    if any(dates_by_split[a] & dates_by_split[b] for a, b in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    )):
        raise ValueError("a service date occurs in more than one split")
    train_max = pd.to_datetime(manifest.loc[manifest["split"].eq("train"), "sched_date"]).max()
    val_dates = pd.to_datetime(manifest.loc[manifest["split"].eq("validation"), "sched_date"])
    test_min = pd.to_datetime(manifest.loc[manifest["split"].eq("test"), "sched_date"]).min()
    if not train_max < val_dates.min() <= val_dates.max() < test_min:
        raise ValueError("split date ranges are not strictly chronological")


def save_split_artifacts(frame: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    manifest, summary = build_split_manifest(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.csv"
    summary_path = output_dir / "split_summary.csv"
    manifest.to_csv(manifest_path, index=False, date_format="%Y-%m-%d")
    summary.to_csv(summary_path, index=False, date_format="%Y-%m-%d")
    return manifest_path, summary_path
