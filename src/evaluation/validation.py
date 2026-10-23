"""Prediction schema, ID, label, and population validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_PREDICTION_COLUMNS = [
    "flight_row_id", "y_true", "y_probability", "split", "model_name", "run_id"
]


def validate_predictions(
    predictions: pd.DataFrame,
    split_manifest: pd.DataFrame,
    labels: pd.DataFrame,
    expected_split: str,
) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError(f"prediction file missing columns: {missing}")
    if predictions["flight_row_id"].isna().any() or predictions["flight_row_id"].duplicated().any():
        raise ValueError("prediction IDs must be non-null and unique")
    probability = predictions["y_probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("prediction probabilities must be finite and in [0, 1]")
    if set(predictions["split"].unique()) != {expected_split}:
        raise ValueError(f"prediction file must contain split={expected_split!r} only")
    expected = split_manifest.loc[
        split_manifest["split"].eq(expected_split), ["flight_row_id", "split"]
    ].merge(labels[["flight_row_id", "substantial_delay"]], on="flight_row_id", validate="one_to_one")
    if set(predictions["flight_row_id"]) != set(expected["flight_row_id"]):
        missing_ids = len(set(expected["flight_row_id"]) - set(predictions["flight_row_id"]))
        extra_ids = len(set(predictions["flight_row_id"]) - set(expected["flight_row_id"]))
        raise ValueError(f"prediction population mismatch: missing={missing_ids}, extra={extra_ids}")
    checked = predictions.merge(expected, on=["flight_row_id", "split"], validate="one_to_one")
    if not np.array_equal(checked["y_true"].astype(int), checked["substantial_delay"].astype(int)):
        raise ValueError("prediction labels do not agree with canonical labels")
    return checked.drop(columns="substantial_delay")
