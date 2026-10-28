"""Historical delay aggregates with a conservative outcome-availability cutoff.

Enforces two strict temporal availability invariants:
1. `sched_date < target_flight.sched_date`: strictly prior calendar date (no same-day leakage).
2. A prior outcome enters only after a conservative completion bound: scheduled
   NYC departure plus 24 hours and any positive arrival delay. This avoids
   treating destination-local scheduled arrival times as New York local times.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def _get_prediction_cutoff_utc(df: pd.DataFrame) -> pd.Series:
    if "prediction_cutoff_utc" in df.columns:
        return pd.to_datetime(df["prediction_cutoff_utc"], utc=True)
    dep_ts = pd.to_datetime(df["sched_dep_ts"], errors="coerce")
    local = dep_ts.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    return local.dt.tz_convert("UTC") - dt.timedelta(minutes=60)


def _outcome_available_utc(df: pd.DataFrame) -> pd.Series:
    """Use an explicit availability timestamp or a conservative source-data bound.

    The supplied arrival clock is destination-local. Without a verified release
    timestamp, a 24-hour flight-duration bound plus positive arrival delay
    withholds an outcome until safely after its expected arrival. Unknown
    arrival delays never enter the historical pool.
    """
    if "actual_arr_utc" in df.columns:
        return pd.to_datetime(df["actual_arr_utc"], utc=True)
    dep_ts = pd.to_datetime(df["sched_dep_ts"], errors="coerce")
    dep_utc = dep_ts.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    delay = pd.to_numeric(df["arr_delay"], errors="coerce") if "arr_delay" in df else pd.Series(np.nan, index=df.index)
    return (dep_utc + dt.timedelta(hours=24) + pd.to_timedelta(delay.clip(lower=0), unit="m")).where(delay.notna())


def _compute_verified_prior(
    df: pd.DataFrame,
    group_cols: list[str],
    prefix: str,
    smoothing: float = 20.0,
) -> pd.DataFrame:
    """Compute prior count and smoothed delay rate with T-60 availability."""
    n = len(df)
    prior_flights = np.zeros(n, dtype=np.int64)
    prior_delays = np.zeros(n, dtype=np.float64)

    # Target fields
    target_dates = pd.to_datetime(df["sched_date"]).values
    target_cutoffs = df["_cutoff_utc"].values

    # Historical candidate pool: rows with a known label and availability bound.
    valid_hist = df["substantial_delay"].notna() & df["_available_utc"].notna()
    hist_pool = df.loc[
        valid_hist,
        ["sched_date", "_available_utc", "substantial_delay"] + group_cols,
    ].copy()

    # Pre-group hist pool by group_cols
    groups_hist = {k: v for k, v in hist_pool.groupby(group_cols, sort=False)}

    for key, target_sub in df.groupby(group_cols, sort=False):
        if key not in groups_hist:
            continue
        g_hist = groups_hist[key]
        g_hist_by_date = {d: v for d, v in g_hist.groupby("sched_date", sort=False)}

        sub_indices = target_sub.index.values
        sub_dates = target_dates[sub_indices]
        sub_cutoffs = target_cutoffs[sub_indices]

        unique_dates = np.unique(sub_dates)

        running_arr = []
        running_y = []

        for d in unique_dates:
            d_mask = sub_dates == d
            idx_sub = sub_indices[d_mask]
            cutoffs_sub = sub_cutoffs[d_mask]

            if len(running_arr) > 0:
                s_arr = np.array(running_arr)
                s_y = np.array(running_y)
                sort_idx = np.argsort(s_arr)
                s_arr = s_arr[sort_idx]
                s_y = s_y[sort_idx]
                cumsum_y = np.cumsum(s_y)

                # Binary search: find how many completed arrivals occurred <= target cutoff
                pos = np.searchsorted(s_arr, cutoffs_sub, side="right")
                valid_pos = pos > 0

                cnt = pos.astype(np.int64)
                dly = np.zeros(len(pos), dtype=np.float64)
                dly[valid_pos] = cumsum_y[pos[valid_pos] - 1]

                prior_flights[idx_sub] = cnt
                prior_delays[idx_sub] = dly

            # Add completed flights from date d to running pool for dates > d
            if d in g_hist_by_date:
                h_d = g_hist_by_date[d]
                running_arr.extend(h_d["_available_utc"].values)
                running_y.extend(h_d["substantial_delay"].values.astype(np.float64))

    # Compute smoothed rate and missing indicator
    res = pd.DataFrame(index=df.index)
    res[f"{prefix}_prior_count"] = pd.Series(prior_flights, index=df.index, dtype="Int64")

    # A fixed symmetric prior has no future or not-yet-available labels.
    smoothed_rate = (prior_delays + smoothing * 0.5) / (
        prior_flights + smoothing
    )
    is_missing = prior_flights == 0
    res[f"{prefix}_delay_rate"] = np.where(~is_missing, smoothed_rate, np.nan)
    res[f"{prefix}_history_missing"] = is_missing

    return res


def add_historical_features(
    flights: pd.DataFrame, smoothing: float = 20.0, history_training_end: str | None = None
) -> pd.DataFrame:
    """Add carrier and route rates using only conservatively available outcomes."""
    df = flights.copy()
    df["_cutoff_utc"] = _get_prediction_cutoff_utc(df)
    df["_available_utc"] = _outcome_available_utc(df)
    if history_training_end is not None:
        # Validation/test labels are evaluation outcomes, never feature inputs.
        df.loc[pd.to_datetime(df["sched_date"]).gt(pd.Timestamp(history_training_end)), "_available_utc"] = pd.NaT

    if "route" not in df.columns and "origin" in df.columns and "dest" in df.columns:
        df["route"] = df["origin"].astype(str) + "_" + df["dest"].astype(str)

    # Carrier rates
    carrier_feats = _compute_verified_prior(
        df, ["carrier"], "carrier", smoothing=smoothing
    )
    # Route rates
    route_feats = _compute_verified_prior(
        df, ["origin", "dest"], "route", smoothing=smoothing
    )

    df["carrier_prior_count"] = carrier_feats["carrier_prior_count"]
    df["carrier_delay_rate"] = carrier_feats["carrier_delay_rate"]
    df["carrier_history_missing"] = carrier_feats["carrier_history_missing"]

    df["route_prior_count"] = route_feats["route_prior_count"]
    df["route_delay_rate"] = route_feats["route_delay_rate"]
    df["route_history_missing"] = route_feats["route_history_missing"]

    df = df.drop(columns=["_cutoff_utc", "_available_utc"])
    return df
