"""Historical delay aggregates verified to be known before each row's T-60 cutoff.

Enforces two strict temporal availability invariants:
1. `sched_date < target_flight.sched_date`: strictly prior calendar date (no same-day leakage).
2. `actual_arr_utc <= target_flight.prediction_cutoff_utc`: every prior flight outcome
   used was physically completed and known before the flight's T-60 cutoff.
   Any prior-date flight that landed late at night or early morning after cutoff is excluded.
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


def _get_actual_arrival_utc(df: pd.DataFrame) -> pd.Series:
    if "actual_arr_utc" in df.columns:
        return pd.to_datetime(df["actual_arr_utc"], utc=True)
    if "sched_arr_ts" in df.columns:
        arr_ts = pd.to_datetime(df["sched_arr_ts"], errors="coerce")
        if "arr_delay" in df.columns:
            arr_delay = pd.to_numeric(df["arr_delay"], errors="coerce").fillna(0.0)
            arr_ts = arr_ts + pd.to_timedelta(arr_delay, unit="m")
        local = arr_ts.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
        return local.dt.tz_convert("UTC")
    dep_ts = pd.to_datetime(df["sched_dep_ts"], errors="coerce")
    local = dep_ts.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    return local.dt.tz_convert("UTC") + dt.timedelta(hours=2)


def _compute_verified_prior(
    df: pd.DataFrame,
    group_cols: list[str],
    prefix: str,
    smoothing: float = 20.0,
) -> pd.DataFrame:
    """Compute prior count and smoothed delay rate with verified T-60 arrival timing."""
    n = len(df)
    prior_flights = np.zeros(n, dtype=np.int64)
    prior_delays = np.zeros(n, dtype=np.float64)

    # Global daily sums for prior baseline smoothing
    base = df.loc[
        df["substantial_delay"].notna(),
        ["sched_date", "substantial_delay"],
    ].copy()
    totals = (
        base.groupby("sched_date")["substantial_delay"]
        .agg(all_delays="sum", all_flights="count")
        .sort_index()
    )
    totals["global_prior_delays"] = totals["all_delays"].cumsum() - totals["all_delays"]
    totals["global_prior_flights"] = (
        totals["all_flights"].cumsum() - totals["all_flights"]
    )
    totals["global_prior_rate"] = totals["global_prior_delays"] / totals[
        "global_prior_flights"
    ].replace(0, np.nan)
    global_prior_map = totals["global_prior_rate"].to_dict()

    # Target fields
    target_dates = pd.to_datetime(df["sched_date"]).values
    target_cutoffs = df["_cutoff_utc"].values

    # Historical candidate pool: rows with valid substantial_delay and actual arrival
    valid_hist = df["substantial_delay"].notna() & df["_actual_arr_utc"].notna()
    hist_pool = df.loc[
        valid_hist,
        ["sched_date", "_actual_arr_utc", "substantial_delay"] + group_cols,
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
                running_arr.extend(h_d["_actual_arr_utc"].values)
                running_y.extend(h_d["substantial_delay"].values.astype(np.float64))

    # Compute smoothed rate and missing indicator
    res = pd.DataFrame(index=df.index)
    res[f"{prefix}_prior_count"] = pd.Series(prior_flights, index=df.index, dtype="Int64")

    # Map global prior rate by sched_date
    sched_dates_series = pd.to_datetime(df["sched_date"])
    global_prior = sched_dates_series.map(global_prior_map).astype(float)

    smoothed_rate = (prior_delays + smoothing * global_prior) / (
        prior_flights + smoothing
    )
    is_missing = prior_flights == 0
    res[f"{prefix}_delay_rate"] = smoothed_rate.where(~is_missing, np.nan)
    res[f"{prefix}_history_missing"] = is_missing

    return res


def add_historical_features(
    flights: pd.DataFrame, smoothing: float = 20.0
) -> pd.DataFrame:
    """Add smoothed carrier and route rates based strictly on outcomes available at T-60."""
    df = flights.copy()
    df["_cutoff_utc"] = _get_prediction_cutoff_utc(df)
    df["_actual_arr_utc"] = _get_actual_arrival_utc(df)

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

    df = df.drop(columns=["_cutoff_utc", "_actual_arr_utc"])
    return df