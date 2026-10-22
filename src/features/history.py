"""Historical delay aggregates using calendar dates strictly before each row."""

from __future__ import annotations

import pandas as pd


def _prior_rate(
    flights: pd.DataFrame, group_cols: list[str], prefix: str, smoothing: float
) -> pd.DataFrame:
    base = flights.loc[
        flights["substantial_delay"].notna(),
        ["flight_row_id", "sched_date", *group_cols, "substantial_delay"],
    ].copy()
    daily = base.groupby(["sched_date", *group_cols], dropna=False)["substantial_delay"].agg(
        daily_delays="sum", daily_flights="count"
    ).reset_index()
    daily = daily.sort_values([*group_cols, "sched_date"])
    grouped = daily.groupby(group_cols, dropna=False, sort=False)
    daily["prior_delays"] = grouped["daily_delays"].cumsum() - daily["daily_delays"]
    daily["prior_flights"] = grouped["daily_flights"].cumsum() - daily["daily_flights"]

    totals = base.groupby("sched_date")["substantial_delay"].agg(
        all_delays="sum", all_flights="count"
    ).sort_index()
    totals["global_prior_delays"] = totals["all_delays"].cumsum() - totals["all_delays"]
    totals["global_prior_flights"] = totals["all_flights"].cumsum() - totals["all_flights"]
    totals["global_prior_rate"] = (
        totals["global_prior_delays"] / totals["global_prior_flights"].replace(0, pd.NA)
    )
    daily = daily.merge(totals[["global_prior_rate"]], left_on="sched_date", right_index=True)
    daily[f"{prefix}_prior_count"] = daily["prior_flights"].astype("Int64")
    daily[f"{prefix}_delay_rate"] = (
        daily["prior_delays"] + smoothing * daily["global_prior_rate"]
    ) / (daily["prior_flights"] + smoothing)
    daily[f"{prefix}_history_missing"] = daily["prior_flights"].eq(0)
    daily.loc[daily[f"{prefix}_history_missing"], f"{prefix}_delay_rate"] = pd.NA
    return daily[["sched_date", *group_cols, f"{prefix}_prior_count", f"{prefix}_delay_rate", f"{prefix}_history_missing"]]


def add_historical_features(flights: pd.DataFrame, smoothing: float = 20.0) -> pd.DataFrame:
    """Add smoothed carrier and route rates based only on earlier dates."""
    df = flights.copy()
    for groups, prefix in ((["carrier"], "carrier"), (["origin", "dest"], "route")):
        rates = _prior_rate(df, groups, prefix, smoothing)
        df = df.merge(rates, on=["sched_date", *groups], how="left", validate="many_to_one")
    return df
