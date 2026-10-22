"""Data profiling: missingness, duplicates and join-key coverage (Phase 1, Step 6).

Everything here is measured at runtime from the supplied files. Reference
counts in docs/contract are expectations, never runtime constants.
"""

from __future__ import annotations

import pandas as pd


def column_missingness(df: pd.DataFrame) -> dict[str, dict]:
    """Missing count and percentage per column (only columns with gaps)."""
    result = {}
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing:
            result[col] = {
                "missing": missing,
                "pct": round(100 * missing / len(df), 4),
            }
    return result


def count_full_duplicates(df: pd.DataFrame) -> int:
    """Rows identical in every column (identifier column excluded)."""
    columns = [c for c in df.columns if c != "flight_row_id"]
    return int(df.duplicated(subset=columns, keep=False).sum())


def duplicate_key_counts(sources: dict) -> dict[str, dict]:
    """Duplicate lookup keys that would multiply rows if joined blindly."""
    checks = {
        "airlines": ("carrier",),
        "airports": ("faa",),
        "planes": ("tailnum",),
        "weather": ("origin", "time_hour"),
    }
    result = {}
    for name, keys in checks.items():
        df = sources[name].df
        dup_keys = df.duplicated(subset=keys, keep=False).sum()
        result[name] = {
            "key_columns": list(keys),
            "duplicate_key_rows": int(dup_keys),
        }
    return result


def _unmatched_rows(flights: pd.DataFrame, column: str, lookup: pd.DataFrame,
                    key: str) -> pd.DataFrame:
    values = flights[column].dropna()
    return values[~values.isin(lookup[key].dropna())]


def join_coverage(flights: pd.DataFrame, sources: dict) -> dict:
    """Join-key coverage for airline, airport, plane and weather lookups.

    This is a Phase 1 profile only; the actual modelling joins (including the
    weather as-of join) are built in Phase 2.
    """
    flights_df = flights
    airlines = sources["airlines"].df
    airports = sources["airports"].df
    planes = sources["planes"].df
    weather = sources["weather"].df

    unmatched_carrier = _unmatched_rows(flights_df, "carrier", airlines, "carrier")
    unmatched_origin = _unmatched_rows(flights_df, "origin", airports, "faa")
    unmatched_dest = _unmatched_rows(flights_df, "dest", airports, "faa")

    tailnum = flights_df["tailnum"].dropna()
    tailnum_matched = tailnum.isin(planes["tailnum"].dropna())

    # Same-hour origin/time weather availability: profile only. Availability
    # at the T-60 cutoff is a Phase 2 concern (as-of join).
    weather_keys = set(
        map(tuple, weather[["origin", "time_hour"]].dropna().astype(str).values)
    )
    flight_keys = list(
        map(tuple, flights_df[["origin", "time_hour"]].astype(str).values)
    )
    weather_same_hour_rows = sum(key in weather_keys for key in flight_keys)

    def _summary(values: pd.Series, label: str) -> dict:
        return {
            label: sorted(values.unique().tolist()),
            "rows": int(len(values)),
        }

    return {
        "airlines": {
            "unmatched_carriers": _summary(unmatched_carrier, "carrier_codes"),
        },
        "airports_origin": {
            "unmatched_airports": _summary(unmatched_origin, "airport_codes"),
        },
        "airports_destination": {
            "unmatched_airports": _summary(unmatched_dest, "airport_codes"),
        },
        "planes": {
            "non_missing_tailnum_rows": int(len(tailnum)),
            "matched_rows": int(tailnum_matched.sum()),
            "unmatched_rows": int((~tailnum_matched).sum()),
        },
        "weather": {
            "rows_with_same_hour_observation": weather_same_hour_rows,
            "rows_without_same_hour_observation": int(
                len(flights_df) - weather_same_hour_rows
            ),
            "note": "profile only; T-60 as-of join is Phase 2",
        },
    }


def date_coverage(flights: pd.DataFrame) -> dict:
    """Scheduled-date coverage of the flights table."""
    dates = flights["sched_date"].dropna()
    if dates.empty:
        return {"min": None, "max": None, "distinct_days": 0}
    return {
        "min": str(dates.min().date()),
        "max": str(dates.max().date()),
        "distinct_days": int(dates.dt.date.nunique()),
    }


def monthly_label_distribution(flights: pd.DataFrame) -> list[dict]:
    """Rows, label counts and prevalence per month (labelled rows only)."""
    known = flights[flights["label_status"] == "known"]
    grouped = known.groupby(known["sched_date"].dt.month, dropna=True)
    rows = []
    for month, group in grouped:
        positive = int((group["substantial_delay"] == 1).sum())
        total = int(len(group))
        rows.append({
            "month": int(month),
            "labelled": total,
            "positive": positive,
            "negative": total - positive,
            "prevalence": round(positive / total, 6) if total else None,
        })
    return sorted(rows, key=lambda r: r["month"])
