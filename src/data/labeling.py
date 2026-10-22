"""Construction of the substantial-delay target (Phase 1, Step 5).

Contract (configs/project_contract.yaml): substantial_delay = 1 when
arr_delay >= 15, 0 when arr_delay < 15, and an unknown label when arr_delay
is missing. Missing labels are excluded from supervised use downstream —
they are never zero-filled and never described as confirmed cancellations.
"""

from __future__ import annotations

import pandas as pd

THRESHOLD_MINUTES = 15

LABEL_KNOWN = "known"
LABEL_UNKNOWN = "unknown"


def add_delay_label(flights: pd.DataFrame, threshold: int = THRESHOLD_MINUTES) -> pd.DataFrame:
    """Add `substantial_delay` (nullable Int64) and `label_status` columns.

    `arr_delay` itself is preserved untouched as an audit-only column; it must
    never reach a model's feature matrix.
    """
    df = flights.copy()
    known = df["arr_delay"].notna()
    label = pd.Series(pd.NA, index=df.index, dtype="Int64")
    label[known & (df["arr_delay"] >= threshold)] = 1
    label[known & (df["arr_delay"] < threshold)] = 0
    df["substantial_delay"] = label
    df["label_status"] = known.map({True: LABEL_KNOWN, False: LABEL_UNKNOWN})
    return df


def label_summary(flights: pd.DataFrame) -> dict:
    """Counts and prevalence for the label, plus the unknown-label split by
    observed departure-time availability (an observable missingness pattern,
    not proof of a cause)."""
    total = len(flights)
    known = flights[flights["label_status"] == LABEL_KNOWN]
    unknown = flights[flights["label_status"] == LABEL_UNKNOWN]
    positive = int((known["substantial_delay"] == 1).sum())
    negative = int((known["substantial_delay"] == 0).sum())
    return {
        "total_rows": int(total),
        "known_labels": int(len(known)),
        "positive": positive,
        "negative": negative,
        "unknown_labels": int(len(unknown)),
        "positive_prevalence": round(positive / len(known), 6) if len(known) else None,
        "unknown_label_detail": {
            "also_missing_dep_time": int(unknown["dep_time"].isna().sum()),
            "dep_time_present": int(unknown["dep_time"].notna().sum()),
        },
    }


def labelled_subset(flights: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known label — the supervised population for later phases.

    Keeps `flight_row_id` so exclusions remain auditable; does not modify the
    input frame.
    """
    return flights[flights["label_status"] == LABEL_KNOWN].copy()
