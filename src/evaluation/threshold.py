"""Validation-only threshold selection (Phase 5, Step 3).

Guarantees strict data isolation:
- Threshold decisions are made on validation predictions only.
- Passing test data into this selection module raises a ValueError.
- Stated precision floor (default 0.70) is evaluated and clearly labeled
  as an exploratory operational assumption.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def select_operating_thresholds(
    y_val: np.ndarray,
    y_val_prob: np.ndarray,
    precision_floor: float = 0.70,
    split_name: str = "validation",
) -> dict:
    """Select decision thresholds using validation data only.

    Parameters
    ----------
    y_val : np.ndarray
        True binary labels from the validation split.
    y_val_prob : np.ndarray
        Predicted probabilities on the validation split.
    precision_floor : float
        Target precision requirement (labeled exploratory).
    split_name : str
        Name of split. Must be 'validation' to prevent leakage.

    Returns
    -------
    dict
        Threshold configuration and metrics achieved on validation data.
    """
    if split_name.lower() != "validation":
        raise ValueError(
            f"Threshold selection is prohibited on '{split_name}' data! "
            f"Phase 0 and Phase 5 contracts mandate validation-only threshold tuning."
        )

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)

    # 1. Best F1 threshold on validation
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * (precisions * recalls) / (precisions + recalls),
        0.0,
    )
    best_f1_idx = int(np.argmax(f1_scores))
    if best_f1_idx < len(thresholds):
        best_f1_thresh = float(thresholds[best_f1_idx])
    else:
        best_f1_thresh = float(thresholds[-1])

    # 2. Precision floor threshold (e.g. >= 0.70 exploratory requirement)
    valid_floor_mask = (precisions[:-1] >= precision_floor) & (recalls[:-1] > 0)

    if np.any(valid_floor_mask):
        valid_indices = np.where(valid_floor_mask)[0]
        # Choose the first threshold meeting floor to maximize recall
        chosen_idx = valid_indices[0]
        floor_thresh = float(thresholds[chosen_idx])
        floor_prec = float(precisions[chosen_idx])
        floor_rec = float(recalls[chosen_idx])
        floor_achieved = True
    else:
        # Fallback if floor is unreachable: highest precision with >= 1% recall
        high_recall_mask = recalls[:-1] >= 0.01
        if np.any(high_recall_mask):
            sub_indices = np.where(high_recall_mask)[0]
            max_p_idx = sub_indices[np.argmax(precisions[sub_indices])]
            floor_thresh = float(thresholds[max_p_idx])
            floor_prec = float(precisions[max_p_idx])
            floor_rec = float(recalls[max_p_idx])
        else:
            floor_thresh = 0.5
            floor_prec = float(precisions[0])
            floor_rec = float(recalls[0])
        floor_achieved = False

    return {
        "precision_requirement": {
            "floor_value": precision_floor,
            "status": "exploratory_operational_assumption",
            "floor_achieved": floor_achieved,
            "threshold": round(floor_thresh, 4),
            "validation_precision": round(floor_prec, 4),
            "validation_recall": round(floor_rec, 4),
        },
        "best_f1_requirement": {
            "threshold": round(best_f1_thresh, 4),
            "validation_precision": round(float(precisions[best_f1_idx]), 4),
            "validation_recall": round(float(recalls[best_f1_idx]), 4),
            "validation_f1": round(float(f1_scores[best_f1_idx]), 4),
        },
        "default_threshold": 0.50,
    }
