"""Choose operating thresholds using validation predictions only."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def select_operating_thresholds(
    y_val: np.ndarray,
    y_val_prob: np.ndarray,
    precision_floor: float = 0.35,
    split_name: str = "validation",
    min_recall: float = 0.05,
    min_alerts: int = 100,
) -> dict:
    """Return a feasible precision-floor threshold and a separate F1 reference.

    A high-precision point based on only one or two alerts is not considered an
    operating point. If the floor is infeasible, its threshold is ``None``.
    """
    if split_name != "validation":
        raise ValueError(f"Threshold selection is prohibited on '{split_name}' data")
    if not 0 < precision_floor <= 1 or not 0 <= min_recall <= 1 or min_alerts < 1:
        raise ValueError("invalid threshold policy")
    y = np.asarray(y_val, dtype=int)
    p = np.asarray(y_val_prob, dtype=float)
    if len(y) != len(p) or len(y) == 0 or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("invalid validation predictions")
    precision, recall, thresholds = precision_recall_curve(y, p)
    if not len(thresholds):
        raise ValueError("no candidate thresholds")
    precision, recall = precision[:-1], recall[:-1]
    alerts = np.rint(recall * y.sum() / np.maximum(precision, 1e-12)).astype(int)
    feasible = np.flatnonzero((precision >= precision_floor) & (recall >= min_recall) & (alerts >= min_alerts))
    floor_idx = max(feasible, key=lambda i: (recall[i], precision[i], thresholds[i])) if len(feasible) else None
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    f1_idx = int(np.argmax(f1))
    return {
        "precision_requirement": {
            "floor_value": precision_floor,
            "status": "exploratory_operational_assumption",
            "floor_achieved": floor_idx is not None,
            "minimum_recall": min_recall,
            "minimum_alerts": min_alerts,
            "threshold": float(thresholds[floor_idx]) if floor_idx is not None else None,
            "validation_precision": float(precision[floor_idx]) if floor_idx is not None else None,
            "validation_recall": float(recall[floor_idx]) if floor_idx is not None else None,
            "validation_alerts": int(alerts[floor_idx]) if floor_idx is not None else None,
        },
        "best_f1_requirement": {
            "threshold": float(thresholds[f1_idx]),
            "validation_precision": float(precision[f1_idx]),
            "validation_recall": float(recall[f1_idx]),
            "validation_f1": float(f1[f1_idx]),
        },
    }
