"""Shared run helpers for the neural steps.

These are exact copies of the Step 5 helpers in
``src/models/classical/train_step5.py``. They live here so the neural code
never imports that module: it loads XGBoost, and on Apple-silicon Macs
loading XGBoost's OpenMP runtime next to PyTorch's can crash the process
with a segmentation fault. Keep them in sync with the Step 5 versions;
``tests/test_step6_neural.py`` checks that they still match.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data import config
from src.models.classical.step5_data import DATE, ID, LABEL, LEAD, Fold

PREDICTION_COLUMNS = [ID, LEAD, "fold_id", "split", "y_true", "y_probability", "model_name", "run_id"]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def code_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT, capture_output=True, text=True,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def write_fold_manifest(path: Path, metadata: pd.DataFrame, folds: list[Fold], final: Fold) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        for fold in [*folds, final]:
            evaluation_split = "retrospective" if fold is final else "validation"
            rows = pd.concat([
                metadata.loc[fold.train_ids, [DATE]].assign(split="train"),
                metadata.loc[fold.evaluation_ids, [DATE]].assign(split=evaluation_split),
            ]).assign(fold_id=fold.fold_id)
            rows.reset_index()[[ID, "fold_id", DATE, "split"]].to_csv(
                stream, index=False, header=fold is folds[0], date_format="%Y-%m-%d"
            )


def average_precision_by_lead(evaluation: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    metrics = {}
    leads = evaluation[LEAD].to_numpy()
    targets = evaluation[LABEL].to_numpy()
    for lead in np.unique(leads):
        mask = leads == lead
        if len(np.unique(targets[mask])) != 2:
            raise ValueError(f"Validation requires both classes at {lead} minutes")
        metrics[str(lead)] = float(average_precision_score(targets[mask], scores[mask]))
    if not metrics:
        raise ValueError("Empty validation population")
    return metrics
