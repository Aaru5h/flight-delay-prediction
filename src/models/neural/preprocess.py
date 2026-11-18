"""Training-only preprocessing for the neural models.

Category vocabularies, numeric medians, missing indicators and scaling are all
learned from the training rows passed to ``fit``. Index 0 is reserved for
categories that are rare or unseen in training, and index 1 for missing values,
so neither can be mistaken for a real category.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

UNKNOWN_INDEX = 0
MISSING_INDEX = 1
RESERVED_INDICES = 2


def category_strings(values: pd.Series) -> np.ndarray:
    """Return canonical strings so 7, 7.0 and "7" map to one category."""
    numeric = pd.to_numeric(values, errors="coerce")
    is_number = numeric.notna() & values.notna()
    integral = is_number & np.isclose(numeric.fillna(0) % 1, 0)
    out = values.astype("string")
    out = out.mask(integral, numeric.where(integral).round().astype("Int64").astype("string"))
    out = out.mask(is_number & ~integral, numeric.astype("string"))
    return out.fillna("<missing>").to_numpy(dtype=object)


def embedding_dim(cardinality: int, max_dim: int) -> int:
    """Use the common 1.6 * n^0.56 rule, capped for a small dataset."""
    return int(max(2, min(max_dim, round(1.6 * cardinality ** 0.56))))


class TabularPreprocessor:
    """Encode categoricals as indices and numerics as imputed, scaled floats."""

    def __init__(
        self, allowlist: list[str], categorical: list[str], *,
        min_category_count: int = 5, numeric_clip: float = 8.0,
    ):
        unknown = set(categorical) - set(allowlist)
        if unknown:
            raise ValueError(f"Categorical columns missing from the allowlist: {sorted(unknown)}")
        if len(set(allowlist)) != len(allowlist):
            raise ValueError("The allowlist contains duplicate columns")
        self.allowlist = list(allowlist)
        self.categorical = [name for name in allowlist if name in set(categorical)]
        self.numeric = [name for name in allowlist if name not in set(categorical)]
        self.min_category_count = int(min_category_count)
        self.numeric_clip = float(numeric_clip)
        self.vocabularies: dict[str, list[str]] = {}
        self.medians: list[float] = []
        self.indicator_columns: list[str] = []
        self.means: list[float] = []
        self.scales: list[float] = []
        self.fitted = False

    def _numeric_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.numeric:
            return np.zeros((len(frame), 0), dtype=np.float64)
        return frame[self.numeric].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        if frame.empty:
            raise ValueError("Cannot fit preprocessing on an empty training frame")
        for name in self.categorical:
            counts = Counter(category_strings(frame[name]))
            counts.pop("<missing>", None)
            self.vocabularies[name] = sorted(
                value for value, count in counts.items() if count >= self.min_category_count
            )
        values = self._numeric_matrix(frame)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(values, axis=0) if values.shape[1] else np.zeros(0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        missing = np.isnan(values).any(axis=0)
        self.medians = medians.tolist()
        self.indicator_columns = [name for name, flag in zip(self.numeric, missing) if flag]
        filled = np.where(np.isnan(values), medians, values)
        means = filled.mean(axis=0) if values.shape[1] else np.zeros(0)
        scales = filled.std(axis=0) if values.shape[1] else np.zeros(0)
        self.means = means.tolist()
        self.scales = np.where(scales > 1e-12, scales, 1.0).tolist()
        self.fitted = True
        return self

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocabularies[name]) + RESERVED_INDICES for name in self.categorical]

    @property
    def numeric_dim(self) -> int:
        return len(self.numeric) + len(self.indicator_columns)

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Fit the preprocessor before transforming")
        missing_columns = set(self.allowlist) - set(frame.columns)
        if missing_columns:
            raise ValueError(f"Frame is missing estimator columns: {sorted(missing_columns)}")
        codes = np.empty((len(frame), len(self.categorical)), dtype=np.int64)
        for position, name in enumerate(self.categorical):
            lookup = {value: index + RESERVED_INDICES for index, value in enumerate(self.vocabularies[name])}
            strings = category_strings(frame[name])
            codes[:, position] = [
                MISSING_INDEX if value == "<missing>" else lookup.get(value, UNKNOWN_INDEX)
                for value in strings
            ]
        values = self._numeric_matrix(frame)
        medians = np.asarray(self.medians, dtype=np.float64)
        is_missing = np.isnan(values)
        filled = np.where(is_missing, medians, values)
        scaled = (filled - np.asarray(self.means)) / np.asarray(self.scales)
        scaled = np.clip(scaled, -self.numeric_clip, self.numeric_clip)
        indicator_positions = [self.numeric.index(name) for name in self.indicator_columns]
        indicators = is_missing[:, indicator_positions].astype(np.float64)
        numeric = np.concatenate([scaled, indicators], axis=1).astype(np.float32)
        if not np.isfinite(numeric).all():
            raise ValueError("Numeric preprocessing produced non-finite values")
        return codes, numeric

    def category_lookup(self, name: str) -> dict[str, int]:
        return {value: index + RESERVED_INDICES for index, value in enumerate(self.vocabularies[name])}

    def state(self) -> dict:
        if not self.fitted:
            raise RuntimeError("Fit the preprocessor before saving it")
        return {
            "allowlist": self.allowlist, "categorical": self.categorical,
            "min_category_count": self.min_category_count, "numeric_clip": self.numeric_clip,
            "vocabularies": self.vocabularies, "medians": self.medians,
            "indicator_columns": self.indicator_columns, "means": self.means, "scales": self.scales,
        }

    @classmethod
    def from_state(cls, state: dict) -> "TabularPreprocessor":
        item = cls(
            state["allowlist"], state["categorical"],
            min_category_count=state["min_category_count"], numeric_clip=state["numeric_clip"],
        )
        item.vocabularies = {name: list(values) for name, values in state["vocabularies"].items()}
        item.medians = list(state["medians"])
        item.indicator_columns = list(state["indicator_columns"])
        item.means = list(state["means"])
        item.scales = list(state["scales"])
        item.fitted = True
        return item
