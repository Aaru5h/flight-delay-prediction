"""Reloadable model pipelines whose preprocessing is fitted on training only."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def _preprocessor(categorical: list[str], numeric: list[str], scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
    return ColumnTransformer(
        [
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]), categorical),
            ("numeric", Pipeline(numeric_steps), numeric),
        ],
        sparse_threshold=1.0,
    )


def build_logistic_pipeline(
    categorical: list[str], numeric: list[str], settings: dict
) -> Pipeline:
    return Pipeline([
        ("preprocess", _preprocessor(categorical, numeric, scale=True)),
        ("model", LogisticRegression(**settings)),
    ])


def build_xgboost_pipeline(
    categorical: list[str], numeric: list[str], settings: dict, seed: int
) -> Pipeline:
    model_settings = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cpu",
        "random_state": seed,
        "n_jobs": -1,
        **settings,
    }
    return Pipeline([
        ("preprocess", _preprocessor(categorical, numeric, scale=False)),
        ("model", XGBClassifier(**model_settings)),
    ])
