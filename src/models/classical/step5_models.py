"""Train-only transforms and chronological early stopping for Step 5."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
)
from xgboost import XGBClassifier

from src.models.classical.step5_data import LABEL


def _category_strings(frame: pd.DataFrame) -> np.ndarray:
    return frame.astype("string").fillna("<missing>").to_numpy(dtype=str)


def make_preprocessor(family: str, allowlist: list[str], categorical: list[str]) -> ColumnTransformer:
    categories = [column for column in categorical if column in allowlist]
    if family == "xgboost":
        non_categories = [column for column in allowlist if column not in categories]
        return ColumnTransformer([
            ("categorical", Pipeline([
                ("strings", FunctionTransformer(_category_strings)),
                ("codes", OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan, dtype=np.float32,
                )),
            ]), categories),
            ("numeric", "passthrough", non_categories),
        ], sparse_threshold=0.0)

    hour = "sched_dep_hour"
    categories = [column for column in categories if column != hour]
    numeric = [column for column in allowlist if column not in categories and column != hour]
    return ColumnTransformer([
        ("categorical", Pipeline([
            ("strings", FunctionTransformer(_category_strings)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
        ]), categories),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]), numeric),
        ("hour_spline", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("spline", SplineTransformer(
                degree=3, knots=np.linspace(0, 24, 9).reshape(-1, 1),
                extrapolation="periodic", include_bias=False,
            )),
        ]), [hour]),
    ], sparse_threshold=1.0)


def sample_candidates(family: str, settings: dict, budget: int, seed: int) -> list[dict]:
    """Use the same deterministic search space draw for every feature set."""
    rng = np.random.default_rng(seed)
    if family == "logistic_regression":
        weights = settings["class_weights"]
        return [
            {"C": float(10 ** rng.uniform(*settings["c_log10_range"])),
             "class_weight": weights[int(rng.integers(len(weights)))]}
            for _ in range(budget)
        ]
    return [
        {
            "max_depth": int(rng.integers(*settings["max_depth"], endpoint=True)),
            "learning_rate": float(10 ** rng.uniform(*settings["learning_rate_log10_range"])),
            "subsample": float(rng.uniform(*settings["subsample"])),
            "colsample_bytree": float(rng.uniform(*settings["colsample_bytree"])),
            "min_child_weight": float(10 ** rng.uniform(*settings["min_child_weight_log10_range"])),
            "reg_lambda": float(10 ** rng.uniform(*settings["reg_lambda_log10_range"])),
        }
        for _ in range(budget)
    ]


@dataclass
class PreparedFold:
    preprocessor: ColumnTransformer
    x_train: object
    y_train: np.ndarray
    x_inner_train: object
    y_inner_train: np.ndarray
    x_holdout: object
    y_holdout: np.ndarray
    x_evaluation: object | None
    feature_types: list[str]


def prepare_fold(
    family: str,
    allowlist: list[str],
    categorical: list[str],
    train: pd.DataFrame,
    inner_train: pd.DataFrame,
    holdout: pd.DataFrame,
    evaluation: pd.DataFrame | None = None,
) -> PreparedFold:
    for part, name in ((train, "training"), (inner_train, "inner fit"), (holdout, "early-stopping holdout")):
        if part[LABEL].nunique() != 2:
            raise ValueError(f"Step 5 requires both classes in the {name} window")
    preprocessor = make_preprocessor(family, allowlist, categorical)
    inner_preprocessor = make_preprocessor(family, allowlist, categorical)
    category_count = sum(column in allowlist for column in categorical)
    return PreparedFold(
        preprocessor=preprocessor,
        x_train=preprocessor.fit_transform(train[allowlist]),
        y_train=train[LABEL].to_numpy(),
        x_inner_train=inner_preprocessor.fit_transform(inner_train[allowlist]),
        y_inner_train=inner_train[LABEL].to_numpy(),
        x_holdout=inner_preprocessor.transform(holdout[allowlist]),
        y_holdout=holdout[LABEL].to_numpy(),
        x_evaluation=preprocessor.transform(evaluation[allowlist]) if evaluation is not None else None,
        feature_types=["c"] * category_count + ["q"] * (len(allowlist) - category_count),
    )


def _fit_saga(
    x, y: np.ndarray, settings: dict, candidate: dict, seed: int, limit: int, *, holdout=None
) -> tuple[LogisticRegression, int, bool]:
    model = LogisticRegression(
        solver=settings["solver"], tol=settings["tol"], random_state=seed,
        warm_start=True, **candidate,
    )
    best_loss = np.inf
    best_round, rounds, stale_checks = 0, 0, 0
    converged = False
    block = settings["iterations_per_check"]
    while rounds < limit:
        cap = min(block, limit - rounds)
        model.set_params(max_iter=cap)
        # Hitting a checkpoint is expected. Record convergence explicitly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x, y)
        completed = int(model.n_iter_[0])
        rounds += completed
        converged = completed < cap
        if holdout is not None:
            x_holdout, y_holdout = holdout
            loss = log_loss(y_holdout, model.predict_proba(x_holdout), labels=[0, 1])
            if loss < best_loss - 1e-6:
                best_loss, best_round, stale_checks = loss, rounds, 0
            else:
                stale_checks += 1
        if converged or stale_checks >= settings["patience_checks"]:
            break
    return model, best_round if holdout is not None else rounds, converged


def fit_candidate(
    family: str, data: PreparedFold, settings: dict, candidate: dict, seed: int
) -> tuple[Pipeline, dict]:
    """Select training length inside the fold, then refit on eligible outer training."""
    if family == "logistic_regression":
        _, best_round, _ = _fit_saga(
            data.x_inner_train, data.y_inner_train, settings, candidate, seed,
            settings["max_iter"], holdout=(data.x_holdout, data.y_holdout),
        )
        model, completed, converged = _fit_saga(
            data.x_train, data.y_train, settings, candidate, seed, best_round,
        )
        diagnostics = {"selected_iterations": best_round, "refit_iterations": completed, "refit_converged": converged}
    else:
        shared = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "enable_categorical": True,
            "feature_types": data.feature_types,
            "n_jobs": settings["n_jobs"],
            "random_state": seed,
            **candidate,
        }
        inner_model = XGBClassifier(
            n_estimators=settings["max_estimators"],
            early_stopping_rounds=settings["early_stopping_rounds"],
            **shared,
        )
        inner_model.fit(
            data.x_inner_train, data.y_inner_train,
            eval_set=[(data.x_holdout, data.y_holdout)], verbose=False,
        )
        best_round = int(inner_model.best_iteration) + 1
        model = XGBClassifier(n_estimators=best_round, **shared).fit(data.x_train, data.y_train)
        diagnostics = {"selected_estimators": best_round}
    return Pipeline([("preprocess", data.preprocessor), ("model", model)]), diagnostics
