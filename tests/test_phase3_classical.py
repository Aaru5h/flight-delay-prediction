"""Phase 3 pipeline, output, reproducibility and leakage-boundary tests."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.classical.pipelines import build_logistic_pipeline, build_xgboost_pipeline
from src.models.classical.train import (
    _prediction_frame,
    assign_split,
    tune_xgboost,
    validation_metrics,
)


@pytest.fixture
def tiny_data():
    X = pd.DataFrame({
        "carrier": ["AA", "UA", "AA", "UA", "AA", "UA", "AA", "UA"],
        "origin": ["EWR", "JFK"] * 4,
        "distance": [100.0, 200.0, np.nan, 180.0, 120.0, 220.0, 140.0, 240.0],
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    return X, y


def test_chronological_split_boundaries():
    dates = pd.Series(["2013-08-31", "2013-09-01", "2013-10-31", "2013-11-01", "2013-12-31"])
    assert assign_split(dates).tolist() == ["train", "validation", "validation", "test", "test"]


@pytest.mark.parametrize("builder", ["logistic", "xgboost"])
def test_models_handle_missing_numeric_and_unseen_categories(tiny_data, builder):
    X, y = tiny_data
    if builder == "logistic":
        model = build_logistic_pipeline(
            ["carrier", "origin"], ["distance"],
            {"C": 1.0, "max_iter": 200, "solver": "liblinear", "random_state": 42},
        )
    else:
        model = build_xgboost_pipeline(
            ["carrier", "origin"], ["distance"],
            {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.2}, 42,
        )
    model.fit(X, y)
    future = pd.DataFrame({"carrier": ["NEW"], "origin": ["LGA"], "distance": [np.nan]})
    probability = model.predict_proba(future)[:, 1]
    assert probability.shape == (1,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_reloaded_pipeline_produces_identical_probabilities(tmp_path, tiny_data):
    X, y = tiny_data
    model = build_logistic_pipeline(
        ["carrier", "origin"], ["distance"],
        {"C": 1.0, "max_iter": 200, "solver": "liblinear", "random_state": 42},
    ).fit(X, y)
    expected = model.predict_proba(X)[:, 1]
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    actual = joblib.load(path).predict_proba(X)[:, 1]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_xgboost_tuning_interface_has_no_test_input(tiny_data):
    X, y = tiny_data
    model, settings, trials = tune_xgboost(
        X.iloc[:6], y.iloc[:6], X.iloc[6:], y.iloc[6:],
        ["carrier", "origin"], ["distance"],
        [{"n_estimators": 5, "max_depth": 2, "learning_rate": 0.2}], 42,
    )
    assert model is not None
    assert settings["n_estimators"] == 5
    assert len(trials) == 1


def test_prediction_contract_and_metric_probability_checks():
    ids = pd.Series(["a", "b"])
    y = pd.Series([0, 1])
    probability = np.array([0.2, 0.8])
    output = _prediction_frame(ids, y, probability, "example", "run-1")
    assert output.columns.tolist() == [
        "flight_row_id", "y_true", "y_probability", "split", "model_name", "run_id"
    ]
    assert output["flight_row_id"].is_unique
    assert set(validation_metrics(y, probability)) == {"pr_auc", "roc_auc", "brier_score"}
    with pytest.raises(ValueError, match="invalid probabilities"):
        validation_metrics(y, np.array([0.2, 1.2]))
