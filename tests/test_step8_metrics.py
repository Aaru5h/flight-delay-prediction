"""Independent arithmetic and temporal/identity contracts for Steps 8-9."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.artifacts_v2 import validate_predictions
from src.evaluation.calibrate_step8 import fit_calibrator
from src.evaluation.calibration import calibration_improves
from src.evaluation.metrics import compute_ece
from src.evaluation.threshold import select_operating_thresholds
from src.evaluation.evaluate_step9 import matched_rows
from src.evaluation.metrics_v2 import (
    _ranking_plan, _weighted_ranking, alert_budget, metrics, paired_day_bootstrap, reliability_bins,
)


def test_alert_budget_counts_exactly_and_breaks_ties_by_id():
    result, selected = alert_budget([0, 1, 1, 0], [0.7] * 4, ["d", "b", "a", "c"], 0.5)
    assert result["alerts"] == 2 and result["precision"] == 1 and result["recall"] == 1
    assert selected.tolist() == [False, True, True, False]
    result, _ = alert_budget([0, 1], [0.3, 0.7], ["a", "b"], 0.1)
    assert result["alerts"] == 0 and result["precision"] is None


def test_brier_skill_uses_training_prevalence_and_equal_mass_bins():
    result = metrics([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9], list("abcd"), 0.2, n_bins=3)
    expected = np.mean((np.array([0, 1, 0, 1]) - 0.2) ** 2)
    assert result["reference_brier_score"] == pytest.approx(expected)
    assert result["brier_skill_score"] == pytest.approx(1 - 0.025 / expected)
    assert [b["count"] for b in result["reliability_bins"]] == [2, 1, 1]
    assert sum(b["count"] for b in reliability_bins([0, 1], [0, 1], 10)) == 2


@pytest.mark.parametrize("p", [[np.nan, 0.1], [-0.1, 1], [0.1, 1.1]])
def test_invalid_probabilities_are_rejected(p):
    with pytest.raises(ValueError):
        metrics([0, 1], p, ["a", "b"], 0.2)


def test_weighted_tie_metrics_match_explicit_day_duplication():
    y = np.array([0, 1, 1, 0, 0, 1])
    p = np.array([0.1, 0.7, 0.7, 0.3, 0.3, 0.2])
    weights = np.array([3, 0, 2, 1, 0, 2])
    ap, auc = _weighted_ranking(_ranking_plan(y, p), weights.astype(float))
    copies = np.repeat(np.arange(len(y)), weights)
    assert ap == pytest.approx(average_precision_score(y[copies], p[copies]))
    assert auc == pytest.approx(roc_auc_score(y[copies], p[copies]))


def test_bootstrap_is_paired_by_day_and_deterministic():
    y, p = [0, 1, 0, 1], np.array([0.2, 0.8, 0.3, 0.7])
    kwargs = dict(y=y, probabilities={"multi_lead/xgboost": p, "mlp": p.copy()},
                  dates=["2013-11-01"] * 2 + ["2013-11-02"] * 2,
                  reference_prevalences={"multi_lead/xgboost": 0.3, "mlp": 0.3}, resamples=25)
    result = paired_day_bootstrap(**kwargs)
    assert result == paired_day_bootstrap(**kwargs)
    assert result["days"] == 2
    difference = result["average_precision_difference_vs_xgboost"]["mlp"]
    assert difference["lower"] == difference["upper"] == 0


def test_calibration_requires_available_population_and_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        fit_calibrator(pd.DataFrame({"y_true": [0] * 10, "y_probability": [0.1] * 10}), "isotonic", 5)


def test_predictions_join_by_identity_and_reject_label_corruption():
    expected = pd.DataFrame({"flight_row_id": ["a", "b"], "lead_time_minutes": [60, 60],
                             "fold_id": ["2013-10"] * 2, "y_true": [0, 1]})
    pred = expected.iloc[::-1].assign(split="validation", y_probability=[0.8, 0.2], model_name="mlp", run_id="run")
    valid = validate_predictions(pred, expected, "run", "mlp")
    assert valid.flight_row_id.tolist() == ["a", "b"]
    pred.loc[pred.flight_row_id.eq("a"), "y_true"] = 1
    with pytest.raises(ValueError, match="labels"):
        validate_predictions(pred, expected, "run", "mlp")


def test_retrospective_population_mismatch_fails():
    ref = pd.DataFrame({"flight_row_id": ["a"], "lead_time_minutes": [60], "fold_id": ["retrospective"],
                        "y_true": [1], "sched_date": ["2013-11-01"], "y_probability_raw": [0.3]})
    with pytest.raises(ValueError, match="disagree"):
        matched_rows(ref, ref.assign(flight_row_id="b"))


def test_calibration_improvement_boundary_and_equal_mass_legacy_api():
    assert not calibration_improves(0.15, 0.149)
    assert calibration_improves(0.15, 0.1489)
    assert not calibration_improves(0.15, 0.16)
    with pytest.raises(ValueError):
        calibration_improves(float("nan"), 0.1)
    y, p = np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4])
    expected = metrics(y, p, list("abcd"), 0.2, n_bins=2)["equal_mass_ece"]
    assert compute_ece(y, p, n_bins=2, strategy="quantile") == pytest.approx(expected)


def test_precision_floor_counts_tied_alerts_directly():
    y, p = np.array([1, 0, 1, 0, 1]), np.array([0.9, 0.9, 0.5, 0.5, 0.1])
    decision = select_operating_thresholds(y, p, precision_floor=0.5, min_recall=0, min_alerts=2)
    selected = decision["precision_requirement"]
    assert selected["minimum_alerts"] == 2
    assert int((p >= selected["threshold"]).sum()) >= 2
