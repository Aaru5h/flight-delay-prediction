"""Exercise v2 folds, real model fitting, and the evaluation handoff."""

from datetime import timedelta
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.classical.step5_data import AVAILABLE, inner_split, make_folds, read_v2_features
from src.models.classical.step5_models import _fit_saga, sample_candidates
from src.models.classical.train_step5 import CONTRACT_PATH, SETTINGS_PATH, run_step5


@pytest.fixture
def step4_files(tmp_path):
    protocol = yaml.safe_load(CONTRACT_PATH.read_text())
    leads = protocol["prediction_time"]["lead_time_minutes"]
    partitions = []
    csv_dir = tmp_path / "step4"
    csv_dir.mkdir()
    names = ["carrier", "sched_dep_hour", "distance", "lead_time_minutes", "temp"]
    manifest_columns = [
        {"name": name, "type": "categorical" if name in {"carrier", "lead_time_minutes"} else "numeric",
         "availability_rule": "origin weather valid by cutoff" if name == "temp" else "published schedule and fixed calendar"}
        for name in names
    ]

    for lead in leads:
        rows = []
        for month in range(1, 13):
            for day in (2, 18, 27):
                for i in range(4):
                    local_date = pd.Timestamp(year=2013, month=month, day=day)
                    scheduled = local_date.tz_localize("UTC") + timedelta(hours=14 + i)
                    available = scheduled + timedelta(hours=5)
                    if (month, day, i) == (4, 27, 0):
                        available = pd.Timestamp("2013-05-04T00:00:00Z")
                    rows.append({
                        "flight_row_id": f"flight-{month:02d}-{day:02d}-{i}",
                        "lead_time_minutes": lead,
                        "sched_date": local_date.date().isoformat(),
                        "prediction_cutoff_utc": scheduled - timedelta(minutes=lead),
                        "outcome_available_utc": available if i != 1 else pd.NaT,
                        "substantial_delay": i % 2 if month < 11 else "sealed",
                        "carrier": "AA" if i < 3 else "UA",
                        "sched_dep_hour": 9 + i,
                        "distance": i * 100,
                        "temp": np.nan if i == 3 else float(i % 2) + 0.01 * lead,
                    })
        path = csv_dir / f"features_{lead}m.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        partitions.append({"lead_time_minutes": lead, "features": path.name, "rows": len(rows)})

    manifest = {
        "protocol_version": protocol["contract_version"],
        "manifest_version": "asof-fixture",
        "label_column": "substantial_delay",
        "allowlist": names,
        "features": manifest_columns,
        "partitions": partitions,
        "source_fingerprints": {"flights": "80ac"},
    }
    (csv_dir / "feature_manifest.json").write_text(json.dumps(manifest))
    return csv_dir, protocol, manifest


def test_walk_forward_uses_earliest_cutoff_and_availability_for_each_fit(step4_files):
    dataset, protocol, _ = step4_files
    features, metadata, _, _ = read_v2_features(dataset, protocol)
    folds, retrospective = make_folds(metadata, protocol)
    assert [fold.fold_id for fold in folds] == protocol["validation"]["validation_months"]
    assert "flight-04-27-0" not in folds[0].train_ids
    assert "flight-04-27-0" in folds[1].train_ids
    assert "flight-04-27-1" not in retrospective.train_ids
    assert features.frame["sched_date"].dt.month.max() == 10
    assert not features.frame["flight_row_id"].isin(retrospective.evaluation_ids).any()
    for fold in [*folds, retrospective]:
        assert metadata.loc[fold.train_ids, AVAILABLE].le(fold.fit_cutoff_utc).all()
        assert fold.fit_cutoff_utc == metadata.loc[fold.evaluation_ids, "first_cutoff_utc"].min()
        inner, holdout = inner_split(metadata, fold, 14)
        assert metadata.loc[inner, AVAILABLE].le(metadata.loc[holdout, "first_cutoff_utc"].min()).all()
        assert metadata.loc[inner, "sched_date"].max() < metadata.loc[holdout, "sched_date"].min()


def test_step5_fits_and_exports_both_feature_sets_without_scoring_sealed_labels(tmp_path, step4_files):
    dataset, protocol, manifest = step4_files
    legacy = tmp_path / "legacy_features.csv"
    frame_60 = pd.read_csv(dataset / "features_60m.csv")
    frame_60.drop(columns="lead_time_minutes").to_csv(legacy, index=False)
    legacy_manifest = tmp_path / "legacy_manifest.json"
    legacy_manifest.write_text(json.dumps({
        "manifest_version": "legacy-fixture",
        "allowlist": [name for name in manifest["allowlist"] if name != "lead_time_minutes"],
        "features": [feature for feature in manifest["features"] if feature["name"] != "lead_time_minutes"],
    }))
    quick_settings = json.loads(SETTINGS_PATH.read_text())
    quick_settings["logistic_regression"].update(max_iter=60, iterations_per_check=10)
    quick_settings["xgboost"].update(max_estimators=12, early_stopping_rounds=3, n_jobs=1)
    config_path = tmp_path / "quick.json"
    config_path.write_text(json.dumps(quick_settings))
    output = tmp_path / "artifacts"
    report = run_step5(
        dataset, output, config_path, legacy_features=legacy, legacy_manifest=legacy_manifest, trial_limit=2,
    )
    assert report == json.loads((output / "run_metadata.json").read_text())
    assert not report["protocol_budget_completed"]
    assert report["full_protocol_budget"] == protocol["model_selection"]["trials_per_model_per_feature_set"]
    fold_manifest = pd.read_csv(output / report["fold_manifest"])
    assert not fold_manifest.duplicated(["flight_row_id", "fold_id"]).any()
    assert "retrospective" in set(fold_manifest["split"])
    assert "y_true" not in fold_manifest

    predictions = {}
    for feature_set, artifacts in report["feature_sets"].items():
        for name, model in artifacts["models"].items():
            pred = pd.read_csv(output / model["validation_predictions"])
            predictions[feature_set, name] = pred
            assert set(pred["split"]) == {"validation"}
            assert set(pred["run_id"]) == {artifacts["run_id"]}
            assert pred["y_probability"].between(0, 1).all()
            assert set(pred["fold_id"]) == set(protocol["validation"]["validation_months"])
            assert not pred.duplicated(["flight_row_id", "lead_time_minutes", "fold_id"]).any()
            assert len(model["trials"]) == 2
            pipeline = joblib.load(output / model["model_file"])
            sample = frame_60[model["feature_allowlist"]].iloc[:3].copy()
            sample["carrier"] = "unseen"
            assert np.isfinite(pipeline.predict_proba(sample)[:, 1]).all()
            if name == "xgboost":
                assert 1 <= model["final_fit_diagnostics"]["selected_estimators"] <= 12
            else:
                spline = pipeline.named_steps["preprocess"].named_transformers_["hour_spline"]
                assert spline.named_steps["spline"].extrapolation == "periodic"
    full = predictions["multi_lead", "xgboost"]
    assert set(full["lead_time_minutes"]) == {60, 120, 360, 1440}
    assert len(full) == 288
    legacy_predictions = predictions["legacy_aml", "xgboost"]
    keys = ["flight_row_id", "lead_time_minutes", "fold_id", "y_true"]
    pd.testing.assert_frame_equal(
        full.loc[full["lead_time_minutes"].eq(60), keys].reset_index(drop=True),
        legacy_predictions[keys].reset_index(drop=True),
    )
    assert "temp" not in report["feature_sets"]["schedule_only"]["feature_allowlist"]


def test_real_saga_settings_converge_and_sampling_is_reproducible():
    settings = json.loads(SETTINGS_PATH.read_text())["logistic_regression"]
    trials = sample_candidates("logistic_regression", settings, 40, 42)
    assert trials == sample_candidates("logistic_regression", settings, 40, 42)
    assert len({trial["C"] for trial in trials}) == 40
    x = np.tile([[-1.0], [1.0]], (40, 1))
    y = np.tile([0, 1], 40)
    _, iterations, converged = _fit_saga(x, y, settings, trials[0], 42, settings["max_iter"])
    assert converged and iterations < settings["max_iter"]


def test_declared_features_cannot_contain_labels(step4_files):
    root, protocol, manifest = step4_files
    manifest["allowlist"].append("substantial_delay")
    manifest["features"].append({"name": "substantial_delay", "type": "numeric"})
    (root / "feature_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leakage-prone"):
        read_v2_features(root, protocol)
