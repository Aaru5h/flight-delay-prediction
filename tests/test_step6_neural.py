"""Step 6 embedding MLP: preprocessing, training, reload and the fold contract."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.classical.step5_data import AVAILABLE, make_folds, read_v2_features
from src.models.neural.fit import (
    NeuralScorer, build_network, mean_log_loss, predict_proba, save_checkpoint, train_network,
)
from src.models.neural.preprocess import MISSING_INDEX, UNKNOWN_INDEX, TabularPreprocessor, category_strings
from src.models.neural.train_step6 import (
    SETTINGS_PATH, STEP5_SETTINGS_PATH, prepare_fold, run_step6, sample_candidates,
)
from tests.neural_helpers import quick_settings, write_step4_fixture


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "carrier": rng.choice(["AA", "UA", "DL"], n),
        "hour": rng.integers(5, 22, n),
        "distance": rng.normal(1000, 300, n),
        "temp": np.where(rng.random(n) < 0.1, np.nan, rng.normal(50, 10, n)),
    })
    logits = (frame["carrier"].eq("UA") * 1.5 + (frame["hour"] > 16) * 1.0 - 1.2).to_numpy()
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(np.float32)
    return frame, y


def _candidate(**overrides):
    candidate = sample_candidates(json.loads(SETTINGS_PATH.read_text()), 1, 42)[0]
    candidate.update(hidden_layers=[16], dropout=0.0, learning_rate=0.01, weight_decay=1e-5,
                     batch_size=64, positive_weighting=None, numeric_encoding="linear")
    candidate.update(overrides)
    return candidate


def _settings():
    settings = json.loads(SETTINGS_PATH.read_text())
    settings["training"].update(max_epochs=8, patience_epochs=2, warmup_epochs=1, eval_batch_size=128)
    return settings


def _encode(preprocessor, frame):
    codes, numeric = preprocessor.transform(frame)
    return {"cat": codes, "num": numeric}


def test_categories_and_numerics_are_learned_from_training_rows_only():
    train = pd.DataFrame({
        "carrier": ["AA"] * 5 + ["UA"] * 5 + ["ZZ"] + [None],
        "distance": [100.0] * 6 + [200.0] * 6,
        "temp": [np.nan] + [10.0] * 11,
    })
    preprocessor = TabularPreprocessor(["carrier", "distance", "temp"], ["carrier"], min_category_count=5).fit(train)
    assert preprocessor.vocabularies["carrier"] == ["AA", "UA"]
    assert preprocessor.indicator_columns == ["temp"]
    future = pd.DataFrame({"carrier": ["NEW", None, "ZZ", "UA"], "distance": [np.nan, 1e9, 150.0, 100.0],
                           "temp": [np.nan, 10.0, 10.0, 10.0]})
    codes, numeric = preprocessor.transform(future)
    assert codes[:, 0].tolist() == [UNKNOWN_INDEX, MISSING_INDEX, UNKNOWN_INDEX, 3]
    assert numeric.shape == (4, preprocessor.numeric_dim) == (4, 3)
    assert numeric[0, 0] == pytest.approx((150.0 - 150.0) / 50.0)
    assert numeric[1, 0] == preprocessor.numeric_clip
    assert numeric[:, 2].tolist() == [1.0, 0.0, 0.0, 0.0]
    restored = TabularPreprocessor.from_state(json.loads(json.dumps(preprocessor.state())))
    np.testing.assert_array_equal(restored.transform(future)[1], numeric)


def test_numbers_and_strings_share_category_codes():
    assert category_strings(pd.Series([7, 7.0, "7", None, "AA"], dtype=object)).tolist() == [
        "7", "7", "7", "<missing>", "AA"
    ]


def test_early_stopping_restores_the_best_holdout_epoch():
    frame, y = _toy()
    preprocessor = TabularPreprocessor(list(frame), ["carrier", "hour"], min_category_count=1).fit(frame[:300])
    train, holdout = _encode(preprocessor, frame[:300]), _encode(preprocessor, frame[300:])
    settings = _settings()
    candidate = _candidate(learning_rate=0.05)
    model, _ = build_network(preprocessor, candidate, settings, float(y[:300].mean()))
    result = train_network(model, train, y[:300], candidate, settings["training"], 42, holdout=(holdout, y[300:]))
    best = int(np.argmin(result.holdout_losses))
    assert result.best_epoch == best + 1
    assert mean_log_loss(result.model, holdout, y[300:], 128, "cpu") == pytest.approx(
        result.holdout_losses[best], abs=1e-6
    )


def test_training_is_reproducible_and_probabilities_are_valid():
    frame, y = _toy()
    preprocessor = TabularPreprocessor(list(frame), ["carrier", "hour"], min_category_count=1).fit(frame)
    arrays = _encode(preprocessor, frame)
    settings = _settings()
    outputs = []
    for encoding in ("periodic", "periodic", "linear"):
        candidate = _candidate(numeric_encoding=encoding, positive_weighting="balanced")
        model, _ = build_network(preprocessor, candidate, settings, float(y.mean()), seed=42)
        trained = train_network(model, arrays, y, candidate, settings["training"], 42, epochs=3).model
        outputs.append(predict_proba(trained, arrays, 128))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    for probabilities in outputs:
        assert np.isfinite(probabilities).all() and ((probabilities > 0) & (probabilities < 1)).all()


def test_checkpoint_reload_scores_raw_rows_identically(tmp_path):
    frame, y = _toy()
    preprocessor = TabularPreprocessor(list(frame), ["carrier", "hour"], min_category_count=1).fit(frame)
    settings = _settings()
    candidate = _candidate(numeric_encoding="periodic")
    model, kwargs = build_network(preprocessor, candidate, settings, float(y.mean()))
    model = train_network(model, _encode(preprocessor, frame), y, candidate, settings["training"], 42, epochs=2).model
    save_checkpoint(tmp_path / "mlp.pt", model, kwargs, preprocessor)
    scorer = NeuralScorer(tmp_path / "mlp.pt")
    unseen = frame.iloc[:5].assign(carrier=["ZZ", None, "AA", "UA", "DL"])
    expected = predict_proba(model, _encode(preprocessor, unseen), 128)
    np.testing.assert_allclose(scorer.predict_proba(unseen), expected, atol=1e-7, rtol=0)


def test_search_is_reproducible_and_matches_the_baseline_holdout():
    settings = json.loads(SETTINGS_PATH.read_text())
    first = sample_candidates(settings, 40, 42)
    assert first == sample_candidates(settings, 40, 42)
    assert len({candidate["learning_rate"] for candidate in first}) == 40
    assert settings["inner_holdout_days"] == json.loads(STEP5_SETTINGS_PATH.read_text())["inner_holdout_days"]


def test_fold_preparation_respects_dates_and_label_availability(tmp_path):
    dataset, protocol, _ = write_step4_fixture(tmp_path)
    features, metadata, _, _ = read_v2_features(dataset, protocol)
    folds, final = make_folds(metadata, protocol)
    settings = json.loads(quick_settings(tmp_path).read_text())
    for fold in folds:
        data = prepare_fold(features, metadata, fold, settings, validation=True)
        assert len(data.y_train) == features.frame["flight_row_id"].isin(fold.train_ids).sum()
        assert metadata.loc[fold.train_ids, AVAILABLE].le(fold.fit_cutoff_utc).all()
        evaluation_dates = metadata.loc[fold.evaluation_ids, "sched_date"]
        assert metadata.loc[fold.train_ids, "sched_date"].max() < evaluation_dates.min()
    data = prepare_fold(features, metadata, final, settings, validation=False)
    assert data.evaluation is None and data.evaluation_frame is None


def test_step6_exports_aligned_predictions_without_reading_sealed_labels(tmp_path):
    dataset, protocol, manifest = write_step4_fixture(tmp_path)
    legacy = tmp_path / "legacy_features.csv"
    frame_60 = pd.read_csv(dataset / "features_60m.csv")
    frame_60.drop(columns="lead_time_minutes").to_csv(legacy, index=False)
    legacy_manifest = tmp_path / "legacy_manifest.json"
    legacy_manifest.write_text(json.dumps({
        "manifest_version": "legacy-fixture",
        "allowlist": [name for name in manifest["allowlist"] if name != "lead_time_minutes"],
        "features": [item for item in manifest["features"] if item["name"] != "lead_time_minutes"],
    }))
    output = tmp_path / "artifacts"
    report = run_step6(dataset, output, quick_settings(tmp_path), legacy_features=legacy,
                       legacy_manifest=legacy_manifest, trial_limit=2)
    assert report == json.loads((output / "run_metadata.json").read_text())
    assert not report["protocol_budget_completed"]
    for field in ("code_commit", "raw_data_fingerprints", "fold_manifest_version", "model_config", "seed",
                  "package_versions"):
        assert field in report
    fold_manifest = pd.read_csv(output / report["fold_manifest"])
    assert "y_true" not in fold_manifest and "retrospective" in set(fold_manifest["split"])

    predictions = {}
    for name, artifacts in report["feature_sets"].items():
        model = artifacts["models"]["embedding_mlp"]
        pred = pd.read_csv(output / model["validation_predictions"])
        predictions[name] = pred
        assert list(pred.columns) == protocol["schemas"]["predictions"]["required_fields"]
        assert set(pred["split"]) == {"validation"} and set(pred["run_id"]) == {artifacts["run_id"]}
        assert set(pred["fold_id"]) == set(protocol["validation"]["validation_months"])
        assert pred["y_probability"].between(0, 1).all()
        assert not pred.duplicated(["flight_row_id", "lead_time_minutes", "fold_id"]).any()
        assert len(model["trials"]) == 2 and model["reload_check_max_abs_diff"] <= 1e-6
        history = model["final_fit_diagnostics"]
        assert len(history["inner_training_loss"]) == history["inner_epochs_run"]
        assert len(history["inner_holdout_loss"]) == history["inner_epochs_run"]
        assert len(history["refit_training_loss"]) == history["refit_epochs"]
        assert np.isfinite(history["inner_holdout_loss"]).all()
        scorer = NeuralScorer(output / model["model_file"])
        sample = frame_60.iloc[:3].copy()
        sample["carrier"] = "unseen"
        assert np.isfinite(scorer.predict_proba(sample[scorer.allowlist])).all()
    full = predictions["multi_lead"]
    assert set(full["lead_time_minutes"]) == {60, 120, 360, 1440} and len(full) == 288
    keys = ["flight_row_id", "lead_time_minutes", "fold_id", "y_true"]
    pd.testing.assert_frame_equal(
        full.loc[full["lead_time_minutes"].eq(60), keys].reset_index(drop=True),
        predictions["legacy_aml"][keys].reset_index(drop=True),
    )


def test_step6_refuses_to_overwrite_a_previous_run(tmp_path):
    dataset, _, _ = write_step4_fixture(tmp_path)
    (tmp_path / "existing").mkdir()
    with pytest.raises(FileExistsError):
        run_step6(dataset, tmp_path / "existing", quick_settings(tmp_path), trial_limit=1)


def test_neural_code_never_loads_xgboost_and_keeps_the_step5_helpers():
    """XGBoost next to PyTorch can segfault on Apple-silicon Macs."""
    import subprocess
    import sys

    code = (
        "import sys; import src.models.neural.train_step7, src.models.neural.train_step6; "
        "sys.exit(int('xgboost' in sys.modules))"
    )
    root = Path(__file__).resolve().parents[1]
    assert subprocess.run([sys.executable, "-c", code], cwd=root).returncode == 0

    # Compare the helpers in a classical-only child, keeping this process free
    # of XGBoost before the remaining neural fits execute.
    check = '''
import inspect
from src.models.classical import train_step5
from src.models.neural import common
pairs = [
    (train_step5._write_json, common.write_json),
    (train_step5._code_commit, common.code_commit),
    (train_step5._write_fold_manifest, common.write_fold_manifest),
    (train_step5._average_precision_by_lead, common.average_precision_by_lead),
]
for original, copy in pairs:
    renamed = inspect.getsource(original).replace(original.__name__, copy.__name__)
    assert renamed == inspect.getsource(copy), copy.__name__
assert train_step5.PREDICTION_COLUMNS == common.PREDICTION_COLUMNS
'''
    assert subprocess.run([sys.executable, "-c", check], cwd=root).returncode == 0
