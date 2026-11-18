"""Step 7 sequence gate, leakage guards and the GRU sequence model."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.classical.step5_data import read_v2_features
from src.models.neural.fit import NeuralScorer, build_network, predict_proba
from src.models.neural.preprocess import TabularPreprocessor
from src.models.neural.sequences import SequenceSource, gate_decision, load_sequences
from src.models.neural.train_step6 import run_step6, sample_candidates
from src.models.neural.train_step7 import run_step7
from tests.neural_helpers import quick_settings, rewrite_sequence_record, write_step4_fixture

PASSING_MLP = {"passed": True, "detail": "fixture"}


def _load(tmp_path, departures=False):
    dataset, protocol, manifest = write_step4_fixture(tmp_path, sequences=True, departures=departures)
    features, _, _, _ = read_v2_features(dataset, protocol)
    raw, checks = load_sequences(dataset, manifest, features.frame)
    return dataset, protocol, manifest, features, raw, checks


@pytest.mark.parametrize("departures", [False, True])
def test_sequences_load_right_aligned_and_pass_every_check(tmp_path, departures):
    _, _, _, features, raw, checks = _load(tmp_path, departures)
    gate = gate_decision(checks, mlp_check=PASSING_MLP, leakage_tests_passed=True, departures_enabled=departures)
    assert gate["decision"] == "go", gate
    frame = features.frame.reset_index(drop=True)
    with_weather = frame["flight_row_id"].str.endswith(("-0", "-1", "-2")).to_numpy()
    assert (raw.wx_len[with_weather] == 3).all() and (raw.wx_len[~with_weather] == 0).all()
    # Oldest first in the leading slots: 150, 90 then 30 minutes before the cutoff.
    np.testing.assert_allclose(raw.wx_hours[with_weather][:, :3], [[2.5, 1.5, 0.5]] * with_weather.sum())
    assert np.isnan(raw.wx_hours[:, 3:]).all()
    assert raw.departures_enabled == departures
    if departures:
        assert (raw.dep_len == 2).all()
        assert sorted(raw.carriers) == ["AA", "UA"]


def test_an_event_after_the_cutoff_means_no_go(tmp_path):
    dataset, protocol, manifest = write_step4_fixture(tmp_path, sequences=True)

    def leak(record):
        record["weather_sequence"]["rows"][-1]["event_utc"] = "2013-02-02T23:59:00Z"

    rewrite_sequence_record(dataset, 60, "flight-02-02-0", leak)
    features, _, _, _ = read_v2_features(dataset, protocol)
    _, checks = load_sequences(dataset, manifest, features.frame)
    gate = gate_decision(checks, mlp_check=PASSING_MLP, leakage_tests_passed=True, departures_enabled=False)
    assert checks.cutoff_violations == 1
    assert gate["decision"] == "no-go"
    assert not gate["conditions"]["timestamps_available_by_cutoff"]["passed"]


def test_a_mask_that_disagrees_with_its_records_means_no_go(tmp_path):
    dataset, protocol, manifest = write_step4_fixture(tmp_path, sequences=True)

    def corrupt(record):
        record["weather_sequence"]["mask"][0] = True

    rewrite_sequence_record(dataset, 120, "flight-03-18-2", corrupt)
    features, _, _, _ = read_v2_features(dataset, protocol)
    _, checks = load_sequences(dataset, manifest, features.frame)
    assert checks.structure_errors == 1
    gate = gate_decision(checks, mlp_check=PASSING_MLP, leakage_tests_passed=True, departures_enabled=False)
    assert gate["decision"] == "no-go"


def test_missing_mlp_or_failed_leakage_tests_block_the_gate(tmp_path):
    _, _, _, _, raw, checks = _load(tmp_path)
    assert gate_decision(checks, mlp_check={"passed": False}, leakage_tests_passed=True,
                         departures_enabled=False)["decision"] == "no-go"
    assert gate_decision(checks, mlp_check=PASSING_MLP, leakage_tests_passed=None,
                         departures_enabled=False)["decision"] == "no-go"


def test_padded_slots_never_change_a_prediction(tmp_path):
    _, _, _, features, raw, _ = _load(tmp_path, departures=True)
    settings = json.loads(quick_settings(tmp_path).read_text())
    frame = features.frame
    tabular = TabularPreprocessor(features.allowlist, features.categorical, min_category_count=1).fit(frame)
    source = SequenceSource(raw, settings)
    rows = np.arange(len(frame))
    sequences = source.fit(rows, tabular)
    candidate = sample_candidates(settings, 1, 42, sequence=True)[0]
    torch.manual_seed(0)
    model, _ = build_network(tabular, candidate, settings, 0.3, {**source.dims(sequences), "gru_hidden": 4})
    codes, numeric = tabular.transform(frame)
    arrays = {"cat": codes, "num": numeric, **source.arrays(sequences, rows)}
    baseline = predict_proba(model, arrays, 64)

    padded = {key: value.copy() for key, value in arrays.items()}
    padded["wx"][:, 3:, :] = 99.0
    padded["dep"][:, 2:, :] = -99.0
    padded["dep_carrier"][:, 2:] = 1
    np.testing.assert_allclose(predict_proba(model, padded, 64), baseline, atol=1e-6, rtol=0)

    changed = {key: value.copy() for key, value in arrays.items()}
    changed["wx"][:, 0, 0] += 5.0
    assert not np.allclose(predict_proba(model, changed, 64)[raw.wx_len > 0], baseline[raw.wx_len > 0])


def test_step7_writes_a_no_go_report_and_trains_nothing_without_an_mlp(tmp_path):
    dataset, _, _ = write_step4_fixture(tmp_path, sequences=True)
    output = tmp_path / "step7"
    report = run_step7(dataset, output, quick_settings(tmp_path), step6_dir=tmp_path / "missing",
                       trial_limit=1, leakage_tests_passed=True)
    assert report["gate"]["decision"] == "no-go"
    assert report["feature_sets"] == {}
    assert "Decision: no-go" in (output / "go_no_go.md").read_text()
    assert not list(output.glob("*.pt"))


def test_step7_trains_the_sequence_model_after_a_go(tmp_path):
    dataset, protocol, _ = write_step4_fixture(tmp_path, sequences=True, departures=True)
    settings = quick_settings(tmp_path)
    run_step6(dataset, tmp_path / "step6", settings, trial_limit=1)
    output = tmp_path / "step7"
    report = run_step7(dataset, output, settings, step6_dir=tmp_path / "step6", trial_limit=2,
                       leakage_tests_passed=True)
    assert report["gate"]["decision"] == "go"
    assert report["gate"]["inputs"] == "weather and same-day departures"
    model = report["feature_sets"]["multi_lead"]["models"]["sequence_model"]
    pred = pd.read_csv(output / model["validation_predictions"])
    assert list(pred.columns) == protocol["schemas"]["predictions"]["required_fields"]
    assert set(pred["model_name"]) == {"sequence_model"} and len(pred) == 288
    assert pred["y_probability"].between(0, 1).all()
    assert all("gru_hidden" in trial["params"] for trial in model["trials"])
    scorer = NeuralScorer(output / model["model_file"])
    assert scorer.sequence_state is not None
    with pytest.raises(ValueError, match="needs sequence inputs"):
        scorer.predict_proba(pd.read_csv(dataset / "features_60m.csv")[scorer.allowlist].iloc[:2])
