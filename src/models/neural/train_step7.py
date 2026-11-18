"""Step 7: sequence-model go/no-go gate, then the GRU sequence model.

The gate is written to ``go_no_go.json`` and ``go_no_go.md`` whatever the
outcome. Training starts only after a "go". The sequence model is the Step 6
embedding MLP plus GRU encoders over the latest origin weather readings and,
when same-day departures are approved, the latest origin departures. It uses
the same folds, budget, selection score and prediction schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from src.data import config
from src.data.loading import file_fingerprint
from src.models.classical.step5_data import make_folds, read_v2_features
from src.models.neural.common import code_commit, write_fold_manifest, write_json
from src.models.neural.fit import NeuralScorer
from src.models.neural.sequences import SequenceSource, gate_decision, load_sequences
from src.models.neural.train_step6 import (
    NEURAL_CODE, load_protocol_and_settings, provenance, train_neural_benchmark,
)

LOGGER = logging.getLogger(__name__)
LEAKAGE_TESTS = ("tests/test_step3_asof.py", "tests/test_step7_sequence.py")
SEQUENCE_CODE = (*NEURAL_CODE, Path(__file__), Path(__file__).with_name("sequences.py"))


def check_step6(step6_dir: Path | None) -> dict:
    """The gate needs a reproducible MLP: a published run whose checkpoint reloads."""
    if step6_dir is None:
        return {"passed": False, "detail": "no Step 6 output directory supplied"}
    metadata_path = Path(step6_dir) / "run_metadata.json"
    if not metadata_path.is_file():
        return {"passed": False, "detail": f"missing {metadata_path}"}
    report = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        model = report["feature_sets"]["multi_lead"]["models"]["embedding_mlp"]
    except KeyError:
        return {"passed": False, "detail": "Step 6 run has no multi-lead embedding MLP"}
    checkpoint = Path(step6_dir) / model["model_file"]
    predictions = Path(step6_dir) / model["validation_predictions"]
    if not checkpoint.is_file() or not predictions.is_file():
        return {"passed": False, "detail": "Step 6 checkpoint or validation predictions are missing"}
    NeuralScorer(checkpoint)
    difference = model.get("reload_check_max_abs_diff")
    passed = difference is not None and difference <= 1e-6
    return {
        "passed": passed,
        "detail": (
            f"run {report['feature_sets']['multi_lead']['run_id']}, reload difference {difference}, "
            f"full trial budget: {report['protocol_budget_completed']}"
        ),
        "step6_run_id": report["feature_sets"]["multi_lead"]["run_id"],
        "step6_full_budget": report["protocol_budget_completed"],
    }


def run_leakage_tests() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *LEAKAGE_TESTS], cwd=config.PROJECT_ROOT,
        capture_output=True, text=True,
    )
    LOGGER.info(result.stdout[-2000:])
    return result.returncode == 0


def _markdown(gate: dict) -> str:
    lines = [
        "# Step 7 sequence-model gate", "",
        f"**Decision: {gate['decision']}.** Inputs: {gate['inputs']}.", "",
        "| Condition | Passed | Detail |", "|---|---|---|",
    ]
    for name, item in gate["conditions"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'yes' if item['passed'] else 'no'} | {item['detail']} |")
    lines += ["", f"Records checked: {gate['records_checked']:,}; development rows: {gate['development_records']:,}.", ""]
    coverage = ", ".join(f"{lead} min {share:.1%}" for lead, share in gate["weather_coverage_by_lead"].items())
    lines.append(f"Rows with at least one weather reading: {coverage or 'none'}.")
    if gate["departure_coverage_by_lead"]:
        coverage = ", ".join(f"{lead} min {share:.1%}" for lead, share in gate["departure_coverage_by_lead"].items())
        lines.append(f"Rows with at least one earlier departure: {coverage}.")
    if gate["examples"]:
        lines += ["", "First problems found:", *(f"- {example}" for example in gate["examples"])]
    return "\n".join(lines) + "\n"


def run_step7(
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    settings_path: Path | None = None,
    *,
    step6_dir: Path | None = None,
    trial_limit: int | None = None,
    device: str = "cpu",
    leakage_tests_passed: bool | None = None,
) -> dict:
    """Run the gate. On "go", train and publish the sequence model."""
    dataset = Path(dataset_dir or config.PROCESSED_DATA_DIR / "step4")
    output = Path(output_dir or config.PROJECT_ROOT / "artifacts" / "step7")
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose a new output directory.")
    protocol, settings, budget, full_budget = load_protocol_and_settings(settings_path, trial_limit)
    seed = protocol["model_selection"]["seed"]
    features, metadata, source_fingerprints, _ = read_v2_features(dataset, protocol)
    manifest = json.loads((dataset / "feature_manifest.json").read_text(encoding="utf-8"))
    raw, checks = load_sequences(dataset, manifest, features.frame)
    if leakage_tests_passed is None:
        leakage_tests_passed = run_leakage_tests()
    gate = gate_decision(
        checks, mlp_check=check_step6(step6_dir), leakage_tests_passed=leakage_tests_passed,
        departures_enabled=raw.departures_enabled,
    )
    LOGGER.info("Sequence gate: %s", gate["decision"])

    base = provenance(protocol, settings, source_fingerprints, budget, full_budget, SEQUENCE_CODE, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".step7-", dir=output.parent) as work:
        stage = Path(work) / "sequence"
        stage.mkdir()
        write_json(stage / "go_no_go.json", gate)
        (stage / "go_no_go.md").write_text(_markdown(gate), encoding="utf-8")
        report = {**base, "code_commit": code_commit(), "gate": gate, "feature_sets": {}}
        if gate["decision"] == "go":
            folds, final = make_folds(metadata, protocol)
            fold_file = stage / "fold_manifest.csv"
            write_fold_manifest(fold_file, metadata, folds, final)
            report["fold_manifest"] = fold_file.name
            report["fold_manifest_sha256"] = file_fingerprint(fold_file)
            item = {
                "feature_set": features.name,
                "feature_manifest_version": features.manifest_version,
                "feature_file_sha256": features.file_fingerprints,
                "sequence_file_sha256": {
                    partition["sequences"]: file_fingerprint(dataset / partition["sequences"])
                    for partition in manifest["partitions"]
                },
                "feature_allowlist": features.allowlist,
                "sequence_inputs": gate["inputs"],
            }
            run_hash = hashlib.sha256(json.dumps({**base, **item}, sort_keys=True).encode()).hexdigest()
            run_id = f"step7-{features.name}-{run_hash[:12]}"
            report["feature_sets"][features.name] = {
                **item,
                "run_id": run_id,
                "models": {
                    "sequence_model": train_neural_benchmark(
                        stage, features, metadata, folds, final, "sequence_model", settings, budget, seed,
                        run_id, device=device, sequences=SequenceSource(raw, settings),
                    )
                },
            }
        write_json(stage / "run_metadata.json", report)
        stage.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7 sequence-model gate and training")
    parser.add_argument("--dataset-dir", type=Path, help="Step 4 export built with --include-sequences")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--settings-path", type=Path)
    parser.add_argument("--step6-dir", type=Path, default=config.PROJECT_ROOT / "artifacts" / "step6")
    parser.add_argument("--trial-limit", type=int, help="Reduced development run for smoke testing")
    parser.add_argument("--device", default="cpu", help="cpu (default, reproducible), mps or cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_step7(**vars(args))
    output = args.output_dir or config.PROJECT_ROOT / "artifacts" / "step7"
    print(f"Gate decision: {report['gate']['decision']}. Wrote {output}")


if __name__ == "__main__":
    main()
