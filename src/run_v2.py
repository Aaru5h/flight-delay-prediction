"""Reproducible, resumable execution of the v2 milestone through Step 9."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def run_pipeline(output_root=None, *, through=9, trial_limit=None, classical_settings=None,
                 neural_settings=None, evaluation_settings=None, device="cpu", run_checks=True):
    output = Path(output_root or ROOT / "artifacts/v2").resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data/processed/step4"
    state_path = output / "pipeline_state.json"
    state = read(state_path) if state_path.exists() else {"protocol": "2.0.0", "steps": {}}
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
           "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/private/tmp/flight-delay-matplotlib"),
           "PYTHONUNBUFFERED": "1"}

    def save():
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    def command(step, module, args=()):
        LOGGER.info("Starting %s: %s", step, module)
        state["steps"][str(step)] = {"status": "running", "module": module, "arguments": list(map(str, args))}
        save()
        with (output / f"step{step}.log").open("a") as log:
            result = subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            state["steps"][str(step)]["status"] = "failed"; save()
            raise RuntimeError(f"Step {step} failed; see {output / f'step{step}.log'}")
        state["steps"][str(step)]["status"] = "completed"; save()

    if run_checks:
        command("checks", "src.checks")
    if not (ROOT / "data/processed/labelled_flights.csv").exists():
        command(1, "src.data.prepare")
    state["steps"]["1"] = {"status": "implemented", "evidence": "Phase 1 labels validated against raw source by Step 4"}
    state["steps"]["2"] = {"status": "implemented", "evidence": "Schedule helpers, alert counts, equal-mass ECE and calibration tests"}
    state["steps"]["3"] = {"status": "implemented", "evidence": "AsOfFeatureBuilder and cutoff perturbation tests"}; save()
    if not (data / "feature_manifest.json").exists():
        command(4, "src.features.build_multi_lead", ["--include-sequences", "--output-dir", data])
    else:
        manifest = read(data / "feature_manifest.json")
        raw_manifest = read(ROOT / "configs/data_manifest.json")
        expected = {item["file"]: item["sha256"] for item in raw_manifest["files"]}
        if sorted(manifest["source_fingerprints"].values()) != sorted(expected.values()):
            raise ValueError("Cached Step 4 source fingerprints differ; rebuild in a fresh directory")
        if any(not part.get("sequences") or not (data / part["features"]).is_file() or not (data / part["sequences"]).is_file() for part in manifest["partitions"]):
            raise ValueError("Cached Step 4 export is incomplete or lacks sequence files")
        state["steps"]["4"] = {"status": "completed", "evidence": str(data / "data_quality.json")}; save()
    if through == 4:
        return state
    legacy = ROOT / "data/processed/phase2_features.csv"
    legacy_manifest = ROOT / "data/processed/phase2_feature_manifest.json"
    if not legacy.exists() or not legacy_manifest.exists():
        command("legacy", "src.features.pipeline")
    budget_args = ["--trial-limit", trial_limit] if trial_limit is not None else []
    for step, module, settings_path in [(5, "src.models.classical.train_step5", classical_settings),
                                        (6, "src.models.neural.train_step6", neural_settings),
                                        (7, "src.models.neural.train_step7", neural_settings)]:
        destination = output / f"step{step}"
        args = ["--dataset-dir", data, "--output-dir", destination, *budget_args]
        if step != 7:
            args += ["--legacy-features", legacy, "--legacy-manifest", legacy_manifest]
        if step != 5:
            args += ["--device", device]
        if step == 7:
            args += ["--step6-dir", output / "step6"]
        if settings_path:
            args += ["--settings-path", settings_path]
        if not destination.exists():
            command(step, module, args)
        else:
            report = read(destination / "run_metadata.json")
            budget = trial_limit if trial_limit is not None else 40
            if report["trials_per_model"] != budget or report["protocol_sha256"] != digest(ROOT / "configs/project_contract_v2.yaml"):
                raise ValueError(f"Cached Step {step} run has a different budget or protocol")
            wanted = read(settings_path or ROOT / f"configs/{'step5_classical' if step == 5 else 'step6_neural'}.json")
            if report["model_config"] != wanted:
                raise ValueError(f"Cached Step {step} settings differ")
            for filename, expected_hash in report.get("code_sha256", {}).items():
                candidates = list(ROOT.joinpath("src").rglob(filename))
                if len(candidates) != 1 or digest(candidates[0]) != expected_hash:
                    raise ValueError(f"Cached Step {step} source changed: {filename}")
            for feature_set in report["feature_sets"].values():
                for filename, expected in feature_set["feature_file_sha256"].items():
                    location = (legacy.parent if feature_set["feature_set"] == "legacy_aml" else data) / filename
                    if digest(location) != expected:
                        raise ValueError(f"Cached Step {step} feature artifact changed: {location}")
                for filename, expected in feature_set.get("sequence_file_sha256", {}).items():
                    if digest(data / filename) != expected:
                        raise ValueError(f"Cached Step {step} sequence artifact changed: {filename}")
            state["steps"][str(step)] = {"status": "completed", "evidence": str(destination / "run_metadata.json")}; save()
        if through == step:
            return state
    args = ["--dataset-dir", data, "--step5-dir", output / "step5", "--step6-dir", output / "step6",
            "--step7-dir", output / "step7", "--output-dir", output / "step8"]
    if evaluation_settings:
        args += ["--settings-path", evaluation_settings]
    if trial_limit is not None and trial_limit < 40:
        args += ["--allow-reduced-budget"]
    if not (output / "step8").exists():
        command(8, "src.evaluation.calibrate_step8", args)
    else:
        from src.evaluation.evaluate_step9 import verify_freeze
        verify_freeze(output / "step8")
        state["steps"]["8"] = {"status": "completed", "evidence": str(output / "step8/frozen_decisions.json")}; save()
    if through == 8:
        return state
    if not (output / "step9").exists():
        command(9, "src.evaluation.evaluate_step9", ["--step8-dir", output / "step8", "--output-dir", output / "step9",
                "--reports-dir", ROOT / "reports" / output.name, "--legacy-features", legacy])
    result = read(output / "step9/evaluation.json")
    if result["frozen_decisions_sha256"] != digest(output / "step8/frozen_decisions.json"):
        raise ValueError("Cached evaluation used different frozen decisions")
    state["development_run"] = result["development_run"]
    state["completed_through"] = 9
    state["full_protocol_experiment_completed"] = not result["development_run"] and result["diagnostics"]["status"] == "completed"
    save()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--through", type=int, choices=range(4, 10), default=9)
    parser.add_argument("--trial-limit", type=int, help="Development only: identical reduced search budgets for every model")
    for name in ("classical-settings", "neural-settings", "evaluation-settings"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-checks", action="store_true", help="Reuse an already completed check run")
    args = vars(parser.parse_args())
    args["run_checks"] = not args.pop("skip_checks")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_pipeline(**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
