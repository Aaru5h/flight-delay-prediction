"""Isolated scoring worker: never load XGBoost and PyTorch together on macOS."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.artifacts_v2 import CONTRACT, read_json
from src.models.classical.step5_data import ID, LEAD, DATE, CUTOFF, LABEL


def read_retrospective(dataset_dir, manifest, allowlist, *, legacy_features=None):
    protocol = yaml.safe_load(CONTRACT.read_text())
    period = protocol["validation"]["retrospective_period"]
    partitions = manifest["partitions"] if legacy_features is None else [{LEAD: 60, "features": str(legacy_features)}]
    frames = []
    for partition in partitions:
        path = Path(dataset_dir) / partition["features"]
        columns = set(allowlist) | {ID, DATE, CUTOFF, LABEL}
        if legacy_features is None:
            columns.add(LEAD)
        for chunk in pd.read_csv(path, usecols=columns, dtype={ID: str}, chunksize=50000):
            dates = pd.to_datetime(chunk[DATE], utc=True).dt.strftime("%Y-%m-%d")
            rows = chunk.loc[dates.between(period["start"], period["end"])].copy()
            rows[DATE] = dates.loc[rows.index]
            rows[LEAD] = partition[LEAD]
            if len(rows):
                rows[LABEL] = pd.to_numeric(rows[LABEL], errors="raise")
                if not rows[LABEL].isin([0, 1]).all():
                    raise ValueError("Unknown retrospective labels")
                frames.append(rows)
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated([ID, LEAD]).any() or frame[[ID, LEAD]].isna().any().any():
        raise ValueError("Invalid retrospective feature keys")
    return frame


def run_request(request):
    manifest = read_json(Path(request["dataset_dir"]) / "feature_manifest.json")
    item = request["model"]
    frame = read_retrospective(request["dataset_dir"], manifest, item["feature_allowlist"],
                               legacy_features=request.get("legacy_features") if item["feature_set"] == "legacy_aml" else None)
    if item["model_path"].endswith(".pt"):
        import torch
        from src.models.neural.fit import NeuralScorer
        torch.set_num_threads(2)
        scorer = NeuralScorer(Path(item["model_path"]))
        if scorer.allowlist != item["feature_allowlist"] or scorer.metadata["run_id"] != item["run_id"]:
            raise ValueError("Neural checkpoint metadata mismatch")
        sequence = None
        if scorer.sequence_state is not None:
            from src.models.neural.sequences import load_sequences
            raw, checks = load_sequences(Path(request["dataset_dir"]), manifest, frame)
            if any((checks.missing_development_records, checks.structure_errors, checks.order_errors,
                    checks.cutoff_violations, checks.availability_violations, checks.offset_mismatches)):
                raise ValueError("Retrospective sequence audit failed")
            sequence = raw.take(np.arange(len(frame)))
        probability = scorer.predict_proba(frame, sequence)
    else:
        import joblib
        scorer = joblib.load(item["model_path"])
        probability = scorer.predict_proba(frame[item["feature_allowlist"]])[:, 1]
    if len(probability) != len(frame) or not (np.isfinite(probability) & (probability >= 0) & (probability <= 1)).all():
        raise ValueError("Invalid retrospective model probabilities")
    result = frame[[ID, LEAD, DATE, LABEL]].rename(columns={LABEL: "y_true"}).assign(
        fold_id="retrospective", split="retrospective", model_name=item["name"],
        feature_set=item["feature_set"], run_id=item["run_id"], y_probability_raw=probability,
    )
    result.to_csv(request["output_path"], index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    run_request(read_json(args.request))


if __name__ == "__main__":
    main()
