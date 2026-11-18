"""Audited prediction-only import of teammate runs; strict native loading stays unchanged."""

from __future__ import annotations

import json
from pathlib import Path
import stat
from zipfile import ZipFile

import pandas as pd
import yaml

from src.data.loading import file_fingerprint
from src.evaluation.artifacts_v2 import CONTRACT, ModelArtifact, read_json, validate_predictions
from src.models.classical.step5_data import FeatureSet, make_folds, ID, LEAD, DATE, AVAILABLE, LABEL


def extract_verified(archive, destination):
    archive, destination = Path(archive), Path(destination)
    identity = file_fingerprint(archive)
    marker = destination / "archive_identity.json"
    if destination.exists():
        if not marker.exists() or read_json(marker)["sha256"] != identity:
            raise ValueError("Import directory already belongs to a different or incomplete archive")
        return
    destination.mkdir(parents=True)
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Unsafe archive path or symbolic link")
        zipped.extractall(destination)
    marker.write_text(json.dumps({"sha256": identity}) + "\n")


def population(dataset):
    cols = [ID, DATE, "prediction_cutoff_utc", AVAILABLE, LABEL]
    frame = pd.read_csv(Path(dataset) / "features_1440m.csv", usecols=cols, dtype={ID: str})
    frame[DATE] = pd.to_datetime(frame[DATE]).dt.normalize()
    frame[AVAILABLE] = pd.to_datetime(frame[AVAILABLE], utc=True)
    frame["first_cutoff_utc"] = pd.to_datetime(frame.pop("prediction_cutoff_utc"), utc=True)
    if frame[ID].isna().any() or frame[ID].duplicated().any() or not frame[LABEL].isin([0, 1]).all():
        raise ValueError("Invalid local flight population")
    return frame.set_index(ID)


def verify_fold_manifest(path, metadata, folds, final):
    assignments = pd.read_csv(path, dtype={ID: str})
    if assignments.duplicated([ID, "fold_id"]).any():
        raise ValueError("Duplicate fold assignments")
    if set(assignments.fold_id) != {f.fold_id for f in [*folds, final]}:
        raise ValueError("Unexpected fold windows")
    for fold in [*folds, final]:
        rows = assignments.loc[assignments.fold_id.eq(fold.fold_id)]
        split = "retrospective" if fold is final else "validation"
        if set(rows.split) != {"train", split} or set(rows.loc[rows.split.eq("train"), ID]) != set(fold.train_ids) or set(rows.loc[rows.split.eq(split), ID]) != set(fold.evaluation_ids):
            raise ValueError("Archive folds differ from availability-safe local folds")
        dates = pd.to_datetime(rows[DATE]).dt.normalize()
        if not dates.eq(rows[ID].map(metadata[DATE])).all():
            raise ValueError("Fold dates disagree with local schedules")


def load_archive_handoff(archive, dataset, destination):
    extract_verified(archive, destination)
    destination, dataset = Path(destination), Path(dataset)
    protocol = yaml.safe_load(CONTRACT.read_text())
    manifest = read_json(dataset / "feature_manifest.json")
    metadata = population(dataset)
    folds, final = make_folds(metadata, protocol)
    frames = []
    for fold in folds:
        for lead in protocol["prediction_time"]["lead_time_minutes"]:
            rows = metadata.loc[fold.evaluation_ids, [DATE, AVAILABLE, LABEL]].reset_index()
            frames.append(rows.rename(columns={LABEL: "y_true"}).assign(lead_time_minutes=lead, fold_id=fold.fold_id))
    expected = pd.concat(frames, ignore_index=True)
    files = {"feature_manifest.json": file_fingerprint(dataset / "feature_manifest.json")}
    files.update({p["features"]: file_fingerprint(dataset / p["features"]) for p in manifest["partitions"]})
    models, budgets, mismatches = [], set(), []
    canonical_folds = None
    for meta_path in sorted(destination.rglob("run_metadata.json")):
        report = read_json(meta_path)
        if report["protocol_sha256"] != file_fingerprint(CONTRACT) or report["raw_data_fingerprints"] != manifest["source_fingerprints"]:
            raise ValueError("Archive protocol or source data differs from this project")
        budgets.add(report["trials_per_model"])
        fold_path = meta_path.parent / report["fold_manifest"]
        digest = file_fingerprint(fold_path)
        if digest != report["fold_manifest_sha256"]:
            raise ValueError("Archive fold checksum mismatch")
        if canonical_folds is None:
            verify_fold_manifest(fold_path, metadata, folds, final)
            canonical_folds = digest
        elif canonical_folds != digest:
            raise ValueError("Models have different folds")
        for name, item in report["feature_sets"].items():
            if item["feature_manifest_version"] != manifest["manifest_version"]:
                raise ValueError("Feature definitions have a different version")
            if item["feature_file_sha256"].get("feature_manifest.json") != files["feature_manifest.json"]:
                raise ValueError("Feature-manifest definitions differ")
            differences = [n for n, sha in item["feature_file_sha256"].items() if files.get(n) != sha]
            if differences:
                mismatches.append({"run_id": item["run_id"], "files": differences,
                                   "status": "Cross-platform rounding explanation supplied; original numeric tables absent, not independently verified"})
            for model_name, info in item["models"].items():
                if len(info["trials"]) != report["trials_per_model"] or info["fit_cutoff_utc"] != final.fit_cutoff_utc.isoformat():
                    raise ValueError("Invalid search budget or fit boundary")
                for filename in [info["model_file"], info["validation_predictions"]]:
                    if not (meta_path.parent / filename).is_file():
                        raise FileNotFoundError(filename)
                if model_name == "sequence_model" and report.get("gate", {}).get("decision") != "go":
                    raise ValueError("Sequence model lacks a passing gate")
                pred = pd.read_csv(meta_path.parent / info["validation_predictions"], dtype={ID: str})
                pred = validate_predictions(pred, expected, item["run_id"], model_name)
                matches = list(destination.rglob(f"{name}_{model_name}_retrospective_predictions.csv"))
                if len(matches) != 1:
                    raise ValueError("Each imported model needs exactly one retrospective prediction file")
                info = {**info, "retrospective_predictions_path": str(matches[0].resolve())}
                models.append(ModelArtifact(model_name, name, meta_path.parent, item["run_id"], report, item, info, pred))
    required = {"schedule_only", "logistic_regression", "xgboost", "embedding_mlp", "sequence_model"}
    if {m.name for m in models} != required or len(models) != 5 or len(budgets) != 1:
        raise ValueError("Expected five distinct models with the same search budget")
    models.sort(key=lambda m: (m.name != "schedule_only", m.name))
    lightweight = metadata.reset_index().assign(lead_time_minutes=60)
    features = FeatureSet("multi_lead", lightweight, manifest["manifest_version"], manifest["allowlist"], [], files)
    audit = {"archive_sha256": file_fingerprint(archive), "feature_table_differences": mismatches,
             "validated": ["raw fingerprints", "feature definitions", "complete availability-safe fold assignments", "validation IDs, labels, splits, probabilities and run IDs"],
             "scope": "Supplied five-model reduced-budget comparison; no full-budget or legacy feature attribution claim"}
    return protocol, features, metadata, folds, final, models, ["Legacy-feature model runs were not supplied"], audit


def read_archived_retrospective(model, dataset):
    metadata = population(dataset)
    truth = metadata.loc[metadata[DATE].between("2013-11-01", "2013-12-31")]
    frame = pd.read_csv(model["retrospective_predictions_path"], dtype={ID: str})
    required = {ID, LEAD, "fold_id", "split", "y_true", "y_probability", "model_name", "run_id"}
    if not required.issubset(frame) or frame[list(required)].isna().any().any() or frame.duplicated([ID, LEAD, "fold_id"]).any():
        raise ValueError("Invalid archived retrospective keys")
    if set(frame.split) != {"retrospective"} or set(frame.fold_id) != {"retrospective"} or set(frame.run_id) != {model["run_id"]} or set(frame.model_name) != {model["name"]}:
        raise ValueError("Retrospective run/split mismatch")
    if not frame.y_probability.between(0, 1).all() or not frame.y_true.eq(frame[ID].map(truth[LABEL])).all():
        raise ValueError("Retrospective labels/probabilities invalid")
    if set(frame[LEAD]) != {60, 120, 360, 1440} or len(frame) != 4 * len(truth):
        raise ValueError("Retrospective population is incomplete")
    for _, group in frame.groupby(LEAD):
        if set(group[ID]) != set(truth.index):
            raise ValueError("Retrospective flights differ across leads")
    frame[DATE] = frame[ID].map(truth[DATE]).dt.strftime("%Y-%m-%d")
    return frame.rename(columns={"y_probability": "y_probability_raw"})
