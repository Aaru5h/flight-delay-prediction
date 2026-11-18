# Steps 6 and 7 handoff — neural models

**Owner:** Aarush · **Protocol:** v2 (`configs/project_contract_v2.yaml`) · **Detail:** [Step 6](step-6-neural-handoff.md), [Step 7](step-7-sequence-handoff.md)

Steps 6 and 7 are implemented and tested. Step 6 trains the embedding MLP that protocol v2 requires. Step 7 runs the sequence-model go/no-go gate and, on a "go", trains a GRU sequence model. Both use the Step 5 folds, availability rules, early-stopping holdout, 40-configuration budget, selection score and prediction schema, so their predictions line up row for row with the baselines.

## What is in the repository

| Path | Purpose |
|---|---|
| `configs/step6_neural.json` | Search space and training limits for both neural models |
| `src/models/neural/preprocess.py` | Training-only category vocabularies, imputation, missing indicators and scaling |
| `src/models/neural/networks.py` | Embedding MLP, periodic numeric embeddings, GRU sequence encoders |
| `src/models/neural/fit.py` | Training loop, chronological early stopping, prediction, checkpoints, `NeuralScorer` |
| `src/models/neural/common.py` | Exact copies of the Step 5 run helpers (see "Mac note") |
| `src/models/neural/train_step6.py` | Step 6 command |
| `src/models/neural/sequences.py` | Sequence loading, independent checks, gate decision, sequence preprocessing |
| `src/models/neural/train_step7.py` | Step 7 command: gate, then the sequence model |
| `tests/test_step6_neural.py`, `tests/test_step7_sequence.py`, `tests/neural_helpers.py` | 18 tests |

## Setup

Use one virtual environment so every package goes to the same Python:

```bash
python3.12 -m venv .venv          # or python3.13, the agreed version, if installed
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests/test_step6_neural.py tests/test_step7_sequence.py   # expect 18 passed
```

**Mac note:** loading XGBoost and PyTorch in the same Python process can crash with a segmentation fault on Apple-silicon Macs, because each brings its own OpenMP runtime. The neural code never imports XGBoost, and a test enforces this. Evaluation code should score classical and neural models in separate processes, or import `torch` first.

## How to run

```bash
# 1. Step 4 data with sequences (about 45 minutes). Step 7 needs the sequence files.
python -m src.features.build_multi_lead --include-sequences --output-dir data/processed/step4

# 2. Step 6, then Step 7 (full protocol budget; plan for an overnight run)
python -m src.models.neural.train_step6
python -m src.models.neural.train_step7
```

- `--trial-limit N` runs fewer configurations for a quick check; the metadata marks it as a reduced budget.
- `--device mps` is faster on a Mac but may not reproduce CPU results bit for bit. CPU is the default.
- Add `--allow-same-day-departures` to step 1 only after the instructor approves same-day departure inputs. Until then Step 7 uses weather sequences only, which is the protocol's fallback.
- For the legacy T−60 comparison, pass `--legacy-features` and `--legacy-manifest` to Step 6 as for Step 5.
- Step 7 peaked at about 5 GB of memory on the full data.

## Outputs for evaluation (Step 8–9)

Step 6 writes to `artifacts/step6/`; Step 7 writes to `artifacts/step7/`.

| File | Contents |
|---|---|
| `*_validation_predictions.csv` | Out-of-fold probabilities for May–October in the protocol v2 schema: `flight_row_id, lead_time_minutes, fold_id, split, y_true, y_probability, model_name, run_id`. Model names `embedding_mlp` and `sequence_model`. |
| `*.pt` | Final model fitted before the retrospective boundary. Score Nov–Dec with `NeuralScorer(path).predict_proba(frame)`; the sequence model also needs `load_sequences(...)[0].take(rows)`. |
| `fold_manifest.csv` | Built by the same function as Step 5, from the same metadata |
| `run_metadata.json` | Run ID, code commit, data fingerprints, manifest and fold versions, config, seed, package versions, every trial's per-fold and per-lead scores and selected epochs |
| `go_no_go.json`, `go_no_go.md` | Step 7 gate decision and the evidence for each condition |

Retrospective (Nov–Dec) outcomes are never read by Steps 6 or 7. Calibration and operating points belong to Step 8 and should use the validation predictions above.

## Status of the checks

- **Tests:** the 18 neural tests pass. In a full-suite run, 87 of 88 tests passed; the one failure was a Phase 0 test that reads `reports/phase0_split_profile.csv`, which is in the repository but was missing from that test copy.
- **Full-data development run** (separate environment, 1 configuration, at most 4 epochs): both commands completed on all 1,309,384 rows. The saved checkpoints reloaded with zero difference.
- **Gate on full data:** go. 1,309,384 sequence records were checked, with 0 events after a cutoff, 0 mask or padding errors and 0 out-of-order events. The leakage tests passed. Weather sequences cover 99.7–100% of rows at each lead time.
- **Indicative validation PR-AUC from that reduced run.** These are not results for the report; report numbers must come from the full runs' saved evaluation files.

  | Lead time | Embedding MLP | Sequence model |
  |---|---|---|
  | 1 h | 0.482 | 0.500 |
  | 2 h | 0.478 | 0.496 |
  | 6 h | 0.459 | 0.484 |
  | 24 h | 0.394 | 0.419 |

## Open items

| Item | Owner | Note |
|---|---|---|
| Full 40-configuration runs of Steps 6 and 7 | Aarush | Overnight; outputs go to Kevish |
| Instructor approval for same-day departure inputs | Aarush | Likely the biggest gain for the sequence model; rebuild Step 4 with the flag if approved |
| Calibration, operating points, comparison grid, intervals | Kevish | Use the validation prediction files; keep XGBoost and PyTorch scoring in separate processes on Mac |
| Report and slides | Meet | Take every number from the saved evaluation results, not from this note |
