# Step 6 embedding MLP

Run this from a checkout containing protocol v2, the completed Step 4 datasets and the Step 5 code.

```bash
python -m src.models.neural.train_step6
```

By default the command reads `data/processed/step4/` and publishes results under `artifacts/step6/`. Use `--dataset-dir` for another Step 4 export or `--output-dir` for a fresh run. The full run draws 40 configurations, as protocol v2 requires. A smoke run can use `--trial-limit 1`, which the metadata records as a reduced-budget run. Training runs on the CPU by default so results are reproducible; `--device mps` (Apple silicon) or `--device cuda` is faster but may not match the CPU run bit for bit.

For the protocol's legacy comparison at T−60, supply the Phase 2 table exactly as for Step 5:

```bash
python -m src.models.neural.train_step6 \
  --legacy-features data/processed/phase2_features.csv \
  --legacy-manifest data/processed/phase2_feature_manifest.json
```

## Same rules as the baselines

The MLP imports the Step 5 reader, fold builder and inner split, so it sees the same flights, folds, outcome-availability rule and 14-day early-stopping holdout as Logistic Regression and XGBoost. The run refuses to start if its holdout length differs from the Step 5 setting. Each candidate picks its epoch count on the inner holdout by log loss, then refits on the eligible outer training window for exactly that many epochs, as the baselines do with iterations and trees. The winner has the highest mean validation average precision across folds and lead times.

## Model

| Part | Choice |
|---|---|
| Categorical inputs | Learned embeddings for every manifest categorical, including `lead_time_minutes` and `sched_dep_hour`. Size min(16, round(1.6 × n^0.56)). Index 0 is rare or unseen, index 1 is missing. Categories need 5 training rows. |
| Numeric inputs | Training medians, a missing indicator for each column with gaps in training, standardisation, clipped at ±8. The search chooses plain inputs or periodic embeddings (Gorishniy et al., 2022). |
| Network | Dense layers of 128–64, 256–128–64 or 512–256–128 with ReLU and dropout; one output whose bias starts at the training log-odds. |
| Training | Binary cross-entropy on logits, optional positive-class weighting, AdamW, one warm-up epoch then cosine decay over a fixed 30-epoch budget, up to 30 epochs with patience 4. |

The search space is in `configs/step6_neural.json`.

## Evaluation handoff

- `fold_manifest.csv` is built by the Step 5 function from the same metadata, so it matches the baselines' manifest.
- `*_validation_predictions.csv` holds the selected candidate's out-of-fold probabilities in the protocol v2 prediction schema, for `multi_lead` and, if requested, `legacy_aml`.
- `*.pt` is the final checkpoint, fitted before the retrospective boundary. `NeuralScorer(path).predict_proba(frame)` reloads it and scores raw feature rows using the saved preprocessing. Each run checks that a reload reproduces the trained probabilities.
- `run_metadata.json` records the protocol fields: run ID, code commit, raw-data fingerprints, manifest and fold-manifest versions, model config, seed and package versions, plus every trial's per-fold, per-lead scores and selected epochs.

Run `python -m pytest -q tests/test_step6_neural.py` for the preprocessing, early-stopping, reload and fold-contract checks.
