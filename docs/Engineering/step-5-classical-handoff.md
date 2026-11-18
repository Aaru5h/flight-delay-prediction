# Step 5 classical baselines

Run this from a checkout containing protocol v2 and the completed Step 4 datasets.

```bash
python -m src.models.classical.train_step5
```

By default the command reads `data/processed/step4/` and publishes results under `artifacts/step5/`. Use `--dataset-dir` for another Step 4 export or `--output-dir` for a fresh run. The full run draws 40 configurations for each benchmark. A CPU smoke run can use `--trial-limit 1`, which is recorded in the metadata as a reduced-budget run.

## Benchmarks and fold rules

| Benchmark | Inputs and estimator |
|---|---|
| Schedule-only | Schedule, calendar, static reference inputs, and lead time from the Step 4 manifest. Logistic Regression with a periodic hour spline. |
| Logistic Regression | Full Step 4 allowlist, train-only one-hot encoding and median imputation, and a periodic spline for scheduled departure hour. |
| XGBoost | The same full allowlist. Train-only category codes, native categorical splits, and native missing-value handling. |

Each May–October fold validates on one local service month and uses earlier dates for training. Model fitting uses only labels whose `outcome_available_utc` is at or before the earliest validation cutoff across the four leads. All lead times for a flight share the same fold membership. The last 14 service days of each training window provide a chronological early-stopping holdout. The inner fit applies the same outcome-availability rule using the earlier holdout cutoff.

SAGA fits in warm-start blocks and checks holdout log loss between blocks. XGBoost uses the holdout as its early-stopping evaluation set. Each candidate is then refit with its selected iteration budget on the eligible outer training window. The candidate with the highest mean validation average precision across folds and leads wins. Ties keep the earliest sampled candidate.

For the legacy feature comparison, first generate the Phase 2 table, then supply both files:

```bash
python -m src.features.pipeline
python -m src.models.classical.train_step5 \
  --legacy-features data/processed/phase2_features.csv \
  --legacy-manifest data/processed/phase2_feature_manifest.json
```

The optional comparison refits Logistic Regression and XGBoost at T−60 on the legacy representation with the same development flights, fold eligibility, and trial budget. Compare their T−60 predictions with the T−60 slice of the new-feature predictions. The all-lead score in metadata averages over different lead-time populations.

## Evaluation handoff

- `fold_manifest.csv` lists eligible training and evaluation assignments per `flight_row_id` and `fold_id`. The retrospective rows contain schedule metadata and no labels.
- `*_validation_predictions.csv` contains the selected candidates' out-of-fold probabilities in the protocol v2 prediction schema.
- `*.joblib` contains a reloadable preprocessing and estimator pipeline fit before the retrospective boundary. The dataset run leaves November–December outcomes for the evaluation owner to score.
- `run_metadata.json` records the fold and source fingerprints, candidate settings, validation scores, early-stopping diagnostics, versions, and artifact filenames.

For later scoring, reload the corresponding pipeline and select its recorded allowlist from the matching feature table. Use the out-of-fold validation predictions for calibration and operating points in Step 8.
