# Step 7 sequence-model gate and GRU model

Step 7 needs a Step 4 export built with sequences and a finished Step 6 run:

```bash
python -m src.features.build_multi_lead --include-sequences --output-dir data/processed/step4
python -m src.models.neural.train_step6
python -m src.models.neural.train_step7
```

Add `--allow-same-day-departures` to the Step 4 command only after the instructor approves same-day departure inputs. Without it, the sequence model uses weather sequences only, which is the protocol's fallback. The Step 7 options match Step 6 (`--dataset-dir`, `--output-dir`, `--trial-limit`, `--device`), plus `--step6-dir` (default `artifacts/step6`).

## The gate

The command always writes `go_no_go.json` and `go_no_go.md`. Training starts only when every condition passes:

| Condition | How it is checked |
|---|---|
| Reproducible MLP | The Step 6 run metadata, checkpoint and predictions exist; the checkpoint reloads; the saved reload check is within 1e-6. |
| Sequences for every lead | Each partition has a sequence file and every development row has a record. |
| Available by the cutoff | Every event time, and every departure's availability time, is at or before that row's cutoff; stated minutes before the cutoff match the timestamps. |
| Masks and padding | Each sequence has its fixed length, the mask matches the records, and padding is only on the left. |
| Time order | Events are strictly increasing in time. |
| Leakage tests | `tests/test_step3_asof.py` and `tests/test_step7_sequence.py` pass when the command runs them. |

These checks are recomputed from the files, independently of the Step 4 builder's own checks.

## Model

The sequence model is the Step 6 network plus two GRU encoders. Real events are moved to the front of each sequence, oldest first, and the GRU reads only those events, so padded slots cannot change a prediction (a test checks this). Weather steps carry the nine readings, scaled on training rows, a missing-value flag and the hours before the cutoff. Departure steps carry minutes before the cutoff, the departure delay clipped to −60 to 300 minutes, and the airline through the shared carrier embedding. An empty sequence gives a zero summary and a "no events" flag.

It uses the same folds, 40-configuration budget, early-stopping holdout, selection score and prediction schema as Step 6, and adds the GRU size (16, 32 or 64 in the config) to the search. Outputs follow the Step 6 layout under `artifacts/step7/`, with model name `sequence_model`.

Scoring with a saved checkpoint needs sequence inputs as well as feature rows: load them with `load_sequences` and pass `RawSequences.take(rows)` to `NeuralScorer.predict_proba`.
