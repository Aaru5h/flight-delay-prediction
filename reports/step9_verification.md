# Verification of the implementation through Step 9

Verified on 19 November 2026 using Python 3.13 and the canonical requirements.

```text
python -m src.checks
85 passed (classical, data, features, archive import and evaluation)
19 passed (neural, in a separate process)
```

The integration test trains and reloads schedule-only, LR, XGBoost, embedding
MLP and sequence models, including the legacy/new feature comparison. It
checks the six monthly folds, label availability, calibration populations,
strictly earlier-month validation calibration, exact alert budgets, common
retrospective populations, paired day-bootstrap intervals, refit ablations,
subgroup tables and figure generation. It also verifies that modifying a frozen
checkpoint is rejected. Separate arithmetic tests check probability bounds,
tie handling, Brier references and weighted bootstrap equivalence.

Sequence tests cover masks, ordering, availability, padding invariance and
duplicate-record rejection. Source fingerprints are checked by the original
data-contract suite. Source compilation and whitespace checks also pass.

The first mixed XGBoost/PyTorch full-suite process crashed with the OpenMP
conflict described in the neural handoff. `src.checks` and evaluation workers
use separate processes; the neural helper-comparison test now also isolates
its classical imports.

Real-data Steps 8-9 have now completed using the supplied five-model archive,
with one configuration per model. All 20 model/lead combinations were
independently recomputed from the exported predictions and agree with the
evaluation to 1e-12 tolerance. Frozen input hashes and every exported alert mask
also pass verification. See `reports/supplied_run/results_verification.json`.

Numerical results are in `reports/supplied_run/RESULTS.md` and
`numerical_results.csv`; full metrics and confidence intervals are in
`evaluation.json`. This completes the supplied-run scope. It does not complete
the 40-configuration experiment or the missing legacy-feature comparison grid.
