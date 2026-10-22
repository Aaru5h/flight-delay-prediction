# Phase 0 — Repository Inventory and Environment Report

**Date:** 2026-10-22
**Contract:** [configs/project_contract.yaml](../configs/project_contract.yaml) (v1.0.0)
**Status:** Phase 0 complete — contracts frozen, environment verified, baseline tests passing.

This report records the repository inventory, source-data fingerprints, agreed
environment, and the split-prevalence inspection required by
[implementation-phase-0-plan.md](../docs/Engineering/implementation-phase-0-plan.md).

## 1. Repository inventory

| Path | Status | Purpose |
|---|---|---|
| `README.md` | existing | Project overview and getting-started guide |
| `docs/Product/`, `docs/Research/`, `docs/Architecture Diagrams/`, `docs/Engineering/` | existing | PRD/design, research notes, diagrams, phase plans |
| `docs/Data/`, `docs/Models/` | empty scaffolds | Reserved; no content yet |
| `data/raw/` | existing (5 CSVs) | Immutable source inputs |
| `data/processed/`, `src/**`, `tests/`, `configs/`, `reports/`, `artifacts/` | scaffolded | Populated from Phase 1 onward |
| `configs/project_contract.yaml` | **new (Phase 0)** | Canonical frozen contract: target, T−60 rules, split, metrics, artifact schemas |
| `configs/data_manifest.json` | **new (Phase 0)** | SHA-256 fingerprints, sizes, row/column counts and headers for all five raw inputs |
| `requirements.txt` | **new (Phase 0)** | Single canonical dependency file |
| `.gitignore` | **new (Phase 0)** | Excludes `.venv/`, caches, `data/processed/`, `artifacts/` |
| `tests/test_phase0_contracts.py` | **new (Phase 0)** | Baseline contract tests |
| `reports/phase0_split_profile.csv` | **new (Phase 0)** | Split-window prevalence evidence (§4) |

No raw CSV was edited; fingerprints in `configs/data_manifest.json` are the
reference the tests enforce.

## 2. Source-data fingerprints (SHA-256, measured 2026-10-22)

See `configs/data_manifest.json` for the full machine-readable record
(headers included). File sizes: `nyc_flights.csv` 30 MB; `nyc_weather.csv`
2.2 MB; `nyc_airports.csv` 102 KB; `nyc_planes.csv` 241 KB;
`nyc_airlines.csv` 386 B.

## 3. Agreed environment

- **Interpreter:** Python 3.13 (dev machine: 3.13.6, arm64 macOS). The Phase 0
  plan suggested 3.11 as a candidate; 3.11/3.12 are not installed on this
  machine, so 3.13 was agreed instead — it is fully supported by the whole stack.
- **Dependency management:** `requirements.txt` is the single canonical file.
- **Verified clean install:** fresh `.venv` created with
  `python3.13 -m venv .venv` + `pip install -r requirements.txt` succeeded.
- **Resolved versions on the verification run:** pandas 2.3.3, numpy 2.5.4,
  scikit-learn 1.9.1, xgboost 3.4.1, torch 2.14.1, matplotlib 3.11.2,
  PyYAML 6.0.3, pytest 9.1.1.
- **Test command:** `python -m pytest` (run via `.venv/bin/python -m pytest`).

## 4. Split inspection (Phase 0 Step 4 evidence)

Monthly labelled-positive prevalence from
[reports/phase0_split_profile.csv](phase0_split_profile.csv):

| Split | Window | Rows | Labelled | Positive | Unknown | Prevalence |
|---|---|---:|---:|---:|---:|---:|
| Train | Jan–Aug 2013 | 224,910 | 217,727 | 57,423 | 7,183 | 26.37% |
| Validation | Sep–Oct 2013 | 56,463 | 55,628 | 8,731 | 835 | 15.70% |
| Test | Nov–Dec 2013 | 55,403 | 53,991 | 13,946 | 1,412 | 25.83% |
| All | 2013 | 336,776 | 327,346 | 80,100 | 9,430 | 24.47% |

Monthly prevalence shows the expected seasonality: June–July and December are
the peaks (31.5–33.5%), September the trough (13.7%).

**Decision:** the proposed boundaries are frozen as-is. The validation window's
notably lower prevalence (15.7%) is a genuine seasonal shift; moving boundaries
to balance prevalence would be outcome-driven tuning. The shift is recorded in
`configs/project_contract.yaml` (`split.distribution_shift_note`) and must be
disclosed in every evaluation report, with prevalence reported alongside all
metrics. Threshold selection on the validation window must account for its
lower prevalence when interpreting recall at a precision floor.

## 5. Decisions and deviations log

| # | Decision | Rationale |
|---|---|---|
| D1 | `requirements.txt` over `pyproject.toml` | Simpler single-file contract for a pipeline project; plan permits either, one is canonical |
| D2 | Python 3.13 instead of the 3.11 candidate | 3.11/3.12 unavailable on the dev machine; 3.13 fully supported by all pinned libraries |
| D3 | YAML for the frozen contract | Machine-readable by tests, human-readable for review |
| D4 | Raw CSVs stay tracked in git | Matches README §10; they are the supplied project inputs |
| D5 | Split frozen with disclosed prevalence shift | Preserving proposed boundaries avoids outcome-driven changes; shift disclosed per plan §6 risk policy |

## 6. Open assumptions

- **TZ-001 (time_hour semantics):** `flights.time_hour` / `weather.time_hour`
  appear to be UTC (local = UTC−5 EST). Recorded in the contract as
  `open_assumptions`; must be verified in Phase 2 before the weather as-of
  join is built. If unresolved, the weather join is blocked rather than guessed.

## 7. Baseline test result

```
$ .venv/bin/python -m pytest tests/ -v
tests/test_phase0_contracts.py::test_raw_files_match_data_manifest PASSED
tests/test_phase0_contracts.py::test_flights_header_matches_documented_schema PASSED
tests/test_phase0_contracts.py::test_contract_freezes_expected_values PASSED
tests/test_phase0_contracts.py::test_split_profile_reconciles_with_flights_csv PASSED
tests/test_phase0_contracts.py::test_reference_profile_matches_supplied_data PASSED
============================== 5 passed in 0.86s ==============================
```

## 8. Phase 0 acceptance checklist

- [x] The environment can be recreated from the documented dependency file.
- [x] No raw CSV has been changed (enforced by fingerprint test).
- [x] The target definition and missing-label policy are unambiguous.
- [x] The proposed date split has been checked against date-level counts and target prevalence (§4).
- [x] Shared schemas and component boundaries are written down (`schemas` section of the contract).
- [x] The repository has one canonical place for config (`configs/`), tests (`tests/`), and generated artifacts (`reports/`, `artifacts/` + gitignore rules).
- [x] The metric/threshold policy protects the test set from tuning.
