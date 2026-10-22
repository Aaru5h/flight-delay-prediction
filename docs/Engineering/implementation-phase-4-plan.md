# Implementation Phase 4 Plan — Neural Model and Temporal Feasibility

**Lead:** Contributor 4  
**Goal:** Implement a compact embedding MLP that can be compared fairly with the classical baselines, then make an evidence-based decision about whether a sequence/temporal model is feasible within scope.

## 1. Entry conditions

- Shared target, row IDs, chronological split and prediction schema are agreed.
- Phase 2 feature matrix and leakage tests pass.
- Contributors 3 and 4 agree on which shared features the MLP will consume and how to export predictions.
- PyTorch and other selected dependencies install in the agreed environment.

## 2. Part A — Required embedding MLP

### Step 1 — Define the model input schema

Use learned embeddings for the categorical variables `carrier`, `origin` and `dest`, concatenated with normalized numeric inputs. Additional categorical inputs may be used only if they are in the approved feature manifest and can be encoded without leaking information. Keep IDs, target and audit fields outside the tensor inputs.

Create a stable category-to-index mapping from training data only. Reserve a code for unknown/unseen categories. Do not build category vocabularies from future test data if that would expose future categories or violate the agreed pipeline policy.

### Step 2 — Build training-only preprocessing

- Fit numeric imputation/scaling using the training split only.
- Define category mappings and numeric feature order as versioned model metadata.
- Apply the same preprocessing to train, validation and test.
- Preserve `flight_row_id` separately so prediction outputs can be aligned with other models.
- Handle missing categories explicitly rather than converting them into a valid real category by accident.

### Step 3 — Implement the MLP

- Create embedding tables for carrier, origin and destination.
- Concatenate embeddings with numeric features.
- Use a modest feed-forward network with documented hidden dimensions, activation, dropout/regularization and output layer.
- Output one probability for the positive class. Use a numerically stable binary classification loss such as logits with binary cross-entropy-with-logits.
- Handle class imbalance only through a documented training/validation experiment; do not modify validation/test labels.

Start with a small architecture. The objective is a reproducible benchmark, not an oversized neural model that is hard to tune on one year's data.

### Step 4 — Train and select on validation data

- Set and record all applicable random seeds.
- Use mini-batches and monitor training and validation loss/metrics.
- Use early stopping based on a validation metric fixed in advance.
- Avoid repeatedly selecting epochs against the test set.
- Save the best validation-selected checkpoint and training history.
- If a supported GPU is unavailable, keep a smaller CPU-compatible run available.

### Step 5 — Export common predictions and artifacts

Export validation probabilities in the shared schema. Persist weights/checkpoint, category mappings, feature order, numeric preprocessing state, model configuration, seed, package versions, split/feature versions and run ID. Verify that a reload produces the same probabilities within a documented tolerance.

## 3. Part B — Optional temporal-model go/no-go gate

A temporal model is an experiment, not a prerequisite for completing the project. Do not start it until the MLP is working end to end.

### Feasibility questions

- Is there a meaningful sequence available for each prediction (for example, earlier observations by route/carrier/origin)?
- Can every sequence element be ordered by event time and shown to be available by the target flight's T−60 cutoff?
- Can labels/outcomes from the target date and all later dates be excluded from historical sequence inputs?
- Are sequences sufficiently populated to compare models fairly without dropping a large, biased subset?
- Can the sequence builder be covered by perturbation tests that prove future events do not affect earlier predictions?
- Is there enough implementation time after the required MLP is reproducible?

### Go criteria

Proceed only when all data-availability questions are resolved, the sequence can be constructed from allowed historical data, the split remains chronological, and automated leakage tests can prove ordering. Define the sequence target, padding/masking and missing-history policy before model training.

### No-go/defer criteria

Defer the temporal model if sequence history depends on ambiguous same-day timing, future outcomes, unreliable timestamps, excessive population loss, or time that would put required integration at risk. Record the reason and retain the required embedding MLP as the neural comparison. Do not claim a temporal model was tested if no valid experiment was completed.

## 4. Tests

- Training-only category maps handle unknown categories.
- Tensor feature order and output row order match the feature manifest/IDs.
- All predicted probabilities are finite and in `[0, 1]`.
- Validation early stopping restores the best checkpoint.
- Artifact reload yields equivalent inference.
- Test labels are not used for early stopping, epoch selection or architecture tuning.
- For any temporal prototype, altering future records does not change earlier sequence features or predictions; target-date outcomes are excluded as defined by the Phase 0 contract.

## 5. Deliverables

Required:
- Reproducible embedding MLP.
- Category maps and training-only numeric preprocessing.
- Saved model/checkpoint, config and run metadata.
- Validation predictions in the common schema.
- Training history and tests.

Optional:
- Temporal feasibility note with go/no-go decision.
- If approved, temporal sequence builder, tests, model code and common-format predictions.

## 6. Acceptance criteria

- [ ] The MLP trains, reloads and produces valid aligned probabilities.
- [ ] Preprocessing and category mappings use training data only.
- [ ] Model selection is based on validation, not test, performance.
- [ ] The MLP uses the shared feature, target and split contracts.
- [ ] The temporal experiment has a written go/no-go decision.
- [ ] No temporal performance claim is made without a leakage-tested run artifact.

## 7. Handoff to Phase 5

Provide the selected MLP artifact, validation predictions and complete run metadata. If an optional temporal model was validly completed, provide it under the same prediction schema and identify it as an additional experiment.
