# Supervised Cell Proportion Recovery Design

## Goal

Recover the original `predict_cell_prop` workflow in `VAEDecon` so the model
learns to predict cell proportions from bulk RNA-seq during training and can
output predicted cell proportions after training without requiring ground-truth
fractions at inference time.

The intended behavior is:

- use `predict_cell_prop=True`
- supervise the cell proportion branch with known training-set cell fractions
- use the predicted cell proportions, not the ground-truth ones, in the
  reconstruction path
- allow inference on unlabeled datasets after training

## Current Situation

The current codebase already contains most of the prediction branch, but it is
only partially active.

What is already present:

- encoders can emit `pred_cell_prop`
- the VAE forward path returns `pred_cell_prop`
- the evaluation workflow can save `predicted_cell_prop.csv`
- the training datasets already load cell-fraction labels when available

What is incomplete or inconsistent:

- the supervised cell proportion loss is computed but not added into the final total loss
- missing-label handling is inconsistent across encoders
- config validation does not clearly define the requirements for
  `predict_cell_prop=True`
- some fallback logic mixes "supplied labels for conditioning" with "predicted
  proportions for inference," which blurs the workflow contract

## Target Workflow

### Training

When `predict_cell_prop=True`:

- the training dataset must provide ground-truth cell proportions
- the encoder predicts cell proportions from the input bulk expression
- the predicted cell proportions are used in the downstream reconstruction path
- the predicted cell proportions are directly supervised against the training
  labels

This keeps the model honest: the branch being optimized is the same branch used
later at inference time.

### Validation

When validation labels are available:

- compute the supervised cell proportion loss
- save and visualize predicted cell proportions as needed
- report label-based cell proportion metrics

When validation labels are not available:

- still run forward prediction
- skip label-based cell proportion losses and metrics
- keep reconstruction-related outputs available

### Inference

When `predict_cell_prop=True`:

- inference must not depend on embedded dataset labels
- the trained prediction head produces cell proportions directly from bulk
  expression
- the workflow saves `predicted_cell_prop.csv`

This is the main recovered user-facing feature.

## Non-goals

This recovery does not introduce a new multi-mode abstraction for both:

- supervised prediction
- fixed or externally supplied conditioning proportions

That second workflow can remain supported later, but this design is focused on
restoring and stabilizing the supervised prediction path first.

## Proposed Behavior

### 1. Restore supervised cell proportion loss

In `vae_model.py`, re-enable the supervised cell proportion term in the total
loss when:

- `predict_cell_prop=True`
- ground-truth labels are available in the batch

The existing `cell_prop` loss weight in `loss_coefficient` remains the switch
that controls the magnitude of this term.

Design rule:

- if `predict_cell_prop=True` and `cell_prop > 0`, the supervised loss must
  contribute to `total_loss`
- if `predict_cell_prop=True` and labels are missing during training, raise a
  clear error instead of silently training an unsupervised proportion head

### 2. Keep reconstruction conditioned on predicted proportions

During supervised training, the model should still reconstruct bulk expression
using the predicted cell proportions, not the ground-truth ones.

Reason:

- this matches the intended inference-time computation graph
- it avoids a train-inference mismatch where the prediction head is supervised
  but not actually used by the reconstruction branch

### 3. Standardize missing-label handling

Introduce one consistent rule for label availability:

- labels are usable only when they are not `None` and `numel() > 0`

Apply this rule in:

- encoder fallback logic
- VAE forward and loss code
- evaluation code

This removes the current ambiguity where an empty tensor can be mistaken for a
real label matrix.

### 4. Clarify workflow contracts in config validation

Add explicit validation rules around `predict_cell_prop`:

- `predict_cell_prop=True` means the model is expected to predict cell
  proportions after training
- training data must contain cell-fraction labels for supervised training
- `loss_coefficient.cell_prop` must be non-negative
- if `predict_cell_prop=False`, the supervised prediction branch is inactive

This design does not require a new config field yet. The existing
`predict_cell_prop` switch remains the public entry point.

### 5. Make evaluation outputs honest

When `predict_cell_prop=True`:

- `predicted_cell_prop.csv` must contain model predictions

When labels are also available:

- save comparison outputs and metrics against the true cell proportions
- plot the predicted cell proportions against the true cell proportions and report the correlation coefficient, RMSE, and CCC as metrics

The workflow should avoid writing ground-truth labels into
`predicted_cell_prop.csv` under the prediction-enabled path.

## Affected Code Areas

### 1. VAE loss assembly

File:

- `vaedecon/models/vae/vae_model.py`

Needed changes:

- ensure the supervised cell proportion loss is included in `total_loss`
- gate that loss on both `predict_cell_prop=True` and usable labels
- raise a clear error during training if supervised prediction is enabled but
  labels are absent

### 2. Encoder fallback behavior

Files:

- `vaedecon/models/nn/mlp.py`
- `vaedecon/models/nn/res_mlp.py`
- `vaedecon/models/nn/transformer.py`
- `vaedecon/models/nn/fused_mlp_gnn.py` (deprecated, no need to update this file)

Needed changes:

- use one shared interpretation of "labels are available"
- do not treat empty tensors as valid conditioning proportions

### 3. Dataset output contract

File:

- `vaedecon/data/datasets.py`

Needed review:

- keep the current batch structure if possible
- make sure downstream code does not confuse empty labels with real labels

No format rewrite is required if the rest of the stack consistently checks
`numel() > 0`.

### 4. Evaluation and inference workflow

Files:

- `vaedecon/workflow/workflow.py`
- `vaedecon/workflow/inference.py`

Needed changes:

- ensure prediction-enabled inference works without labels
- save model-predicted cell proportions
- compute comparison metrics only when true labels are available

### 5. Config validation and documentation

File:

- `vaedecon/configs/default_config.py`

Needed changes:

- clarify the intended semantics of `predict_cell_prop`
- validate the relationship between `predict_cell_prop` and
  `loss_coefficient.cell_prop`

## Recommended Implementation Order

1. Restore the supervised proportion loss in the VAE total loss.
2. Add one helper for checking whether labels are usable.
3. Apply that helper across encoder, loss, and evaluation code paths.
4. Tighten config validation and error messages.
5. Add focused tests for supervised prediction training and unlabeled
   inference.

This order keeps the highest-value behavior change first while limiting the
debug surface.

## Testing Plan

### Unit and config checks

1. Verify that `predict_cell_prop=True` with a positive cell proportion loss
   weight includes the supervised proportion term in `total_loss`.
2. Verify that an empty label tensor is treated as missing labels.
3. Verify that training raises a clear error when supervised prediction is
   enabled but labels are unavailable.

### Small workflow checks

1. Run a small labeled training example with `predict_cell_prop=True`.
2. Confirm that:
   - training completes
   - `predicted_cell_prop.csv` is produced during evaluation
   - predicted proportions differ from ground-truth labels unless the model
     fits perfectly
3. Run inference on an unlabeled dataset after training.
4. Confirm that:
   - inference succeeds without embedded labels
   - predicted cell proportions are saved
   - reconstruction outputs still appear normally

## Risks

The main risks are:

- re-enabling the supervised proportion loss may change training dynamics and
  require retuning the `cell_prop` loss weight
- some evaluation utilities may currently assume labels are always present for
  simulated datasets
- inconsistencies across encoder implementations can create mode-specific bugs
  if they are not normalized together

These risks are manageable because the required changes are localized and can
be covered with targeted tests.

## Recommendation

Recover the supervised cell proportion workflow in one focused pass rather than
trying to redesign all proportion-conditioning modes at the same time.

That means:

- keep `predict_cell_prop` as the current public switch
- restore direct supervision for the prediction head
- ensure inference truly uses predicted proportions after training
- leave broader multi-mode workflow cleanup for a later design if needed
