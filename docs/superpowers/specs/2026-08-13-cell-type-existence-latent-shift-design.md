# Design: cell-type existence latent shift

Last updated: 2026-08-13

## Status

Proposed.

## Goal

Add a training-supervised existence signal that reuses the model's predicted
cell proportions to estimate whether each cell type is effectively present in a
mixed bulk sample and uses that signal to shift the corresponding latent mean.
The goal is not only to classify cell-type existence, but also to encourage
the model to represent low-proportion and high-proportion states differently
in latent space.

## User requirements

The user asked for a design that mirrors the role of the hierarchical code
head, but focuses on cell-type existence:

1. For each cell type, derive a binary existence target from the true cell
   proportion and a threshold.
2. Reuse the existing workflow for predicting cell proportions and transform
   that output into an existence indicator based on a threshold.
3. Apply a sigmoid-style soft thresholding step to the predicted cell
   proportions and add the resulting scalar back to the latent mean so the
   representation is shifted differently for absent and present cell types.
4. Keep the new feature internal for now. It is a training objective and
   representation mechanism, not a new user-facing prediction output.

During design discussion, the centered shift variant was chosen over adding a
raw positive sigmoid value directly. This avoids pushing all latent means in
the same direction and more clearly separates the two states.

## Motivation

The current model already learns cell-type-specific latent means
`mu_types in R^(B x L x C)` and uses them for reconstruction, deconvolution,
and several auxiliary objectives such as repulsion, attractor, and
hierarchical-code losses. However, there is no explicit training signal that
encourages the latent representation for a given cell type to differ depending
on whether that cell type is materially present in the mixed bulk input.

Today, `EncoderResMLP` contains an optional fixed positional-encoding shift
that is triggered when cell proportions exceed a hard-coded threshold of
`0.01`. That mechanism injects existence-like information, but:

- it is not learned,
- it is tied to encoder internals,
- it uses a fixed threshold and fixed shift pattern, and
- it is not explicitly supervised as a binary existence task.

The proposed feature turns this idea into a learned, supervised latent-space
modulation that can better separate "present" and "not present" states for
each cell type, while reusing the existing cell-proportion pathway instead of
adding another predictor head.

## Non-goals

- Add a new inference output file or visualization by default.
- Change the existing matched `sctGEP` supervision workflow.
- Remove or redesign the existing positional-encoding feature in this same
  change.
- Introduce a hard binary gate on decoding.

This design intentionally adds a soft residual shift, not a discrete gate.

## Current behavior

### Cell-type latent means

The default `EncoderResMLP` maps the mixed-bulk input to
`mu_all_types` and `logvar_all_types`, each with shape `(B, L, C)`. If more
than one encoder is configured, `VAE.forward()` fuses these posteriors into a
single `mu_types` and `log_var_types` tensor before sampling and decoding.

### Auxiliary hierarchical supervision

`VAE` already contains one auxiliary head:

- `hierarchical_code_head = nn.Linear(latent_dim, 8)`

It reads `mu_types`, predicts an 8-bit hierarchy code for each cell type, and
contributes `hierarchical_code_loss` to the total objective.

### Existing existence-like encoder shift

Inside `EncoderResMLP.forward()`, if `using_positional_encoding` is enabled and
cell proportions are available, the encoder computes:

- `exists = (cell_prop >= 0.01).float()`

and then adds a fixed position-encoding vector to `mu_all_types` for those
cell types. This behavior is related to the new proposal, but it is fixed,
encoder-local, and not directly supervised by a dedicated loss.

## Proposed change

The proposed feature adds a learned existence-conditioned residual shift on the
cell-type latent means. This section describes the head, the target, the loss,
and where the shift enters the forward pass.

## Overview

Reuse the predicted cell proportions already produced by the current model,
convert them into a soft existence score around the configured threshold, and
add a centered residual shift back to the fused latent means before decoding
and downstream loss evaluation.

The existence signal remains supervised only during training, when true cell
proportions are available. At inference time, the model still computes the
existence probability and latent shift from predicted proportions, but no
existence loss is applied.

## Existence signal construction

The new feature does not add a separate existence head. Instead, it reuses the
existing `cell_prop` output:

```text
pred_cell_prop: (B, C)
```

Convert the predicted proportions to a soft existence logit and probability:

```text
existence_logits = (pred_cell_prop - threshold) / max(threshold, eps)
existence_probs: (B, C) = sigmoid(existence_logits)
```

This keeps the existence mechanism tied directly to the model's own estimate of
cell-type abundance while avoiding a duplicate prediction branch.

## Centered residual shift

Use a centered residual shift rather than a raw positive addition:

```text
shift = cell_type_existence_shift_scale * (existence_probs - 0.5)
mu_types_shifted = mu_types + shift.unsqueeze(1)
```

This design is preferred over:

```text
mu_types + existence_probs
```

for two reasons:

1. It separates absent and present states in opposite directions instead of
   only pushing present states upward.
2. It reduces systematic upward bias in `mu_types`, which is more compatible
   with the VAE prior and existing KL regularization.

The shifted tensor `mu_types_shifted` becomes the latent mean used for:

- latent sampling,
- reconstruction and deconvolution,
- repulsion loss,
- attractor loss,
- hierarchical-code loss,
- and any other downstream logic that already consumes `mu_types`.

The existence score is derived from `pred_cell_prop`, while the resulting shift
is applied to `mu_types` after posterior fusion.

## Target definition

Derive the training target from the true cell proportions:

```text
exist_target[b, c] = 1 if y[b, c] >= threshold else 0
```

To keep the change small and backward compatible, reuse the existing
`data.training_sct_gep_cell_prop_threshold` value as the threshold source for
this new existence target. This field is already the training-side threshold
used for masking matched `sctGEP` supervision, so it is a natural fit for
defining whether a cell type is materially present in a training sample.

This means the threshold now has two aligned training-time uses:

1. mask matched `sctGEP` supervision, and
2. define the binary target for the existence head.

## Threshold semantics

Two threshold concepts matter here, even though the current implementation
reuses one config value for both.

### Matched `sctGEP` masking threshold

This threshold decides when direct matched `sctGEP` supervision is meaningful
enough to include in the loss. If a cell type has a very small true proportion,
its contribution to the mixed bulk may be nonzero in simulation, but still too
small to provide a stable, informative supervision target.

### Cell-type existence threshold

This threshold decides when a cell type is treated as effectively "present" for
the existence objective and latent shift. This is a representation-learning
decision, not a strict statement that the simulated mixture contribution is
exactly zero or nonzero.

### Brief difference

In short:

- the matched `sctGEP` masking threshold asks, "Is this cell type large enough
  to supervise directly?";
- the cell-type existence threshold asks, "Is this cell type large enough to be
  treated as materially present?"

These two thresholds can be different in principle. The simulation floor, for
example `0.0001`, only means the contribution is mathematically nonzero. It
does not automatically mean that the signal is strong enough for direct
supervision or that the model should treat the cell type as clearly present in
latent space.

For the first implementation, the code keeps things simple by reusing
`data.training_sct_gep_cell_prop_threshold` for both decisions. If later
experiments show that these semantics need to diverge, the two thresholds can
be separated into distinct config fields.

## Loss term

Add a new optional loss coefficient:

```python
cell_type_existence_weight: float = 0.0
```

When this weight is positive and labels are available, compute a binary
cross-entropy loss with logits:

```text
existence_loss = BCEWithLogits(existence_logits, exist_target)
```

Reduce the loss by averaging over cell types for each sample, returning a
per-sample vector shaped `(B,)`, then aggregate it into the total loss in the
same style as other auxiliary losses.

When `cell_type_existence_weight == 0`, the existence feature is disabled and
the current workflow remains unchanged.

Because the existence signal is derived from `pred_cell_prop`, this feature is
defined only when `predict_cell_prop=True`.

## Forward-path placement

Apply the existence shift in `VAE.forward()` after encoder fusion and before
reparameterization:

1. Run encoders and collect `mu_all_types`, `logvar_all_types`, and
   `cell_prop`.
2. Fuse encoder outputs into `mu_types`, `log_var_types`, and `cell_prop`.
3. Compute `existence_logits` and `existence_probs` from `cell_prop`.
4. Build `mu_types_shifted` by adding the centered residual shift.
5. Sample `z_types` from `mu_types_shifted` and `log_var_types`.
6. Use `mu_types_shifted` for downstream auxiliary losses.

This placement has two advantages:

- it works for both single-encoder and multi-encoder configurations, and
- it reuses the existing cell-proportion branch instead of introducing a
  separate existence predictor.

## Inference behavior

Inference must remain robust and largely unchanged from the user perspective.

The model will still compute the existence score from predicted proportions and
apply the latent shift at inference time, because that shift is part of the
learned representation. This avoids a train-inference mismatch.

However, the loss term is training-only:

- if labels are available during training and
  `cell_type_existence_weight > 0`, compute the supervised existence loss;
- if `cell_type_existence_weight > 0` but `predict_cell_prop=False`, raise a
  clear configuration error, because the reused existence path does not exist;
- if labels are missing during training and the weight is positive, raise a
  clear error, because the requested supervision target cannot be built;
- if labels are missing during inference or prediction, skip the existence loss
  and return zero for that term, similar to the existing inference-safe
  behavior for training-only matched `sctGEP` supervision.

## Configuration changes

This feature only needs two new config fields: one loss weight to enable the
supervision term, and one small scalar to control the magnitude of the latent
shift.

## New field

Add to `LossCoefficient`:

```python
cell_type_existence_weight: float = 0.0
```

## New field

Add to `ModelConfig`:

```python
cell_type_existence_shift_scale: float = 0.0
```

This scale controls how strongly the predicted existence probability shifts the
latent mean. A small default is recommended because the new shift operates
directly in latent space and will interact with reconstruction and KL terms.

## Backward compatibility

The change remains backward compatible because:

- existing configs do not need to set either field,
- `cell_type_existence_weight` defaults to `0.0`,
- `cell_type_existence_shift_scale` defaults to `0.0`,
- the new logic becomes a no-op when the feature is disabled, and
- old trained models can still run because the new parameters are introduced as
  optional defaults rather than required inputs.

## Implementation plan

The code change is intentionally small and local because it reuses existing
cell-proportion outputs.

### `vaedecon/configs/default_config.py`

- Add `cell_type_existence_weight` to `LossCoefficient`.
- Add `cell_type_existence_shift_scale` to `ModelConfig`.
- Include both fields in non-negative validation.

### `vaedecon/configs/example_config.yaml`

- Document the new loss weight under `model.loss_coefficient`.
- Document the new shift scale under `model`.
- Reuse the existing `training_sct_gep_cell_prop_threshold` comment to note
  that it also defines the existence target.

### `vaedecon/models/vae/vae_model.py`

- Extend `LossTerms` with `cell_type_existence`.
- Add a helper to compute:
  - existence logits from predicted cell proportions,
  - existence probabilities,
  - centered residual shift,
  - and the shifted latent mean.
- Add a helper to compute supervised existence loss from true cell fractions.
- Update `forward()` so the shifted latent mean is used for decoding and
  downstream auxiliary losses.
- Update `loss_function()` to:
  - accept existence logits,
  - build the binary target from `y` and
    `data.training_sct_gep_cell_prop_threshold`,
  - compute existence loss only when supervision is available,
  - and skip the loss cleanly during inference.

### `vaedecon/models/nn/res_mlp.py`

Do not change encoder architecture in the first implementation. The new
existence shift lives in `VAE`, after posterior fusion, so the existing encoder
API remains stable.

The current optional positional-encoding shift remains untouched in this
change. That keeps the implementation surgical and avoids expanding the scope.

## Testing plan

Add focused regression tests that verify:

1. the existence loss is positive and finite when labels are present and the
   feature is enabled;
2. the existence loss is zero during inference when labels are absent;
3. the shifted latent mean differs from the raw latent mean when the feature is
   enabled;
4. the disabled path remains backward compatible when
   `cell_type_existence_weight == 0`;
5. the model raises a clear training-time error if the user enables the
   existence loss but omits cell-fraction labels; and
6. the model raises a clear configuration-time error if the user enables the
   existence feature while `predict_cell_prop=False`.

## Risks and mitigations

### Latent bias

If the shift scale is too large, the model may overuse the existence head and
distort the latent geometry. A small default scale and a separate supervised
loss weight reduce this risk.

### Overlap with positional encoding

If `using_positional_encoding` is also enabled, the model may receive two
existence-related latent shifts. This design does not change that behavior, but
the new feature should be evaluated first with positional encoding disabled so
the effect is interpretable.

### Threshold semantics

Reusing `training_sct_gep_cell_prop_threshold` keeps the design small, but it
means one threshold controls both masking and existence targets. This is
acceptable because both uses express the same training-time notion of material
presence.

## Success criteria

The feature is complete when:

1. the model can learn a supervised existence signal from predicted cell
   proportions and true cell fractions;
2. the derived existence probability produces a centered residual shift in
   `mu_types`;
3. inference remains functional without requiring labels or new training-only
   tensors; and
4. existing configs and tests continue to work when the feature is disabled.
