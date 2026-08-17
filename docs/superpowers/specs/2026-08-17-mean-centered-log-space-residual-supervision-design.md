# Mean-centered log-space residual supervision for SCT-GEP decoding

## Goal

Add a new residual-learning branch controlled by
`model.learn_gep_residual_mode: "mean_centered"` that:

1. keeps the current cell-type-specific mean GEP as the reference baseline,
2. drops the current z-score-style use of precomputed gene-wise std in the
   decoder reconstruction path,
3. predicts residuals in the same scaled log-expression space used by the
   model output, and
4. supervises those residuals directly against matched ground-truth SCT-GEP
   residuals in the same scaled log space.

The first implementation should support the new branch without changing the
current default behavior.

## Status

Proposed.

## Expression-space definitions

This spec uses two related expression spaces and keeps them distinct.

Let:

$$
x^{\log} = \log_2(\mathrm{CPM} + 1)
$$

be the unscaled log-expression space, and let:

$$
x^{\mathrm{slog}} =
\frac{\log_2(\mathrm{CPM} + 1)}{\mathrm{scaling\_factor}}
$$

be the scaled log-expression space used internally by the current model when
`scaling_by_constant=True`.

For consistency with the current implementation:

1. `g_mean` is stored in scaled log space;
2. `true_sct_gep` is cached in scaled log space when scaling is enabled;
3. `to_log_space(x_non_log, scaling_factor)` returns scaled log space;
4. `log_exp2cpm_tensor(...)` expects unscaled log space as input.

Unless explicitly stated otherwise, the new `mean_centered` branch is defined
in **scaled log space**.

## Motivation

The current `learn_gep_residual: true` path treats the decoder output as a
bounded z-score-like residual in non-log CPM space:

$$
\hat{x}_{g,c}^{\mathrm{cpm}} =
\mu_{g,c}^{\mathrm{cpm}} +
\hat{z}_{g,c} \cdot \sigma_{g,c}^{\mathrm{cpm}}
$$

This design has three practical issues for the current project.

First, the saved `*_std` values are a prior reference, not a directly observed
training target. They are helpful as a coarse scale prior, but they are not
required once matched ground-truth SCT-GEP supervision is available.

Second, the current matched SCT-GEP supervision is indirect with respect to the
residual representation. The model predicts a z-score-like output, converts it
through `mean + std * z`, renormalizes to CPM, converts back to scaled log
space, and only then compares against matched SCT-GEP targets. That makes the
supervision path longer than necessary.

Third, the z-score residual mixes two different ideas:

1. a structural prior about the baseline mean expression of each cell type, and
2. a scale normalization based on precomputed std.

For the current scientific goal, the first idea still looks useful, but the
second looks less necessary now that direct matched supervision exists.

## Design summary

Introduce a new config field:

```yaml
model:
  learn_gep_residual: true
  learn_gep_residual_mode: "mean_centered"
```

Semantics:

- `learn_gep_residual: false`
  - keep the current full-GEP decoding path unchanged
- `learn_gep_residual: true` with
  `learn_gep_residual_mode: "zscore"`
  - keep the current residual implementation unchanged
- `learn_gep_residual: true` with
  `learn_gep_residual_mode: "mean_centered"`
  - use the new branch defined in this spec

The new branch predicts a residual in scaled log space:

$$
r_{g,c}^{\mathrm{slog}} =
x_{g,c}^{\mathrm{slog}} - \mu_{g,c}^{\mathrm{slog}}
$$

where:

- $x_{g,c}^{\mathrm{slog}}$ is the target cell-type-specific expression in
  scaled log space,
- $\mu_{g,c}^{\mathrm{slog}}$ is the precomputed cell-type-specific mean in the
  same scaled log space.

The decoder predicts:

$$
\hat{r}_{g,c}^{\mathrm{slog}}
$$

and reconstructs the scaled-log-space SCT-GEP as:

$$
\hat{x}_{g,c}^{\mathrm{slog}} =
\mu_{g,c}^{\mathrm{slog}} + \hat{r}_{g,c}^{\mathrm{slog}}
$$

That reconstructed scaled-log-space SCT-GEP is then multiplied by
`scaling_factor` to recover unscaled log space before converting to CPM for
bulk mixing using the existing `log_exp2cpm_tensor(...)` path.

## Why use mean-centered residuals in scaled log space first

This branch should use mean-centered residuals in scaled log space, not raw CPM
residuals, for the first implementation.

Reasons:

1. residuals remain signed, so the branch can represent both up- and
   down-regulation around the reference mean;
2. scaled log space compresses large expression ranges and avoids the
   instability of direct TPM/CPM residual prediction;
3. the model already uses scaled log expression internally, so this branch fits
   the current architecture with minimal additional complexity;
4. matched SCT-GEP supervision targets are already represented in scaled log
   space, so direct residual supervision becomes straightforward.

This gives a much shorter supervision path:

$$
\hat{r}^{\mathrm{slog}} \leftrightarrow r_{\mathrm{true}}^{\mathrm{slog}}
$$

instead of the current:

$$
\hat{z} \rightarrow \hat{x}^{\mathrm{cpm}} \rightarrow
\hat{x}^{\mathrm{slog}} \leftrightarrow x_{\mathrm{true}}^{\mathrm{slog}}
$$

## Current behavior to preserve

The existing code path should remain available and unchanged unless the new
mode is explicitly enabled.

In particular:

1. `learn_gep_residual: false` remains the default;
2. the current `learn_gep_residual: true` behavior should be preserved under
   an explicit `"zscore"` mode;
3. existing configs that only set `learn_gep_residual: true` should remain
   backward compatible by defaulting to `"zscore"`.

## Proposed config changes

### New field

Add a new model config field:

```python
learn_gep_residual_mode: Literal["zscore", "mean_centered"] = Field(
    default="zscore",
    description=(
        "Residual decoding mode used when learn_gep_residual=True. "
        "'zscore' keeps the current std-scaled residual path; "
        "'mean_centered' predicts scaled-log-space residuals relative to the "
        "cell-type-specific mean."
    ),
)
```

### Validation rule

Add a config validator:

- if `learn_gep_residual` is `false`, the mode is ignored;
- if `learn_gep_residual` is `true`, the mode must be one of:
  - `"zscore"`
  - `"mean_centered"`

No other behavior change is required at config load time.

## Data and reference quantities

The new branch still depends on precomputed cell-type-specific **means**:

- `g_mean` in scaled log space
- `g_mean_non_log` only if another loss still needs it

It no longer needs precomputed `g_std` or `g_std_non_log` for:

1. the forward reconstruction path in `"mean_centered"` mode, or
2. the direct matched residual supervision target in that mode.

This is an important distinction:

- the new branch removes std from the **decoder reconstruction definition**
- it also removes std from the **direct residual supervision definition**
- it does not necessarily remove std from every legacy optional loss in the
  model, so those losses need explicit compatibility rules

## Forward-pass design

### Decoder output interpretation

For `"mean_centered"` mode, the decoder output is interpreted as an unbounded
residual in scaled log space:

$$
\hat{r}^{\mathrm{slog}} \in \mathbb{R}^{B \times G \times C}
$$

Unlike the current `"zscore"` mode:

- do not squash it into a bounded `(-3, 3)` range,
- do not compute `z_min`,
- do not multiply by `g_std_non_log`.

The output head for this mode should be linear.

### Reconstruction formula

Let:

- `recon_residual_log` be the decoder output in scaled log space,
- `g_mean` be the stored cell-type-specific mean in the same scaled log space.

Then:

$$
\hat{x}^{\mathrm{slog}} =
\hat{r}^{\mathrm{slog}} + \mu^{\mathrm{slog}}
$$

Implementation-level tensor shape:

```text
recon_residual_log:   (B, G, C)
g_mean:               (G, C)
recon_x_all_types_log:(B, G, C)
```

After that:

1. if `scaling_by_constant` is active, multiply `recon_x_all_types_log` by
   `scaling_factor` to recover unscaled log space before converting to CPM;
2. convert that unscaled log tensor to `recon_x_all_types_cpm` using the
   existing `log_exp2cpm_tensor(..., transpose=True)` path;
3. continue the current bulk mixing workflow unchanged.

This preserves consistency with the current full-GEP decoding path after the
cell-type-specific expression has been reconstructed.

## Direct matched residual supervision

### Target definition

The direct supervision target should be the matched ground-truth SCT-GEP
residual in scaled log space:

$$
r_{\mathrm{true}}^{\mathrm{slog}} =
x_{\mathrm{true}}^{\mathrm{slog}} - \mu^{\mathrm{slog}}
$$

where `x_true_log` is the matched training target already used by
`cell_type_sct_gep_weight`.

### Predicted quantity

The supervised prediction is:

$$
\hat{r}^{\mathrm{slog}}
$$

not the post-conversion CPM reconstruction and not a z-score-like quantity.

### Loss definition

For the first implementation, use the same masking policy already used by the
matched SCT-GEP supervision branch:

- require `true_sct_gep`
- require `true_sct_gep_present_mask`
- require true cell proportions `y`
- apply `training_sct_gep_cell_prop_threshold`

Then define the residual supervision loss in the same shape as the current
matched supervision:

```text
pred_residual_log: (B, G, C)
true_residual_log: (B, G, C)
active_mask:       (B, 1, C)
```

and compute:

$$
\mathcal{L}_{\mathrm{sct\_res}} =
\operatorname{mean}_{g,c}
\left[
\left(
\hat{r}_{g,c}^{\mathrm{slog}} -
r_{\mathrm{true},g,c}^{\mathrm{slog}}
\right)^2
\cdot m_c
\right]
$$

for the first version.

This means the new mode still uses `loss_coefficient.cell_type_sct_gep_weight`,
but the supervised quantity changes from reconstructed scaled log expression to
direct scaled-log-space residual.

### Why this is better aligned

This directly supervises the representation that the decoder is asked to
produce.

That removes the current mismatch where:

1. the decoder predicts a z-like variable,
2. the model converts it into a full GEP,
3. the loss is applied only after that conversion.

The new branch becomes:

1. predict residual in scaled log space,
2. supervise that same residual directly,
3. reconstruct full scaled-log-space GEP by adding the stored mean,
4. convert to CPM only for mixing and downstream losses.

## Relationship to the current matched SCT-GEP loss

The current helper `_matched_sct_gep_supervision_loss(...)` compares:

```text
to_log_space(recon_x_all_types_cpm, scaling_factor)  vs  true_sct_gep
```

No additional division by the scale factor is needed there. In the current
implementation:

1. `to_log_space(...)` already returns **scaled** log space, and
2. `true_sct_gep` is already cached in the same scaled log space during
   dataset preprocessing when `scaling_by_constant=True`.

So the existing helper already compares like with like in scaled log space.

The new branch should not overload that helper with hidden mode-dependent
behavior if doing so would make the function harder to reason about.

Preferred first implementation:

1. keep the current helper for full-GEP or `"zscore"` paths;
2. add a new helper for direct residual supervision, for example:

```python
_matched_sct_gep_residual_supervision_loss(
    pred_residual_log,
    true_sct_gep,
    true_sct_gep_present_mask,
    true_cell_prop,
    cell_prop_threshold,
)
```

That helper can derive `true_residual_log = true_sct_gep - g_mean` internally.

This separation keeps both paths explicit and easier to test.

## Interaction with other losses

### Bulk reconstruction loss

Keep it unchanged.

Even in `"mean_centered"` mode, bulk reconstruction is still computed from the
reconstructed cell-type-specific CPM outputs mixed by cell proportions.

### Cross-sample gene-variance loss

Keep it unchanged for the first implementation.

It already operates on:

```text
recon_x_all_types_log
```

which remains available in the new branch after reconstructing:

$$
\hat{x}^{\mathrm{slog}} =
\mu^{\mathrm{slog}} + \hat{r}^{\mathrm{slog}}
$$

This makes `cross_sample_gene_var_weight` naturally compatible with the new
mode.

### Gene mean / std regularization losses

For `learn_gep_residual_mode == "mean_centered"`, the first implementation
should not use the z-score-based regularization path at all.

In particular, the following must be treated as incompatible with this mode:

1. `z_score_kl_weight`
2. `z_score_reg_weight`

Recommended first-version policy:

- if either is nonzero while `learn_gep_residual_mode == "mean_centered"`,
  raise a clear config or runtime error instead of silently applying a loss
  with changed semantics.

This keeps the first version conceptually clean and avoids mixing the new
mean-centered branch with the old z-score-specific regularizers.

## Output and logging behavior

The model output should expose enough tensors to debug the new branch clearly.

Recommended additions:

1. `recon_residual_log`
2. `recon_x_all_types_log`

These do not need to be saved by default, but they should be available in the
forward/loss path for tests and future diagnostics.

Metrics:

- keep the existing `cell_type_sct_gep_loss` metric name for continuity, even
  though the supervised quantity becomes the residual in this mode;
- document in code comments and spec that the meaning of that term is
  mode-dependent:
  - full scaled-log-expression supervision in the existing path
  - direct scaled-log-residual supervision in `"mean_centered"` mode

## Non-goals for the first implementation

The first version should not:

1. introduce raw TPM residual prediction;
2. introduce a learned variance head;
3. replace the existing `"zscore"` branch;
4. redesign the cross-sample variance loss;
5. retune all existing loss coefficients automatically.

Those can be follow-up experiments after the branch exists and is testable.

## Backward compatibility

This change should be fully backward compatible.

Rules:

1. existing configs with `learn_gep_residual: false` behave exactly as before;
2. existing configs with `learn_gep_residual: true` and no mode field default
   to `"zscore"`;
3. new configs can explicitly select `"mean_centered"` without affecting old
   runs.

## Implementation outline

1. **Config**
   - add `learn_gep_residual_mode`
   - add validation and documentation

2. **Forward branch**
   - branch inside the current `learn_gep_residual` path on:
     - `"zscore"`
     - `"mean_centered"`
   - compute `recon_x_all_types_log = recon_residual_log + g_mean` for the new
     mode in scaled log space
   - convert back to unscaled log space before CPM mixing

3. **Supervision**
   - add a direct residual supervision helper
   - use it only when:
     - `learn_gep_residual_mode == "mean_centered"`
     - matched SCT-GEP supervision is enabled

4. **Guard rails**
   - reject incompatible z-score regularizers in `"mean_centered"` mode

5. **Tests**
   - config parsing and backward compatibility
   - forward-pass tensor-shape coverage
   - direct residual target construction
   - masking behavior
   - incompatibility checks for z-score-only regularizers

## Testing plan

The first implementation should add regression tests for:

1. config defaults:
   - missing mode + `learn_gep_residual: true` resolves to `"zscore"`

2. explicit new mode:
   - `"mean_centered"` config loads correctly

3. forward reconstruction:
   - in `"mean_centered"` mode the branch reconstructs scaled-log-space
     cell-type GEPs as `residual + mean`

4. matched supervision:
   - residual loss target equals `true_sct_gep - g_mean`
   - masking by cell-type presence and cell-proportion threshold still works

5. guard rails:
   - nonzero `z_score_kl_weight` or `z_score_reg_weight` fails clearly in the
     new mode

## Recommended first experiment

After implementation, the first comparison should be:

1. `learn_gep_residual: false`
2. `learn_gep_residual: true`, `learn_gep_residual_mode: "zscore"`
3. `learn_gep_residual: true`,
   `learn_gep_residual_mode: "mean_centered"`

All three should keep the same:

- bulk training data,
- matched SCT-GEP supervision setup,
- loss coefficients,
- checkpoint selection rule.

That will isolate whether the new residual representation itself improves
held-out SCT-GEP accuracy.
