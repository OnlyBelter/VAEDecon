# Per-Sample Residual Variance Loss Design

Last updated: 2026-08-24

## Status

Drafted for review.

## Goal

Add a new auxiliary supervision term that helps prevent sample-specific cell-type GEP collapse by matching the **per-sample, per-cell-type residual variance across genes** between:

- matched SCT-derived targets, and
- inferred cell-type-specific GEPs reconstructed by the model.

The first implementation is intentionally narrow:

1. keep the existing `cross_sample_gene_var_weight` loss unchanged,
2. add the new loss alongside it,
3. support it only when `learn_gep_residual_mode='mean_centered'`.

## Why This Change

The current model can reconstruct the right approximate cell-type centroid while still collapsing the within-cell-type sample geometry. In practice this appears as:

- `true_vs_true` inter-sample similarity within a cell type being moderate,
- `recon_vs_recon` similarity being near 1.0,
- `true_vs_recon` remaining decent enough to hide the collapse in average per-sample metrics.

The existing `cross_sample_gene_var_weight` loss helps match per-`(gene, cell_type)` variability across samples, but it is still a marginal second-moment constraint. It does not directly penalize the regime where every sample gets nearly the same reconstructed profile within a cell type.

The new loss targets a different signal:

> for a given sample and cell type, how much does the whole inferred residual profile deviate from the reference mean across genes?

This is a direct anti-collapse cue for sample-specific amplitude of deviation.

## Statistic to Supervise

Let:

- `s` index samples,
- `g` index genes,
- `c` index cell types,
- `x_true[s, g, c]` be the matched true SCT GEP in scaled log space,
- `x_pred[s, g, c]` be the inferred GEP in scaled log space,
- `g_mean[g, c]` be the training/reference mean GEP already loaded by the model.

Define true and predicted residuals in scaled log space:

```text
r_true[s, g, c] = x_true[s, g, c] - g_mean[g, c]
r_pred[s, g, c] = x_pred[s, g, c] - g_mean[g, c]
```

Then define the per-sample residual variance across genes:

```text
v_true[s, c] = Var_g(r_true[s, g, c])
v_pred[s, c] = Var_g(r_pred[s, g, c])
```

This yields, for each cell type, a distribution over samples of residual magnitudes.

## Loss Definition

For the first version, supervise the paired scalar values directly rather than using KL divergence on a batch-level distribution estimate.

Recommended default:

```text
L_sample_var
= mean_{(s,c) in Omega}
  | log(v_pred[s, c] + eps) - log(v_true[s, c] + eps) |
```

Where:

- `Omega` is the set of active `(sample, cell_type)` pairs,
- `eps` is a small constant such as `1e-8`.

### Active mask

Use the same supervision mask logic already used for matched sctGEP supervision:

- `true_sct_gep_present_mask[s, c] == True`
- `true_cell_prop[s, c] >= training_sct_gep_cell_prop_threshold`

This keeps the new loss aligned with the existing matched-supervision policy.

## Why Not KL First

Although the scientific intuition is about matching a per-cell-type variance distribution across samples, the available supervision is already paired at the `(sample, cell_type)` level. That makes direct regression a better first implementation because it is:

- easier to debug,
- less batch-size-sensitive,
- easier to weight relative to other losses,
- more interpretable when training fails.

KL divergence or another distribution-matching term can be added later as a second-order regularizer if needed.

## Proposed Config Changes

### 1. `LossCoefficient`

Add:

```python
per_sample_residual_var_weight: float = 0.0
```

Semantics:

- default `0.0` keeps current behavior unchanged,
- non-negative validation matches the existing loss-weight pattern.

### 2. `ModelConfig`

Add:

```python
training_sct_per_sample_residual_var_fp: Optional[Path] = None
```

Semantics:

- required only when `per_sample_residual_var_weight > 0`,
- points to the precomputed target file described below.

### 3. Validation Rules

If `loss_coefficient.per_sample_residual_var_weight > 0`:

1. `learn_gep_residual_mode` must be `"mean_centered"`,
2. `training_sct_per_sample_residual_var_fp` must exist,
3. the file must align with the model's cell-type order.

Reason for restricting to `mean_centered`:

- the statistic is defined directly on scaled-log residuals around `g_mean`,
- that is the cleanest and least ambiguous implementation path,
- it avoids mixing z-score residual semantics with the new supervision target.

## Target File Format

New training artifact:

```text
training_sct_per_sample_residual_variance_log2p1_scaled_by_<factor>.csv
```

Format:

- rows = training/matched SCT sample IDs,
- columns = cell types in dataset order,
- values = per-sample residual variance across genes in scaled log space.

Definition for each entry:

```text
Var_g( x_true[s, g, c] - g_mean[g, c] )
```

Storage:

- save under the model output directory, alongside existing reusable training statistics,
- use `float_format='%g'`.

## Workflow Changes

In `vaedecon/workflow/train.py`, when training SCT targets are available and the new loss may be used:

1. compute the target table from the training SCT data after the same gene alignment and scaled-log transform used elsewhere,
2. save the CSV into the model directory,
3. pass the file path into `model_config.training_sct_per_sample_residual_var_fp`.

The computation should use the same gene order and cell-type order as:

- `dataset.gene_list`
- `dataset.cell_types`

No new branching is needed in the staged training workflow beyond allowing the new weight to be overridden like existing loss coefficients.

## Model Changes

### Target loading

During model initialization:

1. if the new weight is zero, skip loading the target file and skip the loss,
2. if the new weight is positive, load the CSV into a sample-indexed structure that can be queried by batch sample ID.

Because this target is sample-indexed rather than gene-indexed, the model must access the correct target rows for the current batch. The cleanest implementation is:

- keep the full target table outside the static `(G, C)` buffer pattern,
- retrieve batch-aligned rows by sample ID from the dataset outputs or batch metadata,
- move the selected tensor to device during loss computation.

This avoids pretending that a sample-indexed target is a fixed gene-statistics buffer.

### Loss computation

For the batch:

1. compute predicted residuals in scaled log space:

```text
r_pred = recon_residual_log
```

because in `mean_centered` mode the decoder already outputs the residual around `g_mean`.

2. compute:

```text
v_pred[s, c] = Var_g(r_pred[s, :, c])
```

3. retrieve matched target:

```text
v_true[s, c]
```

4. apply the active `(sample, cell_type)` mask,
5. compute masked log-L1 loss,
6. add:

```text
total += per_sample_residual_var_weight * L_sample_var
```

### Logging

Log a separate scalar metric:

```text
per_sample_residual_var_loss
```

This is important because the new loss is specifically intended to diagnose collapse.

## Relationship to Existing Losses

### Kept unchanged

- `cell_type_sct_gep_weight`
- `cross_sample_gene_var_weight`
- `gene_mean_weight`
- `gene_std_weight`

### New term's role

The new loss is complementary:

- `cell_type_sct_gep_weight` anchors gene-wise sample targets,
- `cross_sample_gene_var_weight` matches per-`(gene, cell_type)` variance across samples,
- `per_sample_residual_var_weight` matches the overall residual amplitude of each sample within each cell type.

This still does **not** fully preserve sample-sample geometry. It is a strong first anti-collapse signal, not a complete geometric constraint.

## Alternatives Considered

### 1. Replace `cross_sample_gene_var_weight`

Rejected for the first version.

Reason:

- the old and new terms supervise different moments,
- keeping both makes the ablation cleaner.

### 2. Distribution matching with KL

Deferred.

Reason:

- more complicated,
- batch-sensitive,
- unnecessary when paired scalar targets already exist.

### 3. Inter-sample similarity matrix loss

Promising but deferred.

Reason:

- this would directly attack the collapse seen in `inter_sample_similarity_ccc`,
- but it is a larger change in compute, implementation, and tuning.

If the new variance loss is not enough, similarity-preservation loss is the next design to try.

## Testing Plan

Add tests for:

1. config defaults and non-negative validation for `per_sample_residual_var_weight`,
2. validation failure when the weight is positive but:
   - `learn_gep_residual_mode != 'mean_centered'`,
   - the target file is missing,
3. CSV generation shape and alignment:
   - rows are sample IDs,
   - columns match cell-type order,
4. exact toy-value regression test for the masked log-variance loss,
5. mask behavior test to ensure only active `(sample, cell_type)` pairs contribute,
6. a backward-compatibility test confirming weight `0` leaves old configs functional.

## Success Criteria

The feature is successful if:

1. training remains stable with the new weight set to a small positive value,
2. `recon_vs_recon` inter-sample similarity within cell type decreases away from near-1.0 collapse,
3. `true_vs_recon` quality does not regress materially,
4. the new loss term is interpretable enough to tune from logged metrics alone.

## Recommended First Experiment

Start conservatively:

```yaml
model:
  learn_gep_residual_mode: mean_centered
  training_sct_per_sample_residual_var_fp: <auto-generated path>
  loss_coefficient:
    cell_type_sct_gep_weight: <existing value>
    cross_sample_gene_var_weight: <existing value>
    per_sample_residual_var_weight: 0.1
```

If the term is numerically much smaller than expected, increase gradually to `0.5` or `1.0`.
If it causes optimization conflict, keep it only in `reconstruction_training` and disable it in `joint_finetune` as an ablation.
