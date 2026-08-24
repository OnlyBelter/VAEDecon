# Cell-type-wise inter-sample similarity loss design

Last updated: 2026-08-24

## Status

Drafted for review.

## Goal

Add a new auxiliary supervision term that directly discourages
sample-specific cell-type GEP collapse by matching the **within-cell-type,
inter-sample similarity structure** between:

- matched SCT-derived residual GEPs, and
- inferred residual GEPs reconstructed by the model.

The first implementation is intentionally narrow:

1. support only `learn_gep_residual_mode='mean_centered'`,
2. compute the similarity target from the current batch only,
3. use concordance correlation coefficient (CCC) as the similarity measure,
4. apply cell-type-wise masking using both target availability and cell
   proportion thresholding.

## Why this change

The current collapse failure mode is not mainly a wrong centroid problem. The
model can produce:

- decent `true_vs_recon` agreement,
- extremely high `recon_vs_recon` similarity within a cell type,
- much lower `true_vs_true` inter-sample similarity for the same cell type.

That means the model is learning an approximate average profile for each cell
type while failing to preserve the true variation across samples.

The existing losses do not directly supervise this geometry:

- matched `sctGEP` supervision anchors each sample to its own target,
- `cross_sample_gene_var_weight` constrains marginal gene-wise variance across
  samples,
- `per_sample_residual_var_weight` constrains sample-wise residual amplitude.

None of them explicitly tells the model that if two samples are similar in the
true residual space, they should also be similar in the predicted residual
space, and likewise for dissimilar pairs.

This new loss is designed to target that missing structure directly.

## Mask semantics

One clarification is important because the current naming can be misleading.

`true_sct_gep_present_mask` does **not** come from the cell-proportion
threshold. It indicates whether a matched SCT target exists for a given
`(sample, cell_type)` pair after aligning:

- the bulk sample,
- the sample-to-cell mapping table,
- and the matched SCT GEP dataset.

The actual active supervision mask is formed later during loss computation:

```text
active_mask = true_sct_gep_present_mask & (true_cell_prop >= threshold)
```

So in this design:

- `true_sct_gep_present_mask` means "matched target exists",
- `true_cell_prop >= threshold` means "cell type is active enough to supervise",
- the final similarity loss uses the intersection of both.

## Statistic to supervise

Let:

- `s` index samples in the current batch,
- `g` index genes,
- `c` index cell types,
- `x_true[s, g, c]` be the matched true SCT GEP in scaled log space,
- `x_pred[s, g, c]` be the inferred GEP in scaled log space,
- `g_mean[g, c]` be the reference mean GEP already loaded by the model.

Define mean-centered residuals:

```text
r_true[s, g, c] = x_true[s, g, c] - g_mean[g, c]
r_pred[s, g, c] = x_pred[s, g, c] - g_mean[g, c]
```

In `mean_centered` mode, the decoder already predicts:

```text
r_pred = recon_residual_log
```

For each cell type `c`, define the active sample set within the batch:

```text
A_c = { s :
  true_sct_gep_present_mask[s, c] = 1
  and true_cell_prop[s, c] >= training_sct_gep_cell_prop_threshold
}
```

Only if `|A_c| >= 2`, compute a sample-sample similarity matrix for that cell
type using CCC over genes:

```text
S_true^(c)[i, j] = CCC(r_true[i, :, c], r_true[j, :, c])
S_pred^(c)[i, j] = CCC(r_pred[i, :, c], r_pred[j, :, c])
```

These matrices describe the within-cell-type geometry across samples for the
current batch.

## Loss definition

For each active cell type `c`, compute the loss only on off-diagonal entries:

```text
L_sim^(c) = mean_{i != j in A_c} |S_pred^(c)[i, j] - S_true^(c)[i, j]|
```

Then average over the cell types that have at least two active samples:

```text
L_sim = mean_c L_sim^(c)
```

If no cell type in the batch has at least two active samples, return zero for
that batch.

### Why off-diagonal only

The diagonal of a CCC similarity matrix is always 1 for valid self-pairs, so it
does not add useful supervision. Using only off-diagonal entries keeps the loss
focused on actual inter-sample structure.

### Why CCC

CCC matches the current diagnostic lens already used in
`inter_sample_similarity_ccc`, so it keeps the training objective aligned with
the main collapse readout that motivated this design.

It is more complex than correlation or cosine similarity, but here that
complexity is justified because:

- it is already part of the evaluation language of the project,
- it captures both correlation and scale agreement,
- and it makes the training target directly interpretable against existing
  outputs.

## Why batch-local matrices first

This design uses the current batch rather than a precomputed full training-set
matrix for three reasons:

1. no new large matrix artifact is needed,
2. it reuses the matched targets already present in training batches,
3. it keeps the first implementation smaller and easier to debug.

The tradeoff is that the target is batch-composition-dependent, so the loss can
be noisier when few valid samples for a cell type appear in a batch.

That is acceptable for the first version because the goal is to add a direct
anti-collapse signal with minimal new infrastructure.

## Proposed config changes

### 1. `LossCoefficient`

Add:

```python
inter_sample_similarity_weight: float = 0.0
```

Semantics:

- default `0.0` keeps current behavior unchanged,
- non-negative validation matches the existing loss-weight pattern.

### 2. No new file path in the first version

Because this design uses batch-local true matrices computed from the matched
targets already in memory, the first version does **not** need a new saved
training artifact path.

## Validation rules

If `loss_coefficient.inter_sample_similarity_weight > 0`:

1. `learn_gep_residual` must be `True`,
2. `learn_gep_residual_mode` must be `"mean_centered"`,
3. matched training targets must be available in the dataset,
4. training batches must include:
   - `true_sct_gep`,
   - `true_sct_gep_present_mask`,
   - `labels` containing true cell proportions.

Reason for restricting to `mean_centered`:

- the true and predicted objects are then defined in the same residual space,
- the residual interpretation is direct and biologically meaningful,
- it avoids mixing z-score residual semantics into a similarity loss that is
  intended to diagnose sample geometry.

## Model changes

### Loss computation inputs

Reuse the existing batch fields:

- `recon_residual_log`,
- `true_sct_gep`,
- `true_sct_gep_present_mask`,
- `labels` as `true_cell_prop`.

No new sample ID lookup or precomputed matrix loading is required.

### Core algorithm

For each batch:

1. compute true residuals:

```text
true_residual_log = true_sct_gep - g_mean
```

2. for each cell type, build the active sample mask:

```text
active_mask[:, c] = true_sct_gep_present_mask[:, c] & (true_cell_prop[:, c] >= threshold)
```

3. if fewer than 2 samples are active for a cell type, skip it,
4. otherwise gather:

```text
R_true^(c)  in R^(n_active x n_genes)
R_pred^(c)  in R^(n_active x n_genes)
```

5. compute the pairwise CCC matrix for `R_true^(c)` and `R_pred^(c)`,
6. compute off-diagonal L1 difference between the two matrices,
7. average across active cell types,
8. add to total loss:

```text
total += inter_sample_similarity_weight * L_sim
```

### Numerical notes

- clamp CCC denominator terms with a small epsilon to avoid divide-by-zero,
- if a residual vector is constant across genes, CCC can become unstable, so
  the implementation must explicitly guard the denominator,
- if a cell type has exactly 2 active samples, the loss still works and reduces
  to a single off-diagonal pair value.

## Relationship to existing losses

### Kept unchanged

- `cell_type_sct_gep_weight`
- `cross_sample_gene_var_weight`
- `per_sample_residual_var_weight`
- `gene_mean_weight`
- `gene_std_weight`

### New term's role

The new term is complementary:

- matched `sctGEP` loss anchors gene-wise sample targets,
- `cross_sample_gene_var_weight` constrains marginal gene variability,
- `per_sample_residual_var_weight` constrains sample-wise residual amplitude,
- `inter_sample_similarity_weight` constrains the within-cell-type sample
  geometry.

This is the first loss in the stack that directly supervises the
sample-sample structure that appears collapsed in the current outputs.

## Alternatives considered

### 1. Precomputed full training-set similarity matrices

Deferred.

Reason:

- would require storing and indexing a matrix per cell type,
- adds more infrastructure than is needed for the first attempt,
- makes the training/data contract heavier.

### 2. Cosine or Pearson similarity instead of CCC

Deferred.

Reason:

- simpler to compute,
- but less aligned with the current evaluation outputs and user-facing failure
  diagnosis.

If CCC proves too unstable or expensive, Pearson correlation is the next
fallback to try.

### 3. Neighbor-only or top-k similarity supervision

Deferred.

Reason:

- may reduce noise and compute,
- but introduces more hyperparameters and design choices.

## Testing plan

Add tests for:

1. config defaults and non-negative validation for
   `inter_sample_similarity_weight`,
2. validation failure when the weight is positive but
   `learn_gep_residual_mode != 'mean_centered'`,
3. exact toy-value regression test for the pairwise CCC matrix loss,
4. masking test to ensure low-proportion or missing-target samples are excluded,
5. skip behavior when a cell type has fewer than 2 active samples,
6. backward-compatibility test confirming weight `0` leaves old configs
   functional.

## Success criteria

The feature is successful if:

1. `recon_vs_recon` inter-sample similarity within cell type decreases away
   from near-1.0 collapse,
2. `true_vs_recon` quality does not regress materially,
3. cell proportion performance stays stable,
4. the new metric is numerically stable enough to tune from logs.

## Recommended first experiment

Start conservatively:

```yaml
model:
  learn_gep_residual_mode: mean_centered
  loss_coefficient:
    cell_type_sct_gep_weight: <existing value>
    cross_sample_gene_var_weight: <existing value>
    per_sample_residual_var_weight: <existing value>
    inter_sample_similarity_weight: 0.01
```

Recommended staging policy:

- keep the weight at `0.0` in `cell_prop_predictor_pretrain`,
- enable it in `reconstruction_training`,
- keep it small or disable it in `joint_finetune` for the first ablation.

If the signal is too weak, increase gradually to `0.05` or `0.1`.
If CCC-based optimization proves unstable, keep the same design but switch the
similarity operator to Pearson correlation in a follow-up spec.
