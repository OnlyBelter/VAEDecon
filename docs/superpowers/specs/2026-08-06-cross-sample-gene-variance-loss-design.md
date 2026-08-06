# Cross-sample gene variance loss for VAEDecon + simulation weight knobs

Last updated: 2026-08-06

## Status
Approved for implementation.

## Goal / Motivation

The existing `gene_std_weight` loss already matches the per-(gene, cell_type) marginal standard deviation computed on the batch of reconstructed cell-type-specific GEPs to the reference mean/std buffers loaded from the SCT dataset used for gene-mean/std computation. However, it does **not** preserve sample-level GEP identity, because matching only marginal std is compatible with a “collapsed centroid” regime: the decoder can still output one representative GEP for all samples while producing a similar population-level std across samples by scaling noise.

This design adds a **second, conceptually cleaner loss term** that is explicitly about sample-level variability: a cross-sample gene-variance match. The new weight is named `cross_sample_gene_var_weight`. The name is chosen deliberately to avoid confusion with the existing `gene_std_weight` (marginal std) and the planned but rejected generic `gene_variance_weight` (which was ambiguous).

This spec also documents the user’s requested parameter settings for the next experiment:
1. SimuTME `sorted_bulk_imputation_bulk_weight = 0.01`
2. VAEDecon `hierarchical_code_weight = 0.1` and `z_score_kl_weight = 0.1`
3. VAEDecon new `cross_sample_gene_var_weight > 0` (trainable knob)
4. Use the **same SCT dataset(s)** for training (`data.sct_file_path`) and as the source used to compute gene mean/std (`data.gene_mean_std_sct_gep_file_path`), so the reference statistics are consistent with the targets the model sees.

## Background / Existing behavior

### Existing `gene_std_weight` (kept, no semantic change)
`LossCoefficient.gene_std_weight` and `gene_mean_weight` drive:
```
gm_loss = MSE(recon_gene_mean, g_mean)   # (G,C) -> scalar
gs_loss = MSE(recon_gene_std, g_std)     # (G,C) -> scalar
```
where `recon_gene_mean/std` are computed as `mean(dim=0)`/`std(dim=0)` over the B samples in the current batch at `(B, G, C)`. This matches the **marginal gene statistics** of the reconstructed GEP population.

Buffers `g_mean`, `g_std` are loaded from the CSV at `model_config.gene_mean_std_fp` which is produced by `load_or_compute_gene_mean_std` on the SCT reference path.

### Problem with only marginal std matching
Even if `gs_loss` matches, the decoder can output a collapsed/centroid GEP for every sample and still achieve the target population-level std by increasing residual noise or batch-level scale. The CCC heatmaps in `inter_sample_similarity_ccc/` show exactly this collapse: `recon_vs_recon` off-diagonal CCC is near 0.9–0.99 for many cell types even when `true_vs_true` shows substantially lower off-diagonal CCC.

## Proposed change (minimal, backward compatible)

### 1. Config interface

#### LossCoefficient additions
Add:
```python
cross_sample_gene_var_weight: float = 0.0
```
Validated non-negative with existing `non_negative` validator.

This is the **only new config knob**. Do **not** add `gene_variance_weight`; that name is too easily confused with `gene_std_weight`.

`gene_mean_weight` / `gene_std_weight` remain exactly as they are today (marginal stat matching).

#### LossCoefficient reconciliations / sanity checks
No coupling is needed between `gene_std_weight` and `cross_sample_gene_var_weight`. They target different quantities. If both are set to zero, only the other existing losses (recon MSE, KLD, etc.) remain active, preserving the current baseline.

### 2. Training workflow: compute training-SCT cross-sample gene variance

In `vaedecon/workflow/train.py`, after `dataset` is created and `input_gene_list_fp` / `cell_type_fp` are known, compute one supplementary statistic:
- For every `(gene, cell_type)`, compute the cross-sample variance of the **ground-truth SCT GEPs** used for training, i.e. loaded from `config.data.sct_file_path`. When `sct_file_path` contains multiple `.h5ad`s, concatenate them (sample axis union) and compute variance over the combined sample set.

Output file (for reproducibility and portability, under the model folder):
```
model.model_dir / "training_sct_cross_sample_gene_variances.csv"
```

Format:
- Rows = gene names (same column order/filter as `dataset.gene_list`)
- Columns = cell types (same order as `dataset.cell_types`)
- Values = variance (preferably `ddof=1` sample variance) computed on the SCT samples after alignment to the same log2cpm1p + scaling pipeline as training.

Alignment rules:
1. Use `gene_list` from the GEPDataset as the intersection set. If a gene is missing in an SCT `.h5ad`, treat it as NaN and drop genes that are not present in all SCT files. Alternatively, fill missing with the mean over the available SCT samples; whichever keeps the order identical to `dataset.gene_list` is fine (to guarantee shape matches the model buffers).
2. Cell type order must match `dataset.cell_types`.
3. If a cell type is missing in a given SCT file, variance for `(gene, cell_type)` is computed only from the SCT datasets containing the cell type.

### 3. VAE model: load variance buffer + add loss

In `VAEDeconModel.__init__` / `_load_gene_statistics` path (around where `g_mean`, `g_std` are loaded), register a new persistent buffer:
```
training_sct_cross_sample_gene_var : (G, C) tensor (float32)
```
If `model_config.loss_coefficient.cross_sample_gene_var_weight == 0` **and** the CSV does not exist, allow initializing it to zeros and skip the loss entirely (so old checkpoints/configs still load without changing behavior).

#### Loss term definition
Let `recon_x_all_types_cpm` have shape `(B, G, C)`. Define:
```
recon_cross_sample_var[g, k] = Var_s( recon_x_all_types_cpm[:, g, k] )   # (G, C)
```
using unbiased variance (`unbiased=True` for `torch.var(..., unbiased=True)`, i.e. `ddof=1`) to match the population estimate computed on the training SCT.

Then:
```
loss_cross_sample_gene_var = sum_{g, k} | recon_cross_sample_var[g, k] - training_sct_cross_sample_gene_var[g, k] |
```
Sum over both axes, then divide by `(G * C)` or by `C` so the magnitude is comparable across `n_cell_types` choices; divisor must be explicit and documented. Default divisor is `(G * C)` (mean absolute error per gene per cell type).

#### Loss registration in total loss
Append:
```
total += cross_sample_gene_var_weight * loss_cross_sample_gene_var
```
to the total scalar loss at `vae_model.py:loss_function`.

Also log `loss_cross_sample_gene_var` as a separate prog-bar-friendly metric, like other losses already are.

### 4. Backward compatibility

- Old configs without `cross_sample_gene_var_weight`: default 0 => behavior unchanged.
- Old checkpoints that do not have the new buffer: if the CSV companion exists, load it; if not, buffer is zeros and loss term is skipped for 0-weight case, so loading still succeeds. If a user sets `cross_sample_gene_var_weight > 0` using an old checkpoint without the CSV, raise a clear error: “training_sct_cross_sample_gene_variances.csv is required under model_dir when cross_sample_gene_var_weight > 0”.

### 5. Testing plan

Add tests under `tests/` for:
1. `LossCoefficient` accepts `cross_sample_gene_var_weight` and clamps to non-negative.
2. In a small mocked scenario, `_compute_training_sct_cross_sample_variance` or the wrapper in `train.py` outputs a CSV with rows == gene list, cols == cell types, and matches pandas `var(ddof=1)` on dummy data.
3. Small tensor test: given known `recon_x_all_types_cpm` and known buffer `training_sct_cross_sample_gene_var`, loss output matches expected L1 / (G*C).

### 6. User-facing example configs

Update `vaedecon/configs/example_config.yaml` under the `loss_coefficient` block:
```
loss_coefficient:
  hierarchical_code_weight: 0.1
  z_score_kl_weight: 0.1
  gene_mean_weight: 0
  gene_std_weight: 0          # keep semantics: marginal std matching (currently user keeps off)
  cross_sample_gene_var_weight: 1.0   # NEW; example weight
  low_mean_std_weight: 0
```
and annotate clearly that `cross_sample_gene_var_weight` matches cross-sample variances per (gene, cell_type) to the training SCT distribution.

Also add a commented example in SimuTME `config_example.yaml` showing `sorted_bulk_imputation_bulk_weight: 0.01` for user-requested item (1).

## Open questions resolved during design

- **Should we remove `gene_std_weight` now that we add `cross_sample_gene_var_weight`?**  
  No. They are complementary. User explicitly wants no naming confusion between the two. `gene_std_weight` stays for marginal stat matching; the new term covers sample-level variability.
- **Name options rejected:** `gene_variance_weight` (too easily confused with `gene_std_weight`). Accepted name: `cross_sample_gene_var_weight`.
- **Reference source for variance target:** explicitly `data.sct_file_path` (training SCT datasets), not the separate gene-mean/std ref. Point (4) of user’s plan says both should be the same dataset for the next run. So this design enforces that the variance target is the training SCTs, and separately the user can set the mean/std ref to the same path by configuration.

## File changes list (implementation)

- `VAEDecon/vaedecon/configs/default_config.py` — add `cross_sample_gene_var_weight` field + validation, include in loss to_dict
- `VAEDecon/vaedecon/configs/example_config.yaml` — add example knob for cross_sample_gene_var_weight; add comments for hierarchical_code and z_score_kl weight suggestions
- `VAEDecon/vaedecon/workflow/train.py` — export training-SCT cross-sample gene variance CSV; pass CSV path to model config; ensure model gets it
- `VAEDecon/vaedecon/models/vae/vae_model.py` — load buffer, add loss term, include in total loss, log metric
- `VAEDecon/tests/` — add two new test modules (or extend existing) for config, CSV export, and loss math
- Optional documentation update in `changelog.md`

## Notes

- Simulation setting (1) `sorted_bulk_imputation_bulk_weight = 0.01` is already supported by SimuTME at time of spec writing. This spec only mirrors it here for completeness; no SimuTME code changes are required for this knob.
