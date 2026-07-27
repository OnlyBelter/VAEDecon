# Changelog

This changelog records notable user-facing changes in `VAEDecon`. It focuses
on behavior, configuration, and workflow updates that affect training,
inference, and evaluation.

## July 27, 2026

This update adds activation-aware cell-proportion prediction so you can choose
between the existing Dirichlet workflow and a cancer-remainder sigmoid
workflow.

### Cell proportion activation modes

The cell-proportion head now supports three explicit prediction modes.

- Added `model.cell_prop_activation_function` with `softplus`, `sigmoid`, and
  `softmax` options.
- Added `model.cancer_cell_type_name` so the runtime can resolve the cancer
  cell index from the configured cell type order.
- Kept the existing Dirichlet-based workflow unchanged when
  `cell_prop_activation_function: softplus`.
- Added a sigmoid branch that predicts only non-cancer cell proportions during
  training and inference.
- Calculated the cancer cell proportion as
  `1 - sum(non-cancer cell proportions)` in the sigmoid branch.
- Normalized the final sigmoid-branch cell-proportion vector so each sample
  sums to `1`.

### Training and validation

The new sigmoid mode changes supervision and validation rules to match the new
prediction semantics.

- Limited supervised cell-proportion loss in sigmoid mode to non-cancer cell
  types only.
- Rejected `loss_coefficient.kld_p > 0` when
  `cell_prop_activation_function: sigmoid`, because that branch does not define
  a Dirichlet posterior.
- Added configuration validation for missing `cancer_cell_type_name` in
  sigmoid mode.

### Encoder coverage

The new activation selection now applies consistently across supported encoder
heads.

- Updated the MLP, residual MLP, transformer, hybrid MLP-GNN, and GNN
  proportion heads to use the shared activation-aware cell-proportion builder.
- Added regression tests for sigmoid remainder reconstruction, row-sum
  normalization, and config validation.

### Softmax cell proportion mode

The cell-proportion head now also supports a direct full-cell-type softmax
workflow.

- Added `softmax` as a third `model.cell_prop_activation_function` option.
- Predicted all cell types directly in the softmax branch, including cancer
  cells.
- Enforced per-sample normalization with `softmax`, so predicted cell
  proportions sum to `1` by construction.
- Used full-vector supervised cell-proportion loss in the softmax branch.
- Rejected `loss_coefficient.kld_p > 0` when
  `cell_prop_activation_function: softmax`, because that branch does not define
  a Dirichlet posterior.

## July 25, 2026

This release updates the cell-proportion prediction workflow, extends
test-set configuration, and fixes related configuration handling.

### Cell proportion prediction

The cell-proportion branch now supports a more explicit Dirichlet-based
workflow for supervised and regularized training.

- Restored supervised cell-proportion training when
  `model.predict_cell_prop: true`.
- Added support for a dedicated `loss_coefficient.kld_p` weight to control
  Dirichlet KL regularization explicitly.
- Changed predicted cell-proportion output to the deterministic Dirichlet mean
  `dd_alpha / sum(dd_alpha)` across supported encoder heads.
- Kept Dirichlet KL computation independent of label availability, so unlabeled
  batches can still receive KL regularization when enabled.
- Preserved supervised proportion loss against the normalized Dirichlet mean
  when training labels are available.

### Cell proportion evaluation

The evaluation workflow now saves and reports predicted cell proportions more
consistently.

- Saved predicted cell proportions directly from model outputs during
  evaluation.
- Improved handling for single-sample batches when writing prediction files.
- Added comparison metrics and plots when true cell fractions are available.
- Saved summary prediction metrics to `prediction_metrics.csv`.

### Multiple configured test sets

Inference now supports multiple named test sets from one YAML configuration.

- Added grouped `data.test_sets` configuration support for named test-set
  bundles.
- Preserved backward compatibility with legacy flat test-set path fields.
- Enabled batch prediction across configured test sets when inference is
  called without a direct `data_file_path`.

### Configuration fixes

Several configuration issues were fixed to make the new workflows stable.

- Fixed recursive validation in config reconciliation by avoiding direct
  self-assignment under assignment validation.
- Improved validation and warning behavior around
  `predict_cell_prop`, `loss_coefficient.cell_prop`, and
  `loss_coefficient.kld_p`.

## July 19, 2026

This update improves dataloader configuration for local training workflows.

- Added configurable dataloader worker settings.
- Updated defaults so local training is less likely to hit multiprocessing
  issues.

## June 18, 2026

This update added a pooled scRNA-seq gene-statistics workflow and improved how
reference gene statistics align with the training configuration.

### Pooled scRNA-seq gene statistics

The training workflow can now compute reference gene statistics directly from a
pooled scRNA-seq dataset instead of relying only on the legacy sctGEP path.

- Added a configurable `gene_mean_std_source` switch so training can compute
  reference gene mean and standard deviation on the fly from a pooled
  scRNA-seq `.h5ad`.
- Added `pooled_sc_h5ad_path`, `pooled_sc_cell_type_col`,
  `pooled_sc_cell_subtype_col`, `pooled_sc_sample_size`, and
  `pooled_sc_seed`.
- Saved pooled-sc-derived reference gene statistics into the model directory as
  `gene_mean_std_log2p1_scaled_by_<scaling_factor>.csv`.

### Alignment and lookup changes

The pooled-sc workflow now aligns more consistently with the effective
training dataset.

- Aligned pooled scRNA-seq gene statistics to the training gene list and
  training cell type order before loading them into the VAE.
- Supported both `cell_type` and `cell_subtype` columns when resolving
  training labels from the pooled scRNA-seq reference dataset.

### Fixes

The pooled-sc workflow now handles subtype-backed references more reliably.

- Fixed failures when labels such as `Non-plasma B cells`, `CD8 T effector`,
  `CAFs`, and `Myofibroblasts` exist under `cell_subtype` rather than
  `cell_type`.
- Restored seeded sampling after the pooled-sc workflow refactor.

## May 17, 2026

This update refreshes key dependencies used by the training and data-loading
stack.

- Updated `anndata` from `0.8.0` to `0.12.16`.
- Corrected the `pandas` pin from `3.0.2` to `2.3.3`.

## April 9, 2026

This update adds z-score-based regularization, low-mean and low-std gene
protection, and several training stability improvements.

### Added

This release introduces new regularization and training controls.

- Added per-cell-type z-score KL regularization to constrain the empirical
  z-score distribution of each cell type toward `N(0, 1)` after predicting GEPs
  in TPM or CPM space.
- Added configurable `low_mean_std_gene_loss` to handle genes whose z-scores
  are unreliable.
- Added caching to `find_sct_gep_of_bulk_sample` to avoid repeatedly reading
  large `.h5ad` files during visualization.
- Added optional stochastic depth to residual blocks in `ResidualBlock`.
- Added `model.torch_compile` with error handling for Python 3.12+
  compatibility.
- Added configurable `gradient_clip_val` with a default of `1.0` to reduce
  gradient explosion at higher learning rates.
- Added Kaiming initialization for `EncoderResMLP` and `DecoderResMLP` in
  `models/nn/res_mlp.py`.

### Changed

The training defaults and residual-network implementation are more stable and
consistent.

- Changed the default optimizer from `Adam` to `AdamW`.
- Changed the decoder projector activation in `DecoderResMLP` from `ReLU` to
  `GELU`.
- Changed the hardcoded `1e-6` LayerNorm epsilon to the shared `EPS`
  constant.
- Fixed mixed imports in `res_mlp.py` to use relative imports consistently.

### Fixed

Several training and monitoring issues were corrected.

- Fixed backed `AnnData` slicing by using `.to_memory()` instead of `.copy()`.
- Fixed the 3D-mask versus 2D-tensor mismatch in
  `low_mean_std_gene_loss` calculation.
- Added `low_mean_std_gene_loss` and `z_score_kl_loss` to progress bar
  metrics.
- Fixed `kaiming_normal_` initialization for GELU-based layers by using
  `nonlinearity='relu'`.
- Added error handling around `torch.compile()` for unsupported Dynamo and
  Python combinations.
- Fixed per-cell-type KL calculation so it computes one KL value per cell type
  across batch and genes, then averages them.

### Configuration

This release also extends the exposed configuration surface.

- Added `z_score_kl_weight: float = 0.0` to `LossCoefficient`.
- Added `low_mean_std_weight: float = 1.0` to `LossCoefficient`.
- Added `torch_compile: bool = False` to `ModelConfig`.
- Added `gradient_clip_val: Optional[float] = 1.0` to `TrainingConfig`.

## Next steps

Add new entries at the top of this file when you change user-facing behavior,
configuration semantics, or output artifacts.
