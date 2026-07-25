# Changelog

This changelog records notable user-facing changes in `VAEDecon`. It focuses
on behavior, configuration, and workflow updates that affect training,
inference, and evaluation.

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

## Next steps

Add new entries at the top of this file when you change user-facing behavior,
configuration semantics, or output artifacts.
