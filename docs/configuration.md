# Configuration guide

This guide explains how to use `vaedecon/configs/example_config.yaml` on the
current `main` branch. It focuses on the configuration structure that you
edit most often when training and evaluating VAEDecon.

Use this page when you want to:

- understand what each top-level config section does,
- decide which fields matter for your workflow,
- configure staged training on `main`, or
- find the right place to tune losses and outputs.

For concrete defaults, start with
`vaedecon/configs/example_config.yaml`. For design history and feature
rationale, see the files in `docs/superpowers/specs/`.

## How the config is organized

VAEDecon uses one YAML file with four top-level sections:

- `data`: training inputs, reference inputs, preprocessing, and inference
  datasets,
- `training`: optimizer settings, checkpoint behavior, staged training, and
  logging,
- `model`: architecture, routing, cell-proportion prediction, residual
  learning, and loss weights,
- `evaluation`: plotting, exported outputs, and inference-time settings.

In practice, most runs only need edits in a small subset of fields:

1. Update file paths in `data`.
2. Turn staged training on or off in `training.staged_training`.
3. Choose the predictor and loss weights in `model`.
4. Choose plots and saved outputs in `evaluation`.

## Data section

The `data` section controls what VAEDecon reads and how it prepares training
and inference inputs.

### Core training inputs

These fields define the main training inputs:

- `data.simu_bulk_file_path`: mixed bulk training datasets,
- `data.sct_file_path`: single-cell-type GEP (sctGEP) reference datasets used for reference statistics
  and related workflows,
- `data.data_dir`: base dataset directory when relative paths are used.

The current code accepts lists for `sct_file_path` and
`simu_bulk_file_path`. That keeps the layout flexible for multi-dataset
workflows.

### Preprocessing and scaling

These fields control input preprocessing:

- `data.scaling_by_constant`
- `data.scaling_factor`
- `data.remove_low_var_genes`
- `data.min_var`
- `data.force_reprocess`

Use `remove_low_var_genes` carefully in staged runs. The staged workflow
depends on a shared gene space across stages, so data consistency matters more
than aggressive per-stage filtering.

### Reference gene mean and standard deviation

VAEDecon can compute reference gene mean and standard deviation from either
SCT references or pooled single-cell data:

- `data.gene_mean_std_source: "sct_gep"` uses SCT reference data and is the
  default option.
- `data.gene_mean_std_source: "pooled_sc"` uses a pooled single-cell
  `.h5ad`.

Related fields:

- `data.gene_mean_std_sct_gep_file_path`
- `data.pooled_sc_h5ad_path`
- `data.pooled_sc_cell_type_col`
- `data.pooled_sc_cell_subtype_col`
- `data.pooled_sc_sample_size`
- `data.pooled_sc_seed`

Use the pooled single-cell fields only when
`gene_mean_std_source="pooled_sc"`.

### Matched supervision bundles

`data.training_target_sets` defines optional matched supervision bundles for
training workflows that need aligned bulk, sampled-cell mapping, and SCT
reference inputs.

Each named entry can include:

- `training_set_file_path`
- `training_set_sample2cell_id_file_path`
- `training_sct_gep_file_path`

This section is especially relevant when you enable matched sctGEP
supervision losses.

### Training sctGEP cell-proportion threshold

`data.training_sct_gep_cell_prop_threshold` is used by the current code to
mask matched sctGEP supervision for very small cell-type fractions.

Treat it as a stability threshold, not as a biological definition.

### Named test sets

`data.test_sets` lets you define multiple named inference datasets in one
config. Each named entry can include:

- `test_set_file_path`
- `sct_gep_file_path`
- `test_set_sample2cell_id_file_path`

This is useful when you want one trained model to run across several
benchmark or ablation datasets without rewriting the config each time.

## Training section

The `training` section controls optimization, checkpointing, staged training,
and logging.

### Core optimization settings

The most commonly edited fields are:

- `training.output_dir`
- `training.naming_postfix`
- `training.learning_rate`
- `training.batch_size`
- `training.num_epochs`
- `training.n_early_stopping_patience`
- `training.optimizer_cls`
- `training.saved_model_selection`

`saved_model_selection` controls which checkpoint inference loads:

- `best`: load the lowest validation-loss checkpoint,
- `last`: load the final completed checkpoint.

### Device and data loader settings

Use these fields to control hardware and loader behavior:

- `training.devices`: the number of accelerator devices to use for training
  and inference. In most runs, this means the number of GPUs.
- `training.device`: the backend type to use. The current code accepts
  `auto`, `cuda`, and `cpu`. Use `auto` to let the framework choose an
  available accelerator automatically.
- `training.train_dataloader_num_workers`
- `training.eval_dataloader_num_workers`

### Warmup and scheduler settings

The example config uses a warmup schedule plus a learning-rate scheduler.
The main fields are:

- `training.warmup_epochs`
- `training.scheduler_cls`
- `training.scheduler_params`

### Progress bar metrics

`training.prog_bar_metrics` controls which metrics appear in the progress bar
and logging loop. This only changes reporting, not training behavior.

### Debug overfit mode

`training.debug_overfit` creates a reproducible tiny-subset workflow for
diagnosis. It is useful when you need to answer questions like:

- can the model overfit a tiny dataset,
- is a new loss term wired correctly,
- are predictions saved correctly after training.

You usually do not need this block for normal experiments.

## Staged training on main

The `main` branch supports a three-stage workflow through
`training.staged_training`.

The expected stage order is:

1. `cell_prop_predictor_pretrain`
2. `reconstruction_training`
3. `joint_finetune`

At a high level:

1. Stage 1 trains the cell-proportion predictor first.
2. Stage 2 trains the encoder and decoder for GEP reconstruction while the
   predictor stays frozen.
3. Stage 3 fine-tunes the full system jointly with a smaller learning rate.

This staged workflow reduces the difficulty of optimizing cell-proportion
prediction and cell-type-specific GEP reconstruction jointly from the start.

### Minimal staged training block

```yaml
training:
  staged_training:
    enabled: true
    run_stages:
      - cell_prop_predictor_pretrain
      - reconstruction_training
      - joint_finetune
    stage_init_checkpoints: {}
    stages:
      - name: cell_prop_predictor_pretrain
        max_epochs: 300
        train_modules: ["cell_prop_predictor"]
        freeze_modules: ["encoders", "decoder"]
        learning_rate_scale: 1.0
        loss_overrides:
          cell_prop: 1.0e4
        early_stopping:
          monitor: "val_cell_prop_loss"
          patience: 30
          min_delta: 0.0
      - name: reconstruction_training
        max_epochs: 500
        train_modules: ["encoders", "decoder"]
        freeze_modules: ["cell_prop_predictor"]
        learning_rate_scale: 1.0
        loss_overrides:
          cell_prop: 0.0
        early_stopping:
          monitor: "val_loss"
          patience: 50
          min_delta: 0.0
      - name: joint_finetune
        max_epochs: 100
        train_modules: ["cell_prop_predictor", "encoders", "decoder"]
        freeze_modules: []
        learning_rate_scale: 0.1
        early_stopping:
          monitor: "val_loss"
          patience: 20
          min_delta: 0.0
```

### Important staged-training requirements

The config validators enforce several requirements for staged training:

- `training.staged_training.enabled` requires at least one configured stage,
- stage names must be unique,
- staged training requires `model.cell_prop_predictor_cls`,
- staged training requires `model.predict_cell_prop: true`,
- reconstruction and joint stages require configured bulk and SCT inputs.

When staged training is enabled, the predictor branch is not optional in
practice. The staged workflow is built around explicit cell-proportion
pretraining.

## Model section

The `model` section controls the network architecture, routing, prediction
behavior, residual-learning mode, and loss weights.

### Core architecture

The most common architecture fields are:

- `model.input_dim`
- `model.latent_dim`
- `model.n_cell_types`
- `model.encoders`
- `model.decoders`
- `model.encoder_hidden_dims`
- `model.decoder_hidden_dims`
- `model.encoder_dropout_rate`
- `model.decoder_dropout_rate`

The codebase currently supports multiple encoder backbones, including MLP,
ResMLP, Transformer, pathway-based, GNN, and hybrid variants.

### Encoder aliases and output routing

If you use more than one encoder branch, `model.encoder_aliases` gives each
active encoder a stable name. `model.encoder_output_routing` then decides
which branch provides:

- cell-proportion features,
- the latent posterior,
- decoder conditioning context.

If you only use the standard single-encoder setup, you can usually leave
these fields close to their defaults.

### Stage 1 cell-proportion prediction

Cell-proportion prediction is controlled by:

- `model.predict_cell_prop`
- `model.cell_prop_predictor_cls`
- `model.cell_prop_activation_function`
- `model.cancer_cell_type_name`

On the current `main` branch, staged training is designed around
`DeSideCellPropPredictor` for Stage 1 cell-proportion learning.

For the staged workflow, the common pattern is:

```yaml
model:
  predict_cell_prop: true
  cell_prop_predictor_cls: "DeSideCellPropPredictor"
  cell_prop_activation_function: "sigmoid"
  cancer_cell_type_name: "Cancer Cells"
```

If you set `predict_cell_prop: true`, the validation rules also check that
the cell-proportion settings are internally consistent. For example, the
DeSide predictor currently expects the `sigmoid` activation path.

### DeSide predictor architecture

These fields tune the dedicated Stage 1 predictor:

- `model.deside_pathway_network`
- `model.deside_hidden_dims`
- `model.deside_dropout_rate`
- `model.deside_pathway_hidden_dims`
- `model.deside_pathway_dropout_rate`
- `model.deside_normalization`
- `model.deside_normalization_layer`

You typically edit these only when you are actively tuning the predictor
architecture.

### Residual GEP learning

Residual learning is controlled by:

- `model.learn_gep_residual`
- `model.learn_gep_residual_mode`

When residual learning is enabled, the model predicts sample-specific
deviations around a reference mean GEP instead of always predicting the full
cell-type-specific GEP directly.

The current code supports two residual modes:

- `zscore`
- `mean_centered`

Some auxiliary losses are mode-specific. For example,
`z_score_kl_weight` and `z_score_reg_weight` must be `0` when
`learn_gep_residual_mode="mean_centered"`.

### Loss coefficients

`model.loss_coefficient` is the main place to tune training objectives.

Common fields include:

- `beta`: KL weight for the VAE latent objective,
- `gamma`: hierarchy-aware repulsion strength in latent space,
- `cell_prop`: supervised cell-proportion loss weight,
- `kld_p`: Dirichlet-style regularization for cell proportions,
- `cross_sample_gene_var_weight`: cross-sample gene-variance matching,
- `inter_sample_similarity_weight`: within-cell-type sample-structure
  preservation,
- `cell_type_sct_gep_weight`: matched sctGEP supervision,
- `cell_type_existence_weight`: cell-type existence supervision,
- `hierarchical_code_weight`: hierarchy-code supervision weight.

Two loss terms are especially relevant if you want better cell-type geometry
and better per-sample variation:

- `gamma` controls the hierarchy-aware repulsion term, which pushes unrelated
  cell types farther apart while allowing sibling subtypes to remain more
  similar.
- `inter_sample_similarity_weight` preserves within-cell-type sample
  structure, which helps prevent reconstructed GEPs from collapsing toward a
  single average pattern.

## Evaluation section

The `evaluation` section controls exported outputs and plotting during
validation and inference.

Common fields include:

- `evaluation.n_samples`
- `evaluation.visualize_n_sample`
- `evaluation.cell_prop_threshold`
- `evaluation.plot_cell_proportions`
- `evaluation.plot_single_cell_gep`
- `evaluation.plot_bulk_gep`
- `evaluation.plot_latent_space`
- `evaluation.figure_format`
- `evaluation.save_reconstructed_gep`
- `evaluation.save_cell_type_specific_gep_metrics`
- `evaluation.save_bulk_gep_input`
- `evaluation.save_recon_bulk_gep_conv`

Use this section to decide what artifacts you want after prediction. If you
are running large sweeps, limiting plots and saved outputs can reduce
runtime and disk usage.

## Recommended workflow for editing the config

If you are starting from the example config, use this order:

1. Update the dataset paths in `data`.
2. Confirm whether you want staged training in `training.staged_training`.
3. Set the Stage 1 predictor fields in `model`.
4. Choose the main loss weights in `model.loss_coefficient`.
5. Set the outputs you want in `evaluation`.

This order keeps the most important experiment decisions near the top of your
editing workflow.

## Where to look next

Use these files together:

- `vaedecon/configs/example_config.yaml` for a runnable starting point,
- `README.md` for the project overview,
- `docs/superpowers/specs/` for feature-specific design notes.

## Next steps

After you finish the config:

1. Run a short staged training job to confirm the data paths and stage order.
2. Review the saved checkpoints and staged summary outputs.
3. Tune the loss weights and exported evaluation artifacts for your study.
