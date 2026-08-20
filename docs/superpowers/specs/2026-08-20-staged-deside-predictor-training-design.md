# Staged DeSide Predictor Training Design

## Goal

Add a one-YAML staged training workflow for VAEDecon that:

1. Trains `DeSideCellPropPredictor` first for accurate cell proportion prediction.
2. Freezes that predictor and uses its outputs as:
   - `cell_prop`
   - decoder bulk-sample context
3. Trains the latent encoder and decoder for cell-type-specific GEP prediction.
4. Performs a final low-learning-rate joint fine-tuning stage.
5. Uses early stopping independently within each stage and promotes the best checkpoint from one stage into the next stage.

This design keeps the new DeSide branch focused on its strongest task first, then lets the rest of the model adapt around a stable cell-proportion/context signal.

## Why This Change

The current joint training setup can become unstable when the new DeSide-style predictor branch is optimized together with reconstruction-heavy objectives from the start. In practice, the predictor branch and the decoder branch solve different problems:

- the predictor branch should learn high-accuracy bulk-to-cell-proportion mapping
- the latent encoder and decoder should learn cell-type-specific reconstruction

Training them in stages should improve stability and make the DeSide branch more reliable as a conditioning signal for the decoder.

## Proposed Workflow

The new workflow is a built-in multi-stage training schedule controlled by a single config file.

The workflow should also support running selected stages independently. This is important for debugging and ablation work. For example, the user should be able to:

- run only `cell_prop_predictor_pretrain`
- run only `reconstruction_training` using an existing pretrained predictor checkpoint
- run `cell_prop_predictor_pretrain` followed by `reconstruction_training`
- run all three stages end to end

To support this, staged training should accept:

- an ordered `stages` list that defines all available stage configs
- an optional `run_stages` list that selects which stages to execute in the current run
- an optional `stage_init_checkpoints` mapping that supplies an input checkpoint when a selected stage does not follow a previously executed stage in the same run

If `run_stages` is omitted, all configured stages run in order. If `run_stages` is set, only those stages run, in the order listed there.

### Stage 1: Cell-Prop Predictor Pretrain (`cell_prop_predictor_pretrain`)

Train only `cell_prop_predictor`.

- trainable modules:
  - `cell_prop_predictor`
- frozen modules:
  - `encoders`
  - `decoder`
- active objective:
  - cell proportion supervision
- disabled or zero-weighted objectives:
  - reconstruction-heavy objectives
  - matched sctGEP supervision
  - other auxiliary decoder-side objectives

Expected result:

- stable `cell_prop`
- usable `bulk_context_feature`
- best checkpoint chosen by stage-specific early stopping

### Stage 2: Frozen-Predictor Reconstruction Training

Load the best checkpoint from Stage 1, freeze the DeSide predictor branch, and train the remaining modules.

- trainable modules:
  - `encoders`
  - `decoder`
- frozen modules:
  - `cell_prop_predictor`
- routing remains active:
  - `cell_prop_source: cell_prop_predictor`
  - `decoder_context_source: cell_prop_predictor`
  - `latent_posterior_source`: unchanged from the configured latent encoder path

Expected result:

- the decoder and latent encoder learn around a stable DeSide-derived context
- no drift in cell proportion behavior during this stage
- best checkpoint chosen by stage-specific early stopping

### Stage 3: Low-LR Joint Fine-Tune

Load the best checkpoint from Stage 2, unfreeze all modules, reduce learning rate, and fine-tune jointly.

- trainable modules:
  - `cell_prop_predictor`
  - `encoders`
  - `decoder`
- frozen modules:
  - none
- learning rate:
  - stage-specific low-LR override or multiplicative scale relative to the base LR

Expected result:

- final alignment between the predictor branch and reconstruction branch
- improved decoder use of `bulk_context_feature`
- no need to keep the predictor frozen forever

## Config Design

Add a new optional block under `training`:

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
          cell_type_sct_gep_weight: 0.0
          hierarchical_code_weight: 0.0
          cross_sample_gene_var_weight: 0.0
        early_stopping:
          monitor: val_loss
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
          monitor: val_loss
          patience: 50
          min_delta: 0.0

      - name: joint_finetune
        max_epochs: 100
        train_modules: ["cell_prop_predictor", "encoders", "decoder"]
        freeze_modules: []
        learning_rate_scale: 0.1
        early_stopping:
          monitor: val_loss
          patience: 20
          min_delta: 0.0
```

### Notes on Config Semantics

- `staged_training.enabled: false` preserves current behavior.
- `stages` must be ordered explicitly in execution order.
- `run_stages` is optional. If omitted, run all configured stages in `stages` order.
- `run_stages` may contain any subset of configured stage names.
- when a selected stage depends on a prior stage that is not being run in the same invocation, the required starting checkpoint must be provided through `stage_init_checkpoints`.
- `train_modules` and `freeze_modules` are constrained to supported names:
  - `cell_prop_predictor`
  - `encoders`
  - `decoder`
- `loss_overrides` apply only during that stage and do not permanently mutate the base config.
- `learning_rate_scale` multiplies the base `training.learning_rate`.
- each stage has its own early-stopping configuration.

## Execution Model

The training workflow should:

1. build the dataset once
2. build the model once
3. iterate over configured stages in order
4. before each stage:
   - determine whether the stage should run in this invocation
   - restore the best checkpoint from the previous executed stage if applicable
   - otherwise load an explicit stage-init checkpoint if one is required
   - freeze and unfreeze modules according to stage config
   - rebuild optimizer and scheduler for the current trainable parameter set
   - apply stage-local loss overrides
   - configure stage-local early stopping and checkpointing
5. train until stage early stopping or `max_epochs`
6. promote the best checkpoint from the current stage into the next stage
7. after the last stage, save the final merged model directory as usual

This should be implemented as a single workflow invocation rather than multiple user-managed runs, while still allowing a subset of stages to be selected for a given run.

## Checkpointing and Early Stopping

Each stage should have its own:

- early stopping callback
- best checkpoint file
- last checkpoint file
- metrics log

The next stage must start from the previous stage's best checkpoint, not from the last epoch state.

Suggested saved artifacts:
- `stage_cell_prop_predictor_pretrain/best.ckpt`
- `stage_cell_prop_predictor_pretrain/last.ckpt`
- `stage_cell_prop_predictor_pretrain/metrics.csv`
- `stage_reconstruction_training/best.ckpt`
- `stage_joint_finetune/best.ckpt`

The final model directory should also record:

- stage order
- per-stage effective loss overrides
- per-stage early-stopping settings
- which checkpoint was promoted into each next stage

## Routing Behavior

This design does not add another latent encoder path.

The DeSide branch remains a dedicated predictor/context module that exposes:

- `cell_prop`
- `bulk_context_feature`

During staged training:

- Stage 1 uses the predictor outputs directly for cell proportion supervision.
- Stage 2 continues to route:
  - `cell_prop_source` to the predictor
  - `decoder_context_source` to the predictor
- Stage 3 keeps the same routing and only changes trainability and LR.

`latent_posterior_source` continues to come from the main latent encoder path already configured in the model.

## Error Handling and Validation

Validation should fail early when:

- `staged_training.enabled` is true but `stages` is empty
- `run_stages` contains names that are not present in `stages`
- a stage references unsupported module names
- a stage omits `max_epochs`
- the config enables staged training without `cell_prop_predictor_cls`
- `train_modules` and `freeze_modules` conflict in a way that leaves no trainable parameters
- `cell_prop_predictor_pretrain` does not include `cell_prop_predictor` in `train_modules`
- a later stage refers to the predictor when the predictor is not configured
- `reconstruction_training` or `joint_finetune` is selected without an available upstream checkpoint from either:
  - an earlier stage in the same run, or
  - `stage_init_checkpoints`

Warnings should be emitted when:

- Stage 2 disables all reconstruction-related losses
- Stage 3 uses `learning_rate_scale >= 1.0`
- a stage monitors a metric that is not logged by the trainer

## Logging

Training logs should make stage boundaries explicit.

At minimum, log:

- stage name
- stage index
- active trainable modules
- frozen modules
- effective learning rate
- effective loss coefficients for that stage
- checkpoint promoted into the stage

Metrics should remain easy to compare across stages. A simple approach is:

- separate per-stage CSV logs
- one top-level summary file describing the best epoch and best monitored metric for each stage

## Testing Plan

Add tests for:

1. config validation
   - accepts valid staged training blocks
   - rejects invalid stage names or module names
   - rejects enabled staged training without a predictor branch
   - rejects stage subsets that require missing init checkpoints

2. module freezing behavior
   - Stage 1 trains only predictor parameters
   - Stage 2 freezes predictor and trains encoder/decoder
   - Stage 3 unfreezes all

3. stage override behavior
   - stage-local loss overrides affect only the active stage
   - LR scaling is applied per stage

4. checkpoint promotion
   - Stage 2 loads Stage 1 best checkpoint
   - Stage 3 loads Stage 2 best checkpoint

5. end-to-end smoke behavior
   - a tiny staged run completes
   - final model directory contains stage metadata and expected outputs

## Scope Boundaries

This design does not include:

- automatic stage-specific routing changes beyond existing configured routing
- a separate predictor-only CLI
- per-parameter-group fine-tuning schedules beyond stage-level LR scaling
- changing the DeSide predictor architecture itself

Those can be added later if needed, but they are not required for this staged strategy.

## Recommendation

Implement the staged strategy as a first-class training workflow in `training.staged_training`, with three explicit stages and per-stage early stopping.

This should be the default recommended strategy when using `DeSideCellPropPredictor` as:

- the cell proportion source
- the decoder bulk-context source

while a standard encoder remains responsible for the latent posterior.
