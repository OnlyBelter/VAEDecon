# Checkpoint selection for explicit overfit debugging

## Goal

This design adds one training config field that lets you choose which trained
checkpoint the workflow treats as the saved model after a run with early
stopping: the checkpoint with the lowest monitored validation loss, or the
checkpoint from the last completed epoch.

The goal is to support explicit overfit experiments where you:

1. watch both training and validation loss curves,
2. keep training alive by setting a large early-stopping patience, and
3. run prediction with the last trained model rather than the best-validation
   model.

## Problem summary

The current workflow mixes two different save/load behaviors:

1. Lightning training writes a best-validation `.ckpt` through
   `ModelCheckpoint`.
2. The base model also saves a folder-style final artifact through
   `model.save(...)`.
3. Inference loads the first `.ckpt` file it finds under `model_dir`.

That makes explicit overfit testing awkward because:

- you cannot choose the final trained checkpoint through config,
- inference does not reliably know which checkpoint it should load when
  multiple `.ckpt` files exist, and
- the current loader can silently pick the wrong checkpoint when your
  experiment wants the last trained model.

## Design overview

This design introduces one explicit config field under `training`:

```yaml
training:
  saved_model_selection: "best"
```

Supported values:

- `"best"`: use the checkpoint with the lowest monitored validation metric
- `"last"`: use the checkpoint from the last completed epoch

The workflow will save both checkpoints whenever Lightning training runs:

1. a best-validation checkpoint
2. a last-epoch checkpoint

The workflow will then use `training.saved_model_selection` to decide which
checkpoint path prediction and inference should load by default.

The recommended default remains `"best"` so existing training behavior stays
aligned with normal model selection practice.

## Config changes

This section defines the user-facing knob and its intended semantics.

Add a new field to `TrainingConfig`:

```yaml
training:
  saved_model_selection: "best"  # "best" or "last"
```

Definitions:

- `"best"` means the checkpoint selected by the existing monitored metric,
  which is currently the validation loss path already used by
  `ModelCheckpoint`.
- `"last"` means the checkpoint from the final completed epoch in the current
  training run.

This field does not disable early stopping. If early stopping halts training at
epoch 180, then `"last"` means the checkpoint from epoch 180, not the original
maximum epoch budget. For explicit overfit testing, you can still set a very
large patience value so training reaches a much later epoch budget before
stopping.

## Training behavior

This section describes how the trainer saves checkpoints and records the user's
selection.

Update the Lightning checkpoint configuration in `base_trainer.py` so the run
emits two named checkpoint files:

1. `best_model_epoch={epoch}.ckpt`
2. `last_model.ckpt`

Implementation details:

- Keep the existing monitored best-checkpoint callback for `"best"`.
- Enable Lightning's `save_last=True` behavior so the trainer also writes
  `last_model.ckpt`.
- Do not change the existing early-stopping logic.
- Continue saving the folder-style model artifact through `model.save(...)`.

The training config written into `model_dir` must include
`saved_model_selection` so later inference can resolve the intended checkpoint
without extra CLI arguments.

## Load behavior

This section defines how the workflow resolves the checkpoint path
deterministically.

Update `load_trained_model(model_dir)` so it no longer picks the first `.ckpt`
file from `os.listdir(model_dir)`. Instead it must:

1. read `training_config.json`,
2. inspect `training.saved_model_selection`,
3. resolve the requested checkpoint path explicitly, and
4. raise a clear error if the requested checkpoint file is missing.

Resolution rules:

- If `saved_model_selection == "best"`, load the checkpoint reported by the
  best-checkpoint callback. The implementation should persist the exact
  `ckpt.best_model_path` into a small metadata file under `model_dir` so
  inference does not depend on filename guessing.
- If `saved_model_selection == "last"`, load `last_model.ckpt`.

This change must apply to all inference and post-training prediction paths that
currently rely on `load_trained_model(model_dir)`.

## Backward compatibility

This section limits the change to the new feature and keeps old runs usable.

Existing configs remain valid because the new field defaults to `"best"`.

For older model directories that predate this feature:

- If `training_config.json` does not contain `saved_model_selection`, treat it
  as `"best"`.
- If only one `.ckpt` exists, keep loading that file.
- If multiple `.ckpt` files exist but the intended checkpoint cannot be
  determined, raise a clear error instead of silently choosing an arbitrary
  file.

This keeps old single-checkpoint runs working while making ambiguous
multi-checkpoint directories fail loudly.

## Overfit-debug workflow

This section explains how the new config supports your explicit overfitting
workflow.

For an overfit experiment, configure:

```yaml
training:
  num_epochs: 2000
  n_early_stopping_patience: 2000
  saved_model_selection: "last"
```

or, if you use the dedicated debug-overfit block, keep the same idea with a
large patience override and the new save-selection field.

The expected workflow is:

1. inspect `metrics.csv` or the plotted loss curves for both train and
   validation loss,
2. let training continue long enough to visibly overfit,
3. load `last_model.ckpt` for prediction, and
4. run prediction on the same dataset or subset used for training to inspect
   the overfit outcome directly.

## Non-goals

This section narrows scope so the implementation stays surgical.

The first implementation does not add:

- a new CLI flag for checkpoint selection,
- separate selection rules for training, inference, and plotting,
- automatic detection of "most overfit" epochs from the loss curves, or
- changes to the monitored metric itself.

## Testing plan

This section defines the minimum regression coverage needed for the change.

Add focused tests for:

1. config loading with `training.saved_model_selection`,
2. default config behavior when the field is omitted,
3. trainer checkpoint configuration writes both best and last checkpoints,
4. `load_trained_model(model_dir)` loads the best checkpoint when selection is
   `"best"`,
5. `load_trained_model(model_dir)` loads `last_model.ckpt` when selection is
   `"last"`, and
6. old single-checkpoint model directories still load successfully.

At minimum, verify that the loader no longer depends on arbitrary filesystem
ordering when more than one `.ckpt` file is present.

## Success criteria

This change is successful when:

1. you can set one config field to choose `"best"` or `"last"`,
2. post-training prediction follows that selection automatically,
3. explicit overfit runs can use the last trained model without manual
   checkpoint swapping, and
4. old normal runs still default to best-validation checkpoint behavior.
