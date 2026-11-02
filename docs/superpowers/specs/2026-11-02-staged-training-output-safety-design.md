# Staged training output safety

Protect existing training files and make staged checkpoint configuration
fail fast. This design keeps checkpoint mappings keyed by the target stage.

## Problem

When `run_stages` is omitted, all configured stages currently run. A
`stage_init_checkpoints` entry is ignored when that stage's predecessor runs
in the same invocation. Before each selected stage, the trainer removes its
existing output directory. Final-model promotion also deletes and overwrites
previous results, while config snapshots and error reports can overwrite
existing files earlier in the workflow.

This combination can silently discard a user-provided predictor checkpoint
inside the stage directory before it can be reused.

## Stage configuration validation

Preserve the existing target-stage meaning of `stage_init_checkpoints`. For
example, the predictor checkpoint used to initialize
`reconstruction_training` is keyed by `reconstruction_training`, not by
`cell_prop_predictor_pretrain`. Do not infer a checkpoint's contents or source
stage from its file or directory name.

During configuration validation, reject every checkpoint entry that cannot be
used in the selected stage plan:

- Its key must identify a selected target stage with a direct predecessor.
- That predecessor must not also be selected in the current invocation.
- When a selected stage skips its predecessor, its checkpoint entry remains
  mandatory.
- A nonempty checkpoint map with omitted or empty `run_stages` is invalid
  because the default plan runs every stage and cannot use such entries.

The error identifies the unused target key and selected predecessor, explains
the target-stage convention, and tells the user to select the intended
`run_stages` or remove the conflicting entry. Preserve the current contiguous
stage-order validation and valid configurations that explicitly select later
stages with target-keyed input checkpoints.

## Read-only preflight

Before creating output directories, saving configuration, preprocessing,
constructing a model, or starting training, resolve every supplied checkpoint
path using the same working-directory semantics as loading. Require each
path to exist, be a regular readable file, and report its target stage and
resolved path in any error. Model and weight compatibility are checked when
the checkpoint is loaded; file existence alone does not establish
compatibility.

Run the preflight for both `train_vaedecon` and direct
`VAEDeconTrainer.train()` entry points. Ensure a failure does not write an
error report or mutate output files. Configuration validation and checkpoint
preflight also apply before any existing-final-model reuse shortcut.

Before a run that will train, identify existing user-visible outputs that
the invocation might delete, replace, or overwrite, including selected
stage directories, final-model contents, config snapshots, error reports,
stage summaries, and prediction outputs. Distinguish these from untouched
earlier-stage checkpoint inputs and reusable preprocessing caches. An input
checkpoint may reside under an unselected stage directory; its presence
alone must not require deletion or prevent its use.

Resolve paths and symlinks when checking overlap. If any supplied checkpoint
would be deleted or overwritten by a planned operation, fail without
modifying it, even if overwrite permission was granted. Provide its resolved
path and suggest a separate input checkpoint or a fresh output directory.

## Confirmation and unattended runs

Do not silently remove or overwrite previous files. If planned output
collisions exist, list the affected paths before preprocessing and ask for
explicit confirmation in an interactive terminal. Declining, EOF, or an
unavailable terminal stops the run without modifying the listed files.

Support an explicit one-invocation overwrite authorization for unattended
CLI and programmatic runs. It is not stored as a persistent YAML setting.
Document how to supply it through the supported training CLI entry points
and Python API. Authorization covers only the collisions displayed by the
preflight, not newly appearing files or input checkpoints. Recheck relevant
paths immediately before destructive operations; fail if the preflight no
longer covers the current paths.

Existing completed final-model reuse must be clearly reported as reuse, not
mistaken for training or deleted silently. Explicit overwrite authorization
must not silently convert reuse into a different action.

## Tests

Use temporary directories and sentinel contents to verify that failures,
declined prompts, and noninteractive runs preserve previous files byte for
byte. Cover stage-directory cleanup, final-model promotion, config snapshots,
error reports, and prediction outputs.

Test omitted or empty `run_stages` with a supplied checkpoint, a checkpoint
whose predecessor is also selected, unused keys, and valid later-stage
target-keyed mappings. Test missing and unreadable paths before preprocessing,
input-checkpoint/output overlap including symlinks, and an input under an
unselected stage that must remain reusable.

Test interactive approval and refusal, EOF, explicit per-run authorization,
and a new conflicting file appearing after preflight. Verify both training
entry points and existing completed-model reuse. Run the existing staged
training tests and the full suite where practical.

## Documentation

Update the staged-training configuration guide and example to explain
target-stage keys, stage selection, early checkpoint validation, output
collision confirmation, noninteractive authorization, and fresh output
locations. Update the changelog after implementation.
