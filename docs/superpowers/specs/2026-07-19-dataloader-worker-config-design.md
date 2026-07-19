# DataLoader Worker Config Design

## Goal

Make `VAEDecon` honor explicit DataLoader worker settings from the YAML config
so training can be stabilized on servers with limited shared memory.

This design is motivated by the observed runtime error:

- `DataLoader worker ... is killed by signal: Bus error`
- `It is possible that dataloader's workers are out of shared memory`

## Current Problem

The training config schema already supports these fields:

- `training.train_dataloader_num_workers`
- `training.eval_dataloader_num_workers`

However, the current training workflow does not forward those values when it
constructs the runtime `TrainingConfig` object. As a result:

- the trainer receives `None` for both worker settings
- the DataLoader helper falls back to its automatic worker heuristic
- on Linux, that heuristic may create multiple worker processes
- those worker processes can exhaust `/dev/shm` and crash with `SIGBUS`

## User Config To Update

Target config file:

- `/Users/belter/github/VAEDecon_combined/VAEDecon_example/output/vae/6dsx2_n-neighbor/nbase_30-50-100/18ds_n-neighbor_zk1_lm0_hc1_seed10_nbase30-50-100_averaged_with_sorted_bulk.yaml`

Recommended values:

- `training.train_dataloader_num_workers: 0`
- `training.eval_dataloader_num_workers: 0`

Meaning of `0`:

- no background worker subprocesses are used
- data loading happens in the main process
- this reduces shared-memory pressure
- training may be slower, but is usually much more stable

## Recommended Approach

Use the minimal fix:

1. patch `VAEDecon` so the training workflow forwards both worker fields
2. add both fields to the YAML config and set them to `0`

This keeps package behavior focused on explicit user control, instead of
changing global defaults for all users.

## Scope

### In Scope

- forward worker settings from parsed config into `TrainingConfig`
- add the two worker fields to the provided YAML file
- keep all other training behavior unchanged

### Out Of Scope

- changing the DataLoader heuristic itself
- changing persistent worker defaults globally
- changing batch size, model architecture, or training schedule

## Proposed Package Change

File:

- `vaedecon/workflow/train.py`

Update `_build_trainer_config()` so it includes:

- `train_dataloader_num_workers=self.config.training.train_dataloader_num_workers`
- `eval_dataloader_num_workers=self.config.training.eval_dataloader_num_workers`

This ensures the loader builder receives the user-specified values instead of
falling back to `None`.

## Proposed Config Change

File:

- `VAEDecon_example/output/vae/6dsx2_n-neighbor/nbase_30-50-100/18ds_n-neighbor_zk1_lm0_hc1_seed10_nbase30-50-100_averaged_with_sorted_bulk.yaml`

Add under `training`:

- `train_dataloader_num_workers: 0`
- `eval_dataloader_num_workers: 0`

Place them near the other training runtime settings such as `batch_size`,
`devices`, and `device` so they are easy to find later.

## Expected Result

After this change:

- training should use single-process data loading
- the previous DataLoader bus error should be much less likely
- the YAML file will explicitly document the intended worker behavior

## Risks

The main trade-off is performance:

- using `0` workers can reduce input pipeline throughput
- training may take longer if data loading becomes the bottleneck

This is an acceptable trade-off here because the immediate goal is stability.

## Verification Plan

1. Confirm the edited YAML contains both worker settings.
2. Confirm `_build_trainer_config()` forwards them.
3. Run a small training launch and check logs to verify:
   - no auto-selected worker count is reported
   - training proceeds without the previous `SIGBUS` worker crash

## Recommendation

Implement the minimal explicit-control fix now.

If training later becomes too slow, the next controlled experiment can be:

- increase only `train_dataloader_num_workers` from `0` to `1`
- keep `eval_dataloader_num_workers` at `0`

That should be tested only after the stable baseline is confirmed.
