# Staged CLI duplicate prediction

## Problem

Staged training runs final test-set inference into
`final_model/test_results` before `train_vaedecon()` returns. The
`easy_train_cli.py` wrapper then unconditionally runs inference into that same
directory. Its output-safety check correctly rejects the directory, so a
successful first training run exits with an error after training has completed.

## Design

Keep staged training as the owner of its final test-set evaluation. In the
train-and-predict CLI path, if staged training is enabled and
`final_model/test_results` already exists after the training call, reuse those
results rather than invoking prediction a second time.

Do not apply this shortcut when `--skip-training` is set: that mode explicitly
requests prediction-only behavior and may use new test-set definitions. Keep the
existing prediction path for non-staged training. Preserve `used_config.yaml`
creation for fresh staged runs without replacing an existing config snapshot.

Do not change output preservation semantics or make `--overwrite-outputs`
implicitly authorize duplicate prediction. Keep the authoritative example and
the `VAEDecon_example` copy behaviorally aligned.

## Tests

- Verify the staged train-and-predict path reuses existing final test results
  and does not invoke prediction a second time.
- Verify `--skip-training` still invokes prediction even when staged training
  is configured.
- Verify non-staged train-and-predict behavior remains unchanged.
- Verify a fresh staged run can still save `used_config.yaml` without
  overwriting an existing snapshot.
