# Package easy-train CLI

## Problem

`VAEDecon_example/easy_train_cli.py` contains the combined train-and-predict
workflow, argument parsing, email configuration, and SMTP notification logic.
The package's existing `vaedecon` CLI exposes separate train and predict
commands, but does not provide this combined workflow or its email helpers.
Keeping the implementation in an example script makes it harder to reuse and
maintain.

## Design

Move the combined workflow and its helpers into a new package module:
`vaedecon.cli.easy_train`. Expose `main(argv=None)` for both console use and
tests, along with the email helpers `send_email_notification`,
`load_email_settings`, and `notify_completion`.

Register `vaedecon-easy-train = vaedecon.cli.easy_train:main` as a package
console script. Keep the existing `vaedecon` train/predict command unchanged.
The new command accepts the same options as the current example, including
`--email-config`, `--overwrite-outputs`, `--skip-training`, and `--model-dir`.

Replace both `VAEDecon_example/easy_train_cli.py` and
`VAEDecon/examples/easy_train_cli.py` implementations with short wrappers that
call the package `main`. Keep usage examples in the VAEDecon_example wrapper
and preserve its compatibility with existing script and background commands.
Because the package command uses the current working directory for relative
configuration paths, existing working-directory expectations remain unchanged.

Keep email credentials out of source and configuration defaults. Preserve
explicit YAML and `VAEDECON_` environment-variable configuration, validation,
and nonfatal notification failures. Keep staged training's existing final
prediction reuse behavior so the package refactor does not reintroduce duplicate
inference.

## Tests

- Test parser help and core option parsing through `main(argv=...)`.
- Test valid and invalid email configuration, environment fallback, and
  nonfatal email delivery failures without contacting a real SMTP server.
- Test train/predict orchestration paths, including staged-result reuse and
  `--skip-training`.
- Verify each example wrapper delegates to the package entry point.
- Run the focused CLI and staged-training tests, then the package test suite
  where practical.
