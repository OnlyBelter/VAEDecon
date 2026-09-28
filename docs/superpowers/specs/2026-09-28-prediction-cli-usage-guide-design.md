# Prediction CLI and usage guide design

## Goal

Provide a package-local example CLI without email credentials or notifications,
and give users one guide for the common VAEDecon training and inference
workflows.

## Scope

- Copy `VAEDecon_example/easy_train_cli.py` to
  `VAEDecon/examples/easy_train_cli.py`.
- Preserve CLI arguments, training and prediction behavior, test-set
  overrides, and input preflight validation.
- Remove only email-related imports, the email helper, SMTP configuration, and
  the final notification call from the copied script.
- Create `VAEDecon/docs/usage.md` with setup, training, single-dataset
  inference, configured test sets, prediction-only checkpoint reuse, relative
  path guidance, outputs, and relevant troubleshooting.
- Remove the recently added CLI walkthrough from `docs/configuration.md` and
  replace it with a short link to `usage.md`. Keep configuration field
  descriptions in that guide.
- Keep a concise README entry point and link to the full usage guide.

## Behavior and assumptions

The prediction-only command loads model settings from the saved model
configuration, then overrides its test-set definitions with `data.test_sets`
from the YAML passed with `-c`. `--skip-training` requires `--model-dir`, which
points to the trained model's `final_model/` directory. The guide will state
that users must run from a working directory consistent with relative paths in
their YAML.

The original `VAEDecon_example/easy_train_cli.py` remains unchanged. The new
example removes email functionality without changing the training or
prediction workflow.

## Verification

- Compile the copied Python script.
- Check that email-related imports, credentials, and calls are absent from the
  copied script.
- Verify the CLI argument names and required `--model-dir` behavior against
  its parser.
- Review the README and usage guide for consistent commands and links.
