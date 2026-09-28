# Usage guide

This guide covers the common VAEDecon workflows: installing the package,
training from a YAML config, running inference, and reusing a trained model
with new test sets. For configuration field definitions, see
[the configuration guide](configuration.md).

## Set up the repository

Install VAEDecon in editable mode from the repository root:

```bash
python -m pip install -e ./VAEDecon
```

The CLI example is `VAEDecon/examples/easy_train_cli.py`. The commands below
run from `VAEDecon_example` so the relative paths in its example configs
resolve from the expected working directory:

```bash
cd VAEDecon_example
```

## Configure test sets

Add one entry per inference dataset under `data.test_sets`. Each entry must
provide `test_set_file_path`. For simulated data, add the SCT reference and
sample-to-cell mapping to enable ground-truth GEP evaluation.

```yaml
data:
  test_sets:
    Test_set1:
      test_set_file_path: "./datasets/test_set1.h5ad"
      sct_gep_file_path: "./datasets/sct_reference.h5ad"
      test_set_sample2cell_id_file_path: "./datasets/test1_sample2cell_id.csv"
    Test_set2:
      test_set_file_path: "./datasets/test_set2.h5ad"
```

Real bulk datasets can omit `sct_gep_file_path` and
`test_set_sample2cell_id_file_path`. Keep data paths relative to the working
directory used to launch the command, or provide absolute paths.

## Train and evaluate from a config

The example CLI trains from the selected YAML and then runs its configured
test sets. Run it from `VAEDecon_example`:

```bash
python ../VAEDecon/examples/easy_train_cli.py \
  -c config_20ds_n-neighbor_auto_subtypes.yaml
```

You can also start training through the Python API:

```python
from vaedecon.workflow import train_vaedecon

trained_config = train_vaedecon(
    config_file="vaedecon/configs/example_config.yaml"
)
print(trained_config.model.model_dir)
```

The API example assumes that you run it from the `VAEDecon` package directory
so the example config path resolves.

## Reuse a trained model with new test sets

Use prediction-only mode when you already have a checkpoint and want the
test-set definitions from a separate YAML to override those saved with the
model. `--skip-training` requires `--model-dir` to point to the trained
model's `final_model/` directory.

```bash
MODEL_DIR=output/vae/20ds_n-neighbor/nbase30/\
20ds_auto_subtypes_base58_beta0p5_new_gamma2_ld48_bs128/final_model

python ../VAEDecon/examples/easy_train_cli.py \
  -c config_20ds_n-neighbor_auto_subtypes.yaml \
  --skip-training \
  --model-dir "$MODEL_DIR"
```

The saved model config supplies the model settings. The YAML passed with `-c`
supplies the test sets. The log confirms the override with
`Using test-set definitions from user-provided YAML`.

Without `--skip-training`, the normal training workflow can find an existing
checkpoint and load its saved config. In that case, its older test-set paths
can be used instead of the paths in the YAML passed with `-c`. Use
prediction-only mode to reuse the checkpoint while changing the test sets.

## Run inference from Python

Use `predict_vaedecon` for one input dataset. The model directory must contain
a trained checkpoint and its configuration.

```python
from vaedecon.workflow import predict_vaedecon

results = predict_vaedecon(
    model_dir="./output/vae/my_run/final_model",
    data_file_path="./datasets/test_data.h5ad",
    visualize=True,
)

print(results["pred_cell_prop"].shape)
```

To run all named test sets in a config, omit `data_file_path` and pass a
config whose `data.test_sets` contains the datasets to evaluate:

```python
from pathlib import Path

from vaedecon import predict_vaedecon
from vaedecon.configs import VAEDeconConfig

model_dir = Path("./output/vae/my_run/final_model").resolve()
trained_config = VAEDeconConfig.from_yaml(model_dir / "config.yaml")
evaluation_config = VAEDeconConfig.from_yaml("config_new_test_sets.yaml")
trained_config.data.test_sets = evaluation_config.data.test_sets
trained_config.model.model_dir = str(model_dir)

results = predict_vaedecon(
    model_dir=model_dir,
    output_dir=str(model_dir / "test_results"),
    config=trained_config,
    device="auto",
)
```

The evaluation YAML can contain only the test-set definitions:

```yaml
data:
  test_sets:
    Test_set1:
      test_set_file_path: "./datasets/test_set1.h5ad"
```

Add `sct_gep_file_path` and
`test_set_sample2cell_id_file_path` when you want ground-truth evaluation for
simulated data.

## Find prediction outputs

The CLI writes prediction results under
`<model_dir>/test_results/` and saves the effective config as
`<model_dir>/used_config.yaml`. The Python API uses the same default
`test_results` directory when you omit `output_dir`.

For simulated test sets with matched ground truth, outputs include cell
proportion predictions, reconstructed sample-specific cell-type GEPs, and
ground-truth comparison results. The exact outputs depend on the configured
evaluation options.

## Troubleshoot test-set paths

- If a log reports an old test-set path, check that the YAML passed with
  `-c` is the updated file and that the command uses `--skip-training` when
  reusing an existing checkpoint.
- If the CLI reports a missing test-set file, confirm the path exists on the
  machine running VAEDecon and is relative to the command's working directory,
  or change it to an absolute path.
- If prediction-only mode does not log
  `Using test-set definitions from user-provided YAML`, check that the YAML
  contains a non-empty `data.test_sets` block and that `-c` points to it.
