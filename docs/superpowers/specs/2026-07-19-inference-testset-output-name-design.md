# Inference Output Naming Design

## Goal

Improve the inference output path in `VAEDecon` so that the result subfolder
under `test_results` is named by the inferred test-set file, instead of the
current hard-coded `test_set`.

This change is intended to avoid manual renaming when the same trained model is
used to infer multiple test sets.

## Current Problem

For a trained model directory such as:

- `/Users/belter/github/VAEDecon_combined/VAEDecon_example/output/vae/6dsx2_n-neighbor/nbase30/5loss_terms_zk1_lm0_hc1_seed10_averaged_sorted_bulk`

the default inference outputs are written to:

- `test_results/test_set/`

inside that model result folder.

This is too generic. When the user infers a different test set later, the
subfolder name still remains `test_set`, so the user has to rename it by hand.

## Naming Rule

Use the full input test-set filename stem as the result subfolder name:

- take the basename of the inferred `.h5ad` file
- remove the `.h5ad` suffix
- keep the full remaining string unchanged

Examples:

- `simu_bulk_exp_Test_set1_log2cpm1p.h5ad`
  -> `simu_bulk_exp_Test_set1_log2cpm1p`
- `simu_bulk_exp_Mixed_N100K_segment_without_filtering_hnscc_log2cpm1p_selected_50.h5ad`
  -> `simu_bulk_exp_Mixed_N100K_segment_without_filtering_hnscc_log2cpm1p_selected_50`

## Clarification On Parameter Names

There are two relevant parameter names in the current code path:

- config-side field: `test_set_file_path`
- lower-level Python inference argument: `data_file_path`

Design rule:

- when inference is launched through the config-driven workflow, the naming
  should correspond to the configured `test_set_file_path`
- when inference is called directly through the lower-level API, the naming
  should be derived from the runtime `data_file_path`

These two references point to the same actual input file in normal usage, so
the resulting folder name should be identical.

## Scope

This design changes only the naming of the inner inference result subfolder.

It does not change:

- the top-level parent folder such as `test_results`
- the names of existing output files inside the result folder
- the prediction logic
- the visualization content

## Proposed Behavior

### Default Result Layout

Keep the existing top-level result directory behavior:

- `test_results`
- `tcga_results`
- other `{dataset_type}_results`

Replace the hard-coded inner subfolder:

- from: `test_results/test_set/`
- to: `test_results/<input_file_stem>/`

For the example config whose `test_set_file_path` is:

- `./datasets/simulated_bulk_cell_dataset_subtypes_all_range/segment_12ds_0.95_n_base100_19cancer_pca_0.9_median_gep/simu_bulk_exp_Test_set1_log2cpm1p.h5ad`

the output folder should become:

- `test_results/simu_bulk_exp_Test_set1_log2cpm1p/`

### Nested Output Structure

The current nested folders remain unchanged below the renamed subfolder:

- `<result_set_name>/cell_prop/`
- `<result_set_name>/gep/`
- `<result_set_name>/gep/sc_gep/`

Files such as the following remain in the same relative positions:

- `predicted_cell_prop.csv`
- `mu_embeddings.csv`
- reconstructed GEP CSV files
- generated plots

## Affected Code Areas

### 1. Inference Entry

File:

- `vaedecon/workflow/inference.py`

Needed behavior:

- derive a result-set name from the actual inference input file path
- pass that result-set name into `evaluate_model()`

### 2. Evaluation Output Path

File:

- `vaedecon/workflow/workflow.py`

Needed behavior:

- replace the hard-coded `"test_set"` folder with the passed result-set name

### 3. Dataset Preprocessing Cache

File:

- `vaedecon/workflow/inference.py`

Current inconsistency:

- `_build_gepdataset_config()` supports `dataset_type`
- `predict()` currently does not pass `dataset_type` into it
- this causes preprocessing cache folders to stay under `processed_test`

Recommended small consistency fix:

- pass `dataset_type` through to `_build_gepdataset_config()`

This is not the main feature, but it keeps the inference path behavior more
consistent and avoids confusing processed-data cache names.

## Result-Set Name Helper

Add a small helper function in the inference workflow to derive the result-set
name from an input path.

Expected behavior:

- input: string or `Path`
- output: basename without final suffix
- use `Path(...).stem`

Fallback:

- if the derived stem is empty for any reason, fall back to `"test_set"`

This preserves backward-safe behavior for unusual edge cases.

## API Design

### `predict()`

Keep the existing API:

- input argument remains `data_file_path`

Add internal logic:

- derive `result_set_name = stem(data_file_path)`
- pass `result_set_name` into `evaluate_model()`
- pass `dataset_type` into `_build_gepdataset_config()`

### `predict_and_visualize()`

Keep the same user-facing behavior, but ensure that:

- the result-set name still comes from the actual test-set input path
- visualization outputs are generated under the same renamed subfolder

## Backward Compatibility

This change is intentionally small:

- old runs already stored under `test_set` remain unchanged
- new runs will use input-file-specific folder names
- no existing file content format changes

The only path change is the subfolder name created for future inference runs.

## Logging

Add a concise log line during inference such as:

- `Inference result subfolder: simu_bulk_exp_Test_set1_log2cpm1p`

This makes the chosen output path visible in the console and easier to confirm.

## Testing Plan

### Targeted Checks

1. Run a small inference call with:
   - `data_file_path=.../simu_bulk_exp_Test_set1_log2cpm1p.h5ad`
2. Confirm the output path is:
   - `test_results/simu_bulk_exp_Test_set1_log2cpm1p/`
3. Confirm nested folders are created:
   - `cell_prop/`
   - `gep/`
   - `gep/sc_gep/`
4. Confirm `predicted_cell_prop.csv` is saved in the renamed subfolder.

### Additional Naming Check

Repeat with a file such as:

- `simu_bulk_exp_Mixed_N100K_segment_without_filtering_hnscc_log2cpm1p_selected_50.h5ad`

Expected subfolder:

- `simu_bulk_exp_Mixed_N100K_segment_without_filtering_hnscc_log2cpm1p_selected_50`

### Consistency Check

Verify that:

- preprocessing cache uses `processed_<dataset_type>`
- inference outputs still return valid `result_dir` values to downstream code

## Risks

The main risk is very small:

- any code outside `evaluate_model()` that hard-codes `test_set` would need to
  use the returned path instead

Based on current inspection, the change should be localized, because
`evaluate_model()` already returns `test_set_result_dir` and downstream logic
should rely on that returned path rather than reconstructing it.

## Recommendation

Implement the smallest possible fix:

1. derive the result-set name from the inferred file path
2. replace the hard-coded `test_set` folder name
3. pass through `dataset_type` to preprocessing config for consistency
4. keep everything else unchanged
