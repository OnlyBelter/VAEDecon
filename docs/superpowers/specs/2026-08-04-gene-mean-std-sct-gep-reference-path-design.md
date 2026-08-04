# Design: Dedicated SCT reference for gene mean/std (sct_gep mode)

## Goal

Support a dedicated single-cell reference `.h5ad` for computing gene mean/std when:

- `data.gene_mean_std_source: "sct_gep"`

This reference must be configurable independently from the training SCT datasets listed in `data.sct_file_path`.

## Motivation

In some runs, training may use multiple SCT datasets (`data.sct_file_path`) while the gene mean/std statistics should be computed from one specific, stable SCT reference dataset. This improves reproducibility and reduces configuration coupling.

In addition, the gene mean/std output should remain portable by being saved under the model result directory, so the whole model folder can be moved/packaged without depending on external cache files.

## Non-goals

- Change the pooled-scRNA workflow (`data.gene_mean_std_source: "pooled_sc"`).
- Change the output CSV schema consumed by the model.
- Add complex caching/invalidation logic beyond what already exists.

## Current behavior (baseline)

When `data.gene_mean_std_source != "pooled_sc"` (i.e. `sct_gep` mode), training computes gene mean/std via:

- `load_or_compute_gene_mean_std(...)` in `vaedecon/utility/read_file.py`

The reference dataset currently comes from:

- `data.sct_gep_file_path`

The output file path currently depends on the reference dataset location.

## Target behavior

### 1) New configuration field

Add a new optional field in `DataConfig`:

- `data.gene_mean_std_sct_gep_file_path: str | Path`

Interpretation:

- This field is only used when `data.gene_mean_std_source == "sct_gep"`.
- If it is set (non-empty), it is used as the reference `.h5ad` for computing gene mean/std.
- If it is empty, fall back to the existing `data.sct_gep_file_path` (backward compatible).

### 2) Output location remains under model directory (portability)

When `data.gene_mean_std_source == "sct_gep"`, the computed gene mean/std CSV must be written under:

- `model.model_dir`

This keeps the output self-contained with model artifacts.

### 3) Stable output filename

Use a stable naming rule under `model.model_dir`:

- If `data.scaling_by_constant: true`:
  - `gene_mean_std_log2p1_scaled_by_<scaling_factor>.csv`
- Else:
  - `gene_mean_std_log2p1.csv`

This is intentionally independent of the reference SCT basename (per user preference).

## Data flow

### Reference resolution (sct_gep)

When `data.gene_mean_std_source == "sct_gep"`:

1. Resolve `ref_fp`:
   - if `data.gene_mean_std_sct_gep_file_path` is set: use it
   - else: use `data.sct_gep_file_path`
2. Pass `ref_fp` into `load_or_compute_gene_mean_std(...)`.

### Output path (sct_gep)

When building the `ModelConfig.gene_mean_std_fp`, always set it to a file under `model.model_dir` using the stable naming rule above.

## Validation and failure modes

- If `data.gene_mean_std_source == "sct_gep"` and both:
  - `data.gene_mean_std_sct_gep_file_path` is empty, and
  - `data.sct_gep_file_path` is empty
  then fail fast with a clear error (missing SCT reference for gene mean/std).

## Files to change

- `vaedecon/configs/default_config.py`
  - add `gene_mean_std_sct_gep_file_path` to `DataConfig`
  - add validation ensuring a usable SCT reference exists when `gene_mean_std_source == "sct_gep"`
- `vaedecon/configs/example_config.yaml`
  - add `gene_mean_std_sct_gep_file_path` example (empty by default)
- `vaedecon/workflow/train.py`
  - in `_compute_gene_statistics(...)`, use the resolved `ref_fp` for `sct_gep` mode
  - in `_build_vae_config(...)`, set `gene_mean_std_fp` under `model_dir` for `sct_gep` mode using the stable naming rule

## Backward compatibility

- Existing configs that only set `data.sct_gep_file_path` keep working unchanged.
- The new field is additive and optional.

## Testing plan

Update/extend tests to cover:

1. YAML config loading:
   - the new field parses correctly
2. Reference resolution:
   - when new field is set, it is preferred over `sct_gep_file_path`
   - when new field is empty, fallback to `sct_gep_file_path`
3. Output path rule:
   - `gene_mean_std_fp` is under `model.model_dir` in `sct_gep` mode
   - filename matches the stable naming rule
