# Design: Gene mean/std workflow switch (pooled scRNA-seq)

This document describes how you compute the reference gene mean and standard
deviation across cell types for training, and how you switch between two
workflows without changing existing behavior.

## Goals

- Let you select how the model derives the reference gene mean and standard
  deviation matrix.
- Keep the existing sctGEP-based workflow unchanged.
- Add a pooled scRNA-seq workflow that computes the statistics on-the-fly
  during each training run.
- Ensure the output file format remains compatible with how the VAE loads gene
  statistics.

## Non-goals

- Optimize runtime for very large pooled scRNA-seq datasets.
- Add caching for the pooled scRNA-seq workflow.
- Change loss definitions or how gene statistics are consumed in the model.

## Current behavior (baseline)

Training computes or loads a reference matrix containing per-gene mean and
standard deviation for each cell type.

- Training entry point:
  - `VAEDeconTrainer._compute_gene_statistics(...)` in
    `vaedecon/workflow/train.py`
- Current implementation:
  - `load_or_compute_gene_mean_std(...)` and
    `get_gene_mean_std_across_cell_types(...)` in `vaedecon/utility/read_file.py`
- Reference dataset:
  - `data.sct_gep_file_path` points to an `.h5ad` that stores cell-type
    membership as one-hot indicator columns in `adata.obs`.
- Output:
  - A wide CSV at `model.gene_mean_std_fp` with columns:
    - `<cell_type>_avg` and `<cell_type>_std` for each cell type.

The VAE loads the CSV and registers buffers:
- `g_mean`, `g_std`
- `g_mean_non_log`, `g_std_non_log`

## New behavior (pooled scRNA-seq workflow)

You add a second workflow that computes gene mean/std from a pooled single-cell
RNA-seq dataset during training.

### Inputs

- `data.pooled_sc_h5ad_path`: a pooled scRNA-seq `.h5ad`, for example
  `merged_12_sc_datasets_231003.h5ad`.
- `data.pooled_sc_cell_type_col`: a string column in `adata.obs`, default
  `cell_type`.
- `data.pooled_sc_sample_size`: max cells sampled per cell type, default 1000.
- `data.pooled_sc_seed`: deterministic sampling seed, default 123.

Assumptions:
- `adata.X` is in `log_space` (log2(CPM/TPM + 1)).
- `adata.obs[data.pooled_sc_cell_type_col]` stores cell type names as strings
  matching your training cell type list.

### Computation

For each training cell type:

1. Select cells where `obs[cell_type_col] == <cell_type>`.
2. Sample up to `pooled_sc_sample_size` cells (without replacement).
3. Convert the sampled expression from `log_space` to TPM-like space using the
   existing implementation path:
   - `ReadExp(..., exp_type="log_space").to_tpm()`
4. Align gene order to the training gene list by reusing the existing alignment
   behavior:
   - fill missing genes with zeros and reorder to `dataset.gene_list`.
5. Compute per-gene mean and standard deviation across sampled cells.
6. Apply `log2(x + 1)` and scaling by `scaling_factor` to match existing
   training conventions.

### Output and file naming

- Write the wide CSV into the model directory (same directory as the trained
  model artifacts).
- Use a stable filename:
  - `gene_mean_std_log2p1_scaled_by_<scaling_factor>.csv`

Example:
- `gene_mean_std_log2p1_scaled_by_20.0.csv`

The output must keep the same wide format expected by the model:
- columns: `<cell_type>_avg` and `<cell_type>_std`
- rows: genes in the exact `dataset.gene_list` order
- column order: follow the training `cell_type_list.txt` order

## Configuration interface

Add an explicit switch in the config to select the workflow:

- `data.gene_mean_std_source`: `"sct_gep"` or `"pooled_sc"` (default:
  `"sct_gep"`).

If `gene_mean_std_source == "sct_gep"`, use the existing implementation and
existing output naming logic.

If `gene_mean_std_source == "pooled_sc"`, compute from the pooled `.h5ad` and
write the stable output file in the model directory.

## Failure modes and validation

- Missing or empty pooled `.h5ad` path:
  - Fail fast with a clear error.
- Missing `cell_type` column in `adata.obs`:
  - Fail fast with a clear error that includes the expected column name.
- Training cell types not present in pooled dataset:
  - Fail fast and list missing cell types.
- All sampled cells have zero library size after conversion:
  - Treat as an error because mean/std become degenerate.
- Output CSV does not match the training gene list order:
  - Treat as an error.

## Testing plan

- Add a unit-style test using a tiny synthetic AnnData object with:
  - two cell types,
  - a small gene list with a missing gene to verify zero fill,
  - known values to verify mean/std.
- Add an integration-style test that:
  - runs the training prep path up to `_compute_gene_statistics(...)`,
  - verifies the output file exists and column naming matches expectations.

## Rollout plan

- Default remains `"sct_gep"`, so existing runs behave the same.
- You enable the new workflow by setting:
  - `data.gene_mean_std_source: pooled_sc`
  - `data.pooled_sc_h5ad_path: <path>`
