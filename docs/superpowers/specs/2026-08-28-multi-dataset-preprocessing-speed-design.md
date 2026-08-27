# Multi-dataset preprocessing speed design

Last updated: 2026-08-28

## Status

Drafted for review.

## Goal

Reduce the wall-clock time spent before model training starts when `VAEDecon`
loads many training datasets, especially configurations such as:

- `21ds_n-neighbor_ablation71_conditioned_decoder_direct_sct_cellprop_deside_context_fused_latent_d32_staged_sep_kl.yaml`

The design should improve both:

1. first-time preprocessing from raw `.h5ad` / `.csv` inputs, and
2. repeated runs that reuse the same raw datasets under different ablation
   names or model settings.

## Observed problem

Current startup can take about one hour for a 21-dataset training config before
actual model training begins.

The main reasons are:

1. preprocessing is explicitly forced every run:
   - the example 21-dataset config sets `data.force_reprocess: true`
2. the processed dataset cache is keyed by `training.naming_postfix`
   - different ablations using the same raw inputs do not share the same cache
3. raw file ingestion is serial
   - each `.h5ad` is opened, converted, and materialized one-by-one before the
     final merge

## Current behavior summary

### Cache directory naming

In `vaedecon/workflow/train.py`, the processed training dataset directory is
currently created as:

```text
<data_dir>/processed_training_sets_<training.naming_postfix>
```

This means cache reuse is tied to the training run name, not to the actual
dataset/preprocessing content.

Consequence:

- changing only model hyperparameters still rebuilds the whole processed
  dataset from scratch
- multi-ablation sweeps pay repeated preprocessing cost unnecessarily

### Preprocessing flow

`GEPDataset` currently:

1. loads all source files
2. merges them
3. applies gene filtering
4. applies transformation
5. writes one processed cache

The file loading stage is currently serial and pandas-heavy:

- each `.h5ad` is read through `ReadH5AD.get_df(convert_to_tpm=True)`
- each loaded DataFrame is kept until the final `pd.concat(...)`

This is correct but slow for many independent source files.

## Design goals

1. Preserve current dataset semantics.
2. Preserve current cache file formats.
3. Avoid changing training behavior, loss behavior, or model outputs.
4. Improve repeated-run startup dramatically when raw inputs are unchanged.
5. Improve first-run preprocessing speed for many input files.

## Recommended approach

Use a two-part fix:

1. shared content-addressed preprocessing cache
2. parallel source-file ingestion during first-time preprocessing

## Part 1: shared preprocessing cache

### Problem

Cache reuse is currently blocked by run-name-based cache directories.

### Proposed change

Replace the processed dataset directory naming logic with a stable fingerprint
derived from the dataset inputs and preprocessing parameters rather than from
`training.naming_postfix`.

Example conceptual layout:

```text
<data_dir>/processed_training_sets/<dataset_fingerprint>/
```

### Fingerprint inputs

The fingerprint should include the normalized values that affect the cached
training dataset contents:

- resolved `file_paths`
- `gene_list_file`
- `remove_low_var_genes`
- `min_var`
- `scaling_by_constant`
- `scaling_factor`
- `cell_cell2ave_exp_file_path` if used
- training-target-set definitions if they affect cached matched targets
- namespacing inputs derived from `training_target_sets`

### Fingerprint rules

1. Resolve paths to absolute normalized strings before hashing.
2. Sort dictionaries and lists where ordering is not semantically meaningful.
3. Use a deterministic JSON payload and hash it.
4. Keep a human-readable `metadata.json` inside the cache directory for
   debugging.

### Expected result

If two different ablation configs point to the same training files and the same
preprocessing settings, they will reuse the same processed dataset cache.

This should provide the largest gain for repeated experiments.

## Part 2: parallel source-file ingestion

### Problem

First-time preprocessing still requires reading many raw datasets, and that
stage is currently serial.

### Proposed change

Parallelize the independent per-file loading work inside
`GEPPreprocessor._load_and_merge_data(...)`.

Each worker should:

1. open one source file
2. read expression data
3. read cell-fraction labels if present
4. apply stable sample-ID namespacing
5. return the prepared DataFrames to the main process

The main process should then:

1. collect the per-file results
2. perform the final `pd.concat(...)`
3. continue with the existing filtering and transformation pipeline

### Scope boundary

This change should not alter:

- gene filtering logic
- transformation logic
- training-target alignment semantics
- final cached array layout

### Worker count

Start with a conservative default:

- use a bounded worker count such as `min(4, n_files)`

This avoids overloading I/O or RAM on shared systems while still improving
parallel read throughput.

The first implementation can keep this internal rather than adding a new user
config option.

## Non-goals

This design does not include:

- rewriting the cache format into per-file shards
- changing model training logic
- changing DataLoader worker behavior
- changing `.h5ad` storage formats
- adding a new public CLI interface

## Risks and tradeoffs

### 1. Cache invalidation mistakes

If the fingerprint omits a preprocessing-relevant field, the cache may be
reused incorrectly.

Mitigation:

- keep the fingerprint payload explicit and small
- store the payload or a readable summary in cache metadata
- add regression tests for fingerprint sensitivity

### 2. Parallel file loading can increase memory pressure

Loading multiple `.h5ad` files concurrently may increase peak RAM usage.

Mitigation:

- keep the worker count conservative
- parallelize only the independent file-read stage
- keep later merge/transformation logic unchanged in the main process

### 3. Path-order semantics

If file order matters for namespacing or downstream alignment, hashing must
preserve meaningful ordering.

Mitigation:

- preserve order for `file_paths`
- sort only where order is not semantically meaningful

## Verification plan

### Repeated-run verification

1. Run one training config with `force_reprocess: false`.
2. Confirm the processed dataset cache is created in the fingerprinted shared
   directory.
3. Run a second ablation with the same raw datasets and preprocessing settings
   but a different `naming_postfix`.
4. Confirm the second run reuses the same processed dataset cache instead of
   rebuilding from scratch.

### First-run verification

1. Run a representative multi-dataset config such as the 21-dataset YAML.
2. Compare preprocessing wall time before and after the change.
3. Confirm that the final cached data shape, sample IDs, gene list, labels, and
   matched `true_sct_gep` targets remain unchanged.

### Regression coverage

Add tests for:

1. stable cache directory fingerprinting
2. different hyperparameter-only ablations reusing the same processed dataset
   cache
3. preprocessing-sensitive config changes producing different cache keys
4. parallel load path preserving merged sample IDs and labels

## Recommendation

Implement the balanced fix:

1. shared content-addressed preprocessing cache
2. conservative parallel multi-file ingestion

This is the smallest design that addresses both of the user’s actual needs:

- much faster repeated runs across ablations
- meaningfully faster first-time preprocessing for many training datasets
