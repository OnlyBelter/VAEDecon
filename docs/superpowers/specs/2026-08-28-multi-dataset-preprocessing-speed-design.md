# Multi-dataset preprocessing speed design

Last updated: 2026-08-28

## Status

Approved and partially implemented. Expanded on 2026-08-28 to add
metadata-only common-gene discovery, cache-local persistence of the discovered
common gene list, configurable source-load parallelism, and group-first
preprocessing to reduce peak RAM.

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
3. raw file ingestion still performs expensive full-width reads
   - each `.h5ad` is opened and materialized before the final gene intersection
4. many large DataFrames can coexist before the final merge
   - this increases peak RAM and can cause the process to be killed by the
     server even when Python raises no traceback

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

1. loads source files
2. merges them
3. applies gene filtering
4. applies transformation
5. writes one processed cache

The file loading stage is still pandas-heavy even after the first round of
speedups:

- each `.h5ad` is read at full gene width through `ReadH5AD.get_df(convert_to_tpm=True)`
- final common genes are only enforced after full matrices are already in memory
- group-local merges help, but the loader can still do unnecessary IO and keep
  more columns than needed during the heaviest stage

This is correct but slow for many independent source files.

## Design goals

1. Preserve current dataset semantics.
2. Preserve current cache file formats.
3. Avoid changing training behavior, loss behavior, or model outputs.
4. Improve repeated-run startup dramatically when raw inputs are unchanged.
5. Improve first-run preprocessing speed for many input files.

## Recommended approach

Use a five-part fix:

1. shared content-addressed preprocessing cache
2. metadata-only first pass to discover and persist the final common gene set
3. configurable parallel source-file ingestion during first-time preprocessing
4. group-first preprocessing to bound peak RAM
5. shared in-run `sctGEP` query deduplication for matched targets

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

## Part 2: metadata-only common-gene discovery and persistence

### Problem

The current loader reads full expression matrices before it knows the final gene
intersection that will survive preprocessing.

This wastes:

- disk IO, because genes that will later be dropped are still read from disk
- RAM, because full-width DataFrames are materialized before the intersection
- CPU time, because downstream alignment and conversion operate on more columns
  than necessary

### Proposed change

Add a lightweight first pass that reads only per-file gene metadata.

For each input file:

1. if the source is `.h5ad`, open it in backed mode and read only `var_names`
2. if the source is `.csv`, read only the header row
3. convert those names into a normalized ordered gene list for that source

Then compute one final target gene list for the real load phase:

1. start from the intersection across all training sources
2. if `gene_list_file` is configured, intersect with that too while preserving
   the order defined by `gene_list_file`
3. save this final gene list into the fingerprinted processed-cache directory
4. use the saved cache-local common-gene file when loading all later full
   matrices in that preprocessing run

### Saved artifact

The discovered final gene list should be written as a cache-local text file,
for example:

```text
<processed_cache_dir>/common_gene_list.txt
```

This file becomes the single source of truth for the gene space used by that
fingerprinted preprocessing cache.

That means:

- bulk source loading should read only the genes listed in
  `common_gene_list.txt`
- matched-target `sctGEP` loading should also align directly to
  `common_gene_list.txt`
- the optimized training preprocessing path should not need a later
  `align_with_gene_list(...)` pass just to rediscover or re-slice the same gene
  set again

### Scope boundary

This change should not alter:

- downstream tensor shapes beyond the already expected common-gene reduction
- training-target alignment semantics
- cache file formats
- model behavior

### Expected result

The real load phase should materialize only the final gene space that will be
kept anyway.

That should reduce:

- peak memory during preprocessing
- total bytes read from source matrices
- time spent aligning and transforming unused genes

Because downstream readers will consume the saved common-gene list directly,
the repeated informational message
`16217 common genes will be used, 1617 genes will be removed.` should disappear
from the main optimized preprocessing path.

## Part 3: configurable parallel source-file ingestion

### Problem

First-time preprocessing still requires reading many raw datasets, and the
optimal worker count depends on available RAM.

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

Expose a data config field such as:

```yaml
data:
  max_parallel_source_file_loads: 2
```

Behavior:

- default to a conservative value such as `2`
- clamp the effective worker count to `min(configured_value, n_files)`
- allow `1` on memory-tight servers
- allow higher values on stronger machines

This gives users a direct stability/performance knob without changing dataset
semantics.

## Part 4: group-first preprocessing to bound peak RAM

### Problem

Even with grouping by shared `training_sct_gep_file_path`, preprocessing can
still keep too much data alive if all groups contribute to one large in-memory
merge before later stages complete.

### Proposed change

Process one group at a time after the common-gene first pass.

For each preprocessing group:

1. load only the final target genes for files in that group
2. merge that group's expression and label tables
3. apply the normal transformation pipeline for that group
4. convert to arrays or save a temporary group-level intermediate on disk
5. free group-local pandas objects before moving to the next group

After all groups are processed:

1. concatenate the group-level outputs in the original file order
2. write the final processed dataset cache in the same format as today

### Intermediate storage policy

The recommended implementation should allow temporary group-level intermediates
under the processed dataset directory during one preprocessing run.

These intermediates are:

- implementation details, not part of the public cache contract
- safe to overwrite when `force_reprocess: true`
- safe to delete after the final cache is written

### Scope boundary

This change should not alter:

- the final processed cache layout
- sample ordering relative to the configured training file order
- matched-target tensor alignment
- scaling semantics

### Expected result

Peak preprocessing memory should scale closer to the largest group instead of
the full union of all source files.

## Part 5: shared in-run `sctGEP` query deduplication

### Problem

Matched `sctGEP` supervision can still reopen the same `training_sct_gep_file_path`
multiple times inside one preprocessing run when several `training_target_sets`
share that same source file.

This is common for simulated bulk datasets generated by sampling from one shared
single-cell reference. In that case, each bulk dataset may have its own
`sample2cell` mapping file, but all of them point to the same `sctGEP` `.h5ad`.

### Proposed change

When building dataset-aligned matched `true_sct_gep` targets:

1. group `training_target_sets` by normalized `training_sct_gep_file_path`
2. for each shared `sctGEP` source:
   - read each target set's `sample2cell` mapping
   - collect the union of all referenced `selected_cell_id` values
   - load and gene-align that shared `sctGEP` file once
3. slice the shared aligned DataFrame back into each target set's local
   `true_sct_gep` tensor

### Scope boundary

This change should not alter:

- namespacing by `Train_setX`
- matched-target tensor shapes
- cell-type masking behavior
- on-disk processed dataset cache layout
- cross-run cache semantics

### Expected result

If `Train_set1` to `Train_set5` share one `training_sct_gep_file_path`, the
reference `.h5ad` should be opened once for the union of required cells instead
of being reopened once per training set.

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
- load only the final target gene list after the metadata pass
- process one group at a time instead of one full union merge

### 3. Metadata/header pass can disagree across file formats

If `.h5ad` and `.csv` sources expose genes differently, the computed common gene
set could be wrong.

Mitigation:

- normalize all discovered gene names to strings
- preserve existing case-sensitive matching semantics
- add regression tests that mix `.h5ad` and `.csv` inputs

### 4. Temporary group intermediates can leave residue after failure

If preprocessing crashes mid-run, temporary group artifacts may remain in the
processed cache directory.

Mitigation:

- place them under a dedicated temporary subdirectory
- overwrite them on the next `force_reprocess: true` run
- delete them after successful final cache write

### 5. Path-order semantics

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
4. Confirm that the real load phase only materializes the final target gene set
   instead of the full original width.

### Regression coverage

Add tests for:

1. stable cache directory fingerprinting
2. different hyperparameter-only ablations reusing the same processed dataset
   cache
3. preprocessing-sensitive config changes producing different cache keys
4. metadata-only common-gene discovery across multiple `.h5ad` inputs
5. discovered common genes being saved into the fingerprinted cache directory
6. repeated preprocessing loads reusing the saved common-gene file
7. `gene_list_file` intersection preserving configured gene order
8. mixed `.csv`/`.h5ad` metadata discovery producing the correct common genes
9. configurable worker count being honored by the preprocessing loader
10. grouped preprocessing preserving merged sample IDs and labels in original
   file order
11. temporary group intermediates being cleaned up after a successful run
12. shared `sctGEP` sources being loaded once within one preprocessing run

## Recommendation

Implement the balanced fix:

1. shared content-addressed preprocessing cache
2. metadata-only common-gene discovery plus cache-local persistence before full
   reads
3. configurable conservative parallel multi-file ingestion
4. group-first preprocessing with optional temporary intermediates
5. in-run deduplication for shared matched-target `sctGEP` sources

This is the smallest design that addresses both of the user’s actual needs:

- much faster repeated runs across ablations
- meaningfully faster first-time preprocessing for many training datasets
- lower peak RAM during large multi-dataset preprocessing
- a direct worker-count knob for unstable shared servers
- fewer repeated `sctGEP` reference-file reads inside one matched-target build
