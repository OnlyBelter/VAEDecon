# Design: HVG3000 cosine inter-sample similarity for test-set scGEP outputs

Last updated: 2026-08-13

## Status

Approved in chat, pending implementation.

## Goal

Add a second inter-sample similarity analysis for predicted test-set single-cell
GEP outputs in VAEDecon. The new analysis should run alongside the existing CCC
workflow and save results to a parallel folder that uses cosine similarity
computed on the top 3000 highly variable genes (HVGs).

## User requirements

The user asked for the following behavior:

1. Keep the existing CCC-based inter-sample similarity outputs unchanged.
2. Add a new metric based on cosine similarity over the top 3000 HVGs.
3. Save the new outputs in a parallel result folder by replacing the metric
   name in the directory and file naming.
4. Mirror the general logic of the existing VAEDecon test-set result outputs.
5. Refer to the SimuTME sample-diversity implementation for cosine-based
   comparison behavior, while using similarity outputs for VAEDecon.
6. When defining HVGs, use the SCT reference dataset associated with the
   current test set.
7. Compute HVGs separately for each cell type using only the SCT samples of
   that same cell type, not by pooling multiple cell types together.
8. Consider the possibility that the user may later compare samples across
   different test sets.
9. Reject the earlier idea of silently using duplicated references or unclear
   sources. The design should keep the source of HVGs explicit and local to
   each test set.

## Motivation

The current VAEDecon test-set evaluation already saves inter-sample similarity
results under `inter_sample_similarity_ccc`. These outputs are useful because
they compare whether the reconstructed single-cell GEPs preserve the
sample-to-sample structure of the true matched SCT references for each cell
type.

However, CCC is only one notion of similarity. The user also wants a gene-space
comparison that emphasizes high-variance structure, using cosine similarity on
the top 3000 HVGs. This complements CCC rather than replacing it:

- CCC stays as the current full-gene agreement metric.
- HVG3000 cosine adds a shape-oriented similarity metric on a focused gene set.

## Non-goals

- Do not remove or rename the existing `inter_sample_similarity_ccc` folder.
- Do not change the meaning or file contents of the current CCC outputs.
- Do not add a new user-facing configuration knob for the number of HVGs in
  this change.
- Do not introduce pooled-sc or custom-gene-list HVG sourcing in this change.
- Do not convert the saved VAEDecon matrices to cosine distance (`1 - cosine`).
- Do not implement a new cross-test-set similarity workflow in this same
  change.

This feature is intentionally scoped to one fixed analysis mode:

- gene source: top 3000 HVGs
- HVG reference: current test-set SCT reference dataset
- metric: cosine similarity

## Current behavior

`plot_single_cell_gep()` writes reconstructed and true SCT-like GEP matrices for
each cell type and then saves inter-sample CCC outputs under:

`<sc_gep_result_dir>/inter_sample_similarity_ccc`

For each cell type and thresholded sample subset, it saves:

- `true_vs_true`
- `recon_vs_recon`
- `true_vs_recon`

Each comparison currently produces:

- a CSV matrix
- a heatmap image
- a clustermap image
- an `index.html` gallery for the folder

## Proposed change

Add a second, parallel workflow that computes cosine-similarity matrices on the
top 3000 HVGs derived from the current test set's SCT reference dataset.

### Output directory

Create and populate the folder:

`<sc_gep_result_dir>/inter_sample_similarity_hvg3000_cosine`

This folder is parallel to the existing:

`<sc_gep_result_dir>/inter_sample_similarity_ccc`

### HVG source

For each test set, derive the top 3000 HVGs from that test set's SCT reference
dataset, meaning the same SCT reference already used as the ground truth source
for the single-cell GEP comparison.

The HVGs are test-set-specific, not global across all test sets.

### Cell-type-specific HVG calculation

For a given test set and cell type:

1. subset the test-set SCT reference dataset to that cell type only
2. compute gene-wise variance across those samples
3. rank genes by variance in descending order
4. keep the top 3000 genes
5. intersect those genes with the genes present in the true and reconstructed
   scGEP matrices for that same cell type

This is preferred over computing HVGs across all samples from multiple cell
types in the test-set SCT reference, because the current analysis target is
within-cell-type inter-sample similarity. A pooled HVG list would be dominated
by between-cell-type marker genes and would weaken the cell-type-specific
sample-to-sample comparison.

### Matrix definitions

For each cell type and filtered sample subset, build the same three matrices as
the CCC workflow:

1. `true_vs_true`
2. `recon_vs_recon`
3. `true_vs_recon`

Each matrix entry should be the cosine similarity between the corresponding two
sample vectors after both are restricted to the shared top 3000 HVGs.

### Similarity orientation

Although SimuTME sample-diversity uses `1 - cosine similarity` as a distance to
centroid, VAEDecon should save cosine similarity itself in this new folder,
because this output is an inter-sample similarity analysis rather than a
distance-to-centroid analysis.

### File naming

Mirror the current CCC naming pattern, replacing only the metric tag:

- current example:
  `Cancer Cells_ccc_true_prop_ge_0p01_true_vs_true.csv`
- new example:
  `Cancer Cells_hvg3000_cosine_true_prop_ge_0p01_true_vs_true.csv`

The same naming rule applies to the heatmap and clustermap image files.

### Gallery behavior

Generate a dedicated gallery HTML file inside
`inter_sample_similarity_hvg3000_cosine`, analogous to the existing CCC
gallery. It should:

- group outputs by cell type
- show the same three comparison types
- link to CSV, heatmap, and clustermap artifacts
- keep the layout parallel to the CCC gallery

The title and parsing logic must use the new `hvg3000_cosine` metric tag rather
than hard-coding `ccc`.

### Metadata for reproducibility

To make later cross-test-set comparisons possible, each metric-specific result
folder should also save lightweight metadata describing the HVG basis used for
that test set and cell type. At minimum, this should include:

- the SCT reference dataset path used to derive HVGs
- the cell type name
- the number of requested HVGs
- the number of aligned HVGs actually used
- the exact ordered HVG gene list used for that cell type

This metadata does not itself implement cross-test-set comparison, but it makes
the basis transparent and auditable.

## Data flow

1. Prediction on a test set produces reconstructed scGEP outputs as today.
2. `plot_single_cell_gep()` continues to write the current per-cell-type true
   and reconstructed scGEP tables.
3. The existing CCC workflow runs unchanged.
4. A new HVG3000 cosine workflow runs in parallel:
   - derive top 3000 HVGs from the test-set SCT reference dataset separately
     for each cell type
   - intersect those HVGs with the genes available in both true and
     reconstructed per-cell-type matrices
   - compute pairwise cosine-similarity matrices
   - save CSV, heatmap, clustermap, folder gallery, and HVG metadata

## Cross-test-set comparability

The current implementation is test-set-local. That means the HVG basis for one
cell type in test set A may differ from the HVG basis for the same cell type in
test set B.

This is acceptable for the requested per-test-set outputs, but it means the
resulting cosine-similarity matrices are not automatically on the same gene
basis across test sets.

Therefore:

1. within one test set, the new outputs are directly interpretable as designed
2. across different test sets, users should not assume direct comparability
   unless the HVG lists are confirmed to match or a shared HVG basis is
   explicitly recomputed

If a future feature is added for direct cross-test-set comparison, it should
derive a shared per-cell-type HVG basis across the participating test sets or
use a designated common reference. That workflow is intentionally out of scope
for this change, but the saved HVG metadata should make it straightforward to
add later.

## Edge cases

### Fewer than 3000 available genes

If fewer than 3000 HVGs remain after intersecting the SCT-derived HVGs with the
available gene columns, use all remaining intersected genes instead of failing.

### No HVGs remain after intersection

If no genes remain after alignment, save empty-style outputs consistent with the
current empty CCC handling, and avoid crashing the full prediction workflow.

### Too few SCT samples in a cell type

If a specific cell type in the test-set SCT reference has too few samples for a
meaningful variance ranking, the workflow should still behave gracefully. The
practical fallback is to rank on whatever samples exist and proceed with the
available genes, unless the subset is empty.

### Zero-norm sample vectors

If a sample vector has zero norm after HVG restriction, define its cosine
similarity conservatively using the same defensive pattern as SimuTME's cosine
helper so the workflow does not crash.

## Implementation shape

Keep this change surgical and local to the evaluation layer.

Recommended implementation shape:

1. Add cosine/HVG helper functions in `vaedecon/plot/evaluate_result.py`.
2. Generalize only the small pieces that currently hard-code the `ccc` metric
   tag in naming and gallery generation.
3. Keep the existing CCC computation helper unchanged unless a tiny shared
   utility clearly reduces duplication without broad refactoring.

## Verification

Add regression tests that cover:

1. new result directory and file naming for `hvg3000_cosine`
2. cosine-matrix generation on a tiny toy example
3. gallery generation for the new metric tag
4. metadata output describing the HVG basis
5. preservation of the existing CCC outputs and tests

## Success criteria

This feature is complete when:

1. test-set prediction still writes the existing CCC outputs
2. a parallel folder `inter_sample_similarity_hvg3000_cosine` is created
3. each eligible cell type receives `true_vs_true`, `recon_vs_recon`, and
   `true_vs_recon` cosine-similarity outputs on top 3000 SCT-derived HVGs
4. the new folder includes CSVs, heatmaps, clustermaps, a browseable HTML
   gallery, and HVG metadata
5. the saved outputs clearly record the gene basis used for each cell type so
   later cross-test-set comparison is possible without ambiguity
6. existing CCC behavior remains backward compatible
