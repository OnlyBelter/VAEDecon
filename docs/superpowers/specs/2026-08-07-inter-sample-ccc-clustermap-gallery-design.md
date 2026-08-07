# Inter-Sample CCC Clustermap Gallery Design

## Goal

Improve inspection of per-cell-type inter-sample CCC outputs under
`gep/sc_gep/inter_sample_similarity_ccc/` by:

1. adding clustered CCC heatmaps based on `seaborn.clustermap`, and
2. generating a simple static HTML gallery that organizes all CCC outputs in one place.

This is an additive change only. Existing CSV outputs and existing non-clustered
heatmaps must remain unchanged.

## Motivation

The current outputs are useful but hard to inspect at scale because:

- rows and columns stay in the original sample order, which can hide block
  structure or collapsed clusters in the CCC matrix,
- every cell type produces multiple files, and browsing them file-by-file is slow,
- side-by-side comparison across `true_vs_true`, `recon_vs_recon`, and
  `true_vs_recon` is cumbersome.

Clustered heatmaps help surface sample substructure directly. A lightweight HTML
gallery makes it easier to review all cell types and comparison types in one page.

## Scope

In scope:

- add clustered PNG outputs for each non-empty pairwise CCC matrix,
- keep the existing CSV and standard heatmap outputs,
- add one static `index.html` file to the similarity result directory,
- add focused tests for clustered plotting and the HTML gallery.

Out of scope:

- changing the CCC computation,
- changing existing CSV schemas,
- removing or renaming existing PNG outputs,
- introducing a web server, JavaScript app framework, or new runtime dependency.

## Output Behavior

For each existing CCC matrix output:

- `<prefix>_true_vs_true.csv`
- `<prefix>_true_vs_true.png`
- `<prefix>_recon_vs_recon.csv`
- `<prefix>_recon_vs_recon.png`
- `<prefix>_true_vs_recon.csv`
- `<prefix>_true_vs_recon.png`

the pipeline will additionally write:

- `<prefix>_true_vs_true_clustermap.png`
- `<prefix>_recon_vs_recon_clustermap.png`
- `<prefix>_true_vs_recon_clustermap.png`

If no sample passes the threshold, keep the existing empty placeholder PNG logic
and skip clustered output for that matrix.

## Plotting Design

Add a sibling plotting helper next to the existing standard CCC heatmap helper in
`vaedecon/plot/evaluate_result.py`.

### Standard Heatmap

The current helper remains unchanged in purpose and filename behavior.

### Clustermap

The new helper will:

- call `sns.clustermap(matrix_df, ...)`,
- cluster both rows and columns,
- use the same CCC color scaling rule as the standard heatmap:
  - `vmin = max(0.0, matrix_min)`
  - `vmax = max(vmin, matrix_max)`
- use the same color map as the standard heatmap for consistency,
- save directly to a separate `*_clustermap.png` file.

The clustered plot is only generated for non-empty matrices.

## Gallery Design

Generate a single static `index.html` inside the
`inter_sample_similarity_ccc/` result directory.

The gallery should:

- group outputs by cell type,
- within each cell-type section, show the three matrix types:
  - `true_vs_true`
  - `recon_vs_recon`
  - `true_vs_recon`
- for each matrix type, include links to:
  - the CSV,
  - the standard heatmap PNG,
  - the clustered heatmap PNG when present,
- embed the standard and clustered PNGs inline for quick visual review.

The file must use relative paths only so it can be opened directly in a browser
without a local server.

## Integration Point

Integrate gallery generation at the end of the existing
`_save_selected_sample_similarity_outputs(...)` workflow in
`vaedecon/plot/evaluate_result.py`.

Recommended structure:

1. keep matrix computation unchanged,
2. save CSV,
3. save standard heatmap,
4. save clustered heatmap when applicable,
5. refresh the HTML gallery for the directory.

This keeps the change local to the current plotting pipeline and avoids touching
unrelated evaluation code.

## Error Handling

- Empty/no-sample cases keep the current placeholder heatmap behavior.
- If clustered output is skipped because the matrix is empty, the gallery should
  still include the CSV and standard PNG when present.
- The gallery builder should tolerate partial directories and only render links
  for files that actually exist.

## Testing

Add or update tests in `tests/test_selected_sample_gep_outputs.py`.

Required coverage:

1. clustered heatmap helper passes the expected `vmin` / `vmax` bounds to
   `sns.clustermap`,
2. selected-sample similarity output generation writes the expected
   `*_clustermap.png` filename for non-empty matrices,
3. empty/no-sample paths do not attempt clustered plotting,
4. gallery generation includes expected cell type names and file links.

Verification:

- run `python -m compileall` on touched files,
- run relevant targeted tests when available.

## Files Expected To Change

- `vaedecon/plot/evaluate_result.py`
- `tests/test_selected_sample_gep_outputs.py`

## Acceptance Criteria

- Existing CCC CSV and PNG outputs remain available with unchanged filenames.
- New clustered PNG outputs are written for non-empty matrices.
- The clustered plots use nonnegative, data-driven CCC bounds.
- A static `index.html` is created in the result directory and groups outputs by
  cell type and matrix type.
- The new behavior is covered by focused tests.
