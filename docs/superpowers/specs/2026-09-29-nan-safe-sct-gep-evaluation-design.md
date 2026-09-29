# NaN-safe SCT GEP evaluation design

## Goal

Prevent non-finite values in selected SCT reference profiles from aborting
stage-local test-set visualization while preserving the source expression data
and making excluded observations visible.

## Evidence

For `Non-plasma B cells`, the selected SCT truth CSV has shape `(15184, 50)`
and contains 435 NaNs with no infinities. The reconstructed CSV has the same
shape and contains no NaNs or infinities. The failure therefore originates in
the selected truth expression values, not the reconstructed output.

SimuTME omits cell types with zero sampled cell counts from the sample-to-cell
mapping. Zero cell proportions do not directly create NaNs in selected gene
expression values. The current SCT imputation path validates finite values,
but the workflow can reuse pre-existing original or imputed SCT files without
regenerating them. The source H5ADs still need to be checked on the server that
produced the error.

## Design

- Apply pairwise finite masks when comparing selected truth and reconstructed
  expression values. Scatter points, correlation, and RMSE use the same finite
  pairs.
- Compute each pairwise CCC using only genes finite in both compared sample
  vectors. Preserve undefined scores when too few genes remain or the CCC
  denominator is zero.
- Mask undefined cells in similarity heatmaps. Skip hierarchical clustermap
  generation when its input matrix contains undefined entries rather than
  inventing replacement scores.
- Report the number of excluded gene-value pairs for each cell type. Never
  alter the raw SCT reference or overwrite its CSVs.
- Apply the same finite-pair rule to optional per-sample GEP metrics so
  enabling that output does not reintroduce the failure.
- Keep SimuTME unchanged unless inspection of the original and imputed SCT
  H5ADs on the error-producing server identifies an upstream simulation defect.

## Verification

- Test that all-finite data produces the existing correlation, RMSE, and CCC
  results.
- Test that truth and prediction NaNs are excluded pairwise and reported.
- Test that pairwise CCC uses each sample pair's finite gene intersection.
- Test that undefined similarity entries are masked and do not reach
  clustering.
- On the training server, compare selected cell IDs in the original and
  sorted-bulk-imputed SCT H5ADs, and check the SimuTME log for existing-output
  reuse messages.
