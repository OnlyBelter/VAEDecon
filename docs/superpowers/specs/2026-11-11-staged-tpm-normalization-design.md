# Staged TPM normalization and cache reuse

This design extends the staged gene-filtering correction with a consistent
training and inference input contract and removes redundant bulk conversions.
It applies to newly trained staged models; non-staged workflows and existing
checkpoint inference retain their previous behavior.

## Input contract

The training H5AD expression matrices contain `log2(TPM + 1)`. They are
already gene-length corrected; do not apply gene-length correction again or
infer an input format from expression values.

For each accepted training or inference sample:

1. Select the ordered model genes.
2. Convert their log-expression values to linear TPM values.
3. Renormalize the selected genes to sum to one million.
4. Apply `log2(TPM + 1)`.
5. Divide by `data.scaling_factor` when `scaling_by_constant` is true;
   otherwise retain unscaled log expression.

This applies regardless of the number of input files, pathway settings, or
whether variance filtering is enabled. The same contract applies to bulk
training, additional SCT training rows, and matching inference for newly
trained staged models. Do not apply these rules to non-staged training or
silently change inference for checkpoints without the new contract metadata.

For every inference and test-set prediction, use the model's saved ordered
gene list exactly. Do not calculate or apply variance filtering to the test
set, even when training used variance filtering or a model gene has zero
variance in the test set. Ignore extra input genes.

If model genes are missing from the inference input, normalize the model genes
that are present, then pad missing model genes with zeros in the saved model
order. Reject inputs with no genes in common.

## Variance filtering

When filtering is enabled, first intersect bulk-source genes and normalize
the common-gene profile for each sample to one million in linear space.
Compute sample variance across the pooled valid bulk samples, including
training and validation rows, and retain genes with variance strictly greater
than `data.min_var`. Do not include SCT reference rows or independent test
sets in this calculation.

Use two bounded H5AD streaming passes: one for the global variance and one
to create the final selected-gene model inputs. The selected gene set cannot
be known until the global variance pass finishes. Do not create a full
temporary TPM matrix. When filtering is disabled, create final inputs in one
streaming pass.

Use chunked transformations and reuse buffers where safe. Never modify input
H5AD files, allocate a full-matrix conversion temporary, or use compressed
arrays where memory mapping is required. Keep visible progress logs for both
passes. Measure actual server throughput separately; do not claim a speedup
without a benchmark.

## Pathway profiles

Normalize the gene-expression profile only. Calculate pathway values as sums
of the linear gene TPM values using the existing pathway mask. Pathways can
overlap, so neither the pathway vector nor a concatenated gene-plus-pathway
vector is required to sum to one million.

For example, gene TPM values `[100000, 200000, 700000]` and pathway masks
`[g1, g2]` and `[g2, g3]` produce pathway values `[300000, 900000]`.
The combined gene-plus-pathway values sum to 2.2 million and must not be
renormalized.

Apply `log2(x + 1)` and optional constant scaling to the pathway values
independently, preserving the existing DeSide predictor architecture,
pathway order, weights, and forward semantics. Do not change pathway
normalization merely to optimize the gene preprocessing path.

## Zero-expression samples

Exclude samples whose selected genes have zero total linear expression and
record their source, sample ID, and exclusion reason. Apply this check after
the common-gene intersection and again after final gene filtering, because
filtering can make a previously nonzero sample all-zero.

Samples that become zero only after gene filtering remain part of the
original common-gene variance calculation, but are excluded before the
train/validation split. Never iteratively refilter genes after excluding
those samples. If no valid samples remain, stop with an actionable error.
If exclusions are discovered while writing the final model-feature cache,
finalize its valid rows with bounded memory and no additional H5AD scan.
Do not represent unused preallocated rows as valid samples.

Record exclusions for training and for staged-model inference. Keep the
existing zero-label exclusion and make the recorded reasons distinguishable.
Maintain a compatible `sample_id` column for split restoration. Excluded
inference rows must not produce predictions or enter evaluation metrics.

The stage-two matched-target builder must skip only samples documented in
the stage-one exclusion manifest. Unexpected missing or duplicate sample IDs
must still fail. Apply the same exclusion policy to additional SCT training
rows, without changing the meaning of SCT reference target masks.

## Stage handoff and reuse

Stage one saves the final ordered gene list, cell-type order, valid sample
identities, split assignments, exact processed bulk features, and versioned
preprocessing identity.

Reconstruction and joint fine-tuning reuse the selected genes, bulk labels,
features, and split assignments. They must not reopen the bulk expression
matrices to recompute those features. Metadata reads needed for validation or
matched-target IDs remain allowed. Use the existing disk-backed bulk feature
cache as an immutable input; do not introduce a second full-matrix conversion.

Prepare any additional SCT training rows separately using the same gene-input
contract, and combine them with bulk rows without materializing another
full bulk matrix in RAM. Continue to build matched SCT targets and gene
statistics using their existing conventions and stage-specific requirements.
Keep target arrays disk-backed and chunked for large datasets.

Do not change the configured stage-one checkpoint selection, test-set
prediction, stage-two predictor freezing, or stage-three fine-tuning.
Do not round cell-proportion labels in any stage.

## Compatibility and protection

Version the stage-one cache fingerprint and save an explicit input-contract
marker with newly trained stage checkpoints and the final model. Use that
marker to select matching inference preprocessing. Existing models without
the marker keep their prior inference path.

Verify source identity, preprocessing settings, ordered genes, cell types,
and the split manifest before resumed reconstruction uses cached features or
predictor weights. Reject incompatible or unverifiable inputs with an
actionable error. Never overwrite existing checkpoints or caches without the
established output-protection behavior.

Use a new run name when retraining after this preprocessing change. The
previous log-space-filtered predictor cannot gain newly retained genes by
resuming it.

## Verification

Use small synthetic fixtures for regression tests and keep the user's
server datasets and checkpoints read-only.

- Verify the training and inference gene inputs match an independent TPM
  reference for single and multiple bulk sources, with and without pathway
  branches, variance filtering, and constant scaling.
- Verify stage-local and final-model test-set prediction use the saved ordered
  gene list without test-set variance filtering, including model genes with
  zero variance, extra input genes, and missing model genes.
- Verify gene rows sum to one million before the log transform, within
  floating-point tolerance, and no additional gene-length correction occurs.
- Verify overlapping pathways are calculated from gene TPM without
  renormalizing pathway values or the combined gene-plus-pathway vector.
- Verify common-gene and final-gene zero-expression exclusions, recorded
  reasons, split preservation, inference output alignment, and all-zero
  failure behavior.
- Verify stage two reuses the stage-one bulk cache without loading bulk
  expression matrices, and that extra SCT training rows and matched targets
  remain correctly aligned, scaled, and available.
- Verify compiled and uncompiled stage-one checkpoints, stage-local test
  prediction, stage handoff, resume, cache identity, and output protection.
- Verify missing genes, changed sources, non-finite expression, non-staged
  workflows, and existing checkpoint inference behavior.
- Confirm chunked processing and progress logs with synthetic inputs; validate
  memory use, storage behavior, and performance separately on the computing server.
