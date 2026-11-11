# Staged gene filtering and handoff correction

This design corrects gene filtering and stage handoff for bulk-only staged
training. Stage one filters genes once in linear, million-normalized
expression space. Reconstruction and joint fine-tuning reuse the resulting
ordered list without independently filtering it again.

## Context and decisions

An existing run selected 6,128 stage-one genes but prepared 13,210 genes for
reconstruction. The stage-one variance filter used log-expression while
the reconstruction loader filtered linear CPM. The handoff only compared
the two lists after the full reconstruction dataset and matched targets
were prepared.

The user selected a fresh stage-one training run using the corrected
filtering, rather than recovery of the existing 6,128-gene checkpoint. The
existing outputs and caches must remain untouched.

The user selected normalization over the common input genes before
variance filtering. This differs from the prior direct preprocessing path
for three input files, which normalized each complete source before
intersecting genes. Do not claim exact equivalence with that older path.

## Stage-one preprocessing

Stage one must use only bulk training inputs and their labels. For each source,
preserve source namespaces, validate labels and genes, and exclude rows with
zero total labels as specified by the bulk-only training design.

1. Determine the ordered intersection of genes across the bulk sources and
   apply any explicitly configured gene list or pathway-gene restriction.
2. For each valid sample, convert the input log expression to linear values
   and normalize the retained common genes to a row total of one million.
   The user's inputs are already TPM, so this is TPM renormalization and
   must not apply gene-length correction again.
3. Pool the valid rows from all bulk training sources and compute the
   sample variance of each gene in this linear space, before scaling and
   before splitting into training and validation sets.
4. Retain genes with variance strictly greater than `data.min_var`, in the
   existing order. Apply a configured reference-gene restriction as before.
5. For predictor input, renormalize the selected genes to one million,
   apply `log2(x + 1)`, and apply the configured constant scaling.

Variance filtering uses the combined training and validation population,
matching the user's requested whole-sample filtering. Independent test sets
are not used. Variance calculation and feature writing must use bounded
chunks, with periodic progress logging in both passes.

Do not round input labels or expression as part of the new bulk-only path.
This differs from legacy loaders that round some intermediate expression
values. Persist the filter space, normalization order, ordered gene list, and
cache/preprocessing version with the stage-one artifacts.

When `model.torch_compile` is true, apply CUDA compilation to the stage-one
predictor as in the full-VAE workflow. Preserve the existing non-CUDA and
unavailable-compiler fallback behavior, and do not compile when the option is
false. Ensure sample IDs and other string metadata do not enter the compiled
forward path, and save canonical checkpoints that can load with or without
compilation.

After selecting the best stage-one checkpoint, run the configured test-set
prediction and save its stage-local results before beginning reconstruction.
When no test sets are configured, skip prediction. A stage-one-only run must
not accidentally run the same prediction twice in the same output directory.

## Reconstruction and joint fine-tuning

For a run that includes stage one, its saved gene list is authoritative.
Before building reconstruction datasets, matched SCT targets, and gene
statistics, configure the later-stage data preparation to use that exact list
and disable its independent low-variance filtering. Apply this only to the
stage-transition data preparation; do not mutate the user's original config
or change non-staged training behavior.

Validate that all required genes and cell types are present in each required
input. Fail with actionable details before expensive matched-target creation
when the required ordered genes or cell types cannot be supported. Use the
same ordered genes for bulk features, matched SCT targets, SCT statistics,
the VAE encoder, and the decoder.

Reuse the best stage-one predictor checkpoint and the saved bulk train and
validation assignments. Preserve the exact stage-one bulk feature values in
later stages; do not recompute them using a different normalization. Any extra
SCT training rows are prepared and split at the later stage as before. All
three stages must preserve stored cell-proportion labels without `round(3)`;
verify both the original bulk rows and any additional SCT rows.

Resume from a newly produced predictor-only checkpoint must use its saved
gene list and preprocessing identity. Verify the configuration and source
identity before loading weights or cached bulk features. When the identity
cannot be verified, fail with an actionable error rather than silently
substituting recomputed features. Older checkpoints are not required to be
resumable under the corrected filter.

## Cache and output protection

Version the bulk preprocessing fingerprint so that old log-space-filtered
caches cannot be reused as linear-space-filtered caches. A reconstruction
cache must also distinguish the authoritative ordered stage-one gene list
and its preprocessing settings, including a content-based identity for the
gene list.

Keep existing overwrite protections and preserve old cache and checkpoint
artifacts. A fresh training run must use a new output name. Never treat an
existing final-model checkpoint as a request to retrain.

## Deliberate differences from the previous workflow

The earlier staged workflow prepared the full VAE and SCT inputs before
stage one, split the prepared dataset once, and allowed a reconstruction
gradient through the frozen decoder. The approved bulk-only design instead
trains only the proportion predictor, defers SCT preparation, preserves
label precision, and uses a seeded bulk split. These differences are
intentional and are not reversed by this correction.

The single-source, no-pathway prediction path needs a dedicated
preprocessing parity check. This finding is separate from the requested
corrections and must not be silently represented as fixed by this work.

## Verification

Use small synthetic fixtures and do not modify the server datasets or
existing run outputs.

- Compare filtered genes with an independent float64 reference for linear
  normalized variance, including unequal source gene sets, reordered genes,
  multiple sources, zero-label exclusions, and genes near the threshold.
- Verify normalization happens before variance filtering, both stages reuse
  the same ordered genes, and selected-gene model inputs are normalized and
  transformed as specified.
- Verify the variance pass pools all valid bulk rows before the split and
  does not read SCT data or independent test sets.
- Train a small stage-one-to-stage-two workflow with variance filtering
  enabled and matched SCT supervision. Verify feature values, selected
  genes, SCT targets, statistics, split assignments, and best-weight transfer.
- Verify rare proportions retain their stored precision through all three
  stages, including any additional SCT rows, in both memory and cache paths.
- Verify configured stage-one test sets use the best predictor checkpoint
  before stage two begins, and that omitted test sets remain optional.
- Exercise compiled and uncompiled predictor checkpoint save, load, transfer,
  and inference. Verify string sample IDs do not enter compiled inputs.
- Test resume using a newly created predictor checkpoint, and reject missing
  genes, changed source identity, and incompatible preprocessing.
- Verify cache versioning and ordered-list identity prevent stale cache reuse;
  exercise both in-memory and memmap target paths without large allocations.
- Retest stage-one-only, full-VAE and non-staged behavior, pathway options,
  excluded samples, output protections, and configured test-set inference.
- Confirm that errors and periodic progress logs identify the stage and work
  being performed.
