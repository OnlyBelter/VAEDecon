# Large Matched sctGEP Target-Build Design

## Goal

Reduce runtime and peak memory while building matched single-cell expression
targets when the merged training dataset contains more than 500,000 samples.
Preserve the current dense `.npy` cache shape and the downstream
`(samples, genes, cell_types)` interface.

## Current Behavior

`GEPDataset._build_true_sct_gep_targets` creates a float32 memory-mapped target
array with shape `(n_samples, n_genes, n_cell_types)`. It loads the selected
reference-cell profiles, builds a Python list of one tuple per mapping row, and
writes each matched profile separately. When scaling is enabled, the completed
target tensor is then read and divided by the scaling factor in one whole-array
expression. It also accumulates a duplicate-heavy list of selected reference
cell IDs before deduplicating it. For large datasets, the scaling expression
can allocate a temporary as large as the target tensor, and these operations
have no progress logging.

The matching 1M seed10 configuration enables scaling by 20 and uses four bulk
sets totaling 1,000,000 profiles. A target array of 1M samples, about 13K genes,
and 16 cell types is approximately 0.8 TB in float32. The assignment loop also
reports about 16 million sample-cell-type records.

## Proposed Behavior

Use the optimized path only when the merged `n_samples` is greater than
500,000. Keep the current path for smaller datasets.

For the optimized path:

1. Keep the dense target array's filename, dtype, shape, zero-fill behavior,
   and presence-mask semantics unchanged.
2. Avoid constructing one full list of Python tuples or a duplicate-heavy
   reference-cell ID list. Resolve sample-row, cell-type, and reference-row
   indexes in bounded batches, and maintain unique reference IDs as they are
   collected.
3. For each bounded sample block, fill a temporary block in the existing
   `(samples, genes, cell_types)` order, then write that block contiguously to
   the memory-mapped target. Keep the target block capped at 32 MiB; the
   selected-profile gather is bounded by the same amount, keeping these
   temporary arrays to approximately 64 MiB combined.
4. Apply the configured scaling factor to the aligned reference profiles once
   before assignment. Do not perform a second full-array scaling pass over
   the target tensor. Preserve raw values in the on-disk reference-query cache.
5. Preserve missing-target behavior: unassigned entries remain zero, and the
   corresponding presence-mask entries remain false.
6. Retain existing behavior for compressed-cache serialization. Compression
   remains outside the optimization target because `.npz` does not provide
   true memmap access and can be substantially slower.

## Progress Reporting

Show a persistent `tqdm` progress bar for completed assignments, with an
appropriate refresh interval. Also emit periodic timestamped logger updates
with completed and total assignments, elapsed time, and current phase. Keep
the final completion update in the log. This makes progress visible in an
interactive terminal and in output captured by `nohup`.

## Compatibility and Failure Behavior

The optimized path must preserve target values, scaling, array ordering,
float32 dtype, and mask behavior. The threshold is based on the merged sample
count, not the number of assignment rows. Existing cache paths and readers do
not change. The dense output still requires storage proportional to
`n_samples * n_genes * n_cell_types`; this design reduces temporary memory and
write overhead, not the required dense cache size.

## Verification

- Unit-test the batched writer against the current writer on small fixtures,
  including multiple cell types, missing mappings, and duplicate mapping rows
  after the existing deduplication step.
- Verify scaled and unscaled targets match the existing semantics and that
  cached reference values remain unchanged.
- Verify the optimized branch is selected only when `n_samples > 500,000`.
- Verify chunk boundaries preserve sample order, target values, and masks.
- Test progress updates and final completion reporting without emitting a log
  line for every assignment.
- Run the focused matched-sctGEP tests; benchmark a representative large path
  separately from CI so tests do not create multi-terabyte arrays.
