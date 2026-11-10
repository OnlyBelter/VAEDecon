# Bulk-only cell-proportion training design

This design changes staged training so stage one trains and predicts cell
proportions from bulk expression and its cell-fraction labels alone. It must
also support matching the PyTorch DeSide training and prediction workflow
for the same inputs and settings, except for the train/validation split.
Stage two prepares the SCT inputs and full VAE only when reconstruction
training begins. No additional stage-one mode setting is introduced.

## Motivation and scope

The current configuration treats `training_target_sets` as matched SCT target
bundles, so each entry requires three paths. Training also computes SCT
gene-mean and gene-standard-deviation statistics before model construction.
The VAE constructor and evaluation path then require those statistics even when
only cell proportions are needed.

Existing `cell_prop_predictor_pretrain` freezes the encoder and decoder, but
its loss still includes reconstruction. The predictor therefore receives a
gradient through the frozen decoder. The new workflow intentionally trains
only on the configured cell-proportion objective, so its results and loss
curves are not equivalent to earlier stage-one runs.

Matching DeSide requires more than reusing a similar network. DeSide's current
PyTorch trainer computes MAE and RMSE across all values in a batch; the current
VAEDecon `l1_rmse` sums errors per sample before taking the square root.
DeSide's H5AD training loader preserves label precision, while the current
VAEDecon loader rounds fractions to three decimal places. DeSide also applies
thresholding before prediction normalization. These differences must be
addressed for matched-settings comparisons. VAEDecon retains PyTorch's
`random_split` instead of reproducing DeSide's NumPy split.

The requested inputs have 100,000 and 1,000 samples, respectively. Both
contain the same 16 cell types. Their 9,028 shared genes are used when no
additional gene list or variance filtering is configured. Their sample IDs
overlap, so input identity must include the source dataset name.

## Configuration and stage selection

The existing staged-training configuration remains the entry point. Whenever
`cell_prop_predictor_pretrain` runs, it uses the same proportion-only workflow
whether it is the only selected stage or precedes stages two and three. No
mode flag or selection based on SCT path presence is added.

For example, each training entry contains only its expression file:

```yaml
data:
  training_target_sets:
    Train_set1:
      training_set_file_path: /data/simulated_bulk_100k.h5ad
    Train_set2:
      training_set_file_path: /data/simulated_bulk_1k.h5ad

training:
  staged_training:
    enabled: true
    run_stages:
      - cell_prop_predictor_pretrain
```

The same staged configuration also accepts entries that retain their
`training_set_sample2cell_id_file_path` and `training_sct_gep_file_path`.
Stage one uses only `training_set_file_path`. It logs that SCT-related paths
are deferred and does not open their files or validate their availability.
Entries may mix bulk-only and complete bundles. A legacy `sct_file_path` is
not included as a stage-one training source.

The normal model settings must also select a cell-proportion predictor,
enable `predict_cell_prop`, route cell proportions to that predictor, and
configure a positive cell-proportion loss weight. The selected stage trains
only `cell_prop_predictor`. The example configuration and documentation will
show a complete, runnable proportion-only configuration with these settings.

For stage one, derive the bulk input list from `training_target_sets` when
`simu_bulk_file_path` is empty. If both are supplied, require the resolved
paths to match exactly and reject duplicate entries. Do not include an SCT
file as a training source at this stage, even when an existing configuration
retains it.

Defer SCT-reference requirements and file-existence checks until the first
selected stage that needs them. Stage two or three requires SCT statistics for
the full VAE and requires matched mapping and SCT files when its effective
losses or schedules need matched targets. Report missing or incomplete
dependencies with their stage and configuration names at that stage boundary.
Do not let these deferred requirements prevent stage one from running.

The `DeSideCellPropPredictor` pathway branch requires pathway files when
`deside_pathway_network` is enabled. For parity with a DeSide run that uses
the pathway branch, configure the same pathway files, gene list, branch
architecture, and scaling. An example without pathway files must disable that
branch explicitly; do not silently alter the predictor architecture.

## DeSide behavior

The parity target is the current PyTorch DeSide bulk training and prediction
workflow in the sibling `DeSide` repository, with matching configurations
except for the train/validation split. The simulator and historical
TensorFlow implementation are not replaced. Support the following behavior
without adding another stage-one training mode:

- Align common input genes, optional selected genes, pathway profiles, and
  optional cell-type grouping in the order and preprocessing space used by
  the matched DeSide configuration.
- Preserve native training-label precision. Match DeSide's handling of
  cancer-cell targets, selected cell types, and `one_minus_alpha` when those
  options are configured.
- Support DeSide's batch-wide objective
  `alpha * mean(abs(error)) + (1 - alpha) * sqrt(mean(error ** 2))`.
  Expose it as a cell-proportion loss choice without changing existing
  VAEDecon loss choices. For sigmoid outputs with cancer removed, compute the
  loss on raw non-cancer predictions before inference postprocessing.
- Support matching DeSide's batch size, optimizer and weight decay,
  scheduler, early stopping, model architecture, activation, and dropout
  through the stage-one configuration. Reuse existing configuration fields
  where possible and report effective settings. Do not silently override a
  user's non-DeSide settings.
- Match DeSide's configured inference sequence: optional complement,
  threshold, optional normalization for non-cancer sums above one, and
  optional cancer residual. Preserve the configured output cell-type order
  and make output-file rounding explicit.

For the DeSide example configuration, matching settings include its configured
pathway branch, sigmoid non-cancer output, batch-wide MAE/RMSE loss, 20%
validation split with seed 42, and optimizer and postprocessing settings.
PyTorch and NumPy can select different rows even with the same seed and split
fraction.
Its example groups three pairs of subtypes, producing 13 groups from these
16 input cell types. Cell-type grouping is configuration-dependent and is
not enabled implicitly.
The same training files alone do not guarantee identical fitted weights or
CCC: initialization, batch order, hardware, and software also matter.

## Input processing and identity

Read gene names, cell-type labels, and sample IDs from each H5AD. Determine
the ordered common gene list before loading the expression matrices. For a
matched DeSide configuration, preserve its selected-gene, normalization,
pathway-profile, and scaling order, including its behavior for a single
training file versus multiple files. Preserve native precision rather than
rounding the input expression or fractions during training. Store the
resulting gene order and preprocessing settings for prediction.

Use the named training-set entry as a stable namespace, using the existing
`<training-set-name>::<sample-id>` convention. Preserve all valid,
nonzero-label rows from both sources, including rows whose original sample
IDs overlap. Reject duplicate IDs within a source, duplicate source names or
paths, missing or non-finite
cell-fraction labels, out-of-range fractions, and incompatible cell-type
sets. Align cell-fraction columns in the stored cell-type order instead of
intersecting away missing required cell types. Do not silently renormalize the
stored labels: the 100K file has rounded fractions whose row sums vary
slightly around one. Match DeSide's exclusion of zero-total-label rows and
record any exclusions explicitly.

Keep dataset-source identity available in the cache and output manifest.
Cache fingerprints must distinguish stage-one bulk-only inputs, preprocessing
settings, and source namespaces from later-stage matched-target caches.
Stage one must not construct matched SCT targets or open SCT files. Retain
bounded-memory, chunked processing for large input datasets.

## Stage-one training and validation

Construct and optimize only the configured cell-proportion predictor. Do not
instantiate or run the VAE encoder or decoder, compute SCT statistics, or
create placeholder statistics or reconstructions. Retain the existing
cell-proportion losses and add the batch-wide DeSide loss described above.
Reuse the configured activation, loss weighting, learning-rate scheduler,
early stopping, and progress reporting. The monitored
`val_cell_prop_loss` and `train_cell_prop_loss` remain available, and the
optimized objective contains only the configured proportion loss. Apply the
effective stage-one proportion loss and its schedule; do not execute unrelated
base or later-stage losses or schedules during stage one. Log this stage-local
behavior without changing the configuration used by later stages.

Split the merged dataset with PyTorch's `random_split` using the configured
`train_split`, `val_split`, and an explicitly seeded PyTorch generator. Save
enough information to identify the source, sample ID, and train or validation
assignment for each sample. A matching seed and split fraction do not imply
the same assignments as DeSide. For comparisons, evaluate both models on the
same independent test set, excluded from training and validation for both.
Do not label predictions on training samples as independent test results.
A split is over the merged rows, so the 100K source contributes approximately
100 times as many samples as the 1K source unless the user changes the input
data; no reweighting or resampling is introduced.

## Transition to stages two and three

When reconstruction training follows stage one, first finish and save the
best stage-one predictor checkpoint. At the start of stage two, resolve and
validate the required SCT references, compute or load gene statistics, and
construct matched SCT targets only when the effective stage losses or
schedules require them. Then construct the full VAE and initialize its
predictor with the best stage-one weights before freezing or training modules
according to the stage-two configuration.

Keep the bulk gene order, cell-type order, source namespaces, and bulk
train/validation assignments consistent across the transition. If the
configuration uses additional SCT training rows for a later stage, prepare
and split those rows only at that later stage without changing the recorded
bulk assignments. The later stages retain their existing full-VAE objectives,
stage-specific loss overrides, checkpoint selection, and module trainability.

When stage one is skipped, initialize the selected later stage from its
configured predecessor checkpoint and prepare its required SCT inputs at the
start of that stage. If stage-two or stage-three preparation fails, preserve
the completed stage-one checkpoint and report how to continue with the required
inputs and `stage_init_checkpoints`; do not overwrite completed outputs.

## Checkpoints and prediction

Keep the stage directory, `final_model` directory, checkpoint selection,
training logs, and overwrite protections. Save the best and last predictor
checkpoints, the predictor state dictionary, ordered gene list, ordered
cell-type list, preprocessing settings, effective predictor architecture,
training split manifest, and an explicit model-kind marker for a checkpoint
that contains only the stage-one predictor. These artifacts must not imply
that a decoder or SCT-derived statistics were trained or are available.

Extend the existing VAEDecon prediction and CLI entry points to detect a
stage-one predictor checkpoint and load it without constructing a full VAE or
requiring SCT reference files. Align inference inputs to the saved gene order
using the preprocessing contract saved with the predictor. A DeSide-matched
configuration follows DeSide's input normalization, missing-gene, pathway, and
postprocessing sequence. Export
`predicted_cell_prop.csv` for labeled or unlabeled bulk input. Evaluate
proportions when ground truth is available, including the held-out validation
split, and skip reconstruction, latent-space, and SCT-GEP outputs.
Configured test-set inference remains optional; a training-only invocation
without test sets must succeed.

Preserve compatibility with previously saved full-VAE checkpoints and their
existing inference path. Export predictor weights in the existing
`cell_prop_predictor_state_dict.pt` format so a later reconstruction stage
can initialize its predictor when valid SCT inputs and a full-VAE
configuration are supplied. For newly written stage-one predictor artifacts,
validate ordered genes, cell types, predictor architecture, and
preprocessing compatibility before reuse. Reject a mismatch with an
actionable error rather than relying on matching tensor shapes alone.

## Verification

Use small synthetic H5AD fixtures for automated tests and keep the supplied
large datasets read-only. Cover the following cases:

- Parse stage-one entries with all three paths, with only the bulk path, and
  with both forms in the same configuration. Verify that all use the same
  proportion-only objective without a mode option. With fixed seeds and
  identical bulk inputs, adding unused SCT paths must not change stage-one
  samples, losses, gradients, or predictions.
- Reject missing bulk paths, conflicting file lists, duplicate paths, and
  invalid stage-one predictor settings. Report partial or missing SCT bundles
  only when a later stage needs them.
- Preserve non-staged full-VAE training, existing full-VAE checkpoint
  inference, and the objectives and matched-target behavior of later stages.
- Merge two differently sized input datasets with overlapping sample IDs,
  align their gene intersection and cell-type order, and preserve all valid
  labels without rounding.
- Confirm that stage-one preprocessing and training do not open SCT
  inputs, even when their paths are supplied, construct SCT target caches,
  compute SCT statistics, or instantiate a VAE decoder.
- Compare the proportion-only loss and gradients with the existing supervised
  cell-proportion loss for supported activation and weighting configurations.
- Compare the DeSide batch-wide loss and gradients against DeSide's PyTorch
  implementation, separately from existing VAEDecon loss choices.
- Verify that PyTorch's `random_split` produces reproducible assignments for
  the same input order and seed, and that the saved manifest records them.
- Compare both packages on small fixtures with matching settings: input
  features and labels, selected genes and pathway profiles, cell-type
  grouping, output postprocessing, and a controlled optimizer step using the
  same explicit batch. Do not require identical validation indices. Copy
  identical predictor weights for numerical forward and gradient comparisons.
  For CPU float32 comparisons, use `rtol=1e-5` and `atol=1e-6`; require
  exact sample, gene, and cell-type order for the compared fixtures.
- Keep reproducible reference fixtures or expected values in VAEDecon so core
  parity tests run in continuous integration without the sibling repository. Run direct
  cross-package checks when DeSide is available; do not make it a runtime
  dependency.
- Verify DeSide-matched sigmoid, threshold, and cancer-residual behavior
  without changing the semantics of existing VAEDecon proportion-loss
  settings.
- Train and validate a small predictor, save and reload its best checkpoint,
  and reproduce its predictions through the existing public inference and
  CLI paths. Verify that test sets are optional and prediction output is
  correctly labeled.
- Run a stage-one-to-stage-two integration test that proves SCT preparation
  starts only after stage one, the best predictor weights are transferred,
  and bulk sample splits are preserved. Confirm that a later-stage failure
  leaves the stage-one checkpoint usable.
- Validate checkpoint transfer into a later full-VAE stage, rejecting
  incompatible gene order, cell types, predictor settings, or preprocessing.
- Check the real H5AD metadata in backed mode: 101,000 total samples, 16
  matching cell types, 9,028 common genes, and unique namespaced sample IDs.
- Run relevant staged-training, dataset, inference, CLI, and model tests.
  Performance on an H20 requires a separate measurement on that hardware.

Update the configuration reference, workflow guide, example configuration,
and changelog with stage-specific input requirements and the change from the
historical stage-one objective.
