# Large training performance design

This design improves staged training throughput and progress visibility while
preserving the training objective, the dataset split, and existing checkpoint
and cache formats. It applies to the 350K and 1M configurations and does not
require a dataset-size threshold.

## Observations

The reported 350K run produced a stage-one checkpoint at epoch 57, so the
recompilation warning did not stop training. The warning identifies changing
sample-ID strings as a source of repeated `torch.compile` specialization.
Actual GPU throughput and the relative impact of I/O, computation, and
compilation have not yet been measured.

The dataset currently fetches and collates full matched sctGEP tensors for
every batch, including stages where all corresponding losses are disabled.
The model also converts sample-ID strings inside its compiled forward pass.

Stage one cannot simply skip the frozen encoder and decoder: reconstruction
loss is unconditionally included in the total objective, and its mixture
depends on the predictor's cell proportions. A predictor-only forward would
therefore change gradients and metrics even when `beta` and `gamma` are zero.

## Stage-aware target loading

Create a stage-local dataset view that can omit full matched sctGEP expression
targets without mutating the shared underlying training dataset. Apply it to
the existing training and validation subsets before constructing each stage's
loaders. Preserve sample IDs and all fields needed by error reporting.

Determine the required fields conservatively using the effective stage loss
weights and all configured linear and adaptive schedules. Read full
`true_sct_gep` whenever matched sctGEP supervision or inter-sample similarity
can be positive during the stage. Read the presence mask whenever any active
or potentially active loss needs it, including per-sample residual variance.
Only omit a field when it cannot be needed at any point in that stage. Unknown
dataset types or uncertain requirements retain the existing full batch.

The view must behave correctly with shuffled `Subset` instances, validation,
multiple workers, persistent workers, and different stage transitions. It must
not alter preprocessing, already-built target caches, or direct dataset access.

## Sample IDs and compilation

Resolve per-sample residual-variance targets from sample IDs outside the
compiled tensor-only path. When that loss is active or may become active,
pass the resolved batch-aligned tensor and presence mask into the model. Do
not pass changing string values to the compiled forward in staged training.
Keep the original IDs in the trainer batch for non-finite diagnostics.

Preserve namespaced ID alignment, cell-type order, training-time missing-ID
errors, validation behavior, and the existing direct-model interface for
non-staged and inference callers. Changes must work whether `torch_compile` is
enabled or disabled, and must not rely on raising the recompilation limit.
Other compilation guards may still occur and will be measured separately.

## Training progress

Keep a live Lightning progress bar for training and validation. Add periodic,
timestamped log lines for redirected output, including stage, epoch, phase,
completed and total batches, elapsed time, and throughput. Report phase
completion and keep logging bounded by a time interval so large runs do not
produce one line per batch. Restrict reporting to the main process.

## Compatibility and safety

Retain the existing full forward path, stochastic sampling, loss definitions,
loss and metric names, scheduler behavior, trainable modules, and stage
checkpoint layout. Do not change sample count, batch size, precision, random
split, number of epochs, or early-stopping semantics. Existing test-set
evaluation and inference must continue to work.

No change is made to a running process or to its files. A fresh process is
needed to use code changes. Restarting must follow existing output-protection
and checkpoint-reuse rules.

## Verification

- Compare baseline and optimized per-batch losses, predictor gradients, and
  monitored validation metrics under fixed seeds.
- Test stage-specific target and mask requirements, including zero and
  nonzero scheduled weights and transitions to later stages.
- Test sample-ID lookup parity, namespaced IDs, missing-ID errors, validation,
  and direct non-staged forward calls.
- Test normal and debug subsets, validation, multiple workers, and stage
  transitions. Verify that omitted targets are not accessed or transferred.
- Exercise compiled and eager paths with changing IDs and check whether the
  observed string-ID recompilation guard disappears. Record any remaining
  graph breaks rather than claiming all recompilations are eliminated.
- Verify training and validation progress in interactive and redirected output,
  including epoch and phase completion, and rank-zero-only logging.
- Run the staged-training, matched-sctGEP, model-loss, and trainer tests.
  Compare batches per second and peak CPU/GPU memory by stage on H20 using a
  short, fixed workload; local CPU tests cannot establish GPU speedup.
