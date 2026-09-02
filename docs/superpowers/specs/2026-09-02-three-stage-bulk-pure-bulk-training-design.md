# Three-stage bulk-pure-bulk staged training design

Last updated: 2026-09-01

## Status

This document revises the earlier four-stage prototype-residual proposal into a
new three-stage workflow.

The revised design keeps the current Stage 1 cell-proportion training, then
uses pure `sctGEP` reconstruction as a pretraining stage for the full GEP
decoder, and finally fine-tunes the combined system on mixed bulk data with a
small learning rate.

## Goal

This design aims to improve full `sctGEP` inference by separating the training
curriculum into:

1. cell proportion learning on mixed bulk data,
2. full-GEP pretraining on pure `sctGEP` inputs, and
3. low-learning-rate joint adaptation on mixed bulk data.

The key change from the previous proposal is conceptual:

- the design no longer tries to explicitly split prototypes and residuals into
  separate stages and separate decoders,
- Stage 2 instead treats each pure `sctGEP` as a clean single-cell-type
  reconstruction target, and
- Stage 3 uses only mixed bulk reconstruction to combine the Stage 1
  proportion branch and the Stage 2 full-GEP branch.

## Motivation

The motivation is that a pure `sctGEP` contains one specific cell type only.
That makes Stage 2 a cleaner pretraining task than mixed bulk reconstruction,
because it is not distracted by:

- cell proportion errors,
- interference from other cell types, or
- inaccurate inferred GEPs for the other cell types in the same sample.

In this proposal, Stage 2 acts like pretraining for the whole GEP side of the
model, including the encoder, decoder, and any cell-type full-GEP embeddings.

Stage 3 then combines:

- the Stage 1 cell-proportion predictor,
- the Stage 2 full-GEP generator, and
- mixed bulk reconstruction,

and fine-tunes them jointly with a small learning rate.

## Core design summary

The new workflow has three stages:

1. **Stage 1: cell proportion pretraining**
   - keep the current Stage 1 unchanged,
   - train only on mixed bulk training sets,
   - do not use `sctGEP` training sets here.
2. **Stage 2: pure-sctGEP full-GEP pretraining**
   - freeze Stage 1,
   - train only on pure `sctGEP` training sets,
   - use ground-truth cell proportions for those pure samples,
   - reconstruct the input `sctGEP` directly,
   - do not use any extra matched-`sctGEP` auxiliary supervision.
3. **Stage 3: mixed-bulk joint fine-tuning**
   - reload Stage 1 and Stage 2 weights,
   - train only on non-`sctGEP` mixed bulk training sets,
   - use the cell proportions predicted by the Stage 1 branch,
   - reconstruct only the input mixed bulk GEP,
   - make direct `sctGEP` supervision configurable in this stage and disable it
     by default,
   - fine-tune both the Stage 1 and Stage 2 weights with a small learning
     rate.

## Design goals

This design must:

1. keep the current Stage 1 behavior unchanged,
2. let each stage choose the correct training-data source through
   `training_target_sets`, `sct_file_path`, `require_pure_sct_gep`, and
   `require_mixed_bulk`,
3. use pure `sctGEP` reconstruction as a clean pretraining task for the
   full-GEP branch,
4. disable direct `sctGEP` supervision in the final mixed-bulk stage by
   default, while keeping it configurable for ablation,
5. reuse Stage 1 cell-proportion predictions in the final stage,
6. fine-tune the combined model conservatively in the final stage, and
7. stay close enough to the current staged-training framework that the
   implementation can reuse most existing infrastructure.

## Non-goals

This design does not aim to:

- keep the previous prototype-bank plus residual-decoder decomposition,
- require explicit prototype or residual embeddings,
- fully redesign the preprocessing cache format beyond the required separation
  of pure `sctGEP` data and mixed bulk data handling,
- require matched `sctGEP` supervision in every stage, or
- solve every possible domain-gap issue between pure `sctGEP` and mixed bulk in
  the first implementation.

## High-level model view

The revised design assumes one shared full-GEP inference branch instead of a
prototype branch plus residual branch.

At a high level:

```text
Stage 1:
  mixed bulk x -> pi_hat

Stage 2:
  pure sctGEP x_pure + ground-truth pi_true(one-hot or pure-label proportions)
  -> inferred full GEP -> reconstruct x_pure

Stage 3:
  mixed bulk x -> pi_hat
  mixed bulk x + pi_hat -> inferred cell-type full GEPs
  sum_c pi_hat(sample, c) * GEP_hat(sample, c) -> reconstruct bulk x
```

This means:

- Stage 1 trains the proportion branch,
- Stage 2 pretrains the GEP branch under clean single-cell-type conditions, and
- Stage 3 fine-tunes the two branches together under realistic mixed-bulk
  conditions.

## Dataset-selection design

Each stage must be able to choose the correct data source from the existing
config structure instead of adding a new per-stage dataset-name selector.

The recommended design is:

- use `training_target_sets` for mixed bulk datasets,
- use `sct_file_path` for pure `sctGEP` datasets, and
- let each stage decide which source to consume through semantic flags.

Each stage should also support light semantic validation flags so that a stage
cannot silently receive the wrong kind of data. The recommended stage-local
checks are:

- `require_pure_sct_gep`
- `require_mixed_bulk`

Recommended stage behavior:

- Stage 1:
  - `require_mixed_bulk: true`
  - `require_pure_sct_gep: false`
- Stage 2:
  - `require_pure_sct_gep: true`
  - `require_mixed_bulk: false`
- Stage 3:
  - `require_mixed_bulk: true`
  - `require_pure_sct_gep: false`

This keeps the data contract explicit and prevents accidental reuse of the
wrong training sets.

Concretely:

- if `require_pure_sct_gep: true`, the stage uses all `sctGEP` datasets listed
  in `sct_file_path`,
- if `require_mixed_bulk: true`, the stage uses all mixed bulk datasets listed
  in `training_target_sets`.

The first implementation does not need additional fields such as
`selected_training_sets` or `selected_validation_sets`.

## Three-stage workflow

### Stage 1: cell proportion pretraining

This stage is intentionally unchanged in objective. In staged execution, the
GEP-side modules remain frozen here.

- training data:
  - only mixed bulk training sets,
  - no pure `sctGEP` training sets
- trainable modules:
  - `cell_prop_predictor`
- frozen modules:
  - GEP encoder and decoder modules
- active objective:
  - cell proportion supervision only

Expected result:

- stable cell proportion prediction,
- a strong Stage 1 checkpoint that can be reused in Stage 3.

### Stage 2: pure-sctGEP full-GEP pretraining

This stage uses only pure `sctGEP` datasets.

Each training sample contains one cell-type-specific expression profile, so the
ground-truth cell proportion vector is effectively known. In the simplest case,
it is one-hot over cell types. If the pure `sctGEP` representation uses another
convention, the implementation must still provide the exact ground-truth
proportions for the sample.

- training data:
  - only pure `sctGEP` training sets
- initialization:
  - load the best Stage 1 checkpoint
- trainable modules:
  - encoder-side modules used by full-GEP inference
  - full-GEP decoder
  - cell-type full-GEP embeddings or equivalent decoder-side parameters
- frozen modules:
  - `cell_prop_predictor`
- proportion input:
  - use ground-truth cell proportions, not Stage 1 predictions
- active objective:
  - reconstruct the input pure `sctGEP`
- inactive objectives:
  - cell proportion loss
  - matched `sctGEP` auxiliary losses
  - inter-sample similarity losses
  - residual-specific losses from the previous design

Reconstruction view:

```text
x_pure, pi_true -> GEP_hat -> x_hat_pure
loss_stage2 = recon(x_hat_pure, x_pure)
```

The Stage 2 reconstruction loss should use the same underlying reconstruction
function as the Stage 3 mixed-bulk reconstruction loss. The difference between
the two stages is the input regime, not the reconstruction-loss definition.

Here, "no supervision for `sctGEPs`" is interpreted narrowly: the stage does
not add a separate matched-`sctGEP` supervision branch or auxiliary target
beyond direct reconstruction of the input pure `sctGEP`.

Expected result:

- the GEP branch learns a clean cell-type-specific representation,
- the decoder sees pure cell-type signals without bulk mixture confounding,
- the Stage 2 checkpoint becomes a pretrained initialization for Stage 3.

### Stage 3: mixed-bulk joint fine-tuning

This stage combines the Stage 1 and Stage 2 branches and fine-tunes them on
mixed bulk data only.

- training data:
  - only mixed bulk training sets
  - no pure `sctGEP` training sets
- initialization:
  - reload Stage 1 weights into the proportion branch
  - reload Stage 2 weights into the encoder and full-GEP decoder
- trainable modules:
  - `cell_prop_predictor`
  - encoder-side modules used by full-GEP inference
  - full-GEP decoder
  - cell-type full-GEP embeddings or equivalent decoder-side parameters
- proportion input:
  - use `pi_hat` predicted by the Stage 1 branch
- active objective:
  - reconstruct the input mixed bulk GEP
- inactive objectives:
  - direct `sctGEP` supervision, unless explicitly enabled for ablation
  - matched `sctGEP` auxiliary losses
  - Stage 2 pure-input reconstruction objective
- learning rate:
  - smaller than Stage 1 and Stage 2

Reconstruction view:

```text
x_bulk -> pi_hat
x_bulk + pi_hat -> GEP_hat(sample, c)
x_hat_bulk = sum_c pi_hat(sample, c) * GEP_hat(sample, c)
loss_stage3 = recon(x_hat_bulk, x_bulk)
```

The Stage 3 bulk reconstruction loss should use the same underlying
reconstruction function as Stage 2. Stage 2 applies it to pure `sctGEP`
inputs, while Stage 3 applies it to mixed-bulk inputs.

Expected result:

- Stage 1 and Stage 2 become a single jointly tuned model,
- the GEP branch adapts from pure-input pretraining to mixed-bulk inference,
- final training is conservative enough to retain Stage 2 cell-type structure.

## Loss design by stage

Each stage should keep only the losses needed for its role.

### Stage 1

Active:

- cell proportion loss

Off:

- bulk reconstruction from inferred full GEPs
- pure `sctGEP` reconstruction
- direct `sctGEP` supervision
- inter-sample similarity losses

### Stage 2

Active:

- pure `sctGEP` reconstruction loss
- latent regularization that already belongs to the base VAE path, if needed

Off:

- cell proportion loss
- a separate mixed-bulk reconstruction objective
- direct matched-`sctGEP` auxiliary supervision
- inter-sample similarity losses

The Stage 2 pure-`sctGEP` reconstruction loss and the Stage 3 mixed-bulk
reconstruction loss should share the same reconstruction-loss form and the same
data-scale assumptions. Only the inputs and composition structure differ.

### Stage 3

Active:

- mixed-bulk reconstruction loss
- latent regularization that already belongs to the base VAE path, if needed

Off by default:

- direct `sctGEP` supervision
- matched residual or prototype losses from the previous design
- inter-sample similarity losses that depend on direct `sctGEP` targets
- cell proportion supervision

Optional ablation:

- direct `sctGEP` supervision can be re-enabled in Stage 3 through a dedicated
  config flag if the user wants to test whether a small amount of direct
  supervision improves stability.

One important caution is that Stage 3 can drift if it only sees bulk
reconstruction. For that reason, the first implementation should support two
optional stabilizers as guarded ablations, even if they are off by default:

1. a weak checkpoint-retention penalty that keeps Stage 3 decoder parameters
   near the Stage 2 checkpoint, and
2. a weak cell-proportion retention term that keeps the fine-tuned Stage 3
   predictor near the Stage 1 solution when bulk-only tuning destabilizes it.

These are not part of the default objective, but they are the first fallback if
Stage 3 shows collapse or drift.

## Config design

The staged-training config should move from the current four-stage design to a
three-stage design.

Recommended stage names:

- `cell_prop_predictor_pretrain`
- `pure_sct_gep_pretrain`
- `mixed_bulk_joint_finetune`

Recommended new or emphasized stage-local fields:

- `require_pure_sct_gep`
- `require_mixed_bulk`
- `use_ground_truth_cell_prop`
- `enable_direct_sct_gep_supervision`

Example shape:

```yaml
training:
  staged_training:
    enabled: true
    run_stages:
      - cell_prop_predictor_pretrain
      - pure_sct_gep_pretrain
      - mixed_bulk_joint_finetune
    stages:
      - name: cell_prop_predictor_pretrain
        require_mixed_bulk: true
        require_pure_sct_gep: false
        train_modules: ["cell_prop_predictor"]
        freeze_modules: ["encoders", "full_gep_decoder"]
        learning_rate_scale: 1.0

      - name: pure_sct_gep_pretrain
        require_mixed_bulk: false
        require_pure_sct_gep: true
        train_modules: ["encoders", "full_gep_decoder"]
        freeze_modules: ["cell_prop_predictor"]
        use_ground_truth_cell_prop: true
        enable_direct_sct_gep_supervision: false
        learning_rate_scale: 1.0

      - name: mixed_bulk_joint_finetune
        require_mixed_bulk: true
        require_pure_sct_gep: false
        train_modules: ["cell_prop_predictor", "encoders", "full_gep_decoder"]
        freeze_modules: []
        use_ground_truth_cell_prop: false
        enable_direct_sct_gep_supervision: false
        learning_rate_scale: 0.05
```

In this design:

- `require_pure_sct_gep: true` means the stage consumes all `sctGEP` datasets
  listed in `sct_file_path`,
- `require_mixed_bulk: true` means the stage consumes all mixed bulk datasets
  listed in `training_target_sets`.

If module grouping in the current codebase still uses the older `decoder`
terminology, the first implementation can map `full_gep_decoder` back to the
existing decoder group as long as the staged behavior remains the same.

## Execution model

The execution model should remain close to the current staged-training
workflow.

The workflow should:

1. build the dataset registry once,
2. separate pure `sctGEP` inputs and mixed bulk inputs in the dataset and
   caching workflow,
3. validate the stage-local semantic flags before training starts,
4. run Stage 1 on the mixed-bulk sets from `training_target_sets`,
5. load the best Stage 1 checkpoint into Stage 2,
6. run Stage 2 on the pure `sctGEP` sets from `sct_file_path` with
   ground-truth cell proportions,
7. load the best Stage 1 and Stage 2 weights into Stage 3,
8. run Stage 3 on the mixed-bulk sets from `training_target_sets` with a small
   learning rate, and
9. save per-stage summaries and checkpoints.

## Validation and error handling

Validation should fail early when:

- a stage selects no datasets,
- `pure_sct_gep_pretrain` receives any dataset that is not marked pure
  `sctGEP`, where pure `sctGEP` inputs come from `sct_file_path`,
- `cell_prop_predictor_pretrain` or `mixed_bulk_joint_finetune` receives any
  dataset that is not marked mixed bulk, where mixed bulk inputs come from
  `training_target_sets`,
- Stage 2 is requested without a valid Stage 1 checkpoint,
- Stage 3 is requested without valid Stage 1 and Stage 2 checkpoints,
- `use_ground_truth_cell_prop: true` is set for a stage whose datasets do not
  provide exact cell proportions,
- a stage declares both train and freeze for the same module group.

Warnings should be emitted when:

- Stage 2 pure samples do not map cleanly to one-hot or otherwise exact
  cell-proportion vectors,
- Stage 3 learning rate is not substantially smaller than Stage 2,
- Stage 3 fine-tunes the proportion branch with no retention mechanism and no
  cell-proportion labels are available,
- the user enables Stage 3 bulk-only training on a dataset whose mixture
  composition differs strongly from the Stage 1 training distribution.

## Feedback: benefit side

This proposal has several strong points.

### 1. Stage 2 is a cleaner pretraining task

This is the strongest argument in favor of the new design.

A pure `sctGEP` contains one cell type only, so the decoder can focus on
learning cell-type-specific signal instead of sharing capacity across:

- mixture disentanglement,
- proportion errors, and
- cross-cell-type interference.

That makes Stage 2 a much cleaner curriculum than asking the model to learn
full GEPs from mixed bulk immediately.

### 2. The curriculum is conceptually simple

The proposed order is easy to explain:

1. learn proportions,
2. learn pure cell-type GEP reconstruction,
3. fine-tune everything together on mixed bulk.

This simplicity is appealing because it reduces architectural complexity
relative to the prototype-residual design.

### 3. Stage 3 becomes realistic domain adaptation

Stage 2 pretrains on the easy version of the problem, and Stage 3 adapts that
representation to the real mixed-bulk setting.

That is a reasonable transfer-learning strategy, especially if the current
direct full-GEP mode is struggling because it learns from mixtures too early.

### 4. The design reduces dependence on matched sctGEP labels

The final stage no longer requires direct `sctGEP` supervision.

This is attractive if:

- matched `sctGEP` labels are expensive,
- they are noisy,
- or you want the final stage to align more closely with the inference-time
  problem, which only observes mixed bulk.

## Feedback: risk side

This proposal also has real risks that are worth stating clearly.

### 1. Stage 2 may learn the wrong kind of encoder

A pure `sctGEP` sample and a mixed bulk sample are not the same input domain.

If Stage 2 strongly shapes the encoder using pure inputs only, the learned
representation may transfer imperfectly to Stage 3. In other words, Stage 2
could become excellent at pure-cell reconstruction but still leave a domain gap
for bulk inference.

This is the biggest technical risk in the new design.

### 2. Stage 2 may be too easy

When the proportion vector is one-hot and exact, Stage 2 removes most of the
composition ambiguity.

That is good for clean pretraining, but it also means the model can succeed in
Stage 2 without learning the same abstractions it needs in Stage 3. So Stage 2
pretraining could help the decoder a lot while helping the full mixed-bulk
inference problem only partially.

### 3. Stage 3 bulk-only training can drift

Without direct `sctGEP` supervision in Stage 3, the model can improve bulk
reconstruction while still drifting toward less faithful per-cell-type GEPs.

This is the central tradeoff of the proposal:

- you gain a cleaner and more realistic final-stage objective,
- but you lose a direct guardrail on the inferred `sctGEPs`.

If Stage 3 is too aggressive, it can partially undo the specificity learned in
Stage 2.

### 4. Fine-tuning Stage 1 can help or hurt

Reloading and fine-tuning the Stage 1 proportion predictor in Stage 3 is
reasonable, but it changes the optimization problem.

Pros:

- the two branches can co-adapt,
- Stage 1 can adjust to the decoder learned in Stage 2.

Cons:

- Stage 1 can drift away from the clean supervised solution learned in Stage 1,
- bulk-only optimization may reward proportion changes that help reconstruction
  but reduce cell-proportion accuracy.

### 5. The design gives up explicit interpretability of the previous proposal

Compared with the earlier prototype-residual design, this proposal is less
explicit about separating:

- shared cell-type structure, and
- sample-specific deviation.

That makes it simpler, but also less interpretable.

## Recommendation

This revised design is scientifically reasonable and worth testing.

My recommendation is:

1. adopt this three-stage workflow as a new ablation branch,
2. keep the previous four-stage prototype-residual design as a separate
   reference design in the documentation history,
3. implement the new stage-local dataset selectors first, because they are
   required by the new curriculum, and
4. add one conservative anti-drift safeguard for Stage 3, even if it is off by
   default in the main experiment configuration.

The most important recommendation is practical:

> Treat Stage 2 as pretraining and Stage 3 as adaptation, not as proof that
> the model no longer needs any guardrail on `sctGEP` structure.

If Stage 3 collapses or drifts, the first fixes to test are:

1. a smaller Stage 3 learning rate,
2. shorter Stage 3 fine-tuning,
3. weak retention toward the Stage 2 decoder, and
4. optional weak retention toward the Stage 1 cell-proportion solution.

## Verification plan

The revised design should be evaluated against:

1. the current three-stage baseline,
2. the earlier four-stage prototype-residual proposal, and
3. the direct full-GEP mode without this curriculum.

Check:

1. Stage 1 cell-proportion accuracy on mixed bulk validation sets,
2. Stage 2 pure `sctGEP` reconstruction quality,
3. Stage 3 mixed-bulk reconstruction quality,
4. final cell-proportion accuracy after Stage 3 fine-tuning,
5. final inferred `sctGEP` quality on held-out evaluation sets,
6. within-cell-type inter-sample similarity preservation,
7. whether Stage 3 erodes the Stage 2 cell-type structure.

The design is successful if it produces:

- equal or better cell-proportion accuracy than the current Stage 1 path,
- better final `sctGEP` reconstruction than direct full-GEP training from
  scratch,
- better preservation of cell-type specificity than the current collapsed
  direct mode,
- and stable Stage 3 behavior without requiring strong direct `sctGEP`
  supervision.
