# Four-stage prototype-residual staged training design

Last updated: 2026-08-28

## Status

This document is a proposed design. It is not implemented yet.

## Goal

This design adds a four-stage training workflow that lets VAEDecon learn both:

1. a stable, interpretable prototype GEP for each cell type, and
2. sample-specific residual variation around that prototype.

The main motivation is to resolve the current tradeoff:

- `learn_gep_residual: true` preserves within-cell-type variation better, but
  it does not expose a direct cell-type representation cleanly.
- `learn_gep_residual: false` exposes direct per-cell-type GEP outputs, but it
  often collapses toward highly similar inferred GEPs across cell types.

The proposed workflow makes the decomposition explicit:

```text
sample-specific sctGEP(sample, cell_type)
= prototype(cell_type) + residual(sample, cell_type)
```

## Why this change

The current model mostly asks one decoder branch to solve two different
problems at the same time:

1. represent the canonical expression program of a cell type, and
2. represent sample-specific deviations within that cell type.

Those two objectives do not naturally lead to the same optimum.

When the model directly predicts the full sctGEP, the easiest solution is often
to learn overly similar cell-type outputs that reconstruct the mixture well on
average. When the model predicts residuals relative to a reference mean, it can
preserve variation better, but the prototype itself remains implicit and is not
learned as a first-class object.

This design introduces a dedicated prototype stage before residual learning so
the model has one place to learn the shared signal and another place to learn
the per-sample variation.

## Design goals

This design must:

1. keep Stage 1 cell proportion training unchanged,
2. learn a nearly global prototype GEP for each cell type,
3. learn residuals only after the prototype stage is stable,
4. preserve within-cell-type inter-sample variation,
5. expose interpretable prototype outputs directly, and
6. keep the final joint stage small and conservative.

## Non-goals

This design does not aim to:

- replace the Stage 1 cell proportion predictor,
- remove the existing matched `sctGEP` supervision path,
- change the data preprocessing cache format,
- redesign the latent encoder family, or
- support many alternative prototype architectures in the first version.

## Observed failure mode

The current behavior suggests the model lacks a clean structural split between
shared cell-type signal and sample-specific signal.

### Residual mode

With `learn_gep_residual: true`, the decoder learns a residual around
`g_mean`. This keeps more within-cell-type structure, especially when combined
with residual-aware supervision such as inter-sample similarity losses.

The weakness is that the model does not learn a directly usable prototype head.
The shared cell-type expression comes from the precomputed `g_mean`, not from a
learned prototype bank that can adapt during staged training.

### Direct full-GEP mode

With `learn_gep_residual: false`, the decoder predicts the full per-sample,
per-cell-type GEP. This exposes a direct output, but it often converges toward
highly similar cell-type GEPs because the reconstruction objective can be
satisfied by collapsing much of the variation into mixture weights and shared
decoder behavior.

## Recommended approach

Use a four-stage training workflow with an explicit prototype bank:

1. Stage 1: unchanged cell proportion pretraining
2. Stage 2: prototype-bank training with Stage 1 frozen
3. Stage 3: residual decoder training with Stages 1 and 2 frozen
4. Stage 4: low-learning-rate joint fine-tuning

The key recommendation is architectural:

- Stage 2 should not be a normal sample-conditioned decoder.
- Stage 2 should learn one shared prototype vector per cell type.

That gives the model the strongest possible bias toward a nearly global
cell-type average.

## Alternative approaches considered

This section records the main alternatives and why they are not the preferred
starting point.

### Option 1: keep the current three-stage design

This option keeps the existing staged workflow and tunes only the residual
losses.

Pros:

- smallest implementation change,
- reuses current code paths directly.

Cons:

- does not solve the missing explicit prototype representation,
- does not address direct full-GEP collapse structurally.

### Option 2: sample-conditioned average decoder

This option adds a second-stage decoder that takes a sample input and predicts
an "average" GEP for each cell type.

Pros:

- easier to layer onto the current decoder code,
- may improve reconstruction quickly.

Cons:

- the output is no longer truly an average once it is sample-conditioned,
- Stage 2 can absorb sample-specific information and blur the intended split.

### Option 3: explicit prototype bank plus residual decoder

This option learns one prototype per cell type first, then learns residuals
around those prototypes.

Pros:

- cleanest factorization,
- best interpretability,
- strongest defense against prototype collapse.

Cons:

- more implementation work,
- requires careful residual constraints to stop Stage 3 from taking over.

This document recommends Option 3.

## Proposed model decomposition

The design introduces a learned prototype bank:

```text
P in R^(G x C)
```

where:

- `G` is the number of genes,
- `C` is the number of cell types,
- `P[:, c]` is the prototype GEP for cell type `c`.

The final sample-specific cell-type GEP becomes:

```text
GEP(sample, c) = P[:, c] + R(sample, c)
```

where `R(sample, c)` is the residual predicted in Stage 3.

The bulk reconstruction becomes:

```text
x_hat(sample) = sum_c pi_hat(sample, c) * (P[:, c] + R(sample, c))
```

where `pi_hat(sample, c)` comes from the frozen or fine-tuned Stage 1
cell-proportion branch.

## Four-stage workflow

### Stage 1: cell proportion pretraining

This stage keeps the Stage 1 training objective unchanged. Under the new
architecture, the old generic decoder path is split into a shared
`prototype_bank` and a `residual_decoder`.

- trainable modules:
  - `cell_prop_predictor`
- frozen modules:
  - `encoders`
  - `prototype_bank`
  - `residual_decoder`
- active objective:
  - cell proportion supervision

Expected result:

- accurate cell proportion predictions,
- stable `bulk_context_feature`,
- stable `pi_hat` for later stages.

### Stage 2: prototype-bank training

This stage learns the shared prototype GEP of each cell type while keeping the
Stage 1 predictor frozen.

- trainable modules:
  - `prototype_bank`
  - optional minimal mixing head if needed
- frozen modules:
  - `cell_prop_predictor`
  - `encoders`
  - `residual_decoder`
- input:
  - mixed bulk GEP,
  - frozen inferred `pi_hat` from Stage 1
- output:
  - one prototype GEP per cell type, shared across all samples

Reconstruction path:

```text
x_hat_stage2(sample) = sum_c pi_hat(sample, c) * P[:, c]
```

Expected result:

- a directly readable prototype GEP for each cell type,
- no sample-specific variation modeled yet,
- a clean baseline reconstruction using only cell proportions and prototypes.

### Stage 3: residual learning

This stage learns the sample-specific, cell-type-specific residuals on top of
the frozen prototype bank.

- trainable modules:
  - `encoders`
  - `residual_decoder`
- frozen modules:
  - `cell_prop_predictor`
  - `prototype_bank`
- input:
  - mixed bulk GEP,
  - frozen `pi_hat`,
  - frozen prototypes `P`
- output:
  - residual tensor `R(sample, c)`

Reconstruction path:

```text
x_hat_stage3(sample) = sum_c pi_hat(sample, c) * (P[:, c] + R(sample, c))
```

Expected result:

- better sample-specific sctGEP accuracy,
- preserved within-cell-type inter-sample variation,
- prototype branch remains interpretable because it stays frozen.

### Stage 4: low-learning-rate joint fine-tune

This stage loads the best Stage 3 checkpoint and opens all relevant modules for
small-step joint adaptation.

- trainable modules:
  - `cell_prop_predictor`
  - `prototype_bank`
  - `encoders`
  - `residual_decoder`
- frozen modules:
  - none
- learning rate:
  - much smaller than earlier stages

Expected result:

- mild alignment between proportion prediction, prototype bank, and residual
  decoder,
- improved final reconstruction without losing the prototype-residual split.

## Prototype-bank constraints

This section describes how Stage 2 stays close to a global average prototype
instead of drifting into a hidden sample-conditioned decoder.

### 1. Architectural constraint

The first version should use the strongest constraint:

- `prototype_bank` is a shared learnable parameter tensor,
- it does not take per-sample latent input,
- it does not depend on the current sample directly.

This is the main mechanism that makes the prototypes nearly global.

### 2. Prototype anchoring

If matched `sctGEP` targets are available, Stage 2 should compute empirical
cell-type mean targets and anchor each prototype to them.

Recommended loss:

```text
L_proto_anchor
= alpha * MSE(P_c, mean_target_c)
+ (1 - alpha) * (1 - cosine(P_c, mean_target_c))
```

This keeps the prototype biologically plausible while still letting it adapt.

### 3. Between-cell-type separation

To reduce prototype collapse, add a light repulsion or separation objective
between prototypes of different cell types.

One simple option is a cosine-based penalty:

```text
L_proto_sep = mean_{c != k} cosine(P_c, P_k)
```

This term should be small. It should stop exact collapse without forcing
unrealistic orthogonality.

### 4. Optional soft relaxation

The first implementation should not include per-sample prototype adjustment.
If that is needed later, the only acceptable extension is:

```text
P_sample,c = P_c + delta_sample,c
```

with a strong penalty on:

- `||delta_sample,c||^2`, and
- the sample mean of `delta_sample,c`.

This document does not recommend shipping that relaxed version first.

## Residual constraints

Stage 3 needs explicit guardrails so the residual decoder does not swallow the
full signal and make the prototype bank irrelevant.

### 1. Zero-mean residual pressure

Across samples within a cell type, residuals should stay close to zero mean:

```text
L_res_mean = || mean_sample(R(sample, c)) ||^2
```

This encourages the prototype bank to absorb the shared signal.

### 2. Residual magnitude penalty

Residuals should stay as small as possible while still improving
reconstruction:

```text
L_res_norm = mean ||R(sample, c)||^2
```

### 3. Residual structure supervision

The existing mean-centered residual supervision ideas remain useful here.
Stage 3 can reuse:

- matched residual supervision against true `sctGEP - prototype`,
- inter-sample similarity preservation in residual space,
- per-sample residual variance supervision.

These are already aligned with the user's preferred residual-first diagnostic
workflow.

## Loss design by stage

### Stage 1

Active:

- cell proportion loss

Off or near zero:

- prototype losses,
- residual losses,
- matched `sctGEP` losses,
- inter-sample residual losses.

### Stage 2

Active:

- bulk reconstruction from `pi_hat * prototype_bank`,
- prototype anchor loss,
- prototype separation loss

Off or near zero:

- residual losses,
- direct sample-specific `sctGEP` losses.

### Stage 3

Active:

- bulk reconstruction from `pi_hat * (prototype + residual)`,
- matched residual supervision,
- inter-sample residual similarity,
- per-sample residual variance,
- residual zero-mean and residual norm penalties

Frozen:

- prototype bank stays fixed.

### Stage 4

Active:

- same losses as Stage 3,
- optionally a weaker cell proportion loss to avoid Stage 1 drift,
- optionally a weak prototype-anchor retention loss so fine-tuning does not
  destroy the prototypes.

## Config design

The staged-training config should expand from three to four named stages.

Recommended stage names:

- `cell_prop_predictor_pretrain`
- `prototype_training`
- `residual_training`
- `joint_finetune`

The stage schema can reuse the current structure, but valid stage names and
module groups need to expand.

Recommended new module groups:

- `prototype_bank`
- `residual_decoder`

For the four-stage workflow, the current monolithic decoder group is treated as
split into these two logical parts. The first implementation does not need to
keep the old generic `decoder` stage group active inside the new workflow.

Example shape:

```yaml
training:
  staged_training:
    enabled: true
    run_stages:
      - cell_prop_predictor_pretrain
      - prototype_training
      - residual_training
      - joint_finetune
    stages:
      - name: cell_prop_predictor_pretrain
        train_modules: ["cell_prop_predictor"]
        freeze_modules: ["encoders", "prototype_bank", "residual_decoder"]
        learning_rate_scale: 1.0

      - name: prototype_training
        train_modules: ["prototype_bank"]
        freeze_modules: ["cell_prop_predictor", "encoders", "residual_decoder"]
        learning_rate_scale: 1.0

      - name: residual_training
        train_modules: ["encoders", "residual_decoder"]
        freeze_modules: ["cell_prop_predictor", "prototype_bank"]
        learning_rate_scale: 1.0

      - name: joint_finetune
        train_modules:
          ["cell_prop_predictor", "prototype_bank", "encoders", "residual_decoder"]
        freeze_modules: []
        learning_rate_scale: 0.05
```

## Execution model

The execution model should stay close to the current staged workflow.

The training workflow should:

1. build the dataset once,
2. prepare prototype-anchor targets if needed,
3. build the model once,
4. run the four stages in order or run a selected contiguous suffix,
5. promote the best checkpoint from each stage into the next stage,
6. store stage-local summaries and the final merged checkpoint.

The existing stage checkpoint metadata logic can be reused.

## Validation and error handling

Validation should fail early when:

- `prototype_training` is configured but `prototype_bank` is missing,
- `residual_training` is configured but `residual_decoder` is missing,
- Stage 2 is selected without a valid Stage 1 checkpoint,
- Stage 3 is selected without a valid Stage 2 checkpoint,
- Stage 4 is selected without a valid Stage 3 checkpoint,
- a stage declares both training and freezing for the same new module,
- Stage 2 uses sample-conditioned latent input in the first implementation.

Warnings should be emitted when:

- Stage 2 does not include any prototype anchor loss while matched targets are
  available,
- Stage 3 residual penalties are all zero,
- Stage 4 learning rate is not substantially lower than Stage 3,
- prototype separation is weighted so strongly that it may distort biology.

## Risks and tradeoffs

### Benefit side

This design offers:

- explicit and interpretable cell-type prototypes,
- better preservation of within-cell-type variation,
- a clearer decomposition of shared versus sample-specific signal,
- a cleaner answer to the current `learn_gep_residual` versus direct full-GEP
  tradeoff.

### Risk side

This design also introduces real risks:

1. Stage 2 can still learn biased prototypes if Stage 1 proportions are wrong.
2. Stage 3 can still dominate if residual penalties are too weak.
3. Stage 4 can partially undo the decomposition if the fine-tuning step is too
   aggressive.
4. More stages mean more checkpoints, more hyperparameters, and more tuning
   cost.

## Recommendation

Implement the strict version first:

1. keep Stage 1 unchanged,
2. add a true shared `prototype_bank` for Stage 2,
3. add a separate `residual_decoder` for Stage 3,
4. keep Stage 2 prototypes frozen during Stage 3,
5. use a short low-learning-rate Stage 4.

This is the smallest version that directly addresses the user's core problem:

- residual mode preserves variation but hides the prototype,
- direct full-GEP mode exposes a prototype-like output but collapses diversity.

The explicit prototype-plus-residual factorization is a more faithful match to
the biological and modeling intent than either current mode alone.

## Verification plan

The first implementation should be evaluated against the existing staged
workflow on the same dataset splits.

Check:

1. cell proportion accuracy after Stage 1,
2. prototype quality after Stage 2 against empirical cell-type average targets,
3. matched `sctGEP` accuracy after Stage 3,
4. within-cell-type inter-sample variance preservation after Stage 3,
5. prototype drift after Stage 4,
6. whether inferred cell-type prototypes remain more distinct than the current
   `learn_gep_residual: false` baseline.

The design is successful if it produces:

- a readable prototype per cell type,
- better sample-specific `sctGEP` reconstruction than direct full-GEP mode,
- stronger cell-type separation than the current collapsed direct mode,
- and less loss of inter-sample variation than the current direct mode.
