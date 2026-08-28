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

The proposed workflow makes the decomposition explicit in gene space and also
defines an optional visualization-space summary:

```text
prototype_GEP(cell_type) = prototype_decoder(z_proto(cell_type))
sample_specific_GEP(sample, cell_type)
= prototype_GEP(cell_type) + residual_GEP(sample, cell_type)
z_vis(sample, cell_type) = z_proto(cell_type) + z_res(sample, cell_type)
```

This design uses `z_vis(sample, cell_type)` only as a low-dimensional summary
for downstream visualization and analysis. It is not the primary object used to
decode or reconstruct the full sample-specific sctGEP in the first
implementation.

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
5. expose interpretable prototype outputs directly,
6. provide a low-dimensional prototype embedding and a low-dimensional residual
   embedding in the same space, and
7. keep the final joint stage small and conservative.

## Non-goals

This design does not aim to:

- replace the Stage 1 cell proportion predictor,
- remove the existing matched `sctGEP` supervision path,
- change the data preprocessing cache format,
- redesign the latent encoder family completely, or
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

This document recommends Option 3: an explicit prototype bank plus a residual
decoder. The design uses four stages:

1. Stage 1: unchanged cell proportion pretraining
2. Stage 2: prototype-bank training with Stage 1 frozen
3. Stage 3: residual decoder training with Stages 1 and 2 frozen
4. Stage 4: low-learning-rate fine-tuning with prototypes still frozen by
   default

The key recommendation is architectural:

- Stage 2 learns one shared prototype embedding per cell type.
- Stage 2 also learns a simple prototype decoder that maps that embedding to a
  prototype GEP.
- Stage 3 learns a sample-specific residual embedding that has the same
  dimension and compatible semantics as the Stage 2 prototype embedding, but it
  is still a different learned vector.
- The combined embedding `z_proto + z_res` is used only as a visualization
  summary, not as the primary decoding path.

This keeps the prototype nearly global while still supporting a low-dimensional
representation that can be added to the Stage 3 residual embedding and used for
visualization.

Here, "compatible semantics" means the two embeddings are designed to be added
for visualization in a common low-dimensional coordinate system. They do not
share the same values or parameters. Stage 2 learns the baseline prototype
representation, and Stage 3 learns an offset in a matching coordinate system.
The first implementation does not require decoding from `z_proto + z_res`.

Stage 2 can also use a larger batch size than the later stages. That is a good
fit for prototype learning because it stabilizes batch-mean supervision and
reduces noise in the estimated cell-type average target.

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
an average GEP for each cell type.

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
- strongest defense against prototype collapse,
- directly supports a shared embedding space for prototype and residual
  representations.

Cons:

- more implementation work,
- requires careful residual constraints to stop Stage 3 from taking over.

This document recommends Option 3.

## Proposed model decomposition

The design introduces a learned prototype embedding bank:

```text
Z_proto in R^(d x C)
```

where:

- `d` is the prototype and residual embedding dimension,
- `C` is the number of cell types, and
- `Z_proto[:, c]` is the prototype embedding for cell type `c`.

The prototype GEP is reconstructed by a Stage 2 prototype decoder:

```text
P[:, c] = D_proto(Z_proto[:, c])
```

The residual branch predicts a sample-specific embedding in the same space:

```text
Z_res(sample, c) in R^d
```

The final sample-specific cell-type GEP is modeled in gene space as:

```text
GEP(sample, c) = P[:, c] + R(sample, c)
```

The bulk reconstruction becomes:

```text
x_hat(sample) = sum_c pi_hat(sample, c) * GEP(sample, c)
```

where `pi_hat(sample, c)` comes from the frozen or fine-tuned Stage 1
cell-proportion branch.

This factorization has two advantages:

1. `Z_proto[:, c]` is a reusable low-dimensional representation of the cell
   type prototype.
2. `Z_proto[:, c] + Z_res(sample, c)` can be used as a reusable visualization
   summary of the full sample-specific sctGEP if the two embedding branches are
   kept in a compatible coordinate system.

## Four-stage workflow

Each stage has a specific role. The first implementation should keep those
roles explicit instead of letting one branch absorb multiple jobs.

### Stage 1: cell proportion pretraining

This stage keeps the Stage 1 training objective unchanged. Under the new
architecture, the old generic decoder path is split into a shared
`prototype_bank` and a `residual_decoder`, but neither one is trained here.

- trainable modules:
  - `cell_prop_predictor`
- frozen modules:
  - `encoders`
  - `prototype_bank`
  - `prototype_decoder`
  - `residual_decoder`
- active objective:
  - cell proportion supervision

Expected result:

- accurate cell proportion predictions,
- stable `pi_hat` for later stages.

If a conditioned architecture still exposes an intermediate
`bulk_context_feature`, that feature is an internal encoder-side representation.
It is not the same object as the predicted cell proportions. `pi_hat` is the
actual inferred cell proportion vector used in reconstruction.

### Stage 2: prototype-bank training

This stage learns the shared prototype GEP of each cell type while keeping the
Stage 1 predictor frozen.

- trainable modules:
  - `prototype_bank`
  - `prototype_decoder`
  - optional lightweight encoder adaptor if prototype decoding benefits from
    encoder-side context
- frozen modules:
  - `cell_prop_predictor`
  - the main Stage 1 head
  - `residual_decoder`
- input:
  - mixed bulk GEP,
  - frozen inferred `pi_hat` from Stage 1,
  - optionally a frozen or lightly trainable encoder context feature
- output:
  - one prototype embedding per cell type, shared across all samples,
  - one prototype GEP per cell type

Reconstruction path:

```text
P[:, c] = D_proto(Z_proto[:, c])
x_hat_stage2(sample) = sum_c pi_hat(sample, c) * P[:, c]
```

Expected result:

- a directly readable prototype GEP for each cell type,
- a low-dimensional prototype embedding with the same dimension as the Stage 3
  residual embedding,
- no sample-specific variation modeled yet,
- a clean baseline reconstruction using only cell proportions and prototypes.

The prototype decoder in this stage should be simpler than the current
full-sctGEP decoder. Stage 2 only needs to reconstruct shared prototypes, so
the mapping is easier and should benefit from a smaller decoder with fewer ways
to overfit sample-specific structure.

The recommended training setup for this stage uses a larger batch size than the
later stages. Batch sizes such as `512` or `1024` can improve stability if the
hardware supports them.

The default recommendation is to keep Stage 2 largely independent from the main
encoder stack. In the first implementation, Stage 2 should act mainly as a
prototype provider for later stages rather than as a jointly trained feature
extractor. That makes the learned prototypes more stable and easier to reuse in
Stage 3 and Stage 4.

If Stage 2 is independent from Stage 3, the combined embedding is still
meaningful as a visualization summary, but only under a narrower definition.
It is not treated as the literal decoded full sctGEP. Instead, it is an
additive low-dimensional summary where:

- `Z_proto` gives the baseline position of each cell type,
- `Z_res` gives the sample-specific offset around that baseline, and
- `Z_vis = Z_proto + Z_res` gives a compact representation of the inferred
  sample-specific state.

For that to work well, the two branches do not need joint training, but they do
need:

- the same embedding dimension,
- comparable scale through normalization or explicit scaling control, and
- optionally a light alignment loss so `Z_res` behaves like an offset from
  `Z_proto` instead of an unrelated code.

If Stage 2 needs sample-conditioned context to improve prototype learning, add
only a lightweight dedicated prototype encoder or adaptor for Stage 2. Do not
silently reuse or unfreeze the full Stage 1 or Stage 3 encoder stack in the
first implementation.

### Stage 3: residual learning

This stage learns the sample-specific, cell-type-specific residuals on top of
the frozen prototype bank.

- trainable modules:
  - `encoders`
  - `residual_decoder`
- frozen modules:
  - `cell_prop_predictor`
  - `prototype_bank`
  - `prototype_decoder`
- input:
  - mixed bulk GEP,
  - frozen `pi_hat`,
  - frozen prototype embeddings `Z_proto`,
  - frozen prototype GEPs `P`
- output:
  - residual embedding tensor `Z_res(sample, c)`,
  - residual GEP tensor `R(sample, c)`,
  - optional visualization embedding `Z_vis(sample, c)`

Reconstruction path:

```text
GEP_stage3(sample, c) = P[:, c] + R(sample, c)
x_hat_stage3(sample) = sum_c pi_hat(sample, c) * GEP_stage3(sample, c)
```

Optional visualization summary:

```text
Z_vis(sample, c) = Z_proto[:, c] + Z_res(sample, c)
```

Expected result:

- better sample-specific sctGEP accuracy,
- preserved within-cell-type inter-sample variation,
- prototype branch remains interpretable because it stays frozen.

The residual branch in this stage is learned in a supervised way. The primary
supervision target is the full matched sctGEP across cell types after adding
the prototype and residual together. A residual-only target can still be used
as a secondary auxiliary term, but the main Stage 3 target should be the full
sample-specific sctGEP because it is more stable when the learned prototype
bank does not match the empirical mean perfectly.

The Stage 3 residual branch can reuse most of the current decoder logic, but it
should emit a residual embedding for visualization plus a residual GEP for
reconstruction. The residual embedding does not need to decode the final GEP
directly in the first implementation. Instead, it only needs to live in a
compatible low-dimensional coordinate system with `Z_proto` so the two can be
added for visualization.

Here, `prototype_bank` is not an encoder. It is a learnable lookup table or
parameter bank containing one prototype embedding per cell type. The
cell-type identity is carried by `Z_proto` because each cell type has its own
indexed prototype embedding, its own decoded prototype GEP, and its own anchor
loss against the corresponding cell-type mean target. Prototype separation loss
further discourages different cell types from collapsing to the same prototype.

In practice, that means:

- `Z_proto[:, c]` is the baseline identity vector for cell type `c`,
- `Z_res(sample, c)` only needs to model sample-specific deviation around that
  identity, and
- `Z_vis(sample, c) = Z_proto[:, c] + Z_res(sample, c)` gives a compact
  visualization summary even though reconstruction still happens in gene space
  through `P[:, c] + R(sample, c)`.

### Stage 4: low-learning-rate fine-tune

This stage loads the best Stage 3 checkpoint and performs small-step joint
adaptation.

- trainable modules:
  - `cell_prop_predictor`
  - `encoders`
  - `residual_decoder`
- frozen modules by default:
  - `prototype_bank`
  - `prototype_decoder`
- optional trainable modules in a later ablation:
  - `prototype_bank`
  - `prototype_decoder`
- learning rate:
  - much smaller than earlier stages

Expected result:

- mild alignment between proportion prediction, residual inference, and bulk
  reconstruction,
- improved final reconstruction without losing the prototype-residual split.

The default recommendation is to keep the prototype bank frozen in Stage 4.
That preserves interpretability and reduces the chance that fine-tuning drifts
the learned prototype toward sample-specific behavior.

## Prototype-bank constraints

This section describes how Stage 2 stays close to a global average prototype
instead of drifting into a hidden sample-conditioned decoder.

### 1. Architectural constraint

The first version should use a shared prototype embedding bank:

- `prototype_bank` is a shared learnable embedding tensor,
- it does not take per-sample latent input,
- it does not depend on the current sample directly,
- it is decoded into a gene-space prototype by `prototype_decoder`.

This is the main mechanism that makes the prototypes nearly global.

This still supports a low-dimensional prototype representation. The prototype
embedding itself is the low-dimensional object. Stage 3 then learns
sample-specific residual embeddings in the same space, and the combined
embedding `Z_proto + Z_res` becomes the representation of the full
sample-specific sctGEP.

### 2. Prototype anchoring

If matched `sctGEP` targets are available, Stage 2 should anchor each
prototype to an empirical cell-type mean target.

Recommended loss:

```text
L_proto_anchor
= alpha * MSE(P_c, mean_target_c)
+ (1 - alpha) * (1 - cosine(P_c, mean_target_c))
```

This keeps the prototype biologically plausible while still letting it adapt.

The preferred target is the empirical global mean for each cell type computed
from the matched training set. A large batch mean can be used as a stochastic
estimator of that target, especially with batch sizes such as `512` or `1024`.
If batch mean supervision is used, the implementation should stabilize it with
either:

- large batches,
- a running exponential moving average, or
- periodic recomputation of the full training mean target.

### 3. Between-cell-type separation

To reduce prototype collapse, add a light repulsion or separation objective
between prototypes of different cell types.

One simple option is a cosine-based penalty:

```text
L_proto_sep = mean_{c != k} cosine(P_c, P_k)
```

This term should be small. It should stop exact collapse without forcing
unrealistic orthogonality.

Orthogonal initialization of the prototype embeddings is a reasonable starting
point, but an orthogonality loss should not be the default objective. Hard or
strong orthogonality can impose geometry that is not biologically realistic.

The current `kld_type: sep` mechanism remains more appropriate for the sample
latent space than for the Stage 2 prototype bank. If a later ablation adds a
probabilistic prototype embedding, it can explore a cell-type-specific prior
vector, but that is not the first implementation.

### 4. Optional soft relaxation

After the strict shared-prototype version is working, a soft relaxation can be
added:

```text
Z_proto_sample(sample, c) = Z_proto[:, c] + delta(sample, c)
```

with a strong penalty on:

- `||delta(sample, c)||^2`, and
- the sample mean of `delta(sample, c)`.

This is the preferred extension if the strict global prototype proves too
rigid. It keeps the prototype near the global average while still allowing a
small amount of adaptive movement.

The first implementation can keep this branch disabled by default and expose it
as an ablation.

## Residual constraints

Stage 3 needs explicit guardrails so the residual decoder does not swallow the
full signal and make the prototype bank irrelevant.

### 1. Zero-mean residual pressure

Across samples within a cell type, residuals should stay close to zero mean:

```text
L_res_mean = || mean_sample(Z_res(sample, c)) ||^2
```

This encourages the prototype bank to absorb the shared signal.

Larger Stage 3 batch sizes help this term because the batch mean is a less
noisy estimator of the sample mean. It is useful to increase the batch size as
far as memory permits, but this is less important than in Stage 2 because Stage
3 still has to model sample-specific detail. If needed, a running average can
be used to stabilize this term.

### 2. Residual magnitude penalty

Residuals should stay as small as possible while still improving
reconstruction:

```text
L_res_norm = mean ||Z_res(sample, c)||^2
```

### 3. Residual structure supervision

The existing mean-centered residual supervision ideas remain useful here.
Stage 3 can reuse:

- matched supervision against the full target `GEP_true(sample, c)`,
- optional matched residual supervision against
  `GEP_true(sample, c) - P[:, c]`,
- inter-sample similarity preservation in residual space,
- per-sample residual variance supervision.

These are already aligned with the current residual-first diagnostic workflow.

## Loss design by stage

Each stage should activate only the losses needed for that stage's role.

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

- bulk reconstruction from `pi_hat * P`,
- prototype anchor loss,
- prototype separation loss

Off or near zero:

- residual losses,
- direct sample-specific `sctGEP` losses.

Stage 2 prototype supervision can use either:

1. a precomputed empirical cell-type mean target, or
2. a large-batch mean target as an online estimator.

The default recommendation is to use the precomputed empirical mean when it is
available, and use the batch mean only as a practical approximation or online
update signal.

### Stage 3

Active:

- bulk reconstruction from `pi_hat * GEP_stage3`,
- matched full-sctGEP supervision on `GEP_stage3(sample, c)`,
- optional matched residual supervision on `Z_res` or
  `GEP_stage3(sample, c) - P[:, c]`,
- inter-sample residual similarity,
- per-sample residual variance,
- residual zero-mean and residual norm penalties

Frozen:

- prototype bank stays fixed,
- prototype decoder stays fixed.

The primary supervised target in Stage 3 is the full sctGEP after adding the
prototype and residual. Residual-only supervision is best treated as an
auxiliary loss.

### Stage 4

Active:

- same losses as Stage 3,
- optionally a weaker cell proportion loss to avoid Stage 1 drift,
- optionally a weak prototype-anchor retention loss so fine-tuning does not
  destroy the prototypes.

Default frozen modules:

- `prototype_bank`,
- `prototype_decoder`.

## Embedding outputs and visualization

The model should expose three embedding views for analysis:

1. `Z_proto[:, c]` for prototype-level visualization,
2. `Z_res(sample, c)` for residual-only variation analysis,
3. `Z_vis(sample, c) = Z_proto[:, c] + Z_res(sample, c)` as an optional
   low-dimensional visualization summary of the final sample-specific
   cell-type representation.

The final stage and the prediction pipeline on test sets should save
`Z_vis(sample, c)` so the user can visualize the inferred sample-specific
cell-type state directly. Prototype-only and residual-only views can also be
saved as auxiliary outputs.

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
- `prototype_decoder`
- `residual_decoder`

For the four-stage workflow, the current monolithic decoder group is treated as
split into these logical parts. The first implementation does not need to keep
the old generic `decoder` stage group active inside the new workflow.

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
        freeze_modules:
          ["encoders", "prototype_bank", "prototype_decoder", "residual_decoder"]
        learning_rate_scale: 1.0

      - name: prototype_training
        train_modules: ["prototype_bank", "prototype_decoder"]
        freeze_modules: ["cell_prop_predictor", "residual_decoder"]
        learning_rate_scale: 1.0
        batch_size_override: 512

      - name: residual_training
        train_modules: ["encoders", "residual_decoder"]
        freeze_modules:
          ["cell_prop_predictor", "prototype_bank", "prototype_decoder"]
        learning_rate_scale: 1.0

      - name: joint_finetune
        train_modules: ["cell_prop_predictor", "encoders", "residual_decoder"]
        freeze_modules: ["prototype_bank", "prototype_decoder"]
        learning_rate_scale: 0.05
```

If Stage 2 training benefits from a lightweight encoder adaptor, it can be
added as a separate module group instead of silently unfreezing the full Stage
1 encoder stack.

The default recommendation is to keep Stage 2 independent and use it as a
prototype provider for the later stages. That gives the cleanest separation of
roles:

- Stage 2 learns stable cell-type prototypes,
- Stage 3 learns sample-specific residuals around those prototypes,
- Stage 4 fine-tunes the residual path and cell-proportion path without
  rewriting the prototype bank.

A later ablation can add a dedicated Stage 2 prototype encoder and let it
participate in later stages, but that should be an explicit experimental branch
rather than the default design. Otherwise, the prototype stage can drift toward
sample-specific behavior and lose the interpretability benefit that motivated
this design.
## Execution model

The execution model should stay close to the current staged workflow.

The training workflow should:

1. build the dataset once,
2. prepare prototype-anchor targets if needed,
3. build the model once,
4. run the four stages in order or run a selected contiguous suffix,
5. promote the best checkpoint from each stage into the next stage,
6. save stage-local summaries, learned prototype outputs, and learned embedding
   outputs,
7. export combined embeddings for final-stage training summaries and test-set
   prediction outputs.

The existing stage checkpoint metadata logic can be reused.

## Validation and error handling

Validation should fail early when:

- `prototype_training` is configured but `prototype_bank` is missing,
- `prototype_training` is configured but `prototype_decoder` is missing,
- `residual_training` is configured but `residual_decoder` is missing,
- Stage 2 is selected without a valid Stage 1 checkpoint,
- Stage 3 is selected without a valid Stage 2 checkpoint,
- Stage 4 is selected without a valid Stage 3 checkpoint,
- a stage declares both training and freezing for the same new module,
- Stage 2 receives direct per-sample latent input in the first implementation.

Warnings should be emitted when:

- Stage 2 does not include any prototype anchor loss while matched targets are
  available,
- Stage 3 residual penalties are all zero,
- Stage 4 learning rate is not substantially lower than Stage 3,
- prototype separation is weighted so strongly that it may distort biology,
- the Stage 2 batch size is too small for stable batch-mean anchoring when that
  approximation is enabled.

## Risks and tradeoffs

### Benefit side

This design offers:

- explicit and interpretable cell-type prototypes,
- a shared low-dimensional embedding space for prototypes and residuals,
- better preservation of within-cell-type variation,
- a clearer decomposition of shared versus sample-specific signal,
- a cleaner answer to the current `learn_gep_residual` versus direct full-GEP
  tradeoff.

### Risk side

This design also introduces real risks:

1. Stage 2 can still learn biased prototypes if Stage 1 proportions are wrong.
2. Stage 3 can still dominate if residual penalties are too weak.
3. Stage 4 can still partially undo the decomposition if fine-tuning is too
   aggressive or if prototypes are accidentally unfrozen.
4. Large Stage 2 batches improve stability but increase memory pressure.
5. More stages mean more checkpoints, more hyperparameters, and more tuning
   cost.

## Recommendation

Implement the strict shared-embedding version first:

1. keep Stage 1 unchanged,
2. add a true shared `prototype_bank` for Stage 2,
3. add a `prototype_decoder` that is simpler than the current full decoder,
4. add a separate `residual_decoder` for Stage 3,
5. learn residuals in the same embedding space as the prototypes,
6. keep Stage 2 prototypes frozen during Stage 3 and Stage 4 by default,
7. use a short low-learning-rate Stage 4.

This is the smallest version that directly addresses the core problem:

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
3. stability of Stage 2 prototypes when using large-batch supervision,
4. matched `sctGEP` accuracy after Stage 3,
5. within-cell-type inter-sample variance preservation after Stage 3,
6. prototype drift after Stage 4,
7. whether the combined embedding `Z_proto + Z_res` separates samples and cell
   types in a biologically meaningful way,
8. whether inferred cell-type prototypes remain more distinct than the current
   `learn_gep_residual: false` baseline.

The design is successful if it produces:

- a readable prototype per cell type,
- a stable low-dimensional prototype embedding,
- a useful combined embedding for final sample-specific sctGEP visualization,
- better sample-specific `sctGEP` reconstruction than direct full-GEP mode,
- stronger cell-type separation than the current collapsed direct mode,
- and less loss of inter-sample variation than the current direct mode.

## Stage 4 representation strategy

The first implementation must keep a clean distinction between
reconstruction-space modeling and visualization-space summarization.

In Stage 4, it is possible to fine-tune the system while still using a
combined prototype-plus-residual representation, but the safest version does
not try to decode the summed embedding directly into the full sctGEP.

That direct-decoding idea is not recommended in the first implementation for
three reasons:

1. the prototype decoder and residual decoder solve different subproblems and
   do not naturally define one shared de novo decoder,
2. Stage 4 is intentionally short and conservative, so it is a poor place to
   train a brand-new fusion decoder from scratch, and
3. forcing direct decoding from `Z_proto + Z_res` too early can collapse the
   factorization and reduce the interpretability gained from separating shared
   signal and residual variation.

The default Stage 4 design is therefore:

- keep gene-space reconstruction as `P[:, c] + R(sample, c)`,
- keep `Z_vis(sample, c) = Z_proto[:, c] + Z_res(sample, c)` as a
  visualization-only summary,
- fine-tune only the residual path and the cell-proportion path, while keeping
  prototype decoding fixed by default.

If a stronger joint representation is needed later, there are three reasonable
extensions, listed in recommended order.

### Option A: keep additive visualization only

This is the recommended default.

In this option:

- `Z_proto` and `Z_res` are aligned only enough to support visualization,
- decoding still happens through `P + R` in gene space, and
- no new Stage 4 decoder is introduced.

This is the lowest-risk design and best preserves interpretability.

### Option B: add a lightweight fusion head for representation only

This is the recommended first ablation if the simple sum is not expressive
enough.

Instead of decoding `Z_proto + Z_res` directly, learn a small fusion module:

```text
Z_fused(sample, c) = F([Z_proto[:, c], Z_res(sample, c)])
```

where `F` is a shallow MLP or gated linear layer.

Use `Z_fused` only for:

- visualization,
- clustering,
- nearest-neighbor retrieval, or
- auxiliary metric-learning losses.

Do not use it as the primary reconstruction decoder input in the first
ablation. This keeps the representation more flexible without destabilizing the
main factorization.

### Option C: add a fusion decoder in a later version

If the long-term goal is to decode the full sctGEP directly from a single
combined embedding, the clean way is not to bolt that decoder onto Stage 4
alone. Instead:

1. add a dedicated fusion embedding `Z_fused`,
2. add a dedicated fusion decoder `D_fused`,
3. train `D_fused` with supervision from the full matched sctGEP, and
4. introduce this path before Stage 4, or train it as an auxiliary branch over
   both Stage 3 and Stage 4.

That path is viable, but it is a separate model design, not a small Stage 4
fine-tune tweak.

From a deep-learning perspective, this is the compression question: if the
model already factorizes the signal into a prototype code and a residual code,
you only gain something from a fused decoder if it learns a shorter and more
stable description than the existing `P + R` path. Otherwise, it just adds
capacity and can blur the decomposition.

For that reason, the first implementation should treat combined embeddings as
analysis artifacts, not as the primary reconstruction interface.
