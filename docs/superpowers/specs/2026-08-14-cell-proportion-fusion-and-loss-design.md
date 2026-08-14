# Design: cell-proportion fusion, head, loss, and scheduling

Last updated: 2026-08-14

## Status

Proposed.

## Goal

Improve cell-proportion prediction accuracy in `VAEDecon` by redesigning the
cell-proportion branch while keeping the latent posterior branch stable and
probabilistically well-founded.

The main goals are:

1. replace post-activation averaging of multi-encoder cell-proportion outputs
   with a feature-fusion-based prediction path
2. add a deeper, reusable cell-proportion head
3. replace the current pure-MSE cell-proportion loss with a more
   composition-aware objective
4. support scheduled ramp-up of auxiliary loss weights and related training
   knobs
5. preserve backward compatibility in configuration and existing workflows

## User requirements

The user asked for the following behavior and design properties:

1. when multiple encoders are used, avoid smoothing caused by averaging final
   cell-proportion predictions
2. fuse encoder features earlier for the cell-proportion branch
3. design a reusable dense building block that can be used for the new head and
   related fusion components
4. keep the whole-VAE fusion story coherent, especially for `mu` and `logvar`
5. try `L1 + 0.5 * KL(target || pred)` as the first replacement for the
   current cell-proportion loss
6. support a reusable training workflow where some loss weights begin at `0`
   and are gradually increased during training
7. write the implementation as an explicit spec before code changes

## Motivation

The current model uses different fusion behaviors for different branches when
multiple encoders are enabled:

- `mu_types` and `log_var_types` are fused through posterior fusion
- `cell_prop` is computed by averaging already-normalized predictions

This asymmetry is partly reasonable and partly problematic.

For the latent branch, posterior fusion is principled because `mu/logvar`
parameterize a Gaussian latent distribution. Product-of-Experts (PoE) or
posterior averaging has a meaningful probabilistic interpretation.

For the cell-proportion branch, however, averaging final `softmax` outputs is
not ideal. It tends to oversmooth predictions, especially for low-abundance
cell types near zero. That is directly harmful when predicted proportions are
later used to infer cell-type existence.

At the same time, the current cell-proportion head is shallow and the current
supervised loss is pure MSE, which is not a good inductive match for simplex
targets.

The proposed design therefore changes the **cell-proportion branch** first
while keeping the **latent posterior branch** mostly unchanged for the first
iteration.

## Non-goals

This design does not:

- replace the current posterior fusion of `mu/logvar` with a full shared
  pre-latent architecture in the first iteration
- redesign the decoder or reconstruction path beyond what is required to use
  the improved cell-proportion predictions
- change the semantics of matched `sctGEP` supervision itself
- change the existing meaning of `fusion_strategy` for the latent branch
- require every existing config to opt into the new behavior automatically

The first implementation should be a controlled, backward-compatible extension
rather than a full architectural rewrite.

## Current behavior

## Encoder outputs

Each encoder currently emits:

- `mu_all_types`
- `logvar_all_types`
- `mu_mean`
- `logvar_mean`
- `cell_prop`
- optionally `dd_alpha`

For single-encoder runs, the VAE uses that encoder output directly.

For multi-encoder runs:

- `mu/logvar` are fused by `_fuse_encoder_posteriors(...)`
- `cell_prop` is averaged across encoders after each encoder has already
  produced its final normalized output

This means the latent branch already has an explicit fusion strategy, but the
cell-proportion branch still uses a simple linear opinion pool.

## Latent fusion

The current latent fusion strategies are:

- `poe_posterior`
- `avg_posterior`
- `pre_latent` legacy/experimental option

`poe_posterior` is the default and is the preferred current strategy for
multiple encoders.

## Cell-proportion head

The current encoder-side cell-proportion head is effectively a single linear
projection from the encoder feature to the proportion output dimension.

This is lightweight, but likely underpowered for the task.

## Cell-proportion loss

The current supervised cell-proportion loss is elementwise MSE summed across
cell types.

That loss is easy to optimize, but it does not model the geometry of
compositional targets well and tends to over-emphasize larger components.

## Auxiliary loss scheduling

The codebase already supports scheduling for some gene-statistic weights, but
there is no general reusable scheduler for auxiliary weights such as:

- `cell_type_sct_gep_weight`
- `hierarchical_code_weight`
- `cell_type_existence_weight`
- `cell_type_existence_shift_scale`

## Proposed change

## Overview

Add a new feature-fusion-based cell-proportion pathway that is independent from
the latent posterior fusion policy.

The core design choice is:

- keep posterior-aware fusion for `mu/logvar`
- add feature-level fusion for cell-proportion prediction

This produces a consistent whole-VAE policy:

1. use probabilistic fusion where uncertainty matters
2. use feature fusion where predictive sharpness matters

## Branch-specific fusion policy

### Latent branch (`mu`, `logvar`)

Keep the current latent fusion behavior:

- default: `poe_posterior`
- baseline: `avg_posterior`

This branch remains responsible for:

- latent sampling
- reconstruction
- repulsion and attractor terms
- hierarchical-code loss
- existence-shift application after the posterior is fused

### Cell-proportion branch

Add a new branch-specific fusion policy for cell-proportion prediction.

Supported strategies:

1. `legacy_output_average`
   - current behavior
   - average encoder-level final cell-proportion outputs
   - kept for backward compatibility and ablation

2. `shared_feature_mean`
   - each encoder exposes a feature vector
   - project features into a shared fusion dimension
   - mean-pool the projected features
   - apply one shared cell-proportion head

3. `shared_feature_gated`
   - same shared feature projection as above
   - learn sample-specific encoder weights
   - compute a weighted fused feature
   - apply one shared cell-proportion head

Recommended default for new experiments:

- `shared_feature_gated` when `n_encoders > 1`
- `shared_feature_mean` as the simplest new baseline

Recommended default for backward compatibility:

- if the new cell-proportion fusion config is absent, preserve
  `legacy_output_average`

## Encoder feature exposure

To support feature-level fusion, each encoder must expose a sample-level
feature vector before its current cell-proportion output layer.

This design introduces a standard encoder-side contract:

```text
encoder(x) -> {
  ...existing outputs...,
  cell_prop_feature: (B, D_e)
}
```

For existing encoders, `cell_prop_feature` should usually be the final hidden
representation before the current `fc_mu_logvar` and `fc_dd_alpha` heads.

This avoids duplicating encoder computation and provides a clean, reusable
boundary for the new fusion path.

## Shared feature projection

Because encoders may produce features of different dimensionality, the VAE
should project each encoder feature into a shared fusion space:

```text
h_e      = cell_prop_feature_e
z_e      = projector_e(h_e) in R^(d_fuse)
```

Each projector uses a reusable dense block.

## Reusable dense block

Add a reusable dense building block:

```text
Linear -> LayerNorm -> GELU -> Dropout
```

Recommended code shape:

- `MLPBlock`
- used by:
  - cell-proportion feature projectors
  - the shared cell-proportion head
  - optionally future auxiliary heads

This block should not replace every linear layer in the project immediately.
The first implementation should use it only in the new components.

## Shared cell-proportion head

Add a reusable multi-layer head:

```text
z_fused -> MLPBlock* -> output_linear
```

The output dimension depends on the activation mode:

- `softmax`: `n_cell_types`
- `softplus`: `n_cell_types`
- `sigmoid`: `n_cell_types - 1`

The final activation logic should continue to reuse the existing
`build_cell_prop_from_head_output(...)` helper so activation semantics remain
consistent across the codebase.

## Gated fusion

For `shared_feature_gated`, learn sample-specific encoder weights:

```text
z_e      = projector_e(h_e)
alpha    = softmax(g(concat(z_1, ..., z_E)))
z_fused  = sum_e alpha_e * z_e
```

This allows the model to trust different encoders differently for each sample.

The gating network should be intentionally small:

- one or two `MLPBlock`s at most
- final linear output to `n_encoders`

The goal is adaptive fusion, not a large extra branch.

## Single-encoder behavior

When `n_encoders == 1`:

- `legacy_output_average` remains equivalent to the current behavior
- `shared_feature_mean` and `shared_feature_gated` should both reduce to using
  the single exposed feature directly

This keeps the implementation unified and avoids special-purpose code paths.

## Cell-proportion loss redesign

Replace the current pure-MSE supervised loss with a configurable loss family.

### First supported new mode

Add:

- `cell_prop_loss_type: "mse" | "l1_kl"`

For `l1_kl`, define:

```text
loss = L1(target, pred) + lambda_kl * KL(target || pred)
```

Recommended first default for new experiments:

- `cell_prop_loss_type: "l1_kl"`
- `cell_prop_loss_kl_weight: 0.5`

Recommended backward-compatible default:

- if these fields are absent, preserve current MSE behavior

### Numerical stability

For `l1_kl`:

- clamp `pred` and `target` to at least `eps`
- renormalize both to sum to 1 across cell types
- compute per-sample L1 and KL

This is especially important for low-abundance cell types near zero.

### Future extension

Do not implement rare-type weighting in the first step unless the unweighted
`l1_kl` result is still unsatisfactory.

The design should leave room for a later `weighted_l1_kl` option, but that is
out of scope for the first implementation.

## Auxiliary weight scheduling

Add a reusable scheduling workflow for selected scalar config values that are
important during staged training.

### Primary targets

The first implementation should support scheduling for:

- `cell_type_sct_gep_weight`
- `hierarchical_code_weight`
- `cell_type_existence_weight`
- `cell_type_existence_shift_scale`

### Schedule definition

Each scheduled value should support:

- `start_epoch`
- `end_epoch`
- `start_value`
- `end_value`
- `type`

For the first implementation, only `linear` scheduling is required.

### Semantics

For epoch `t`:

```text
progress = clamp((t - start_epoch) / (end_epoch - start_epoch), 0, 1)
value(t) = start_value + progress * (end_value - start_value)
```

### Recommended config shape

```yaml
training:
  aux_loss_schedules:
    cell_type_sct_gep_weight:
      type: linear
      start_epoch: 20
      end_epoch: 60
      start_value: 0.0
      end_value: 1.0
    hierarchical_code_weight:
      type: linear
      start_epoch: 40
      end_epoch: 80
      start_value: 0.0
      end_value: 0.2
    cell_type_existence_weight:
      type: linear
      start_epoch: 60
      end_epoch: 100
      start_value: 0.0
      end_value: 0.2
    cell_type_existence_shift_scale:
      type: linear
      start_epoch: 80
      end_epoch: 120
      start_value: 0.0
      end_value: 0.5
```

If a schedule is absent, preserve the configured constant value.

### Where schedules apply

Because these targets live in both `training` and `model.loss_coefficient`
space, the implementation should centralize schedule application in the
training loop rather than scatter it across modules.

A small scheduler utility should:

1. read configured schedules
2. compute current values at epoch boundaries
3. update the corresponding live config/model values before each epoch

## Config additions

## Model config

Add optional fields:

- `cell_prop_fusion_strategy`
  - allowed:
    - `legacy_output_average`
    - `shared_feature_mean`
    - `shared_feature_gated`

- `cell_prop_fusion_dim`
  - integer fusion-space size

- `cell_prop_head_hidden_dims`
  - list of hidden dims for the shared head

- `cell_prop_head_dropout_rate`
  - dropout used in the shared head

- `cell_prop_loss_type`
  - `mse` or `l1_kl`

- `cell_prop_loss_kl_weight`
  - scalar KL multiplier for `l1_kl`

Recommended defaults for backward compatibility:

- `cell_prop_fusion_strategy: legacy_output_average`
- `cell_prop_loss_type: mse`
- `cell_prop_loss_kl_weight: 0.5`

## Training config

Add optional field:

- `aux_loss_schedules`

This should be fully optional and empty by default.

## Backward compatibility

Existing configs should continue to work unchanged.

If none of the new config fields are provided:

- current posterior fusion behavior remains unchanged
- current cell-proportion averaging remains unchanged
- current MSE loss remains unchanged
- all auxiliary weights remain constant

This is important because the user has a strong preference for backward
compatibility in configuration.

## Affected code areas

### 1. Encoder output contract

Files:

- `vaedecon/models/nn/mlp.py`
- `vaedecon/models/nn/res_mlp.py`
- `vaedecon/models/nn/transformer.py`
- `vaedecon/models/nn/fused_mlp_gnn.py` if still considered active
- any other encoder used in `VAE`

Needed changes:

- expose `cell_prop_feature`
- keep existing outputs unchanged

### 2. Base helper utilities

Files:

- `vaedecon/models/base/base_utils.py`
- possibly a new helper module under `vaedecon/models/base/`

Needed changes:

- add reusable `MLPBlock`
- add shared cell-proportion head helper module or shared component
- keep activation semantics centralized

### 3. VAE forward path

File:

- `vaedecon/models/vae/vae_model.py`

Needed changes:

- add cell-proportion feature fusion utilities
- keep latent posterior fusion unchanged
- switch cell-proportion path based on `cell_prop_fusion_strategy`
- use the shared head when configured

### 4. Loss computation

File:

- `vaedecon/models/vae/vae_model.py`

Needed changes:

- support `cell_prop_loss_type`
- implement stable `l1_kl`
- keep `mse` as a backward-compatible path

### 5. Training-time scheduling

Files:

- `vaedecon/trainers/base_trainer.py`
- possibly a new scheduling helper file

Needed changes:

- add a reusable scalar schedule updater
- apply updates at epoch boundaries

### 6. Config schema and validation

Files:

- `vaedecon/configs/default_config.py`
- possibly related config exports/tests

Needed changes:

- define new optional fields
- validate allowed enum values and schedule structures

## Recommended implementation order

1. add config schema for new cell-proportion fusion/head/loss fields
2. expose `cell_prop_feature` from encoders
3. implement reusable `MLPBlock`, projector, gated fusion, and shared head
4. wire the new cell-proportion fusion path into `VAE.forward()`
5. implement `l1_kl` cell-proportion loss
6. add reusable auxiliary-weight scheduling
7. add tests

This order keeps the architecture changes isolated and makes it easier to
debug each layer of the design.

## Testing plan

### Unit tests

1. verify that encoder outputs include `cell_prop_feature`
2. verify that `shared_feature_mean` and `shared_feature_gated` produce
   correctly shaped fused features
3. verify that single-encoder runs reduce correctly under the new fusion path
4. verify that `build_cell_prop_from_head_output(...)` still works unchanged
   with the new shared head output
5. verify that `l1_kl` returns finite values when targets contain very small
   proportions
6. verify that absent schedules preserve constant values
7. verify that scheduled values ramp correctly across epochs

### Integration tests

1. multi-encoder config with `cell_prop_fusion_strategy=legacy_output_average`
   reproduces current behavior shape-wise
2. multi-encoder config with `cell_prop_fusion_strategy=shared_feature_mean`
   trains and returns predicted cell proportions
3. multi-encoder config with `cell_prop_fusion_strategy=shared_feature_gated`
   trains and returns predicted cell proportions
4. latent posterior fusion still uses `poe_posterior` or `avg_posterior` as
   configured
5. `cell_prop_loss_type=l1_kl` runs end-to-end without NaN/Inf failures

### Experiment-level checks

Recommended first experiment ladder:

1. `EncoderMLP` only, `softmax`, `mse`, no schedules
2. `EncoderMLP` only, `softmax`, `l1_kl`, no schedules
3. same model plus deeper shared head
4. multi-encoder with `shared_feature_mean`
5. multi-encoder with `shared_feature_gated`
6. then ramp in auxiliary schedules

## Acceptance criteria

This feature is complete when:

1. existing configs without new fields still run unchanged
2. the VAE supports a new shared-feature cell-proportion path for one or
   multiple encoders
3. the whole-VAE fusion story is explicit:
   - latent branch uses posterior-aware fusion
   - cell-proportion branch can use feature-level fusion
4. the new shared head is configurable and reusable
5. `l1_kl` cell-proportion loss is available and numerically stable
6. auxiliary scalar schedules can ramp configured values during training
7. tests cover the new fusion path, loss path, and schedule path

## Practical recommendation

For the first actual implementation and experiment, I recommend enabling only:

- `cell_prop_fusion_strategy=shared_feature_mean` or `shared_feature_gated`
- `cell_prop_loss_type=l1_kl`

while keeping:

- latent `fusion_strategy=poe_posterior`
- `cell_type_existence_shift_scale=0.0`
- auxiliary scheduled weights initially at `0`

This keeps the first behavioral change focused on the branch most likely to
improve cell-proportion accuracy.
