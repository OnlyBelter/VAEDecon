# Adaptive Multitask Loss Scheduling for Cell-Prop and SCT-GEP Supervision

## Goal

Improve held-out cell-type-specific GEP reconstruction accuracy by replacing the
current static multitask weighting scheme with an adaptive scheduling mechanism
for:

1. `loss_coefficient.cell_prop`
2. `loss_coefficient.cell_type_sct_gep_weight`

The target use case is the current conditioned-decoder setting, where:

- the cell-proportion task already reaches reasonably high CCC
- but held-out SCT-GEP reconstruction accuracy still plateaus
- and collapse-like homogenization remains visible on held-out evaluation even
  when the cell-proportion branch is disabled

This suggests that the next improvement should focus less on removing the
cell-proportion branch entirely and more on controlling **when** it dominates
training and **when** the SCT-GEP objective becomes the primary optimization
target.

## Problem Summary

The current training setup uses fixed scalar weights for multitask objectives.
In particular:

- `loss_coefficient.cell_prop`
- `loss_coefficient.cell_type_sct_gep_weight`

remain constant throughout training unless manually changed between runs.

This has two limitations:

1. the model may continue spending optimization budget on the cell-proportion
   task even after that task is already well learned
2. the SCT-GEP task, which is harder and more directly tied to the current
   scientific goal, may not become dominant at the right time

The recent larger-dataset comparison strengthens this diagnosis:

- enabling active cell-proportion prediction changes held-out mean GEP accuracy
  very little
- disabling active cell-proportion prediction also does not remove the held-out
  collapse gap
- therefore the core bottleneck is not simply "cell-prop on versus off"
- instead, the system likely needs a better **allocation of optimization
  emphasis over time**

## Design Overview

The first implementation will introduce an **adaptive multitask loss schedule**
that can update selected loss weights during training based on:

1. a predefined numeric range
2. an initial direction encoded by that range
3. optional validation-plateau triggers after a minimum epoch

The main initial targets are:

- `cell_prop`
- `cell_type_sct_gep_weight`

The schedule is designed to support the user-requested behavior:

- a weight may increase first and later decrease
- another weight may decrease first and later increase
- both changes happen within predefined ranges
- direction changes may be triggered by validation-loss plateau after a
  user-configured epoch threshold

This is intentionally more flexible than a one-way linear ramp.

## Core Idea

Instead of a static pair such as:

```yaml
model:
  loss_coefficient:
    cell_prop: 500.0
    cell_type_sct_gep_weight: 10.0
```

the training loop will maintain a live state for each scheduled target.

For a target weight `w`, define a range:

$$
[w_{\mathrm{start}}, w_{\mathrm{end}}]
$$

The order of the range encodes the initial direction:

- `[10, 30]` means increase from `10` toward `30`
- `[30, 10]` means decrease from `30` toward `10`

At each schedule update event, the weight is advanced by a configurable step or
interpolation rule, then clipped to remain within the closed interval bounded
by the two endpoints.

When a configured trigger condition is met, the direction may be reversed.

This gives a bounded, reversible schedule rather than a monotonic one.

## Why Adaptive Triggering Instead of Pure Fixed Ramps

The intended benefit of the method is not merely to "increase one weight and
decrease the other" according to a fixed calendar. The intended benefit is:

1. let the cell-proportion task dominate while it is still useful
2. reduce its influence after it becomes relatively saturated
3. let SCT-GEP supervision dominate when that objective is still improving more
   slowly and remains the main bottleneck

Using only fixed epoch ramps risks switching too early on some runs and too
late on others.

Because the current training pipeline already uses plateau-aware learning-rate
reduction, validation-based triggers are a natural next extension for the loss
weighting system.

Reusing the **idea** of validation-based plateau detection is a good choice for
this schedule, because it matches the current training workflow and keeps the
trigger tied to observed optimization behavior rather than to an arbitrary
epoch. However, the first implementation should not directly reuse the
learning-rate scheduler object itself as the loss-schedule controller. The loss
schedule should maintain its own plateau-tracking state, even if it follows the
same metric, patience, and `min_delta` semantics as the existing
`ReduceLROnPlateau`-style logic.

This separation is important because:

1. learning-rate decay and loss-weight reversal are related but not identical
   control actions
2. the user may later want different patience or cooldown settings for the loss
   schedule than for the learning-rate scheduler
3. separate state makes debugging and logging much clearer

## Proposed First Implementation

### Primary trigger type

The first implementation should support:

- `trigger: "val_plateau"`

where plateau is defined using:

- a monitored metric, initially `val_loss`
- a minimum epoch before plateau detection is allowed
- patience in epochs
- minimum required improvement (`min_delta`)
- optional cooldown after each direction change

This keeps the first version aligned with the existing training workflow and
the current user request.

### Update granularity

The first implementation should update weights at **epoch boundaries**, not at
every batch/step.

Reasons:

1. current trainer-side scheduling utilities already operate at epoch level
2. validation metrics are naturally available once per epoch
3. epoch-level scheduling is easier to log, debug, and compare across runs

The config may still describe values as changing "over training steps" in the
general conceptual sense, but the concrete first implementation should use
epoch-based updates.

If finer-grained step-based updates are needed later, that can be a follow-up.

### State-machine view

Each scheduled target maintains:

- current value
- lower bound
- upper bound
- current direction (`+1` or `-1`)
- last change epoch
- whether it has ever reversed

For the paired two-target schedule, the intended behavior is:

1. initialize both targets from their ranges
2. advance them according to their current directions
3. after `min_epoch_before_trigger`, monitor the configured validation metric
4. when plateau is detected:
   - reverse the current direction of each target that allows reversal
   - or hand off dominance according to a paired policy
5. enforce cooldown before another reversal is allowed

This makes the system interpretable and easy to debug.

The first implementation must also log the live values of both scheduled
weights at every epoch, even if no direction change occurs in that epoch.

## Pairing Policy for the First Version

The first implementation should support an explicit paired schedule for:

- `cell_prop`
- `cell_type_sct_gep_weight`

The default policy should be:

- when one target increases, the paired target decreases
- each target still respects its own configured range
- plateau-triggered reversals apply to both targets together

This matches the motivating intuition:

> keep one dominant optimization target at a time

More general independently-triggered schedules can be supported later, but the
first implementation should prioritize the paired case because that is the
scientifically motivated use case right now.

## Config Design

### New training field

Add a new optional field under `training`:

This field should be designed as an **independent optional extension** to the
current config, not as a replacement for the existing fixed loss coefficients.

That means:

1. if `adaptive_aux_loss_schedule` is absent, the current config behaves
   exactly as it does today
2. if `adaptive_aux_loss_schedule.enabled` is `false`, the current fixed values
   in `model.loss_coefficient` remain active throughout training
3. if `adaptive_aux_loss_schedule.enabled` is `true`, the schedule takes
   control of only the explicitly listed targets, and all other loss
   coefficients remain unchanged

So this design is fully backward compatible and does not affect older configs
unless the new field is explicitly added and enabled.

```yaml
training:
  adaptive_aux_loss_schedule:
    enabled: true
    monitor: "val_loss"
    min_epoch_before_trigger: 10
    trigger_patience: 5
    trigger_min_delta: 0.001
    cooldown_epochs: 15
    update_interval_epochs: 1
    pair_targets: true
    targets:
      cell_prop:
        range: [500.0, 100.0]
        step_size: 25.0  # the absolute delta to move each epoch
        # the unit is the same as the loss coefficient
        reverse_on_plateau: true
      cell_type_sct_gep_weight:
        range: [10.0, 30.0]
        step_size: 1.0
        reverse_on_plateau: true
```

Here, `cooldown_epochs` means:

- after a plateau-triggered reversal happens, the schedule must wait this many
  epochs before it is allowed to reverse again

This prevents rapid back-and-forth oscillation when validation loss is noisy or
when the monitored metric hovers near a plateau boundary.

### Meaning of the range

For each target:

- `range[0]` is the starting value
- `range[1]` is the initial destination
- the order encodes the initial direction

Examples:

- `range: [10, 30]`
  - initialize at `10`
  - initial direction is upward
- `range: [30, 10]`
  - initialize at `30`
  - initial direction is downward

The valid numeric interval is always:

$$
[\min(range), \max(range)]
$$

### Why explicit `step_size`

The first implementation should use `step_size` instead of percentage-based
movement because:

1. the affected targets already live on different numeric scales
2. explicit absolute deltas are easier to interpret experimentally
3. they make ablation logs easier to compare

If later needed, support for interpolation-based or percentage-based updates
can be added as an extension.

## Validation Rules

The config should enforce:

1. only supported targets can be scheduled in the first version
   - `cell_prop`
   - `cell_type_sct_gep_weight`
2. `step_size > 0`
3. `range` must contain exactly two finite numbers
4. if `cell_prop` is scheduled upward above `0`, then `predict_cell_prop` must
   be `true`
5. if `cell_type_sct_gep_weight` is scheduled above `0`, required matched-SCT
   supervision inputs must be present
6. `cooldown_epochs >= 0`
7. `min_epoch_before_trigger >= 0`
8. `trigger_patience >= 1`
9. `update_interval_epochs >= 1`

## Trainer Behavior

### Initialization

At training start:

1. parse the adaptive schedule config
2. set the live values in `model.model_config.loss_coefficient` from each
   target's `range[0]`
3. create a trainer-side state object recording:
   - start value
   - end value
   - bounds
   - direction
   - last reversal epoch
   - best monitored metric since last reversal
   - epochs since improvement

### Per-epoch update flow

At the end of each epoch:

1. collect the monitored validation metric
2. check whether the current epoch is at an update boundary
3. update plateau counters
4. if plateau conditions are met and cooldown has expired:
   - reverse directions for the paired targets
   - reset plateau tracker
5. advance each target by one `step_size` in its current direction
6. clip each value into its allowed interval
7. write the new values back into the live model config before the next epoch

### Logging requirements

At minimum, log per epoch:

- current `cell_prop` weight
- current `cell_type_sct_gep_weight`
- current direction for each target
- whether a plateau trigger fired this epoch
- the monitored metric value

These values should appear in:

- Lightning/CSV logs
- saved training metadata
- an explicit schedule trace CSV in `final_model/`

This is critical for debugging and for interpreting experiment outcomes.

In other words, the training logs must make it possible to reconstruct the full
weight trajectory epoch by epoch, not only the epochs where a trigger fired.

## Relationship to Existing `aux_loss_schedules`

The current codebase already supports:

- `training.aux_loss_schedules`
- linear epoch-based schedules for selected targets

The new feature should **not** replace that mechanism immediately.

Instead:

1. keep `aux_loss_schedules` unchanged for simple one-way schedules
2. add a separate adaptive scheduling mechanism for plateau-triggered,
   reversible, range-driven schedules
3. disallow configuring the same target in both systems at once

That avoids ambiguity and preserves backward compatibility.

## Expected Scientific Behavior

If the idea is correct, the adaptive schedule should produce a training process
that looks like:

1. early stage
   - stronger emphasis on learning accurate cell proportions
   - stable decomposition and cell-type presence cues
2. later stage
   - reduced pressure on the already-strong cell-proportion task
   - stronger direct pressure on sample-specific SCT-GEP reconstruction
3. after plateau
   - optional reversal or handoff to prevent one task from dominating forever

The key success criterion is **not** better cell-proportion CCC alone.

The key success criterion is:

> higher held-out cell-type-specific GEP accuracy and a smaller held-out
> collapse gap, without destroying acceptable cell-proportion prediction.

## Recommended First Ablation Plan

The first ablations should stay close to the current realistic conditioned-
decoder setup and only vary the loss schedule.

### Baseline

Use the current fixed-weight multitask run:

- `predict_cell_prop: true`
- `cell_prop: 500.0`
- `cell_type_sct_gep_weight: 10.0`

### Ablation A: One-way handoff

Use:

- `cell_prop.range: [500, 100]`
- `cell_type_sct_gep_weight.range: [10, 30]`
- no reversal after hitting the second endpoint

This tests whether a simple early-to-late handoff already helps.

### Ablation B: Plateau-triggered reversal

Use the same ranges, but enable plateau-triggered reversal after a minimum
epoch.

This tests the full proposed method.

### Ablation C: Stronger SCT-GEP emphasis without reversal

Use a fixed run with:

- `cell_prop: 100.0`
- `cell_type_sct_gep_weight: 20.0` or `30.0`

This is the simplest comparator to determine whether the gain comes from the
adaptive schedule itself or simply from ending at a better final weighting.

## Code Changes

### 1. `vaedecon/configs/default_config.py`

Add:

1. a new config model for adaptive target schedules
2. a new config model for the paired adaptive schedule
3. validation logic for:
   - supported targets
   - range semantics
   - step size
   - incompatibility with `aux_loss_schedules`

### 2. `vaedecon/trainers/base_trainer.py`

Add:

1. a trainer-side state object to track adaptive schedule progress
2. plateau tracking logic based on validation metrics
3. per-epoch weight updates
4. logging of current target weights and trigger events

This is the natural integration point because trainer-side code already applies
epoch-based scalar schedules.

### 3. `vaedecon/workflow/train.py`

Ensure:

1. the full adaptive scheduling config is preserved in saved training artifacts
2. the schedule trace CSV is saved into the final model directory

### 4. `vaedecon/configs/example_config.yaml`

Add a documented example block showing:

- fixed multitask baseline
- adaptive paired schedule example

## Non-goals for the First Version

The first version should explicitly avoid:

1. batch-level / step-level scheduling
2. metric-specific scheduling for many different validation metrics
3. fully independent per-target adaptive triggers
4. automatic search over schedule hyperparameters
5. replacing the existing learning-rate scheduler

Those can be useful later, but the first implementation should stay narrow and
interpretable.

## Testing Plan

Add focused tests for:

1. config parsing and validation of adaptive schedule ranges
2. direction initialization from ordered ranges
3. plateau-triggered reversal behavior
4. bound clipping when a step overshoots the allowed interval
5. incompatibility with overlapping `aux_loss_schedules`
6. trainer logging of updated live weights

At minimum, verify:

- `[10, 30]` initializes at `10` and moves upward
- `[30, 10]` initializes at `30` and moves downward
- a plateau event can reverse both paired targets
- targets never move outside their configured numeric bounds

## Recommended First Implementation Choice

Although the general idea can support many variants, the first implementation
should choose the following defaults:

- epoch-based updates
- monitor `val_loss`
- paired targets
- bounded ranges with direction encoded by endpoint order
- plateau-triggered reversal only after a minimum epoch

This is the smallest design that fully captures the proposed method while
remaining testable and scientifically interpretable.
