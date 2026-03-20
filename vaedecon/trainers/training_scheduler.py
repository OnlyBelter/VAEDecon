import torch
import torch.nn as nn
from ..configs import TrainingConfig
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    SequentialLR,
)


# ── Option A: Warmup + CosineAnnealingLR ────────────────────────────────────
# Best for: fixed num_epochs, smooth decay, no plateau detection needed.

def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float = 1e-7,
):
    """
    Phase 1 (epochs 0 → warmup_epochs):
        lr linearly increases from ~0 to base_lr (set in optimizer).
    Phase 2 (epochs warmup_epochs → total_epochs):
        lr follows cosine annealing down to eta_min.

    Args:
        optimizer:      The optimizer (base_lr = optimizer's lr).
        warmup_epochs:  Number of warmup epochs.
        total_epochs:   Total training epochs.
        eta_min:        Minimum lr at end of cosine decay.

    Returns:
        SequentialLR scheduler.
    """
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,          # start from near-zero
        end_factor=1.0,             # end at base_lr
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],  # switch at this epoch
    )
    return scheduler


# ── Option B: Warmup + ReduceLROnPlateau ────────────────────────────────────
# Best for: unknown convergence speed, need adaptive decay on val_loss plateau.
# NOTE: ReduceLROnPlateau is NOT compatible with SequentialLR directly,
#       so we handle it manually with a wrapper.

class WarmupThenReduceOnPlateau:
    """
    Manual wrapper: LinearLR warmup, then ReduceLROnPlateau.

    Usage:
        scheduler = WarmupThenReduceOnPlateau(optimizer, warmup_epochs=20, ...)
        for epoch in range(num_epochs):
            train(...)
            val_loss = validate(...)
            scheduler.step(val_loss)   # always pass val_loss
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-7,
        verbose: bool = True,
    ):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # Get base lr from optimizer
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        self.warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=patience,
            factor=factor,
            min_lr=min_lr,
            verbose=verbose,
        )

    def step(self, val_loss: float = None):
        if self.current_epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            if val_loss is None:
                raise ValueError(
                    "val_loss must be provided after warmup phase."
                )
            self.plateau_scheduler.step(val_loss)
        self.current_epoch += 1

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "current_epoch"      : self.current_epoch,
            "warmup_scheduler"   : self.warmup_scheduler.state_dict(),
            "plateau_scheduler"  : self.plateau_scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.current_epoch = state["current_epoch"]
        self.warmup_scheduler.load_state_dict(state["warmup_scheduler"])
        self.plateau_scheduler.load_state_dict(state["plateau_scheduler"])


def build_scheduler(optimizer, cfg: TrainingConfig):
    sched_cls    = cfg.scheduler_cls
    sched_params = cfg.scheduler_params
    warmup       = cfg.warmup_epochs
    total        = cfg.num_epochs

    if sched_cls == "WarmupCosine":
        return build_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=warmup,
            total_epochs=total,
            eta_min=sched_params.get("eta_min", 1e-7),
        )
    elif sched_cls == "WarmupReduceOnPlateau":
        return WarmupThenReduceOnPlateau(
            optimizer,
            warmup_epochs=warmup,
            **sched_params,
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_cls}")

# ── Wrapper: makes ReduceLROnPlateau work after a warmup phase ───────────────
class _WarmupReduceOnPlateauScheduler(lr_scheduler.LRScheduler):
    """
    A LRScheduler-compatible wrapper that:
      - Phase 1 (epoch < warmup_epochs): linear warmup via LinearLR
      - Phase 2 (epoch >= warmup_epochs): delegates to ReduceLROnPlateau

    Lightning calls .step(metrics) on plateau schedulers automatically,
    so this wrapper just needs to route the call to the right phase.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int,
                 plateau_sched: lr_scheduler.ReduceLROnPlateau):
        # NOTE: do NOT call super().__init__() here —
        # ReduceLROnPlateau itself doesn't follow the standard LRScheduler
        # interface, and we manage epoch counting manually.
        super().__init__(optimizer)
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.plateau_sched = plateau_sched
        self._epoch        = 0

        self._warmup_sched = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

    # Lightning calls this every epoch (passes `metrics` for plateau schedulers)
    def step(self, metrics=None):
        if self._epoch < self.warmup_epochs:
            self._warmup_sched.step()
        else:
            if metrics is not None:
                self.plateau_sched.step(metrics)
        self._epoch += 1

    # Required by Lightning's LR monitor
    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "_epoch"        : self._epoch,
            "_warmup_sched" : self._warmup_sched.state_dict(),
            "plateau_sched" : self.plateau_sched.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self._epoch = state["_epoch"]
        self._warmup_sched.load_state_dict(state["_warmup_sched"])
        self.plateau_sched.load_state_dict(state["plateau_sched"])


# ── Quick visual test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model     = nn.Linear(10, 1)
    base_lr   = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    # ── Plot Option A ──
    sched_A = build_warmup_cosine_scheduler(
        optimizer, warmup_epochs=50, total_epochs=500, eta_min=1e-7
    )
    lrs_A = []
    for _ in range(500):
        lrs_A.append(optimizer.param_groups[0]["lr"])
        sched_A.step()

    # ── Plot Option B ──
    optimizer.param_groups[0]["lr"] = base_lr   # reset
    sched_B = WarmupThenReduceOnPlateau(
        optimizer, warmup_epochs=50, patience=10, factor=0.5, min_lr=1e-7
    )
    lrs_B = []
    simulated_val_loss = [1.0 - 0.001 * i + (0.05 if i > 200 else 0)
                          for i in range(500)]
    for i in range(500):
        lrs_B.append(sched_B.get_last_lr()[0])
        sched_B.step(val_loss=simulated_val_loss[i])

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, lrs, title in zip(
        axes,
        [lrs_A, lrs_B],
        ["Option A: Warmup + CosineAnnealing", "Option B: Warmup + ReduceLROnPlateau"],
    ):
        ax.plot(lrs, color="steelblue", linewidth=1.5)
        ax.axvline(50, color="orange", linestyle="--", label="warmup end")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
