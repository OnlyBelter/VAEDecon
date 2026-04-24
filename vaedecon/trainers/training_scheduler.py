import torch
import torch.nn as nn
import inspect
from ..configs import TrainingConfig
from torch.optim.lr_scheduler import (
    LRScheduler,
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

class WarmupThenReduceOnPlateau(LRScheduler):
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
        self._last_lr = self.get_last_lr()

        # Get base lr from optimizer
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        self.warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        plateau_kwargs = {
            "mode": "min",
            "patience": patience,
            "factor": factor,
            "min_lr": min_lr,
        }
        if "verbose" in inspect.signature(ReduceLROnPlateau.__init__).parameters:
            plateau_kwargs["verbose"] = verbose
        self.plateau_scheduler = ReduceLROnPlateau(optimizer, **plateau_kwargs)

    def step(self, metrics: float = None, **kwargs):
        if metrics is None and "val_loss" in kwargs:
            metrics = kwargs["val_loss"]
        if self.current_epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            if metrics is None:
                raise ValueError(
                    "val_loss must be provided after warmup phase."
                )
            self.plateau_scheduler.step(metrics)
        self._last_lr = self.get_last_lr()
        self.current_epoch += 1

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def get_lr(self):                           # required by LRScheduler ABC
        return self._last_lr

    def state_dict(self):
        return {
            "current_epoch"      : self.current_epoch,
            "_last_lr": self._last_lr,
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
        raise ValueError(f"Unknown scheduler: {sched_cls}, only WarmupCosine and WarmupReduceOnPlateau are supported")


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
