# src/scheduler/factory.py

from typing import Dict, Optional
import torch
from torch import optim
from .noop import NoOpScheduler

# TODO: Add wrapper that allows step() to activate only every n step() calls
# class SchedulerWrapper:
#     def __init__(self, scheduler, update_every):
#         self.scheduler = scheduler
#         self.update_every = update_every
#         self.step_count = 0

#     def step(self, *args, **kwargs):
#         self.step_count += 1
#         # Check if it's time to take a step. Don't step if count == 0.
#         if self.step_count > 0 and self.step_count % self.update_every == 0:
#             self.scheduler.step(*args, **kwargs)




VALID_SCHEDULERS = {
    "none",
    "steplr",
    "multisteplr",
    "cosine",
    "reduce_on_plateau",
    "onecycle",
}


def build_scheduler(
    optimizer,
    cfg: Dict,
    steps_per_epoch: Optional[int] = None,
):
    """
    Build a learning rate scheduler from config.

    Expected YAML config:

    scheduler:
      name: "cosine"
      params:
        T_max: 50
        eta_min: 1e-6

    If 'scheduler' missing → returns NoOpScheduler
    """
    #TODO : add warmup option. Should work both with and without scheduler.
    sched_cfg = cfg.get("scheduler", None)

    # --- case 1: no scheduler in config ---
    if sched_cfg is None:
        return NoOpScheduler()

    name = sched_cfg.get("name", "none").strip().lower()
    params = sched_cfg.get("params", {})

    if name not in VALID_SCHEDULERS:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            f"Valid options are: {sorted(list(VALID_SCHEDULERS))}"
        )

    # --- case 2: explicit "none" ---
    if name == "none":
        return NoOpScheduler()

    # ===============================
    #   Concrete Schedulers
    # ===============================

    # --- StepLR ---
    if name == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get("step_size", 30),
            gamma=params.get("gamma", 0.1),
        )

    # --- MultiStepLR ---
    if name == "multisteplr":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=params.get("milestones", [50, 100]),
            gamma=params.get("gamma", 0.1),
        )

    # --- CosineAnnealingLR ---
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get("T_max", 50),
            eta_min=params.get("eta_min", 0.0),
        )

    # --- ReduceLROnPlateau ---
    if name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=params.get("mode", "min"),
            factor=params.get("factor", 0.1),
            patience=params.get("patience", 10),
            threshold=params.get("threshold", 1e-4),
            min_lr=params.get("min_lr", 0.0),
        )

    # --- OneCycleLR ---
    if name == "onecycle":
        if steps_per_epoch is None:
            raise ValueError(
                "OneCycleLR requires steps_per_epoch. "
                "Pass len(train_loader) to build_scheduler()."
            )

        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params.get("max_lr", 1e-3),
            epochs=params.get("epochs"),
            steps_per_epoch=steps_per_epoch,
            pct_start=params.get("pct_start", 0.3),
            anneal_strategy=params.get("anneal_strategy", "cos"),
            div_factor=params.get("div_factor", 25.0),
            final_div_factor=params.get("final_div_factor", 1e4),
        )

    raise RuntimeError("Scheduler config matched nothing — this should be unreachable.")
