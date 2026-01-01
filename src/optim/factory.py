from typing import Dict
import torch
from torch import optim




def build_optimizer(model, cfg: Dict):
    """
    cfg example:

    optimizer:
      name: "adamw"
      params:
        lr: 1e-3
        weight_decay: 1e-4
        momentum: 0.9   # (for SGD only)
    """

    

    opt_cfg = cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adam").lower()
    params = opt_cfg.get("params", {})

    VALID_OPTIMIZERS = {"adam", "adamw", "sgd"}
    if name not in VALID_OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Valid options are: {sorted(list(VALID_OPTIMIZERS))}"
        )


    lr = params.get("lr", 1e-3)
    wd = params.get("weight_decay", 0.0)



    if name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

    if name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=params.get("momentum", 0.9),
            nesterov=params.get("nesterov", False),
        )
    

    raise ValueError(f"Unhandled optimizer name: {name}")
