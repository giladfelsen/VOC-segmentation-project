"""
Initializes the loss functions for training and evaluation from config file.
"""
from typing import Dict

from .segmentation import cross_entropy_loss, dice_loss, combined_ce_dice_loss, class_weighted_cross_entropy_loss

def build_loss_function(cfg: Dict):
    """
    cfg example:

    loss:
      name: "combined_ce_dice"
      params:
        alpha: 0.5
        ignore_index: 255
    """

    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "cross_entropy").lower()
    params = loss_cfg.get("params", {})

    VALID_LOSSES = {
        "cross_entropy",
        "dice",
        "combined_ce_dice",
        "class_weighted_cross_entropy"
    }
    if name not in VALID_LOSSES:
        raise ValueError(
            f"Unknown loss function '{name}'. "
            f"Valid options are: {sorted(list(VALID_LOSSES))}"
        )

    if name == "cross_entropy":
        return cross_entropy_loss

    if name == "dice":
        return dice_loss

    if name == "combined_ce_dice":
        alpha = params.get("alpha", 0.5)
        ignore_index = params.get("ignore_index", 255)
        return lambda logits, targets: combined_ce_dice_loss(
            logits,
            targets,
            alpha=alpha,
            ignore_index=ignore_index
        )

    if name == "class_weighted_cross_entropy":
        class_weights = params.get("class_weights", None)
        ignore_index = params.get("ignore_index", None)
        if class_weights is None:
            raise ValueError("class_weights must be provided for class_weighted_cross_entropy loss.")
        return lambda logits, targets: class_weighted_cross_entropy_loss(
            logits,
            targets,
            class_weights=class_weights,
            ignore_index=ignore_index
            )