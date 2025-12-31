import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, targets, ignore_index=None):
    return F.cross_entropy(
        logits,
        targets,
        ignore_index=ignore_index
    )


def dice_loss(logits, targets, eps=1e-7):
    """
    logits: (B, C, H, W)
    targets: (B, H, W)
    """

    num_classes = logits.shape[1]
    preds = torch.softmax(logits, dim=1)

    # one-hot encode GT
    targets_1h = torch.nn.functional.one_hot(
        targets,
        num_classes=num_classes
    ).permute(0, 3, 1, 2).float()

    # intersection + union
    intersection = (preds * targets_1h).sum(dim=(0, 2, 3))
    union = preds.sum(dim=(0, 2, 3)) + targets_1h.sum(dim=(0, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_ce_dice_loss(logits, targets, alpha=0.5, ignore_index=None):
    ce = cross_entropy_loss(logits, targets, ignore_index)
    dl = dice_loss(logits, targets)
    return alpha * ce + (1 - alpha) * dl


def class_weighted_cross_entropy_loss(logits, targets, class_weights, ignore_index=None):
    """
    class_weights: (C,) tensor
    creates class_weights on the fly so that the total potential loss for each
    class is balanced.
    """
    #TODO 
    pass 
    return F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        ignore_index=ignore_index
    )