"""
Pure PyTorch segmentation metrics.

Implements:
- Pixel Accuracy
- Mean Class Accuracy
- Mean IoU (mIoU)
- Per-class IoU
- Dice (macro)

Features:
- Stateless
- GPU-friendly
- Works with ignore_index (e.g., 255)
- No torchmetrics dependency
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
from torch import Tensor


def _to_preds(preds: Tensor) -> Tensor:
    """
    Convert logits to class predictions.
    Expects preds: (B, C, H, W)
    Returns: (B, H, W)
    """
    if preds.dim() != 4:
        raise ValueError(f"Expected preds (B,C,H,W), got {preds.shape}")
    return torch.argmax(preds, dim=1)


def _confusion_matrix(
    preds: Tensor,
    targets: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> Tensor:
    """
    Returns C x C confusion matrix.
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]

    k = (targets >= 0) & (targets < num_classes)
    inds = num_classes * targets[k] + preds[k]

    return torch.bincount(
        inds,
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def compute_segmentation_metrics(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> Dict[str, float]:
    """
    Args:
        logits  (B, C, H, W)
        targets (B, H, W)

    Returns dict of floats + per-class IoU list.
    """

    preds = _to_preds(logits).long()
    targets = targets.long()

    hist = _confusion_matrix(preds, targets, num_classes, ignore_index)

    tp = hist.diag()
    pos = hist.sum(1)         # true pixels per class
    pred_pos = hist.sum(0)    # predicted pixels per class
    total = hist.sum()

    # ----------------------
    # Pixel Accuracy
    # ----------------------
    pixel_acc = tp.sum() / total if total > 0 else torch.tensor(0.0)

    # ----------------------
    # Mean Class Accuracy
    # ----------------------
    class_acc = tp / (pos + 1e-7)
    mean_class_acc = class_acc.mean()

    # ----------------------
    # IoU
    # ----------------------
    iou = tp / (pos + pred_pos - tp + 1e-7)
    miou = iou.mean()

    # ----------------------
    # Dice (macro)
    # ----------------------
    dice = (2 * tp) / (pos + pred_pos + 1e-7)
    dice_macro = dice.mean()

    return {
        "pixel_acc": float(pixel_acc.item()),
        "mean_class_acc": float(mean_class_acc.item()),
        "miou": float(miou.item()),
        "dice": float(dice_macro.item()),
        "per_class_iou": [float(x) for x in iou],
    }
