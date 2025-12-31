# src/models/model_factory.py
from typing import Any, Dict
import torch.nn as nn

from .unet import ResNet34UNet, SegmentationWrapper


def build_model(name: str, **kwargs):
    name = name.lower()

    if name == "resnet34_unet":
        model = ResNet34UNet(**kwargs)

    else:
        raise ValueError(f"Unknown model name: {name}")

    if kwargs.get("wrap_output", True):
        model = SegmentationWrapper(model)

    return model

