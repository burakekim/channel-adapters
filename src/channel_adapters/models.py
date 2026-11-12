"""Build timm models and locate their first (input) layer.

timm records the name of the input convolution of every pretrained model in
``pretrained_cfg['first_conv']`` (``'conv1'`` for ResNets, ``'stem.0'`` for
ConvNeXt, ``'patch_embed.proj'`` for ViT and DeiT). ``load_pretrained`` in
``timm/models/_builder.py`` uses that name to decide which weight tensor to
pass through ``adapt_input_conv`` when ``in_chans != 3``. We use the same name
to swap the layer ourselves.
"""

from __future__ import annotations

import torch.nn as nn
import timm


def first_layer_name(model: nn.Module) -> str:
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    name = cfg.get("first_conv")
    if name is None:
        raise ValueError("model has no pretrained_cfg['first_conv']")
    if isinstance(name, (tuple, list)):
        name = name[0]
    return name


def first_layer_module(model: nn.Module) -> nn.Conv2d:
    return model.get_submodule(first_layer_name(model))


def set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, module)


def build_model(
    name: str,
    num_classes: int,
    img_size: int | None = None,
    pretrained: bool = True,
) -> nn.Module:
    """An RGB (in_chans=3) pretrained backbone with a fresh classifier head.

    ``img_size`` is forwarded only to models that accept it (ViTs); timm
    resamples the position embedding of the pretrained checkpoint.
    """
    kwargs = dict(pretrained=pretrained, num_classes=num_classes, in_chans=3)
    if img_size is not None and name.startswith(("vit_", "deit_")):
        kwargs["img_size"] = img_size
    return timm.create_model(name, **kwargs)


def head_parameters(model: nn.Module):
    head = model.get_classifier()
    return list(head.parameters())
