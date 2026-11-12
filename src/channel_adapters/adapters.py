"""Seven ways to feed N bands to a network that was trained on three.

Every adapter takes an RGB-pretrained timm model (in_chans=3) and returns the
same model modified in place so that its forward pass accepts a tensor with
``len(bands)`` channels, in the order given by ``bands``. ``rgb`` gives the
positions of the red, green and blue bands inside that order.

Adapters
--------
timm-repeat    timm's default: tile the R,G,B kernels across the N bands and
               scale by 3/N (``adapt_input_conv`` in timm/models/_manipulate.py).
random-init    a freshly initialized first layer with N input channels.
mean-rgb-init  every band gets the mean of the R, G, B kernels, scaled by 3/N.
rgb-only       keep the model as is and feed it only the R, G, B bands.
pseudo-rgb     a 1x1 projection N -> 3 in front of the untouched first layer,
               initialized to select R, G, B (so it starts as rgb-only).
split-stem     R, G, B through the original first layer, the other N-3 bands
               through a parallel zero-initialized copy; outputs are summed
               (so it also starts as rgb-only).
warmup         timm-repeat initialization; the training loop first trains only
               the first layer and the head, then everything (see train.py).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from .models import first_layer_module, first_layer_name, set_submodule


# ----------------------------------------------------------------------------
# The timm default, re-implemented in four lines for the post. tests/ checks it
# matches timm.models._manipulate.adapt_input_conv exactly.
# ----------------------------------------------------------------------------
def repeat_rescale(weight: torch.Tensor, in_chans: int) -> torch.Tensor:
    """Tile the pretrained (O, 3, kH, kW) kernels across ``in_chans`` and rescale."""
    repeat = math.ceil(in_chans / 3)
    weight = weight.repeat(1, repeat, 1, 1)[:, :in_chans]   # R,G,B,R,G,B,...
    return weight * (3 / in_chans)


def kernel_inherited_by_band(band_index: int) -> str:
    """Which pretrained kernel a band inherits under timm's tiling."""
    return "RGB"[band_index % 3]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _clone_conv(conv: nn.Conv2d, in_channels: int, bias: bool | None = None) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None) if bias is None else bias,
        padding_mode=conv.padding_mode,
    )


def _replace_first_layer(model: nn.Module, new_weight: torch.Tensor, keep_bias: bool = True) -> nn.Conv2d:
    old = first_layer_module(model)
    new = _clone_conv(old, new_weight.shape[1])
    with torch.no_grad():
        new.weight.copy_(new_weight)
        if old.bias is not None and keep_bias:
            new.bias.copy_(old.bias)
    set_submodule(model, first_layer_name(model), new)
    return new


class BandSelect(nn.Module):
    """Pick a subset / permutation of channels, then run the wrapped module."""

    def __init__(self, module: nn.Module, indices: Sequence[int]):
        super().__init__()
        self.module = module
        self.register_buffer("indices", torch.as_tensor(list(indices), dtype=torch.long))

    def forward(self, x):
        return self.module(x[:, self.indices])


class PseudoRGB(nn.Module):
    """1x1 projection from N bands to 3 pseudo-RGB channels, then the original layer."""

    def __init__(self, original: nn.Conv2d, n_bands: int, rgb: Sequence[int]):
        super().__init__()
        self.proj = nn.Conv2d(n_bands, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            for out_c, in_c in enumerate(rgb):
                self.proj.weight[out_c, in_c, 0, 0] = 1.0
        self.original = original

    def forward(self, x):
        return self.original(self.proj(x))


class SplitStem(nn.Module):
    """RGB through the original layer, the remaining bands through a zero-init copy."""

    def __init__(self, original: nn.Conv2d, n_bands: int, rgb: Sequence[int]):
        super().__init__()
        others = [i for i in range(n_bands) if i not in set(rgb)]
        self.register_buffer("rgb_idx", torch.as_tensor(list(rgb), dtype=torch.long))
        self.register_buffer("other_idx", torch.as_tensor(others, dtype=torch.long))
        self.original = original
        self.extra = _clone_conv(original, len(others), bias=False)
        with torch.no_grad():
            self.extra.weight.zero_()

    def forward(self, x):
        out = self.original(x[:, self.rgb_idx])
        if self.other_idx.numel() > 0:
            out = out + self.extra(x[:, self.other_idx])
        return out


# ----------------------------------------------------------------------------
# adapters
# ----------------------------------------------------------------------------
def _timm_repeat(model, n_bands, rgb):
    old = first_layer_module(model)
    _replace_first_layer(model, repeat_rescale(old.weight.detach(), n_bands))


def _random_init(model, n_bands, rgb):
    old = first_layer_module(model)
    new = _clone_conv(old, n_bands)  # nn.Conv2d default (kaiming-uniform) init
    set_submodule(model, first_layer_name(model), new)


def _mean_rgb_init(model, n_bands, rgb):
    old = first_layer_module(model)
    w = old.weight.detach().mean(dim=1, keepdim=True).repeat(1, n_bands, 1, 1) * (3 / n_bands)
    _replace_first_layer(model, w)


def _rgb_only(model, n_bands, rgb):
    old = first_layer_module(model)
    set_submodule(model, first_layer_name(model), BandSelect(old, rgb))


def _pseudo_rgb(model, n_bands, rgb):
    old = first_layer_module(model)
    set_submodule(model, first_layer_name(model), PseudoRGB(old, n_bands, rgb))


def _split_stem(model, n_bands, rgb):
    old = first_layer_module(model)
    set_submodule(model, first_layer_name(model), SplitStem(old, n_bands, rgb))


ADAPTERS = {
    "timm-repeat": _timm_repeat,
    "random-init": _random_init,
    "mean-rgb-init": _mean_rgb_init,
    "rgb-only": _rgb_only,
    "pseudo-rgb": _pseudo_rgb,
    "split-stem": _split_stem,
    "warmup": _timm_repeat,  # same initialization; the schedule differs
}


def adapt_model(model: nn.Module, adapter: str, n_bands: int, rgb: Sequence[int]) -> nn.Module:
    """Modify ``model`` in place so it accepts ``n_bands`` input channels."""
    if adapter not in ADAPTERS:
        raise KeyError(f"unknown adapter {adapter!r}; choose from {sorted(ADAPTERS)}")
    if len(rgb) != 3:
        raise ValueError("rgb must give the positions of the R, G, B bands")
    ADAPTERS[adapter](model, n_bands, list(rgb))
    return model


def first_layer_parameters(model: nn.Module):
    """Parameters of the (possibly wrapped) first layer, for the warmup stage."""
    return list(first_layer_module(model).parameters())
