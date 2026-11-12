import math

import pytest
import timm
import torch
from timm.models._manipulate import adapt_input_conv

from channel_adapters.adapters import (
    ADAPTERS, adapt_model, kernel_inherited_by_band, repeat_rescale, first_layer_parameters,
)
from channel_adapters.models import build_model, first_layer_module, first_layer_name
from channel_adapters.data import SBD_BANDS, EUROSAT_BANDS, RGB_NAMES, Split, subsample, band_orders
from channel_adapters.diagnostics import linear_cka

MODELS = ["resnet50", "vit_small_patch16_224"]


@pytest.mark.parametrize("in_chans", [2, 4, 12, 13, 25])
def test_repeat_rescale_matches_timm(in_chans):
    w = torch.randn(64, 3, 7, 7)
    ours = repeat_rescale(w, in_chans)
    theirs = adapt_input_conv(in_chans, w.clone())
    assert torch.allclose(ours, theirs, atol=1e-6)
    # band i inherits kernel i % 3 ...
    for i in range(in_chans):
        assert torch.allclose(ours[:, i] * in_chans / 3, w[:, i % 3])
    # ... and the total input weight is preserved only when N is a multiple of 3
    if in_chans % 3 == 0:
        assert torch.allclose(ours.sum(1), w.sum(1), atol=1e-4)
    else:
        assert not torch.allclose(ours.sum(1), w.sum(1), atol=1e-4)


def test_band_to_kernel_table():
    assert kernel_inherited_by_band(0) == "R"
    # Similar-but-Different order: blue (index 1) gets the G kernel, green gets B, red gets R
    assert [kernel_inherited_by_band(SBD_BANDS.index(b)) for b in ("B02", "B03", "B04")] == ["G", "B", "R"]
    assert [kernel_inherited_by_band(SBD_BANDS.index(b)) for b in ("B11", "B12")] == ["G", "B"]
    assert [kernel_inherited_by_band(EUROSAT_BANDS.index(b)) for b in ("B11", "B12", "B8A")] == ["G", "B", "R"]


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("adapter", sorted(ADAPTERS))
def test_adapters_forward(name, adapter):
    torch.manual_seed(0)
    model = build_model(name, num_classes=10, img_size=64, pretrained=False)
    n = 12
    rgb = [SBD_BANDS.index(b) for b in RGB_NAMES]
    adapt_model(model, adapter, n_bands=n, rgb=rgb)
    x = torch.randn(2, n, 64, 64)
    out = model(x)
    assert out.shape == (2, 10)
    assert len(first_layer_parameters(model)) >= 1


@pytest.mark.parametrize("name", MODELS)
def test_pseudo_rgb_and_split_stem_start_as_rgb_only(name):
    torch.manual_seed(0)
    rgb = [SBD_BANDS.index(b) for b in RGB_NAMES]
    x = torch.randn(2, 12, 64, 64)
    outs = {}
    for adapter in ("rgb-only", "pseudo-rgb", "split-stem"):
        torch.manual_seed(1)  # identical random backbone each time
        model = build_model(name, num_classes=10, img_size=64, pretrained=False).eval()
        adapt_model(model, adapter, n_bands=12, rgb=rgb)
        with torch.no_grad():
            outs[adapter] = model(x)
    assert torch.allclose(outs["rgb-only"], outs["pseudo-rgb"], atol=1e-5)
    assert torch.allclose(outs["rgb-only"], outs["split-stem"], atol=1e-5)


@pytest.mark.parametrize("name", MODELS)
def test_timm_repeat_equals_create_model_in_chans(name):
    """Our timm-repeat adapter reproduces timm.create_model(..., in_chans=12) weights."""
    torch.manual_seed(0)
    model = build_model(name, num_classes=10, img_size=64, pretrained=False)
    w3 = first_layer_module(model).weight.detach().clone()
    adapt_model(model, "timm-repeat", n_bands=12, rgb=[3, 2, 1])
    assert torch.allclose(first_layer_module(model).weight, adapt_input_conv(12, w3), atol=1e-6)


def test_first_layer_names():
    assert first_layer_name(timm.create_model("resnet50", pretrained=False)) == "conv1"
    assert first_layer_name(timm.create_model("vit_small_patch16_224", pretrained=False)) == "patch_embed.proj"


def test_subsample_is_stratified():
    y = torch.arange(10).repeat_interleave(100)
    s = subsample(Split(torch.zeros(1000, 1, 2, 2), y), 0.1, seed=0)
    assert len(s.y) == 100 and (s.y.bincount() == 10).all()


def test_band_orders_are_permutations():
    for k, o in band_orders(SBD_BANDS).items():
        assert sorted(o) == list(range(12)), k
    assert band_orders(SBD_BANDS)["rgb-first"][:3] == [3, 2, 1]


def test_cka_bounds():
    x = torch.randn(200, 16)
    assert math.isclose(linear_cka(x, x), 1.0, abs_tol=1e-5)
    assert math.isclose(linear_cka(x, 3 * x @ torch.eye(16)), 1.0, abs_tol=1e-5)
    assert 0.0 <= linear_cka(x, torch.randn(200, 16)) < 0.5
