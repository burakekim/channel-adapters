"""Cheap diagnostics at initialization: first-layer output statistics and CKA.

For each adapter we compare the first-layer activations of the adapted model
on all bands against the activations of the untouched RGB model on the RGB
bands of the same images. Linear CKA (Kornblith et al., 2019) on the flattened
activations measures how similar the two representations are; the ratio of
activation standard deviations measures the scale mismatch the rest of the
network sees.
"""

from __future__ import annotations

import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xty = (x.T @ y).norm() ** 2
    xtx = (x.T @ x).norm()
    yty = (y.T @ y).norm()
    return float(xty / (xtx * yty))


@torch.no_grad()
def first_layer_outputs(layer: torch.nn.Module, x: torch.Tensor, pool: int = 4) -> torch.Tensor:
    """Flattened first-layer activations, average-pooled spatially to limit dimensionality."""
    out = layer(x)
    if out.ndim == 4 and pool > 1:
        out = torch.nn.functional.adaptive_avg_pool2d(out, out.shape[-1] // pool)
    return out.flatten(1).float()


@torch.no_grad()
def compare_first_layers(reference: torch.Tensor, adapted: torch.Tensor) -> dict:
    return {
        "cka": linear_cka(reference, adapted),
        "std_ratio": float(adapted.std() / reference.std()),
        "mean_shift": float((adapted.mean() - reference.mean()) / reference.std()),
    }
