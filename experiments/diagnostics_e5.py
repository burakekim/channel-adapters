#!/usr/bin/env python
"""E5: first-layer output statistics and linear CKA at initialization.

For each dataset, model and adapter: run 2,048 training images through the
adapted first layer (all bands) and through the untouched RGB first layer
(RGB bands only); report linear CKA, the activation std ratio and mean shift.
Writes results/E5/<dataset>_<model>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from channel_adapters.adapters import ADAPTERS, adapt_model  # noqa: E402
from channel_adapters.data import band_orders, load_dataset  # noqa: E402
from channel_adapters.diagnostics import compare_first_layers, first_layer_outputs  # noqa: E402
from channel_adapters.models import build_model, first_layer_module  # noqa: E402
from channel_adapters.train import prepare, resize  # noqa: E402

MODELS = ["resnet50.a1_in1k", "vit_small_patch16_224.augreg_in21k"]
N, IMG = 2048, 128


def main() -> None:
    device = torch.device("cuda")
    out_dir = ROOT / "results" / "E5"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ("sbd", "eurosat"):
        ds = load_dataset(dataset)
        order = band_orders(ds.bands)["native"]
        splits, rgb = prepare(ds, order, device)
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(splits["train"].y), generator=g)[:N].to(device)
        x = resize(splits["train"].x[idx], IMG)
        for model_name in MODELS:
            ref_model = build_model(model_name, num_classes=len(ds.classes), img_size=IMG).to(device).eval()
            ref = first_layer_outputs(first_layer_module(ref_model), x[:, rgb])
            rows = {}
            for adapter in ADAPTERS:
                torch.manual_seed(0)
                m = build_model(model_name, num_classes=len(ds.classes), img_size=IMG)
                adapt_model(m, adapter, n_bands=len(order), rgb=rgb)
                m.to(device).eval()
                act = first_layer_outputs(first_layer_module(m), x)
                rows[adapter] = compare_first_layers(ref, act)
                print(dataset, model_name, adapter, rows[adapter], flush=True)
            (out_dir / f"{dataset}_{model_name.split('.')[0]}.json").write_text(
                json.dumps({"dataset": dataset, "model": model_name, "n_images": N, "img_size": IMG, "adapters": rows}, indent=1))


if __name__ == "__main__":
    main()
