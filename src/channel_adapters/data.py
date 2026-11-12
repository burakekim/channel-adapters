"""Datasets as in-memory tensors.

Two datasets, both small enough to hold on the GPU as float tensors:

* Similar-but-Different (HF ``calebrob6/similar-but-different``): 12 Sentinel-2
  L2A bands, 32x32, 10 classes, fixed scene-disjoint split from ``splits.json``.
* EuroSAT (torchgeo, HF mirror ``torchgeo/eurosat``): 13 Sentinel-2 L1C bands,
  64x64, 10 classes, the train/val/test split used by torchgeo.

The first call decodes the raw files once and caches ``uint16`` arrays under
``$CHADAPT_DATA`` (default ``~/.cache/channel-adapters``). Later calls load
the cache.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATA_ROOT = Path(os.environ.get("CHADAPT_DATA", Path.home() / ".cache" / "channel-adapters"))

SBD_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
EUROSAT_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A")
RGB_NAMES = ("B04", "B03", "B02")  # red, green, blue

SBD_CLASSES = (
    "Bare_sparse_veg", "Built-up", "Cropland", "Grassland", "Herbaceous_wetland",
    "Mangroves", "Moss_and_lichen", "Permanent_water", "Shrubland", "Tree_cover",
)


@dataclass
class Split:
    x: torch.Tensor  # (N, C, H, W) uint16 on CPU
    y: torch.Tensor  # (N,) long


@dataclass
class Dataset:
    name: str
    bands: tuple[str, ...]
    classes: tuple[str, ...]
    train: Split
    val: Split
    test: Split

    @property
    def rgb_indices(self) -> list[int]:
        return [self.bands.index(b) for b in RGB_NAMES]

    @property
    def n_bands(self) -> int:
        return len(self.bands)


# ----------------------------------------------------------------------------
# Similar-but-Different
# ----------------------------------------------------------------------------
def _decode_tif(buf: bytes) -> np.ndarray:
    import rasterio
    from rasterio.io import MemoryFile

    with MemoryFile(io.BytesIO(buf).read()) as mem, mem.open() as src:
        return src.read()  # (12, 32, 32) uint16


def _build_sbd(cache: Path) -> None:
    import pandas as pd
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download("calebrob6/similar-but-different", repo_type="dataset"))
    splits = json.loads((local / "splits.json").read_text())
    class_to_idx = splits["class_to_idx"]
    pid_to_split = {}
    for split in ("train", "val", "test"):
        for cls, pids in splits[split].items():
            for pid in pids:
                pid_to_split[pid] = split
    out = {s: ([], []) for s in ("train", "val", "test")}
    for shard in sorted((local / "shards").glob("*.parquet")):
        df = pd.read_parquet(shard, columns=["patch_id", "wc_mode_label", "tif"])
        for pid, label, buf in zip(df.patch_id, df.wc_mode_label, df.tif):
            split = pid_to_split.get(pid)
            if split is None:
                continue
            out[split][0].append(_decode_tif(buf))
            out[split][1].append(class_to_idx[label])
    cache.mkdir(parents=True, exist_ok=True)
    for split, (xs, ys) in out.items():
        np.save(cache / f"{split}_x.npy", np.stack(xs).astype(np.uint16))
        np.save(cache / f"{split}_y.npy", np.asarray(ys, dtype=np.int64))
    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    (cache / "classes.json").write_text(json.dumps(classes))


# ----------------------------------------------------------------------------
# EuroSAT (torchgeo)
# ----------------------------------------------------------------------------
def _build_eurosat(cache: Path) -> None:
    from torchgeo.datasets import EuroSAT

    root = DATA_ROOT / "eurosat_raw"
    cache.mkdir(parents=True, exist_ok=True)
    classes = None
    for split in ("train", "val", "test"):
        ds = EuroSAT(root=str(root), split=split, download=True, checksum=False)
        xs, ys = [], []
        for i in range(len(ds)):
            s = ds[i]
            xs.append(s["image"].numpy().astype(np.uint16))
            ys.append(int(s["label"]))
        np.save(cache / f"{split}_x.npy", np.stack(xs))
        np.save(cache / f"{split}_y.npy", np.asarray(ys, dtype=np.int64))
        classes = list(ds.classes)
    (cache / "classes.json").write_text(json.dumps(classes))


# ----------------------------------------------------------------------------
# public
# ----------------------------------------------------------------------------
_BUILDERS = {"sbd": (_build_sbd, SBD_BANDS), "eurosat": (_build_eurosat, EUROSAT_BANDS)}


def load_dataset(name: str) -> Dataset:
    builder, bands = _BUILDERS[name]
    cache = DATA_ROOT / name
    if not (cache / "test_y.npy").exists():
        builder(cache)
    classes = tuple(json.loads((cache / "classes.json").read_text()))
    splits = {}
    for split in ("train", "val", "test"):
        x = torch.from_numpy(np.load(cache / f"{split}_x.npy").astype(np.int32))  # uint16 -> int32 for torch
        y = torch.from_numpy(np.load(cache / f"{split}_y.npy"))
        splits[split] = Split(x, y)
    return Dataset(name, bands, classes, **splits)


def band_stats(split: Split) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-band mean and std of the raw digital numbers."""
    x = split.x.float()
    return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))


def subsample(split: Split, fraction: float, seed: int) -> Split:
    """A stratified subset of ``fraction`` of the split (at least one per class)."""
    if fraction >= 1.0:
        return split
    g = torch.Generator().manual_seed(seed)
    keep = []
    for c in split.y.unique():
        idx = (split.y == c).nonzero().flatten()
        n = max(1, int(round(len(idx) * fraction)))
        keep.append(idx[torch.randperm(len(idx), generator=g)[:n]])
    keep = torch.cat(keep)
    return Split(split.x[keep], split.y[keep])


# band orders used by the permutation experiment (E0)
def band_orders(bands: tuple[str, ...], seeds: tuple[int, ...] = (0, 1)) -> dict[str, list[int]]:
    n = len(bands)
    rgb = [bands.index(b) for b in RGB_NAMES]
    rest = [i for i in range(n) if i not in rgb]
    orders = {
        "native": list(range(n)),
        "rgb-first": rgb + rest,
        "reversed": list(range(n))[::-1],
    }
    for s in seeds:
        g = torch.Generator().manual_seed(1000 + s)
        orders[f"random-{s}"] = torch.randperm(n, generator=g).tolist()
    return orders
