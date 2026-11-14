#!/usr/bin/env python
"""Setup illustrations for the post (no training results involved).

fig_sbd_pairs   Similar-but-Different test pairs from the dataset's own eval/test_pairs.parquet:
                close in RGB, different class, different near-infrared. Each pair is shown with
                one fixed reflectance stretch per row, so visual similarity is not an artefact
                of per-image contrast.
fig_options     Schematic of the nine options: what the first layer gives each of the 12
                Similar-but-Different bands at the start of training, and how it is trained.

Writes posts/channel-adapters/{fig_sbd_pairs,fig_options}{,_dark}.png
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "posts" / "channel-adapters"
PAIRS = [0, 2, 3]   # row indices in eval/test_pairs.parquet (cross-label pairs with the largest NIR difference)
SBD_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
KCOL = {"R": "#e34948", "G": "#1baf7a", "B": "#2a78d6"}


def theme(dark):
    return dict(bg="#1a1a19" if dark else "#ffffff", ink="#ffffff" if dark else "#0b0b0b",
                muted="#c3c2b7" if dark else "#52514e", faint="#3a3a38" if dark else "#e6e4df",
                grey="#8a8984" if dark else "#a9a8a2")


# ----------------------------------------------------------------------------
# Similar-but-Different pairs
# ----------------------------------------------------------------------------
def load_pairs():
    import pandas as pd
    import rasterio
    from huggingface_hub import snapshot_download
    from rasterio.io import MemoryFile

    root = Path(snapshot_download("calebrob6/similar-but-different", repo_type="dataset"))
    pairs = pd.read_parquet(root / "eval" / "test_pairs.parquet").iloc[PAIRS]
    want = set(pairs.patch_a) | set(pairs.patch_b)
    tifs = {}
    for shard in (root / "shards").glob("*.parquet"):
        df = pd.read_parquet(shard, columns=["patch_id", "tif"])
        for pid, buf in zip(df.patch_id, df.tif):
            if pid in want:
                with MemoryFile(buf) as m, m.open() as src:
                    tifs[pid] = src.read().astype(np.float32) / 10000.0   # surface reflectance
    return pairs, tifs


def fig_pairs(pairs, tifs, dark):
    t = theme(dark)
    fig, axes = plt.subplots(len(pairs), 4, figsize=(8.0, 2.3 * len(pairs)), facecolor=t["bg"])
    for r, (_, p) in enumerate(pairs.iterrows()):
        a, b = tifs[p.patch_a], tifs[p.patch_b]
        # one stretch per pair and per view, so the two patches are shown on the same scale
        rgb_lo, rgb_hi = np.percentile(np.concatenate([a[[3, 2, 1]].ravel(), b[[3, 2, 1]].ravel()]), [1, 99])
        nir_lo, nir_hi = np.percentile(np.concatenate([a[7].ravel(), b[7].ravel()]), [1, 99])
        for c, (x, label) in enumerate([(a, p.label_a), (b, p.label_b)]):
            rgb = np.clip((x[[3, 2, 1]].transpose(1, 2, 0) - rgb_lo) / (rgb_hi - rgb_lo), 0, 1)
            nir = np.clip((x[7] - nir_lo) / (nir_hi - nir_lo), 0, 1)
            axes[r, 2 * c].imshow(rgb, interpolation="nearest")
            axes[r, 2 * c + 1].imshow(nir, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[r, 2 * c].set_title(f"{label}\nRGB", fontsize=9.5, color=t["ink"])
            axes[r, 2 * c + 1].set_title(f"{label}\nnear-infrared (B08)", fontsize=9.5, color=t["ink"])
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(t["muted"]); sp.set_linewidth(0.6)
    fig.subplots_adjust(wspace=0.1, hspace=0.45)
    return fig


# ----------------------------------------------------------------------------
# options schematic
# ----------------------------------------------------------------------------
OPTIONS = [
    # (name, group, cells, note); cells: one entry per band: R/G/B = that pretrained kernel x 3/12,
    # "o" = original kernel unchanged, "avg" = mean kernel, "rand" = random, "0" = zero, "-" = band not used
    ("timm-repeat", "Change the first layer at the start", "RGBRGBRGBRGB", "kernel i mod 3, × 3/12 (timm default)"),
    ("wavelength-init", "Change the first layer at the start", None, "nearest-wavelength kernel, × 3/12"),
    ("mean-rgb-init", "Change the first layer at the start", ["avg"] * 12, "mean of R, G, B kernels, × 3/12"),
    ("random-init", "Change the first layer at the start", ["rand"] * 12, "new random first layer"),
    ("rgb-only", "Keep the pretrained model unchanged at the start", None, "only R, G, B are used"),
    ("pseudo-rgb", "Keep the pretrained model unchanged at the start", None, "starts by passing R, G, B through"),
    ("split-stem", "Keep the pretrained model unchanged at the start", None, "extra bands: new parallel conv, starts at zero"),
    ("warmup", "Train the first layer differently", "RGBRGBRGBRGB", "epochs 1-3: first layer + head only, then all"),
    ("stem-lr10", "Train the first layer differently", "RGBRGBRGBRGB", "first layer learning rate × 10"),
]


def option_cells(name):
    from channel_adapters.adapters import kernel_by_wavelength
    rgb_pos = {"B04": "o", "B03": "o", "B02": "o"}
    if name == "wavelength-init":
        return [kernel_by_wavelength(b) for b in SBD_BANDS]
    if name == "rgb-only":
        return [("o" if b in rgb_pos else "-") for b in SBD_BANDS]
    if name == "split-stem":
        return [("o" if b in rgb_pos else "0") for b in SBD_BANDS]
    if name == "pseudo-rgb":
        return ["mix"] * 12
    return None


def fig_options(dark):
    t = theme(dark)
    W, H = 13.0, 9.6
    fig = plt.figure(figsize=(W, H), facecolor=t["bg"])
    X0, CW, GAP = 0.255, 0.032, 0.004
    CH = CW * W / H
    row_h = 0.068
    y = 0.925
    fig.text(X0 + 6 * (CW + GAP) - GAP / 2, y + 0.03, "first layer, per band, at the start of training", ha="center",
             fontsize=10.5, color=t["muted"])
    for k, b in enumerate(SBD_BANDS):
        fig.text(X0 + k * (CW + GAP) + CW / 2, y, b, ha="center", va="center", fontsize=8.5, color=t["ink"])
    group = None
    y -= 0.045
    for name, grp, cells, note in OPTIONS:
        if grp != group:
            group = grp
            y -= 0.025
            fig.text(0.02, y, grp, fontsize=11, color=t["muted"], va="center")
            fig.add_artist(plt.Line2D([0.02, 0.98], [y - 0.018, y - 0.018], color=t["faint"], lw=0.8))
            y -= 0.048
        cells = cells if cells is not None else option_cells(name)
        cells = list(cells)
        yc = y - CH / 2
        fig.text(0.035, y, name, fontsize=11, color=t["ink"], va="center", family="monospace")
        for k, c in enumerate(cells):
            x = X0 + k * (CW + GAP)
            if c in "RGB" and len(c) == 1:
                fig.patches.append(Rectangle((x, yc), CW, CH, transform=fig.transFigure, fc=KCOL[c], ec="none", alpha=0.55))
                fig.text(x + CW / 2, yc + CH / 2, c, ha="center", va="center", fontsize=9, color=t["ink"])
            elif c == "o":
                col = KCOL[{"B04": "R", "B03": "G", "B02": "B"}[SBD_BANDS[k]]]
                fig.patches.append(Rectangle((x, yc), CW, CH, transform=fig.transFigure, fc=col, ec=t["ink"], lw=1.4))
                fig.text(x + CW / 2, yc + CH / 2, {"B04": "R", "B03": "G", "B02": "B"}[SBD_BANDS[k]], ha="center",
                         va="center", fontsize=9, color="white")
            elif c == "avg":
                fig.patches.append(Rectangle((x, yc), CW, CH, transform=fig.transFigure, fc=t["grey"], ec="none", alpha=0.7))
                fig.text(x + CW / 2, yc + CH / 2, "avg", ha="center", va="center", fontsize=7, color=t["ink"])
            elif c == "rand":
                fig.patches.append(Rectangle((x, yc), CW, CH, transform=fig.transFigure, fc="none", ec=t["muted"],
                                             hatch="xxxx", lw=0.8))
            elif c == "0":
                fig.patches.append(Rectangle((x, yc), CW, CH, transform=fig.transFigure, fc=t["bg"], ec=t["muted"], lw=0.8))
                fig.text(x + CW / 2, yc + CH / 2, "0", ha="center", va="center", fontsize=9, color=t["muted"])
            elif c == "-":
                fig.patches.append(Rectangle((x, yc), CW, CH, transform=fig.transFigure, fc="none", ec=t["faint"],
                                             lw=0.8, ls="--"))
        if name == "pseudo-rgb":
            x_end = X0 + 12 * (CW + GAP) - GAP
            fig.patches.append(Polygon([[X0, yc + CH], [x_end, yc + CH], [x_end, yc], [X0, yc]], closed=True,
                                       transform=fig.transFigure, fc=t["grey"], ec=t["muted"], lw=0.8, alpha=0.35))
            fig.text((X0 + x_end) / 2, yc + CH / 2, "1×1 conv: 12 bands → 3 channels, then the original R, G, B kernels",
                     ha="center", va="center", fontsize=8.5, color=t["ink"])
        fig.text(X0 + 12 * (CW + GAP) + 0.015, yc + CH / 2, note, fontsize=10, color=t["muted"], va="center")
        y -= row_h
    # legend
    ly = 0.012
    items = [("R", "pretrained kernel, scaled"), ("o", "pretrained kernel, unchanged"), ("avg", "mean kernel"),
             ("rand", "random"), ("0", "zero"), ("-", "band not used")]
    lx = 0.035
    for c, txt in items:
        if c == "R":
            fig.patches.append(Rectangle((lx, ly), 0.016, 0.026, transform=fig.transFigure, fc=KCOL["R"], ec="none", alpha=0.55))
        elif c == "o":
            fig.patches.append(Rectangle((lx, ly), 0.016, 0.026, transform=fig.transFigure, fc=KCOL["R"], ec=t["ink"], lw=1.4))
        elif c == "avg":
            fig.patches.append(Rectangle((lx, ly), 0.016, 0.026, transform=fig.transFigure, fc=t["grey"], ec="none", alpha=0.7))
        elif c == "rand":
            fig.patches.append(Rectangle((lx, ly), 0.016, 0.026, transform=fig.transFigure, fc="none", ec=t["muted"], hatch="xxxx", lw=0.8))
        elif c == "0":
            fig.patches.append(Rectangle((lx, ly), 0.016, 0.026, transform=fig.transFigure, fc=t["bg"], ec=t["muted"], lw=0.8))
        elif c == "-":
            fig.patches.append(Rectangle((lx, ly), 0.016, 0.026, transform=fig.transFigure, fc="none", ec=t["faint"], lw=0.8, ls="--"))
        fig.text(lx + 0.022, ly + 0.013, txt, fontsize=9.5, color=t["muted"], va="center")
        lx += 0.022 + 0.0068 * len(txt) + 0.035
    return fig


def save(fig, name, dark, bg):
    fig.savefig(OUT / f"{name}{'_dark' if dark else ''}.png", dpi=150, facecolor=bg, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    pairs, tifs = load_pairs()
    print(pairs[["label_a", "label_b", "rgb_l2", "nir_mean_diff"]].to_string())
    for dark in (False, True):
        bg = theme(dark)["bg"]
        save(fig_pairs(pairs, tifs, dark), "fig_sbd_pairs", dark, bg)
        save(fig_options(dark), "fig_options", dark, bg)
    print("wrote setup figures")


if __name__ == "__main__":
    main()
