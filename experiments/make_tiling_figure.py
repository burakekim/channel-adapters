#!/usr/bin/env python
"""Step-through figure for the "How timm does it" section.

Steps 1-3: a natural RGB photo goes through one first-layer filter of an
ImageNet-pretrained ResNet-50; each colour channel has its own part.
Steps 4-9: timm's adapt_input_conv turns that filter into a 13-band filter for
Sentinel-2: repeat the three parts, cut to 13, scale by 3/13, then each band is
convolved with its part and the results are added.

Everything is computed from real data and weights:
* natural image: "Chelsea the cat" from scikit-image (CC0, Stefan van der Walt)
* satellite image: EuroSAT test patch (MIT), 13 bands in torchgeo order
* filter: output channel 20 of conv1 in timm's resnet50.a1_in1k
* tiling: timm.models._manipulate.adapt_input_conv itself

Writes posts/channel-adapters/tiling/{light,dark}_01.png ... _09.png.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "posts" / "channel-adapters" / "tiling"
DATA = Path(os.environ.get("CHADAPT_DATA", Path.home() / ".cache" / "channel-adapters")) / "eurosat"

FILTER = 20
PATCH = 4270
N = 13
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
KCOL = {"R": "#e34948", "G": "#1baf7a", "B": "#2a78d6"}
KNAME = {"R": "red", "G": "green", "B": "blue"}
W, H = 14, 7


def compute():
    import timm
    from skimage.data import chelsea
    from timm.models._manipulate import adapt_input_conv

    model = timm.create_model("resnet50.a1_in1k", pretrained=True).eval()
    w3 = model.conv1.weight.detach().float()
    w13 = adapt_input_conv(N, w3.clone())
    # the repeat step on its own, before cutting and scaling, for display
    repeat = math.ceil(N / 3)
    w15 = w3.repeat(1, repeat, 1, 1)
    assert torch.allclose(w15[:, :N] * 3 / N, w13, atol=1e-6)
    cfg = model.pretrained_cfg

    cat = chelsea()
    h, wd, _ = cat.shape
    cat = cat[:, (wd - h) // 2:(wd - h) // 2 + h]
    t = torch.from_numpy(cat).permute(2, 0, 1).float()[None] / 255
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    tn = (t - torch.tensor(cfg["mean"])[None, :, None, None]) / torch.tensor(cfg["std"])[None, :, None, None]
    cat_out = F.conv2d(tn, w3, stride=2, padding=3)[0, FILTER].numpy()

    x_all = np.load(DATA / "train_x.npy", mmap_mode="r")
    mean = np.array([x_all[:, b].mean() for b in range(N)], dtype=np.float32)
    std = np.array([x_all[:, b].std() for b in range(N)], dtype=np.float32)
    sat = np.load(DATA / "test_x.npy", mmap_mode="r")[PATCH].astype(np.float32)
    s = torch.from_numpy((sat - mean[:, None, None]) / std[:, None, None])[None]
    s = F.interpolate(s, size=(224, 224), mode="bilinear", align_corners=False)
    sat_out = F.conv2d(s, w13, stride=2, padding=3)[0, FILTER].numpy()

    def stretch(a):
        lo, hi = np.percentile(a, [2, 98])
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    crop = lambda a: a[3:-3, 3:-3]  # drop the zero-padding border of the convolution output
    return dict(cat=t[0].permute(1, 2, 0).numpy(), cat_out=crop(cat_out), sat_out=crop(sat_out),
                parts15=w15[FILTER].numpy(), parts13=w13[FILTER].numpy(), filt=w3[FILTER].numpy(),
                bands=[stretch(sat[b]) for b in range(N)],
                sat_rgb=np.stack([stretch(sat[3]), stretch(sat[2]), stretch(sat[1])], -1))


class Canvas:
    def __init__(self, dark):
        self.bg = "#1a1a19" if dark else "#ffffff"
        self.ink = "#ffffff" if dark else "#0b0b0b"
        self.muted = "#c3c2b7" if dark else "#52514e"
        self.faint = "#3a3a38" if dark else "#d8d6cf"

    def new(self):
        self.fig = plt.figure(figsize=(W, H), dpi=100, facecolor=self.bg)

    def img(self, x, y, w, data, cmap=None, vmin=None, vmax=None, border=None, lw=3, alpha=1.0):
        ax = self.fig.add_axes([x, y, w, w * W / H])
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", alpha=alpha)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(self.bg)
        for sp in ax.spines.values():
            sp.set_color(border or self.muted); sp.set_linewidth(lw if border else 0.8); sp.set_alpha(alpha)
        return ax

    def text(self, x, y, s, size=12, color=None, ha="center", va="center", weight="normal", alpha=1.0):
        self.fig.text(x, y, s, fontsize=size, color=color or self.ink, ha=ha, va=va, fontweight=weight, alpha=alpha)

    def arrow(self, a, b, lw=1.5, head=True):
        self.fig.patches.append(FancyArrowPatch(a, b, transform=self.fig.transFigure,
                                                arrowstyle="-|>" if head else "-", mutation_scale=13, lw=lw, color=self.muted))

    def line(self, xs, ys, lw=1.3):
        self.fig.add_artist(plt.Line2D(xs, ys, transform=self.fig.transFigure, color=self.muted, lw=lw))

    def save(self, path):
        self.fig.savefig(path, facecolor=self.bg)
        plt.close(self.fig)


# ----------------------------------------------------------------------------
# steps 1-3: the photo
# ----------------------------------------------------------------------------
def photo(cv, d, step):
    cv.new()
    v = float(np.abs(d["filt"]).max())
    cv.img(0.04, 0.3, 0.2, d["cat"])
    cv.text(0.14, 0.25, "photo (red, green, blue)", 12, color=cv.muted)
    ys = [0.70, 0.39, 0.08]
    for c, k in enumerate("RGB"):
        cv.arrow((0.245, 0.5), (0.315, ys[c] + 0.1))
        cv.img(0.32, ys[c], 0.1, d["cat"][..., c], cmap="gray", vmin=0, vmax=1, border=KCOL[k])
        cv.text(0.37, ys[c] + 0.225, f"{KNAME[k]} channel", 11, color=cv.muted)
        if step >= 2:
            cv.text(0.445, ys[c] + 0.1, "⊛", 20)
            cv.img(0.47, ys[c], 0.1, d["filt"][c], cmap="RdBu_r", vmin=-v, vmax=v, border=KCOL[k])
            cv.text(0.52, ys[c] + 0.225, f"{KNAME[k]} part of the filter", 11, color=KCOL[k])
        if step >= 3:
            cv.arrow((0.575, ys[c] + 0.1), (0.635, 0.5))
    if step >= 3:
        cv.text(0.655, 0.5, "+", 30)
        cv.arrow((0.675, 0.5), (0.715, 0.5))
        o = float(np.abs(d["cat_out"]).max())
        cv.img(0.72, 0.3, 0.2, d["cat_out"], cmap="RdBu_r", vmin=-o, vmax=o)
        cv.text(0.82, 0.25, "output of this filter", 12, color=cv.muted)


# ----------------------------------------------------------------------------
# steps 4-9: from 3 parts to 13
# ----------------------------------------------------------------------------
X0, CW, GAP = 0.05, 0.054, 0.006
TH = CW * W / H
BAND_Y, PART_Y = 0.74, 0.47
MID = X0 + (12 * (CW + GAP) + CW) / 2   # horizontal centre of the 13 columns


def cx(i):
    return X0 + i * (CW + GAP)


def thirteen(cv, d, step):
    cv.new()
    v = float(np.abs(d["filt"]).max())          # one colour scale for every filter part, before and after scaling
    for i in range(N):
        cv.img(cx(i), BAND_Y, CW, d["bands"][i], cmap="gray", vmin=0, vmax=1)
        cv.text(cx(i) + CW / 2, BAND_Y + TH + 0.03, BANDS[i], 11, weight="bold")
    cv.img(cx(13), 0.69, 2 * CW + GAP, d["sat_rgb"])
    cv.text(cx(13) + CW + GAP / 2, 0.655, "same patch\nin RGB", 10, color=cv.muted)

    n_parts = 3 if step == 4 else 15 if step in (5, 6) else 13
    for i in range(n_parts):
        k = "RGB"[i % 3]
        dropped = step == 6 and i >= N
        part = d["parts15"][i] if step in (4, 5, 6) else d["parts13"][i]
        a = 0.25 if dropped else 1.0
        cv.img(cx(i), PART_Y, CW, part, cmap="RdBu_r", vmin=-v, vmax=v, border=KCOL[k], alpha=a)
        cv.text(cx(i) + CW / 2, PART_Y - 0.03, KNAME[k], 10.5, color=KCOL[k], weight="bold", alpha=a)
        if dropped:
            cv.line([cx(i) + 0.006, cx(i) + CW - 0.006], [PART_Y + 0.01, PART_Y + TH - 0.01], lw=2.5)
            cv.line([cx(i) + 0.006, cx(i) + CW - 0.006], [PART_Y + TH - 0.01, PART_Y + 0.01], lw=2.5)
    if step == 4:
        cv.text(cx(3) + 0.01, PART_Y + TH / 2, "the three parts of one pretrained filter", 12, color=cv.muted, ha="left")
    if step in (5, 6):
        for c in range(5):
            x1, x2 = cx(3 * c) + 0.004, cx(3 * c + 2) + CW - 0.004
            yb = PART_Y - 0.07
            cv.line([x1, x1, x2, x2], [yb + 0.015, yb, yb, yb + 0.015])
            cv.text((x1 + x2) / 2, yb - 0.035, f"copy {c + 1}", 11, color=cv.muted)
    if step == 6:
        cv.text(cx(13) + CW + GAP / 2, PART_Y - 0.17, "dropped", 11, color=cv.muted)
    if step == 7:
        cv.text(MID, PART_Y - 0.1, "every part × 3/13", 13)
    if step >= 8:
        for i in range(N):
            cv.text(cx(i) + CW / 2, (BAND_Y + PART_Y + TH) / 2, "⊛", 16)
    if step == 9:
        for i in range(N):
            cv.line([cx(i) + CW / 2, MID], [PART_Y - 0.055, 0.37])
        cv.text(MID, 0.34, "+", 26)
        cv.arrow((MID, 0.315), (MID, 0.255))
        o = float(np.abs(d["sat_out"]).max())
        cv.img(MID - 0.05, 0.045, 0.1, d["sat_out"], cmap="RdBu_r", vmin=-o, vmax=o)
        cv.text(MID + 0.065, 0.145, "output of this filter", 12, color=cv.muted, ha="left")
    # fixed colour scale for the filter parts
    cax = cv.fig.add_axes([0.05, 0.09, 0.16, 0.025])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-v, v))
    cb = cv.fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_edgecolor(cv.muted); cax.tick_params(colors=cv.muted, labelsize=9)
    cv.text(0.13, 0.155, "filter weight (same scale in every step)", 10, color=cv.muted)


def main():
    d = compute()
    OUT.mkdir(parents=True, exist_ok=True)
    for dark in (False, True):
        cv = Canvas(dark)
        name = "dark" if dark else "light"
        for step in range(1, 10):
            (photo if step <= 3 else thirteen)(cv, d, step)
            cv.save(OUT / f"{name}_{step:02d}.png")
    print("wrote", len(list(OUT.glob("*.png"))), "images,", f"{sum(p.stat().st_size for p in OUT.glob('*.png')) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
