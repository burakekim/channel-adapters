#!/usr/bin/env python
"""Animation for the "How timm does it" section.

Shows timm's adapt_input_conv turning one pretrained RGB filter into a 13-band
filter for Sentinel-2: repeat the three parts, keep the first 13, scale by 3/13,
then convolve each band with its part and add the results.

Everything is computed from real data and weights:
* image: EuroSAT test patch (MIT), Sentinel-2 Level-1C, 13 bands in torchgeo order
* filter: output channel 20 of conv1 in timm's resnet50.a1_in1k
* tiling: timm.models._manipulate.adapt_input_conv itself

Writes posts/channel-adapters/tiling.gif and tiling_dark.gif.
"""

from __future__ import annotations

import io
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
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "posts" / "channel-adapters"
DATA = Path(os.environ.get("CHADAPT_DATA", Path.home() / ".cache" / "channel-adapters")) / "eurosat"

FILTER, PATCH, N = 20, 4270, 13
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
KCOL = {"R": "#e34948", "G": "#1baf7a", "B": "#2a78d6"}
KNAME = {"R": "red", "G": "green", "B": "blue"}
W, H = 14, 6
X0, CW, GAP = 0.155, 0.049, 0.0063
TH = CW * W / H
BAND_Y, PART_Y = 0.66, 0.43
MID = X0 + (12 * (CW + GAP) + CW) / 2


def cx(i):
    return X0 + i * (CW + GAP)


def compute():
    import timm
    from timm.models._manipulate import adapt_input_conv

    model = timm.create_model("resnet50.a1_in1k", pretrained=True).eval()
    w3 = model.conv1.weight.detach().float()
    w13 = adapt_input_conv(N, w3.clone())
    w15 = w3.repeat(1, math.ceil(N / 3), 1, 1)
    assert torch.allclose(w15[:, :N] * 3 / N, w13, atol=1e-6)

    x_all = np.load(DATA / "train_x.npy", mmap_mode="r")
    mean = np.array([x_all[:, b].mean() for b in range(N)], dtype=np.float32)
    std = np.array([x_all[:, b].std() for b in range(N)], dtype=np.float32)
    sat = np.load(DATA / "test_x.npy", mmap_mode="r")[PATCH].astype(np.float32)
    s = torch.from_numpy((sat - mean[:, None, None]) / std[:, None, None])[None]
    s = F.interpolate(s, size=(224, 224), mode="bilinear", align_corners=False)
    out = F.conv2d(s, w13, stride=2, padding=3)[0, FILTER].numpy()[3:-3, 3:-3]

    def stretch(a):
        lo, hi = np.percentile(a, [2, 98])
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    return dict(parts=w15[FILTER].numpy(), out=out, bands=[stretch(sat[b]) for b in range(N)],
                rgb=np.stack([stretch(sat[3]), stretch(sat[2]), stretch(sat[1])], -1))


class Canvas:
    def __init__(self, dark):
        self.bg = "#1a1a19" if dark else "#ffffff"
        self.ink = "#ffffff" if dark else "#0b0b0b"
        self.muted = "#c3c2b7" if dark else "#52514e"

    def new(self):
        self.fig = plt.figure(figsize=(W, H), dpi=100, facecolor=self.bg)

    def img(self, x, y, w, data, cmap=None, vmin=None, vmax=None, border=None, lw=2.5, alpha=1.0):
        ax = self.fig.add_axes([x, y, w, w * W / H])
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", alpha=alpha)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(self.bg)
        for sp in ax.spines.values():
            sp.set_color(border or self.muted); sp.set_linewidth(lw if border else 0.8); sp.set_alpha(alpha)

    def text(self, x, y, s, size=11, color=None, ha="center", va="center", alpha=1.0):
        self.fig.text(x, y, s, fontsize=size, color=color or self.ink, ha=ha, va=va, alpha=alpha)

    def line(self, xs, ys, lw=1.2, alpha=1.0):
        self.fig.add_artist(plt.Line2D(xs, ys, transform=self.fig.transFigure, color=self.muted, lw=lw, alpha=alpha))

    def arrow(self, a, b):
        self.fig.patches.append(FancyArrowPatch(a, b, transform=self.fig.transFigure, arrowstyle="-|>",
                                                mutation_scale=12, lw=1.4, color=self.muted))

    def grab(self):
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", facecolor=self.bg)
        plt.close(self.fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def frame(cv, d, n_copies=1, cut_alpha=1.0, show_cut=False, scale=1.0, conv=False, total=False):
    """One frame. n_copies: copies of the 3 parts shown (1-5); cut_alpha: opacity of parts 14-15
    (0 hides them); scale: factor applied to all parts; conv/total: later stages."""
    cv.new()
    v = float(np.abs(d["parts"]).max())   # one colour scale for every part, before and after scaling
    cv.img(0.012, BAND_Y + TH / 2 - 0.13, 0.115, d["rgb"])
    cv.text(0.0695, BAND_Y + TH / 2 - 0.165, "Sentinel-2 L1C", 11, color=cv.muted)
    cv.text(MID, BAND_Y + TH + 0.105, "spectral channels", 12, color=cv.muted)
    for i in range(N):
        cv.img(cx(i), BAND_Y, CW, d["bands"][i], cmap="gray", vmin=0, vmax=1)
        cv.text(cx(i) + CW / 2, BAND_Y + TH + 0.035, BANDS[i], 10.5)
    cv.text(0.135, PART_Y + TH / 2, "filter\nparts", 11, color=cv.muted, ha="right")
    for i in range(3 * n_copies):
        a = cut_alpha if i >= N else 1.0
        if a <= 0:
            continue
        k = "RGB"[i % 3]
        cv.img(cx(i), PART_Y, CW, d["parts"][i] * scale, cmap="RdBu_r", vmin=-v, vmax=v, border=KCOL[k], alpha=a)
        cv.text(cx(i) + CW / 2, PART_Y - 0.035, KNAME[k], 10, color=KCOL[k], alpha=a)
        if i >= N and show_cut:
            cv.line([cx(i) + 0.005, cx(i) + CW - 0.005], [PART_Y + 0.01, PART_Y + TH - 0.01], lw=2.2)
            cv.line([cx(i) + 0.005, cx(i) + CW - 0.005], [PART_Y + TH - 0.01, PART_Y + 0.01], lw=2.2)
    if cut_alpha > 0 and n_copies > 1:
        for c in range(n_copies):
            x1, x2 = cx(3 * c) + 0.004, cx(3 * c + 2) + CW - 0.004
            yb = PART_Y - 0.085
            cv.line([x1, x1, x2, x2], [yb + 0.018, yb, yb, yb + 0.018])
            cv.text((x1 + x2) / 2, yb - 0.04, f"copy {c + 1}", 10.5, color=cv.muted)
    if show_cut:
        cv.text(cx(13) + CW + GAP / 2, PART_Y + TH + 0.035, "dropped", 10.5, color=cv.muted)
    if scale < 1.0 and not total:
        cv.text(MID, PART_Y - 0.12, f"every part × 3/13" if scale <= 3 / N + 1e-6 else "", 12)
    if conv:
        for i in range(N):
            cv.text(cx(i) + CW / 2, (BAND_Y + PART_Y + TH) / 2, "⊛", 14)
    if total:
        for i in range(N):
            cv.line([cx(i) + CW / 2, MID], [PART_Y - 0.06, 0.29])
        cv.text(MID, 0.255, "+", 24)
        cv.arrow((MID, 0.225), (MID, 0.19))
        o = float(np.abs(d["out"]).max())
        cv.img(MID - 0.035, 0.01, 0.07, d["out"], cmap="RdBu_r", vmin=-o, vmax=o)
        cv.text(MID + 0.045, 0.09, "output of this filter", 11, color=cv.muted, ha="left")
    return cv.grab()


def frames(dark, d):
    cv = Canvas(dark)
    f = []
    f.append((frame(cv, d, 1), 2200))                                        # three pretrained parts
    f.append((frame(cv, d, 5), 2600))                                        # .repeat(1, 5, 1, 1): all 15 parts at once
    f.append((frame(cv, d, 5, show_cut=True, cut_alpha=0.3), 2200))           # keep the first 13
    f.append((frame(cv, d, 5, cut_alpha=0), 700))
    steps = 8
    for k in range(1, steps + 1):                                            # scale by 3/13, gradually
        s = 1 + (3 / N - 1) * k / steps
        f.append((frame(cv, d, 5, cut_alpha=0, scale=s), 90 if k < steps else 2400))
    f.append((frame(cv, d, 5, cut_alpha=0, scale=3 / N, conv=True), 2000))   # each band with its part
    f.append((frame(cv, d, 5, cut_alpha=0, scale=3 / N, conv=True, total=True), 3800))  # add them up
    return f


def save(fr, path):
    imgs = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im, _ in fr]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=[t for _, t in fr], loop=0, optimize=True)
    print("wrote", path.name, f"{path.stat().st_size / 1e6:.1f} MB,", len(fr), "frames,", sum(t for _, t in fr) / 1000, "s")


def main():
    d = compute()
    for dark in (False, True):
        fr = frames(dark, d)
        save(fr, OUT / ("tiling_dark.gif" if dark else "tiling.gif"))
        if not dark:
            for k in (0, 1, 2, 11, 13):
                fr[k][0].save(f"/tmp/claude-29309/anim/g{k}.png")


if __name__ == "__main__":
    main()
