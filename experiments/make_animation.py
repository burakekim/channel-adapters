#!/usr/bin/env python
"""Animation for the "How timm does it" section.

Act 1: a natural RGB photo goes through one first-layer filter of an
ImageNet-pretrained ResNet-50; each colour channel has its own part of the filter.
Act 2: a 13-band Sentinel-2 patch goes through the same filter after timm's
adapt_input_conv has tiled it to 13 channels. Every image, filter and output in
the animation is computed here from real data and real weights:

* natural image: "Chelsea the cat" from scikit-image (CC0, Stefan van der Walt)
* satellite image: EuroSAT test patch (MIT), all 13 bands, torchgeo band order
* filter: output channel 20 of conv1 in timm's resnet50.a1_in1k
* tiling: timm.models._manipulate.adapt_input_conv, the function the post quotes

Writes posts/channel-adapters/tiling.gif and tiling_dark.gif.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "posts" / "channel-adapters"
DATA = Path(os.environ.get("CHADAPT_DATA", Path.home() / ".cache" / "channel-adapters")) / "eurosat"

FILTER = 20          # conv1 output channel shown
PATCH = 4270         # EuroSAT test index (class River)
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
MEANING = ["coastal", "blue", "green", "red", "red edge", "red edge", "red edge", "NIR", "water vap.", "cirrus", "SWIR", "SWIR", "NIR"]
MISMATCH = [(1, 2), (10, 11)]  # (blue, green), (SWIR, SWIR): first and last column of each highlight box
KCOL = {"R": "#e34948", "G": "#1baf7a", "B": "#2a78d6"}
W, H = 14, 8  # figure inches; fig-coordinate squares need height = width * W / H


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def compute():
    import timm
    from skimage.data import chelsea
    from timm.models._manipulate import adapt_input_conv

    model = timm.create_model("resnet50.a1_in1k", pretrained=True).eval()
    w3 = model.conv1.weight.detach().float()                 # (64, 3, 7, 7)
    w13 = adapt_input_conv(13, w3.clone())                   # timm's own tiling
    cfg = model.pretrained_cfg

    cat = chelsea()                                          # (300, 451, 3) uint8
    h, wd, _ = cat.shape
    left = (wd - h) // 2
    cat = cat[:, left:left + h]
    t = torch.from_numpy(cat).permute(2, 0, 1).float()[None] / 255
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    cat_img = t[0].permute(1, 2, 0).numpy()
    tn = (t - torch.tensor(cfg["mean"])[None, :, None, None]) / torch.tensor(cfg["std"])[None, :, None, None]
    cat_out = F.conv2d(tn, w3, stride=2, padding=3)[0, FILTER].numpy()

    x_all = np.load(DATA / "train_x.npy", mmap_mode="r")
    mean = np.array([x_all[:, b].mean() for b in range(13)], dtype=np.float32)
    std = np.array([x_all[:, b].std() for b in range(13)], dtype=np.float32)
    sat = np.load(DATA / "test_x.npy", mmap_mode="r")[PATCH].astype(np.float32)    # (13, 64, 64)
    s = torch.from_numpy((sat - mean[:, None, None]) / std[:, None, None])[None]
    s = F.interpolate(s, size=(224, 224), mode="bilinear", align_corners=False)
    sat_out = F.conv2d(s, w13, stride=2, padding=3)[0, FILTER].numpy()
    rgb_out = F.conv2d(s[:, [3, 2, 1]], w3, stride=2, padding=3)[0, FILTER].numpy()

    crop = lambda a: a[3:-3, 3:-3]  # drop the zero-padding border of the convolution output
    cat_out, sat_out, rgb_out = crop(cat_out), crop(sat_out), crop(rgb_out)

    def stretch(a):
        lo, hi = np.percentile(a, [2, 98])
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    sat_bands = [stretch(sat[b]) for b in range(13)]
    sat_rgb = np.stack([stretch(sat[3]), stretch(sat[2]), stretch(sat[1])], -1)
    return dict(cat=cat_img, cat_out=cat_out, filt=w3[FILTER].numpy(), tiled=w13[FILTER].numpy(),
                sat_bands=sat_bands, sat_rgb=sat_rgb, sat_out=sat_out, rgb_out=rgb_out)


# ----------------------------------------------------------------------------
# drawing helpers (figure coordinates)
# ----------------------------------------------------------------------------
class Canvas:
    def __init__(self, dark: bool):
        self.dark = dark
        self.bg = "#1a1a19" if dark else "#ffffff"
        self.ink = "#ffffff" if dark else "#0b0b0b"
        self.muted = "#c3c2b7" if dark else "#52514e"
        self.hl = "#eda100" if dark else "#c98500"

    def new(self):
        self.fig = plt.figure(figsize=(W, H), dpi=90, facecolor=self.bg)
        return self.fig

    def img(self, x, y, w, data, cmap=None, vmin=None, vmax=None, border=None, lw=3):
        ax = self.fig.add_axes([x, y, w, w * W / H])
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(border or self.muted); sp.set_linewidth(lw if border else 0.8)
        return ax

    def text(self, x, y, s, size=13, color=None, ha="center", va="center", weight="normal"):
        self.fig.text(x, y, s, fontsize=size, color=color or self.ink, ha=ha, va=va, fontweight=weight)

    def arrow(self, a, b, color=None, lw=1.6):
        self.fig.patches.append(FancyArrowPatch(a, b, transform=self.fig.transFigure, arrowstyle="-|>",
                                                mutation_scale=14, lw=lw, color=color or self.muted))

    def box(self, x, y, w, h, color, ls="--"):
        self.fig.patches.append(Rectangle((x, y), w, h, transform=self.fig.transFigure, fill=False,
                                          ec=color, lw=2.2, ls=ls))

    def grab(self):
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", facecolor=self.bg)
        plt.close(self.fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def sym(a):
    v = float(np.abs(a).max())
    return dict(cmap="RdBu_r", vmin=-v, vmax=v)


# ----------------------------------------------------------------------------
# act 1: natural image
# ----------------------------------------------------------------------------
def act1(cv: Canvas, d, step: int):
    fig = cv.new()
    cv.text(0.5, 0.94, "What the pretrained first layer was built for: a photo with three channels", 16, weight="bold")
    cv.text(0.5, 0.885, "One filter of ResNet-50's first layer. It has a separate 7 × 7 part for red, green and blue.", 12, color=cv.muted)
    cv.img(0.04, 0.27, 0.2, d["cat"])
    cv.text(0.14, 0.22, "natural image (RGB)", 12, color=cv.muted)
    names = ["red", "green", "blue"]
    ys = [0.64, 0.415, 0.19]
    fs = sym(d["filt"])
    if step >= 1:
        cmaps = ["Reds_r", "Greens_r", "Blues_r"]
        for c in range(3):
            cv.arrow((0.245, 0.445), (0.315, ys[c] + 0.0875))
            cv.img(0.32, ys[c], 0.1, d["cat"][..., c], cmap="gray", vmin=0, vmax=1, border=KCOL["RGB"[c]])
            cv.text(0.37, ys[c] + 0.19, f"{names[c]} channel", 11, color=cv.muted)
    if step >= 2:
        for c in range(3):
            cv.text(0.445, ys[c] + 0.0875, "⊛", 20)
            cv.img(0.47, ys[c], 0.1, d["filt"][c], border=KCOL["RGB"[c]], **fs)
            cv.text(0.52, ys[c] + 0.19, f"{names[c]} part of filter", 11, color=KCOL["RGB"[c]])
    if step >= 3:
        for c in range(3):
            cv.arrow((0.575, ys[c] + 0.0875), (0.635, 0.445))
        cv.text(0.655, 0.445, "+", 30)
        cv.arrow((0.675, 0.445), (0.715, 0.445))
        cv.img(0.72, 0.27, 0.2, d["cat_out"], **sym(d["cat_out"]))
        cv.text(0.82, 0.22, "output of this filter", 12, color=cv.muted)
        cv.text(0.5, 0.06, "Each colour channel is read by the part of the filter that was trained for it.", 13)
    return cv.grab()


# ----------------------------------------------------------------------------
# act 2: Sentinel-2
# ----------------------------------------------------------------------------
X0, TW, GAP = 0.155, 0.058, 0.0065
TH = TW * W / H
BAND_Y, FILT_Y = 0.6, 0.39


def col_x(i):
    return X0 + i * (TW + GAP)


def act2(cv: Canvas, d, n_filters: int, highlight=False, final=False):
    fig = cv.new()
    cv.text(0.5, 0.94, "With in_chans=13, timm gives Sentinel-2 the same three filter parts", 16, weight="bold")
    cv.text(0.5, 0.885, "adapt_input_conv repeats red, green, blue, red, green, blue, ... until every band has one, then multiplies by 3/13.",
            12, color=cv.muted)
    cv.img(0.015, 0.6, 0.11, d["sat_rgb"])
    cv.text(0.07, 0.565, "Sentinel-2 patch\n(shown as RGB)", 10.5, color=cv.muted)
    fs = sym(d["filt"])
    for i in range(13):
        x = col_x(i)
        cv.img(x, BAND_Y, TW, d["sat_bands"][i], cmap="gray", vmin=0, vmax=1)
        cv.text(x + TW / 2, BAND_Y + TH + 0.045, BANDS[i], 11, weight="bold")
        cv.text(x + TW / 2, BAND_Y + TH + 0.018, MEANING[i], 9.5, color=cv.muted)
        if i < n_filters:
            k = "RGB"[i % 3]
            cv.text(x + TW / 2, FILT_Y + TH + 0.055, "⊛", 15)
            cv.img(x, FILT_Y, TW, d["filt"]["RGB".index(k)], border=KCOL[k], **fs)
            cv.text(x + TW / 2, FILT_Y - 0.03, {"R": "red", "G": "green", "B": "blue"}[k], 10.5, color=KCOL[k], weight="bold")
    cv.text(0.135, FILT_Y + TH / 2, "filter part\nit gets", 10.5, color=cv.muted, ha="right")
    if highlight or final:
        for a, b in MISMATCH:
            cv.box(col_x(a) - 0.004, FILT_Y - 0.05, col_x(b) + TW - col_x(a) + 0.008, BAND_Y + TH + 0.075 - FILT_Y + 0.05, cv.hl)
    if highlight and not final:
        cv.text(0.5, 0.25, "The blue band is read by the green part, the green band by the blue part,", 13.5)
        cv.text(0.5, 0.205, "and both shortwave infrared bands by parts that were trained on visible light.", 13.5)
        cv.text(0.5, 0.13, "Which part a band gets depends only on its position in the array.", 12.5, color=cv.muted)
    if final:
        cv.text(0.5, 0.31, "× 3/13 each, then add all 13 responses", 13)
        cv.arrow((0.5, 0.29), (0.6, 0.245))
        cv.img(0.6, 0.02, 0.12, d["sat_out"], **sym(d["sat_out"]))
        cv.text(0.735, 0.13, "output with the\ntiled filter,\nall 13 bands", 11, ha="left")
        cv.img(0.3, 0.02, 0.12, d["rgb_out"], **sym(d["rgb_out"]))
        cv.text(0.285, 0.13, "for comparison:\nthe original filter\non R, G, B only", 11, ha="right", color=cv.muted)
    return cv.grab()


def frames(dark: bool, d):
    cv = Canvas(dark)
    out = []
    out += [(act1(cv, d, 0), 1400), (act1(cv, d, 1), 1200), (act1(cv, d, 2), 1600), (act1(cv, d, 3), 3200)]
    out += [(act2(cv, d, 0), 1800)]
    out += [(act2(cv, d, n), 380 if n < 13 else 1000) for n in range(1, 14)]
    out += [(act2(cv, d, 13, highlight=True), 4200), (act2(cv, d, 13, final=True), 4500)]
    return out


def save(fr, path):
    imgs = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im, _ in fr]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=[t for _, t in fr], loop=0, optimize=True)
    print("wrote", path.name, f"{path.stat().st_size / 1e6:.1f} MB")


def main():
    d = compute()
    save(frames(False, d), OUT / "tiling.gif")
    save(frames(True, d), OUT / "tiling_dark.gif")
    # still frames for review
    for dark, sfx in ((False, ""), (True, "_dark")):
        fr = frames(dark, d)
        for k in (3, 18, 19):
            fr[k][0].save(f"/tmp/claude-29309/anim/frame{k}{sfx}.png")


if __name__ == "__main__":
    main()
