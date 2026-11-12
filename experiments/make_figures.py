#!/usr/bin/env python
"""Regenerate every figure, table and quoted number of the post from results/.

Never trains. Figures that lack results are skipped with a message. Output:
posts/channel-adapters/fig*.png (light) and fig*_dark.png, tables/*.md and
numbers.json next to the post.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
POST = ROOT / "posts" / "channel-adapters"
TABLES = POST / "tables"
TABLES.mkdir(exist_ok=True)

ADAPTERS = ["timm-repeat", "random-init", "mean-rgb-init", "rgb-only", "pseudo-rgb", "split-stem", "warmup"]
# categorical slots in fixed order (reference palette, light / dark steps)
LIGHT = {"timm-repeat": "#2a78d6", "random-init": "#eb6834", "mean-rgb-init": "#1baf7a", "rgb-only": "#52514e",
         "pseudo-rgb": "#e87ba4", "split-stem": "#008300", "warmup": "#4a3aa7"}
DARK = {"timm-repeat": "#3987e5", "random-init": "#d95926", "mean-rgb-init": "#199e70", "rgb-only": "#c3c2b7",
        "pseudo-rgb": "#d55181", "split-stem": "#008300", "warmup": "#9085e9"}
MARKERS = {"timm-repeat": "o", "random-init": "s", "mean-rgb-init": "D", "rgb-only": "x",
           "pseudo-rgb": "^", "split-stem": "v", "warmup": "P"}
MODELS = {"resnet50.a1_in1k": "ResNet-50", "vit_small_patch16_224.augreg_in21k": "ViT-S/16"}
NUMBERS: dict[str, float] = {}


LR_CHOICE = json.loads((RESULTS / "lr_choice.json").read_text()) if (RESULTS / "lr_choice.json").exists() else {}


def chosen_lr(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the runs at the learning rate selected from E1 for that model x adapter."""
    if df.empty or not LR_CHOICE:
        return df
    keep = [abs(r.lr - LR_CHOICE.get(r.model, {}).get(r.adapter, r.lr)) < 1e-12 for r in df.itertuples()]
    return df[keep]


def load(exp: str) -> pd.DataFrame:
    rows = []
    for f in sorted((RESULTS / exp).glob("*.json")):
        r = json.loads(f.read_text())
        row = dict(r["config"])
        row.update({k: r.get(k) for k in ("test_acc", "val_acc", "best_val_acc", "epochs_to_95pct", "train_time_s", "diverged")})
        row["curve"] = r.get("curve")
        row["probe"] = r.get("probe")
        rows.append(row)
    return pd.DataFrame(rows)


def theme(dark: bool):
    plt.rcParams.update({
        "figure.facecolor": "#1a1a19" if dark else "white", "axes.facecolor": "#1a1a19" if dark else "white",
        "axes.edgecolor": "#c3c2b7" if dark else "#52514e", "axes.labelcolor": "#ffffff" if dark else "#0b0b0b",
        "xtick.color": "#c3c2b7" if dark else "#52514e", "ytick.color": "#c3c2b7" if dark else "#52514e",
        "text.color": "#ffffff" if dark else "#0b0b0b", "grid.color": "#3a3a38" if dark else "#e6e4df",
        "axes.grid": True, "axes.grid.axis": "y", "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "legend.frameon": False, "savefig.dpi": 200, "lines.linewidth": 2, "lines.markersize": 7,
    })
    return DARK if dark else LIGHT


def save(fig, name: str, dark: bool):
    fig.savefig(POST / (f"{name}_dark.png" if dark else f"{name}.png"), bbox_inches="tight")
    plt.close(fig)


def both(fn, name, *args):
    for dark in (False, True):
        colors = theme(dark)
        fig = fn(colors, *args)
        if fig is None:
            print("skip", name)
            return
        save(fig, name, dark)
    print("wrote", name)


# ----------------------------------------------------------------------------
# Figure 1: tiling schematic (from code facts, no results needed)
# ----------------------------------------------------------------------------
SBD = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
KCOL = {"R": "#e34948", "G": "#1baf7a", "B": "#2a78d6"}


def fig_tiling(colors):
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2))
    for ax, (title, k) in zip(axes, [("ResNet stem: conv1, 7x7 kernels", 7), ("ViT patch embedding: patch_embed.proj, 16x16 kernels", 16)]):
        ax.set_xlim(-1.2, 13.2); ax.set_ylim(-0.6, 2.4); ax.axis("off"); ax.set_title(title, loc="left", fontsize=10)
        for j, c in enumerate("RGB"):
            ax.add_patch(Rectangle((j * 0.9, 1.5), 0.8, 0.8, color=KCOL[c])); ax.text(j * 0.9 + 0.4, 1.9, c, ha="center", va="center", color="white", fontsize=9)
        ax.text(3.0, 1.9, f"pretrained ({k}x{k} each)", va="center", fontsize=9)
        ax.annotate("", xy=(6.0, 0.9), xytext=(1.3, 1.45), arrowprops=dict(arrowstyle="->", lw=1.2, color=plt.rcParams["text.color"]))
        ax.text(6.3, 1.15, "repeat(1, 4, 1, 1)[:, :12]  then  x 3/12", fontsize=9, va="center")
        for i, band in enumerate(SBD):
            c = "RGB"[i % 3]
            ax.add_patch(Rectangle((i * 1.05, 0.0), 0.95, 0.8, color=KCOL[c], alpha=0.85))
            ax.text(i * 1.05 + 0.47, 0.4, c, ha="center", va="center", color="white", fontsize=9)
            ax.text(i * 1.05 + 0.47, -0.35, band, ha="center", va="center", fontsize=7.5)
    return fig


# ----------------------------------------------------------------------------
# Figure 3: adapter schematics
# ----------------------------------------------------------------------------
def fig_adapters(colors):
    fig, axes = plt.subplots(1, 7, figsize=(14, 3.2))
    ink = plt.rcParams["text.color"]
    specs = {
        "timm-repeat": [("N bands", None), ("first layer\n(tiled RGB kernels)", "pre-mod"), ("rest of network", "pre")],
        "random-init": [("N bands", None), ("first layer\n(random)", "new"), ("rest of network", "pre")],
        "mean-rgb-init": [("N bands", None), ("first layer\n(mean RGB kernel)", "pre-mod"), ("rest of network", "pre")],
        "rgb-only": [("R, G, B only", None), ("first layer\n(original)", "pre"), ("rest of network", "pre")],
        "pseudo-rgb": [("N bands", None), ("1x1 conv N->3\n(init: select RGB)", "new"), ("first layer\n(original)", "pre"), ("rest of network", "pre")],
        "split-stem": [("RGB | other bands", None), ("original|zero-init", "split"), ("rest of network", "pre")],
        "warmup": [("N bands", None), ("first layer\n(tiled, trained first)", "pre-mod"), ("rest of network\n(frozen, then trained)", "pre")],
    }
    style = {"pre": dict(fc="#d8d6cf", ec=ink), "pre-mod": dict(fc="#d8d6cf", ec=ink, hatch="////", lw=1), "new": dict(fc=colors["timm-repeat"], ec=ink), None: dict(fc="none", ec="none")}
    for ax, name in zip(axes, ADAPTERS):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(name, fontsize=10, color=colors[name], fontweight="bold")
        blocks = specs[name]; h = 0.16; gap = (0.95 - h * len(blocks)) / (len(blocks) + 1)
        for i, (label, kind) in enumerate(blocks):
            y = 0.95 - (i + 1) * (h + gap) + gap
            if kind == "split":
                ax.add_patch(Rectangle((0.05, y), 0.42, h, **style["pre"])); ax.add_patch(Rectangle((0.53, y), 0.42, h, **style["new"]))
                ax.text(0.26, y + h / 2, "first layer\n(original)", ha="center", va="center", fontsize=6.5, color=ink)
                ax.text(0.74, y + h / 2, "copy\n(zero init)", ha="center", va="center", fontsize=6.5, color="white")
                ax.text(0.5, y - 0.012, "+", ha="center", va="top", fontsize=9, color=ink)
            else:
                ax.add_patch(Rectangle((0.05, y), 0.9, h, **style[kind]))
                bbox = dict(fc="#d8d6cf", ec="none", pad=1.5) if kind == "pre-mod" else None
                ax.text(0.5, y + h / 2, label, ha="center", va="center", fontsize=7.5, color="white" if kind == "new" else ink, bbox=bbox)
            if i < len(blocks) - 1:
                ax.annotate("", xy=(0.5, y - gap + 0.01), xytext=(0.5, y - (0.035 if kind == "split" else 0.005)), arrowprops=dict(arrowstyle="->", lw=1, color=ink))
    fig.text(0.5, -0.04, "grey: pretrained, untouched   hatched: pretrained, rewritten   blue: new, trained from a chosen initialisation", ha="center", fontsize=8.5)
    return fig


# ----------------------------------------------------------------------------
# helpers for result figures
# ----------------------------------------------------------------------------
def dot_range(ax, df, group_col, order, colors, label_fmt="{:.1f}"):
    """Dot = mean over seeds, bar = min..max."""
    for i, g in enumerate(order):
        sub = df[df[group_col] == g]["test_acc"] * 100
        if sub.empty:
            continue
        m, lo, hi = sub.mean(), sub.min(), sub.max()
        c = colors[g] if g in colors else plt.rcParams["text.color"]
        ax.plot([lo, hi], [i, i], color=c, lw=2, alpha=0.6)
        ax.plot(m, i, marker=MARKERS.get(g, "o"), color=c, ms=8, ls="none")
        ax.annotate(label_fmt.format(m), (hi, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=plt.rcParams["text.color"])
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.invert_yaxis()
    lo, hi = ax.get_xlim(); ax.set_xlim(lo, hi + 0.12 * (hi - lo))
    ax.grid(axis="x"); ax.grid(False, axis="y")


def fig_main(colors, e2):
    if e2.empty:
        return None
    e2 = e2[(e2.budget == 1.0) & (e2.band_order == "native")]
    if e2.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for ax, (m, name) in zip(axes, MODELS.items()):
        d = e2[e2.model == m]
        dot_range(ax, d, "adapter", ADAPTERS, colors)
        rgb = d[d.adapter == "rgb-only"]["test_acc"].mean() * 100
        if np.isfinite(rgb):
            ax.axvline(rgb, ls="--", lw=1, color=colors["rgb-only"])
        ax.set_title(name, loc="left"); ax.set_xlabel("test accuracy (%)")
        for a in ADAPTERS:
            v = d[d.adapter == a]["test_acc"]
            if len(v):
                NUMBERS[f"E2_{name}_{a}_full_mean"] = round(v.mean() * 100, 1)
                NUMBERS[f"E2_{name}_{a}_full_std"] = round(v.std(ddof=0) * 100, 1)
    axes[1].set_yticklabels([])
    return fig


def fig_budget(colors, e2):
    if e2.empty:
        return None
    e2 = e2[e2.band_order == "native"]
    if e2.empty or e2.budget.nunique() < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (m, name) in zip(axes, MODELS.items()):
        d = e2[e2.model == m]
        for a in ADAPTERS:
            g = d[d.adapter == a].groupby("budget")["test_acc"].agg(["mean", "min", "max"]).sort_index() * 100
            if g.empty:
                continue
            ax.plot(g.index * 100, g["mean"], marker=MARKERS[a], color=colors[a], label=a)
            ax.fill_between(g.index * 100, g["min"], g["max"], color=colors[a], alpha=0.12, lw=0)
            for b, v in g["mean"].items():
                NUMBERS[f"E2_{name}_{a}_b{b:g}_mean"] = round(v, 1)
        ax.set_xscale("log"); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1%", "10%", "100%"])
        ax.set_title(name, loc="left"); ax.set_xlabel("label budget (fraction of training split)")
    axes[0].set_ylabel("test accuracy (%)"); axes[1].legend(fontsize=8, loc="lower right")
    return fig


def fig_lr(colors, e1):
    if e1.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (m, name) in zip(axes, MODELS.items()):
        d = e1[e1.model == m]
        for a in ADAPTERS:
            g = d[d.adapter == a].sort_values("lr")
            if g.empty:
                continue
            ax.plot(g.lr, g.val_acc * 100, marker=MARKERS[a], color=colors[a], label=a)
            NUMBERS[f"E1_{name}_{a}_range"] = round((g.val_acc.max() - g.val_acc.min()) * 100, 1)
        ax.set_xscale("log"); ax.set_title(name, loc="left"); ax.set_xlabel("learning rate")
    axes[0].set_ylabel("validation accuracy (%)"); axes[1].legend(fontsize=8)
    return fig


def fig_convergence(colors, e2):
    if e2.empty:
        return None
    e2 = e2[(e2.budget == 1.0) & (e2.band_order == "native")]
    if e2.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (m, name) in zip(axes, MODELS.items()):
        d = e2[e2.model == m]
        for a in ADAPTERS:
            curves = [pd.DataFrame(c) for c in d[d.adapter == a]["curve"] if c]
            if not curves:
                continue
            acc = np.mean([c.val_acc.values for c in curves], axis=0) * 100
            ax.plot(curves[0].epoch, acc, color=colors[a], label=a, marker=MARKERS[a], ms=4)
            NUMBERS[f"E2_{name}_{a}_epochs95"] = round(float(d[d.adapter == a]["epochs_to_95pct"].mean()), 1)
        ax.set_title(name, loc="left"); ax.set_xlabel("epoch")
    axes[0].set_ylabel("validation accuracy (%)"); axes[1].legend(fontsize=8, loc="lower right")
    return fig


def fig_permutation(colors, e0):
    if e0.empty:
        return None
    orders = ["native", "rgb-first", "reversed", "random-0", "random-1"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True)
    for ax, (m, name) in zip(axes, MODELS.items()):
        d = e0[e0.model == m]
        dot_range(ax, d, "band_order", orders, {o: colors["timm-repeat"] for o in orders})
        ax.set_title(name, loc="left"); ax.set_xlabel("test accuracy (%)")
        g = d.groupby("band_order")["test_acc"].mean() * 100
        if len(g):
            NUMBERS[f"E0_{name}_spread"] = round(g.max() - g.min(), 1)
            for o, v in g.items():
                NUMBERS[f"E0_{name}_{o}"] = round(v, 1)
    axes[1].set_yticklabels([])
    return fig


def fig_cka(colors, e2):
    if e2.empty:
        return None
    files = list((RESULTS / "E5").glob("sbd_*.json"))
    e2 = e2[(e2.budget == 1.0) & (e2.band_order == "native")]
    if not files or e2.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for f in files:
        r = json.loads(f.read_text()); name = MODELS[r["model"]]
        d = e2[e2.model == r["model"]].groupby("adapter")["test_acc"].mean() * 100
        best = d.max()
        for a, diag in r["adapters"].items():
            if a in d.index:
                ax.plot(diag["cka"], best - d[a], marker="o" if name.startswith("ResNet") else "s", color=colors[a], ls="none", ms=8)
                ax.annotate(f"{a} ({name[:3]})", (diag["cka"], best - d[a]), fontsize=7, xytext=(4, 3), textcoords="offset points")
                NUMBERS[f"E5_{name}_{a}_cka"] = round(diag["cka"], 2)
    ax.set_xlabel("linear CKA with the RGB first layer, at initialisation"); ax.set_ylabel("gap to best adapter after fine-tuning (points)")
    return fig


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------
def table_probe(exp: str, path: Path, budgets=(1.0, 0.1, 0.01)):
    df = load(exp)
    if df.empty:
        return
    cols = [f"{n}, {int(b*100)}%" for n in MODELS.values() for b in budgets]
    lines = ["| Adapter | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    cells = {}
    for _, r in df.iterrows():
        for p in r["probe"]:
            cells.setdefault((r.adapter, MODELS[r.model], p["budget"]), []).append(p["test_acc"] * 100)
    for a in ["rgb-only", "timm-repeat", "mean-rgb-init", "random-init", "pseudo-rgb", "split-stem"]:
        vals = []
        for n in MODELS.values():
            for b in budgets:
                v = cells.get((a, n, b))
                vals.append(f"{np.mean(v):.1f}" if v else "n/a")
                if v:
                    NUMBERS[f"{exp}_{n}_{a}_b{b:g}"] = round(float(np.mean(v)), 1)
        lines.append(f"| `{a}` | " + " | ".join(vals) + " |")
    _bold_best(lines, path)


def table_eurosat(path: Path):
    ft, pr = chosen_lr(load("E4")), load("E4p")
    if ft.empty:
        return
    cols = [f"{n}, {k}" for n in MODELS.values() for k in ("fine-tune", "probe")]
    lines = ["| Adapter | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for a in ADAPTERS:
        vals = []
        for m, n in MODELS.items():
            f = ft[(ft.model == m) & (ft.adapter == a)]["test_acc"] * 100
            vals.append(f"{f.mean():.1f} ± {f.std(ddof=0):.1f}" if len(f) else "n/a")
            if len(f):
                NUMBERS[f"E4_{n}_{a}_mean"] = round(f.mean(), 1)
            p = pr[(pr.model == m) & (pr.adapter == a)] if not pr.empty else pr
            if len(p):
                v = [x["test_acc"] for x in p.iloc[0]["probe"] if x["budget"] == 1.0]
                vals.append(f"{v[0]*100:.1f}"); NUMBERS[f"E4p_{n}_{a}"] = round(v[0] * 100, 1)
            else:
                vals.append("n/a")
        lines.append(f"| `{a}` | " + " | ".join(vals) + " |")
    _bold_best(lines, path)


def _bold_best(lines, path):
    rows = [l.split("|")[2:-1] for l in lines[2:]]
    ncol = len(rows[0])
    for j in range(ncol):
        vals = []
        for i, r in enumerate(rows):
            try:
                vals.append((float(r[j].strip().split(" ")[0]), i))
            except ValueError:
                pass
        if vals:
            _, i = max(vals)
            rows[i][j] = f" **{rows[i][j].strip()}** "
    out = lines[:2] + ["|" + lines[k + 2].split("|")[1] + "|" + "|".join(rows[k]) + "|" for k in range(len(rows))]
    path.write_text("\n".join(out) + "\n")
    print("wrote", path.name)


def derived(e0, e2):
    """Gaps the prose quotes: adapter minus timm-repeat and minus rgb-only, per model and budget; permutation spread."""
    if e2.empty:
        return
    for m, name in MODELS.items():
        d = e2[(e2.model == m) & (e2.band_order == "native")]
        for b in sorted(d.budget.unique()):
            g = d[d.budget == b].groupby("adapter")["test_acc"].mean() * 100
            sd = d[d.budget == b].groupby("adapter")["test_acc"].std(ddof=0) * 100
            for a in g.index:
                NUMBERS[f"E2_{name}_{a}_b{b:g}_std"] = round(sd[a], 1)
                NUMBERS[f"gap_{name}_{a}_vs_timm_b{b:g}"] = round(g[a] - g.get("timm-repeat", float("nan")), 1)
                NUMBERS[f"gap_{name}_{a}_vs_rgb_b{b:g}"] = round(g[a] - g.get("rgb-only", float("nan")), 1)
            NUMBERS[f"spread_{name}_nonbottleneck_b{b:g}"] = round(g.drop(["pseudo-rgb", "rgb-only"], errors="ignore").max() - g.drop(["pseudo-rgb", "rgb-only"], errors="ignore").min(), 1)
        NUMBERS[f"E2_{name}_n_seeds"] = int(d[d.budget == 1.0].groupby("adapter").size().max())
    for exp in ("E3", "E4p"):
        probe = load(exp)
        if probe.empty:
            continue
        acc = {}
        for _, r in probe.iterrows():
            for p_ in r["probe"]:
                acc.setdefault((MODELS[r.model], r.adapter, p_["budget"]), []).append(p_["test_acc"] * 100)
        for (name, a, b), v in acc.items():
            NUMBERS[f"{exp}_{name}_{a}_b{b:g}"] = round(float(np.mean(v)), 1)
            ref = acc.get((name, "rgb-only", b))
            if ref:
                NUMBERS[f"gap_{exp}_{name}_{a}_vs_rgb_b{b:g}"] = round(float(np.mean(v) - np.mean(ref)), 1)


def main():
    both(fig_tiling, "fig1_tiling")
    both(fig_adapters, "fig3_adapters")
    e0, e1, e2 = chosen_lr(load("E0")), pd.concat([load("E1"), load("E1b")], ignore_index=True), chosen_lr(load("E2"))
    both(fig_permutation, "fig2_permutation", e0)
    both(fig_main, "fig4_main", e2)
    both(fig_budget, "fig5_budget", e2)
    both(fig_lr, "fig6_lr", e1)
    both(fig_convergence, "fig7_convergence", e2)
    both(fig_cka, "fig8_cka", e2)
    table_probe("E3", TABLES / "table2_probe_sbd.md")
    table_eurosat(TABLES / "table3_eurosat.md")
    derived(e0, e2)
    (POST / "numbers.json").write_text(json.dumps(dict(sorted(NUMBERS.items())), indent=1))
    print(len(NUMBERS), "numbers written")


if __name__ == "__main__":
    main()
