#!/usr/bin/env python
"""Every result figure, table and quoted number of the post, from results/ only. Never trains.

Inclusion rules (applied to every fine-tuning result):
* the learning rate must be the one selected for that model and option in results/lr_choice.json;
* runs at label budgets below 100% must carry "lr_warmup_steps" (fraction-based warmups, see decisions.md);
* probes must use 5 labelled subsets.
Comparisons with timm-repeat are paired by seed (same labelled subset) and reported as the mean
difference with a 95% t-interval (n = 5, 4 degrees of freedom).

Outputs in posts/channel-adapters/: fig_*.png (+ _dark), tables/*.md, numbers.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
POST = ROOT / "posts" / "channel-adapters"
TABLES = POST / "tables"
TABLES.mkdir(exist_ok=True)
LR = json.loads((RESULTS / "lr_choice.json").read_text())

MODELS = {"resnet50.a1_in1k": "ResNet-50", "vit_small_patch16_224.augreg_in21k": "ViT-S/16"}
GROUPS = {
    "Change the first layer at the start": ["wavelength-init", "mean-rgb-init", "random-init"],
    "Keep the pretrained model unchanged at the start": ["rgb-only", "pseudo-rgb", "split-stem"],
    "Train the first layer differently": ["warmup", "stem-lr10"],
}
OPTIONS = ["timm-repeat"] + [a for g in GROUPS.values() for a in g]
# frozen, pseudo-rgb and split-stem compute exactly what rgb-only computes, so they are not separate rows
PROBE_OPTIONS = ["timm-repeat", "wavelength-init", "mean-rgb-init", "random-init", "rgb-only"]
BUDGETS = [1.0, 0.1, 0.01]
# categorical slots 1-3 of the reference palette (validated for all pairs), light / dark steps
GCOL = {False: ["#2a78d6", "#eb6834", "#1baf7a"], True: ["#3987e5", "#d95926", "#199e70"]}
NUM: dict[str, float] = {}


def theme(dark):
    plt.rcParams.update({
        "figure.facecolor": "#1a1a19" if dark else "white", "axes.facecolor": "#1a1a19" if dark else "white",
        "axes.edgecolor": "#c3c2b7" if dark else "#52514e", "axes.labelcolor": "#ffffff" if dark else "#0b0b0b",
        "xtick.color": "#c3c2b7" if dark else "#52514e", "ytick.color": "#c3c2b7" if dark else "#52514e",
        "text.color": "#ffffff" if dark else "#0b0b0b", "grid.color": "#3a3a38" if dark else "#e6e4df",
        "axes.grid": True, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
    })


def save(fig, name, dark):
    fig.savefig(POST / f"{name}{'_dark' if dark else ''}.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load(exp, finetune=True):
    rows = []
    for f in sorted((RESULTS / exp).glob("*.json")):
        r = json.loads(f.read_text()); c = r["config"]
        if finetune:
            if abs(c["lr"] - LR[c["model"]][c["adapter"]]) > 1e-12:
                continue
            if c["budget"] < 1 and "lr_warmup_steps" not in r:
                continue
            rows.append(dict(model=c["model"], adapter=c["adapter"], order=c["band_order"], budget=c["budget"],
                             seed=c["seed"], test=100 * r["test_acc"], val=100 * r["val_acc"],
                             e95=r["epochs_to_95pct"], time=r["train_time_s"]))
        else:
            if len(c["probe_seeds"]) != 5:
                continue
            for p in r["probe"]:
                rows.append(dict(model=c["model"], adapter=c["adapter"], budget=p["budget"], seed=p["seed"],
                                 test=100 * p["test_acc"], time=r["train_time_s"]))
    return pd.DataFrame(rows)


def paired(df, adapter, model, budget, ref="timm-repeat"):
    a = df[(df.adapter == adapter) & (df.model == model) & (df.budget == budget)].set_index("seed").test
    b = df[(df.adapter == ref) & (df.model == model) & (df.budget == budget)].set_index("seed").test
    seeds = sorted(set(a.index) & set(b.index))
    d = (a[seeds] - b[seeds]).values
    if len(d) < 2:
        return np.nan, np.nan, np.nan, len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    return m, m - h, m + h, len(d)


def fmt(v, nd=1):
    return f"{v:.{nd}f}"


# ----------------------------------------------------------------------------
# main comparison: absolute table + paired differences figure
# ----------------------------------------------------------------------------
def main_table(df, dataset_key, budgets, path):
    cols = [(m, b) for m in MODELS for b in budgets]
    head = "| Option | " + " | ".join(f"{MODELS[m]}, {int(b * 100)}%" for m, b in cols) + " |"
    lines = [head, "|---|" + "---:|" * len(cols)]
    best = {}
    for m, b in cols:
        g = df[(df.model == m) & (df.budget == b)].groupby("adapter").test.mean()
        best[(m, b)] = set(g[g.round(1) == g.round(1).max()].index) if len(g) else set()
    for a in OPTIONS:
        cells = []
        for m, b in cols:
            v = df[(df.adapter == a) & (df.model == m) & (df.budget == b)].test
            if len(v) == 0:
                cells.append("n/a"); continue
            s = f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"
            cells.append(f"**{s}**" if a in best[(m, b)] else s)
            NUM[f"{dataset_key}_{MODELS[m]}_{a}_b{b:g}_mean"] = round(v.mean(), 1)
            NUM[f"{dataset_key}_{MODELS[m]}_{a}_b{b:g}_std"] = round(v.std(ddof=1), 1)
            NUM[f"{dataset_key}_{MODELS[m]}_{a}_b{b:g}_n"] = int(len(v))
        lines.append(f"| `{a}` | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def diff_figure(df, dataset_key, budgets, name, dark):
    theme(dark)
    cols = GCOL[dark]
    others = [a for a in OPTIONS if a != "timm-repeat"]
    fig, axes = plt.subplots(len(MODELS), len(budgets), figsize=(3.3 * len(budgets) + 1.6, 5.6), sharey=True, squeeze=False)
    for r, m in enumerate(MODELS):
        for c, b in enumerate(budgets):
            ax = axes[r, c]
            ax.axvline(0, color=plt.rcParams["axes.edgecolor"], lw=1)
            for i, a in enumerate(others):
                gi = [k for k, g in enumerate(GROUPS.values()) if a in g][0]
                mu, lo, hi, n = paired(df, a, m, b)
                if np.isnan(mu):
                    continue
                NUM[f"{dataset_key}_diff_{MODELS[m]}_{a}_b{b:g}"] = round(mu, 1)
                NUM[f"{dataset_key}_diff_{MODELS[m]}_{a}_b{b:g}_lo"] = round(lo, 1)
                NUM[f"{dataset_key}_diff_{MODELS[m]}_{a}_b{b:g}_hi"] = round(hi, 1)
                clear = lo > 0 or hi < 0
                ax.plot([lo, hi], [i, i], color=cols[gi], lw=2.2, solid_capstyle="round")
                ax.plot(mu, i, "o", ms=7, color=cols[gi], mfc=cols[gi] if clear else plt.rcParams["axes.facecolor"], mew=2)
            ax.set_ylim(len(others) - 0.5, -0.5)
            ax.set_yticks(range(len(others))); ax.set_yticklabels(others, family="monospace")
            ax.grid(axis="y", visible=False)
            if r == 0:
                ax.set_title(f"{int(b * 100)}% of labels", fontsize=10.5)
            if r == len(MODELS) - 1:
                ax.set_xlabel("difference from timm-repeat\n(accuracy points, 95% CI)")
        axes[r, 0].annotate(MODELS[m], xy=(-0.62, 0.5), xycoords="axes fraction", rotation=90, va="center",
                            ha="center", fontsize=11)
    handles = [plt.Line2D([], [], color=cols[k], marker="o", lw=2.2, label=g) for k, g in enumerate(GROUPS)]
    handles += [plt.Line2D([], [], color=plt.rcParams["axes.edgecolor"], marker="o", ls="none", mfc="none", mew=2,
                           label="interval includes 0")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.12), fontsize=9.5)
    fig.subplots_adjust(wspace=0.12, hspace=0.25)
    save(fig, name, dark)


# ----------------------------------------------------------------------------
# learning-rate sensitivity
# ----------------------------------------------------------------------------
def lr_figure(dark):
    theme(dark)
    rows = []
    for exp in ("E1", "E1b"):
        for f in (RESULTS / exp).glob("*.json"):
            r = json.loads(f.read_text()); c = r["config"]
            rows.append(dict(model=c["model"], adapter=c["adapter"], lr=c["lr"], val=100 * r["val_acc"]))
    d = pd.DataFrame(rows)
    cols = GCOL[dark]
    fig, axes = plt.subplots(len(MODELS), len(GROUPS), figsize=(11.5, 5.8), sharey="row", squeeze=False)
    ref = plt.rcParams["axes.edgecolor"]
    markers = ["o", "s", "^"]
    for r, m in enumerate(MODELS):
        for c, (g, opts) in enumerate(GROUPS.items()):
            ax = axes[r, c]
            t = d[(d.model == m) & (d.adapter == "timm-repeat")].sort_values("lr")
            ax.plot(t.lr, t.val, color=ref, ls="--", lw=1.5, label="timm-repeat")
            for k, a in enumerate(opts):
                s = d[(d.model == m) & (d.adapter == a)].sort_values("lr")
                ax.plot(s.lr, s.val, color=cols[c], marker=markers[k], ms=6, alpha=[1, 0.75, 0.5][k], label=a)
                NUM[f"lr_range_{MODELS[m]}_{a}"] = round(s.val.max() - s.val.min(), 1)
            NUM[f"lr_range_{MODELS[m]}_timm-repeat"] = round(t.val.max() - t.val.min(), 1)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(g, fontsize=9.5)
            if r == len(MODELS) - 1:
                ax.set_xlabel("learning rate")
            ax.legend(fontsize=8, loc="lower center", prop={"family": "monospace", "size": 8})
        axes[r, 0].set_ylabel(f"{MODELS[m]}\nvalidation accuracy (%)")
    fig.subplots_adjust(wspace=0.08, hspace=0.3)
    save(fig, "fig_lr", dark)


# ----------------------------------------------------------------------------
# band order
# ----------------------------------------------------------------------------
def band_order(e0, e2, path):
    orders = ["native", "rgb-first", "reversed", "random-0", "random-1"]
    lines = ["| Band order | ResNet-50 | ViT-S/16 |", "|---|---:|---:|"]
    for o in orders:
        cells = []
        for m in MODELS:
            v = e0[(e0.model == m) & (e0.order == o)].test
            cells.append(f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) > 1 else "n/a")
            NUM[f"E0_{MODELS[m]}_{o}_mean"] = round(v.mean(), 1)
        lines.append(f"| {o} | " + " | ".join(cells) + " |")
    for m in MODELS:
        groups = [e0[(e0.model == m) & (e0.order == o)].test.values for o in orders]
        if min(len(g) for g in groups) < 2:
            continue
        F, p = stats.f_oneway(*groups)
        means = [g.mean() for g in groups]
        NUM[f"E0_{MODELS[m]}_spread"] = round(max(means) - min(means), 1)
        NUM[f"E0_{MODELS[m]}_anova_p"] = round(float(p), 2)
        NUM[f"E0_{MODELS[m]}_anova_F"] = round(float(F), 2)
        NUM[f"E0_{MODELS[m]}_seed_std"] = round(float(np.mean([g.std(ddof=1) for g in groups])), 1)
    cells = []
    for m in MODELS:
        v = e2[(e2.model == m) & (e2.adapter == "wavelength-init") & (e2.budget == 1.0)].test
        cells.append(f"{v.mean():.1f} ± {v.std(ddof=1):.1f}")
    lines.append("| `wavelength-init` (any order) | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# frozen backbone
# ----------------------------------------------------------------------------
def probe_table(df, key, path, budgets):
    cols = [(m, b) for m in MODELS for b in budgets]
    lines = ["| Option | " + " | ".join(f"{MODELS[m]}, {int(b * 100)}%" for m, b in cols) + " |", "|---|" + "---:|" * len(cols)]
    best = {}
    for m, b in cols:
        g = df[(df.model == m) & (df.budget == b)].groupby("adapter").test.mean()
        best[(m, b)] = set(g[g.round(1) == g.round(1).max()].index) if len(g) else set()
    for a in PROBE_OPTIONS:
        cells = []
        for m, b in cols:
            v = df[(df.adapter == a) & (df.model == m) & (df.budget == b)].test
            s = f"{v.mean():.1f}" if b >= 1 else f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"
            cells.append(f"**{s}**" if a in best[(m, b)] else s)
            NUM[f"{key}_{MODELS[m]}_{a}_b{b:g}"] = round(v.mean(), 1)
            ref = df[(df.adapter == "rgb-only") & (df.model == m) & (df.budget == b)].test.mean()
            NUM[f"{key}_vs_rgb_{MODELS[m]}_{a}_b{b:g}"] = round(v.mean() - ref, 1)
            reft = df[(df.adapter == "timm-repeat") & (df.model == m) & (df.budget == b)].test.mean()
            NUM[f"{key}_vs_timm_{MODELS[m]}_{a}_b{b:g}"] = round(v.mean() - reft, 1)
        lines.append(f"| `{a}` | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def compute_summary(frames):
    runs = sum(len(f) for f in frames)
    hours = sum(f.time.sum() for f in frames) / 3600
    return runs, hours


def main():
    e2, e0, e4 = load("E2"), load("E0"), load("E4")
    e3, e4p = load("E3", finetune=False), load("E4p", finetune=False)
    main_table(e2, "E2", BUDGETS, TABLES / "sbd_finetune.md")
    main_table(e4, "E4", [1.0], TABLES / "eurosat_finetune.md")
    for m in MODELS:   # EuroSAT differences are small, so they are quoted in text rather than plotted
        for a in OPTIONS:
            mu, lo, hi, n = paired(e4, a, m, 1.0)
            if not np.isnan(mu):
                NUM[f"E4_diff_{MODELS[m]}_{a}_b1"] = round(mu, 1)
                NUM[f"E4_diff_{MODELS[m]}_{a}_b1_lo"] = round(lo, 1)
                NUM[f"E4_diff_{MODELS[m]}_{a}_b1_hi"] = round(hi, 1)
    for dark in (False, True):
        diff_figure(e2, "E2", BUDGETS, "fig_diff_sbd", dark)
        lr_figure(dark)
    band_order(e0, e2, TABLES / "band_order.md")
    for m in MODELS:   # paired differences against rgb-only, quoted in the text
        for b in BUDGETS:
            for a in OPTIONS:
                mu, lo, hi, n = paired(e2, a, m, b, ref="rgb-only")
                if not np.isnan(mu):
                    NUM[f"E2_vsrgb_{MODELS[m]}_{a}_b{b:g}"] = round(mu, 1)
    probe_table(e3, "E3", TABLES / "sbd_probe.md", BUDGETS)
    probe_table(e4p, "E4p", TABLES / "eurosat_probe.md", [1.0])
    # count every run that produced a file, including sweeps, for the compute statement
    allruns = 0; secs = 0.0
    for exp in ("E0", "E1", "E1b", "E2", "E3", "E4", "E4p"):
        for f in (RESULTS / exp).glob("*.json"):
            allruns += 1; secs += json.loads(f.read_text()).get("train_time_s", 0)
    NUM["compute_runs"] = allruns
    NUM["compute_gpu_hours"] = round(secs / 3600, 1)
    (POST / "numbers.json").write_text(json.dumps(dict(sorted(NUM.items())), indent=1))
    print("runs counted:", allruns, "GPU-hours (training time only):", round(secs / 3600, 1))
    print(len(NUM), "numbers")


if __name__ == "__main__":
    main()
