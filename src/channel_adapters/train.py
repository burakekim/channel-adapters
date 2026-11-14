"""One experimental run: a full fine-tune or a linear probe.

The whole dataset lives on the GPU as a float tensor; batches are indexed,
augmented (flips, 90-degree rotations), bilinearly resized to ``img_size`` and
fed to the model. A run writes one JSON file with its configuration, the
per-epoch validation curve, the final test accuracy and timing.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .adapters import FIRST_LAYER_LR_MULT, adapt_model, first_layer_parameters
from .data import Dataset, Split, band_orders, band_stats, load_dataset, subsample
from .models import build_model, head_parameters

LR_GRID = (1e-4, 3e-4, 1e-3, 3e-3)
EPOCHS_FOR_BUDGET = {1.0: 20, 0.1: 100, 0.01: 200}


@dataclass
class RunConfig:
    dataset: str = "sbd"
    model: str = "resnet50.a1_in1k"
    adapter: str = "timm-repeat"
    band_order: str = "native"          # key of data.band_orders
    budget: float = 1.0
    seed: int = 0
    lr: float = 1e-3
    mode: str = "finetune"              # or "probe"
    epochs: int | None = None
    warmup_epochs: int = 3              # only used by adapter == "warmup"
    img_size: int = 128
    batch_size: int = 256
    weight_decay: float = 0.05
    lr_warmup_epochs: int = 1
    probe_budgets: tuple[float, ...] = (1.0, 0.1, 0.01)
    probe_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    exp: str = "dev"
    tag: str = ""

    def run_id(self) -> str:
        parts = [self.dataset, self.model.split(".")[0], self.adapter, self.band_order,
                 f"b{self.budget:g}", f"s{self.seed}", f"lr{self.lr:g}", self.mode]
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# data on device
# ----------------------------------------------------------------------------
def prepare(ds: Dataset, order: list[int], device) -> tuple[dict[str, Split], list[int]]:
    """Reorder bands, z-score with train statistics, move to ``device``."""
    mean, std = band_stats(ds.train)
    mean, std = mean[order], std[order].clamp_min(1e-6)
    out = {}
    for name in ("train", "val", "test"):
        s = getattr(ds, name)
        x = (s.x[:, order].float() - mean[None, :, None, None]) / std[None, :, None, None]
        out[name] = Split(x.to(device), s.y.to(device))
    rgb = [order.index(i) for i in ds.rgb_indices]  # positions of R,G,B in the new order
    return out, rgb


def augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(()) < 0.5:
        x = x.flip(-1)
    if torch.rand(()) < 0.5:
        x = x.flip(-2)
    k = int(torch.randint(0, 4, ()))
    return torch.rot90(x, k, dims=(-2, -1)) if k else x


def resize(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(model, split: Split, img_size: int, batch_size: int = 512) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(split.y), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(resize(split.x[i:i + batch_size], img_size))
        correct += (logits.argmax(1) == split.y[i:i + batch_size]).sum().item()
    return correct / len(split.y)


@torch.no_grad()
def features(model, split: Split, img_size: int, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(split.y), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.forward_head(model.forward_features(resize(split.x[i:i + batch_size], img_size)), pre_logits=True)
        out.append(f.float().cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------
# fine-tune
# ----------------------------------------------------------------------------
def set_trainable(model, params_on) -> None:
    on = {id(p) for p in params_on}
    for p in model.parameters():
        p.requires_grad_(id(p) in on)


def finetune(cfg: RunConfig, model, splits, device) -> dict:
    train, val, test = splits["train"], splits["val"], splits["test"]
    n = len(train.y)
    steps_per_epoch = max(1, math.ceil(n / cfg.batch_size))
    epochs = cfg.epochs or EPOCHS_FOR_BUDGET.get(cfg.budget, 20)
    total_steps = epochs * steps_per_epoch
    warm_steps = cfg.lr_warmup_epochs * steps_per_epoch

    mult = FIRST_LAYER_LR_MULT.get(cfg.adapter, 1.0)
    if mult != 1.0:
        stem = {id(q) for q in first_layer_parameters(model)}
        groups = [{"params": [q for q in model.parameters() if id(q) in stem], "lr": cfg.lr * mult},
                  {"params": [q for q in model.parameters() if id(q) not in stem], "lr": cfg.lr}]
    else:
        groups = [{"params": list(model.parameters()), "lr": cfg.lr}]
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm_steps if s < warm_steps
        else 0.5 * (1 + math.cos(math.pi * (s - warm_steps) / max(1, total_steps - warm_steps))))

    stage1 = cfg.adapter == "warmup"
    if stage1:
        set_trainable(model, first_layer_parameters(model) + head_parameters(model))

    curve, step, t0 = [], 0, time.time()
    for epoch in range(epochs):
        if stage1 and epoch == cfg.warmup_epochs:
            set_trainable(model, list(model.parameters()))
            stage1 = False
        model.train()
        perm = torch.randperm(n, device=device)
        running = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            x = resize(augment(train.x[idx]), cfg.img_size)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), train.y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            running += loss.item() * len(idx)
        curve.append({"epoch": epoch + 1, "train_loss": running / n,
                      "val_acc": evaluate(model, val, cfg.img_size), "time": time.time() - t0})
    final_val = curve[-1]["val_acc"]
    reach = next((c["epoch"] for c in curve if c["val_acc"] >= 0.95 * final_val), epochs)
    return {
        "val_acc": final_val,
        "best_val_acc": max(c["val_acc"] for c in curve),
        "test_acc": evaluate(model, test, cfg.img_size),
        "epochs": epochs,
        "steps": total_steps,
        "epochs_to_95pct": reach,
        "curve": curve,
        "train_time_s": time.time() - t0,
        "diverged": not math.isfinite(curve[-1]["train_loss"]),
    }


# ----------------------------------------------------------------------------
# linear probe (multinomial logistic regression, full-batch L-BFGS on the GPU)
# ----------------------------------------------------------------------------
def fit_logreg(x, y, n_classes, l2, iters=300):
    w = torch.zeros(x.shape[1], n_classes, device=x.device, requires_grad=True)
    b = torch.zeros(n_classes, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w + b, y) + 0.5 * l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def probe(cfg: RunConfig, model, splits, ds: Dataset) -> dict:
    t0 = time.time()
    device = splits["train"].x.device
    feats = {k: torch.from_numpy(features(model, v, cfg.img_size)).to(device) for k, v in splits.items()}
    ys = {k: v.y for k, v in splits.items()}
    n_classes = len(ds.classes)
    results = []
    for budget in cfg.probe_budgets:
        for seed in cfg.probe_seeds:
            sub = subsample(Split(feats["train"], ys["train"]), budget, seed)
            mu, sd = sub.x.mean(0, keepdim=True), sub.x.std(0, keepdim=True).clamp_min(1e-6)
            xtr, xva, xte = ((a - mu) / sd for a in (sub.x, feats["val"], feats["test"]))
            best = None
            for l2 in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):   # weight decay; small l2 = large C
                w, b = fit_logreg(xtr, sub.y, n_classes, l2)
                va = ((xva @ w + b).argmax(1) == ys["val"]).float().mean().item()
                if best is None or va > best[0]:
                    best = (va, l2, ((xte @ w + b).argmax(1) == ys["test"]).float().mean().item())
            results.append({"budget": budget, "seed": seed, "l2": best[1], "val_acc": best[0], "test_acc": best[2]})
            if budget >= 1.0:
                break  # the full budget has no sampling randomness
    return {"probe": results, "feature_dim": int(feats["train"].shape[1]), "train_time_s": time.time() - t0}


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
def run(cfg: RunConfig, out_dir: Path | str = "results") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    ds = load_dataset(cfg.dataset)
    order = band_orders(ds.bands)[cfg.band_order]
    splits, rgb = prepare(ds, order, device)
    if cfg.mode == "finetune":
        splits["train"] = subsample(splits["train"], cfg.budget, cfg.seed)
    model = build_model(cfg.model, num_classes=len(ds.classes), img_size=cfg.img_size)
    adapt_model(model, cfg.adapter, n_bands=len(order), rgb=rgb, bands=[ds.bands[i] for i in order])
    model.to(device)

    record = {"config": asdict(cfg), "run_id": cfg.run_id(), "git_sha": git_sha(),
              "bands": [ds.bands[i] for i in order], "rgb_positions": rgb,
              "n_train": int(len(splits["train"].y)), "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    if cfg.mode == "finetune":
        record.update(finetune(cfg, model, splits, device))
    elif cfg.mode == "probe":
        record.update(probe(cfg, model, splits, ds))
    else:
        raise ValueError(cfg.mode)

    out = Path(out_dir) / cfg.exp
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg.run_id()}.json").write_text(json.dumps(record, indent=1))
    return record
