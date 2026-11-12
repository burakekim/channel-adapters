#!/usr/bin/env python
"""Expand an experiment grid and run one, all, or a slurm array of its runs.

    python experiments/run.py --exp E1 --list          # show the runs
    python experiments/run.py --exp E1 --index 3       # run one (used by slurm)
    python experiments/run.py --exp E1 --all           # run all sequentially here
    python experiments/run.py --exp E1 --slurm         # submit as a job array
    python experiments/run.py --pick-lr                # write results/lr_choice.json from E1

Runs whose JSON already exists in results/<exp>/ are skipped, so a partially
finished experiment can be resubmitted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from channel_adapters.train import RunConfig, run  # noqa: E402

CONFIGS = ROOT / "experiments" / "configs" / "experiments.yaml"
RESULTS = ROOT / "results"
LR_CHOICE = RESULTS / "lr_choice.json"
GRID_KEYS = ("model", "adapter", "band_order", "budget", "seed", "lr")


def load_lr_choice() -> dict:
    if not LR_CHOICE.exists():
        raise SystemExit("results/lr_choice.json missing: run E1, then `run.py --pick-lr`")
    return json.loads(LR_CHOICE.read_text())


def expand(exp: str) -> list[RunConfig]:
    spec = yaml.safe_load(CONFIGS.read_text())
    common, block = spec.get("common", {}), spec[exp]
    from_e1 = block.get("lr") == "from_E1"
    lists = {k: (block[k] if isinstance(block[k], list) else [block[k]]) for k in GRID_KEYS if k != "lr" or not from_e1}
    if from_e1:
        lists["lr"] = [None]
    lr_choice = load_lr_choice() if from_e1 else None
    runs = []
    for values in itertools.product(*(lists[k] for k in GRID_KEYS)):
        kw = dict(zip(GRID_KEYS, values))
        if kw["lr"] is None:
            kw["lr"] = lr_choice[kw["model"]][kw["adapter"]]
        kw.update({k: v for k, v in block.items() if k not in GRID_KEYS and k != "lr"})
        kw.update(common)
        runs.append(RunConfig(exp=exp, **kw))
    return runs


def done(cfg: RunConfig) -> bool:
    return (RESULTS / cfg.exp / f"{cfg.run_id()}.json").exists()


def pick_lr() -> None:
    choice: dict[str, dict[str, float]] = {}
    for f in (RESULTS / "E1").glob("*.json"):
        r = json.loads(f.read_text())
        c = r["config"]
        if r.get("diverged"):
            continue
        cur = choice.setdefault(c["model"], {}).get(c["adapter"])
        if cur is None or r["val_acc"] > cur[1]:
            choice[c["model"]][c["adapter"]] = (c["lr"], r["val_acc"])
    out = {m: {a: lr for a, (lr, _) in d.items()} for m, d in choice.items()}
    LR_CHOICE.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


SLURM = """#!/bin/bash
#SBATCH --job-name=chadapt-{exp}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --qos={qos}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-{last}%{max_parallel}
#SBATCH --output={logdir}/%A_%a.out
cd {root}
export CHADAPT_DATA={data}
{python} experiments/run.py --exp {exp} --index $SLURM_ARRAY_TASK_ID
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp")
    p.add_argument("--list", action="store_true")
    p.add_argument("--index", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--slurm", action="store_true")
    p.add_argument("--partition", default="vision-beery-l40s")
    p.add_argument("--account", default="vision-beery")
    p.add_argument("--qos", default="vision-beery-main")
    p.add_argument("--max-parallel", type=int, default=16)
    p.add_argument("--pick-lr", action="store_true")
    a = p.parse_args()

    if a.pick_lr:
        pick_lr()
        return
    runs = expand(a.exp)
    if a.list:
        for i, r in enumerate(runs):
            print(i, "done" if done(r) else "todo", r.run_id())
        print(len(runs), "runs,", sum(done(r) for r in runs), "done")
    elif a.index is not None:
        cfg = runs[a.index]
        if done(cfg):
            print("skip", cfg.run_id())
            return
        rec = run(cfg, RESULTS)
        print(cfg.run_id(), {k: rec.get(k) for k in ("test_acc", "val_acc", "train_time_s")})
    elif a.all:
        for cfg in runs:
            if not done(cfg):
                rec = run(cfg, RESULTS)
                print(cfg.run_id(), {k: rec.get(k) for k in ("test_acc", "val_acc", "train_time_s")}, flush=True)
    elif a.slurm:
        logdir = ROOT / "slurm_logs"
        logdir.mkdir(exist_ok=True)
        script = SLURM.format(exp=a.exp, partition=a.partition, account=a.account, qos=a.qos, last=len(runs) - 1, max_parallel=a.max_parallel,
                              logdir=logdir, root=ROOT, python=sys.executable,
                              data=os.environ.get("CHADAPT_DATA", Path.home() / ".cache" / "channel-adapters"))
        path = logdir / f"{a.exp}.sbatch"
        path.write_text(script)
        print(subprocess.check_output(["sbatch", str(path)], text=True))


if __name__ == "__main__":
    main()
