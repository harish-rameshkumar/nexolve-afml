"""Pretraining hand-off figures + table from the runs' metrics.jsonl files.

    python scripts/plot_pretrain_curves.py --runs outputs/pretrain --out outputs/pretrain_report

Writes:
    val_loss_vs_tokens.png   frozen-val loss vs tokens, one line per seed, colour = arm
    train_loss_vs_tokens.png smoothed training loss
    pretrain_summary.csv     per run: final val loss, tokens, FLOPs, GPU-hours, mean tokens/s
    pretrain_arm_means.csv   per arm: mean/min/max final val loss across seeds
Colours: first three slots of the validated categorical palette (T blue, D orange, P aqua).
Aqua is low-contrast on white, so every line is also direct-labelled and the arms differ in line style.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"T": ("#2a78d6", "-"), "D": ("#eb6834", "--"), "P": ("#1baf7a", ":")}
NAME = {"T": "T standard", "D": "D DIFF V2", "P": "P param-matched"}


def load_runs(root):
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, "*_s*", "metrics.jsonl"))):
        name = os.path.basename(os.path.dirname(path))
        arm, seed = name.split("_s")
        rows = [json.loads(l) for l in open(path) if l.strip()]
        done = os.path.exists(os.path.join(os.path.dirname(path), "done.json"))
        runs[(arm, int(seed))] = (rows, done)
    return runs


def smooth(y, k=5):
    y = np.asarray(y, float)
    return np.array([y[max(0, i - k + 1): i + 1].mean() for i in range(len(y))])


def _draw(ax, runs, key, rows_type, smooth_k, x_min=None):
    """One line per (arm, seed). Seed 0 is solid and bold; later seeds are thinner and fainter."""
    seeds_by_arm = defaultdict(list)
    for arm, seed in runs:
        seeds_by_arm[arm].append(seed)
    ymin, ymax = np.inf, -np.inf
    for (arm, seed), (rows, _) in sorted(runs.items()):
        r = [x for x in rows if x["type"] == rows_type]
        if not r:
            continue
        x = np.array([a["tokens"] for a in r]) / 1e6
        y = smooth([a[key] for a in r], smooth_k)
        c, ls = STYLE[arm]
        first = seed == min(seeds_by_arm[arm])
        lw, alpha = (1.8, 0.95) if first else (1.2, 0.6)
        n = len(seeds_by_arm[arm])
        label = f"{NAME[arm]} ({n} seed{'s' if n > 1 else ''})" if first else None
        ax.plot(x, y, color=c, ls=ls, lw=lw, alpha=alpha, label=label)
        if x_min is not None:
            m = x >= x_min
            if m.any():
                ymin, ymax = min(ymin, y[m].min()), max(ymax, y[m].max())
    if x_min is not None:
        ax.set_xlim(x_min, None)
        if np.isfinite(ymin):
            pad = 0.05 * (ymax - ymin + 1e-9)
            ax.set_ylim(ymin - pad, ymax + pad)


def plot(runs, key, rows_type, ylabel, title, out, smooth_k=1, ylim=None, zoom_from=None):
    """Full-range panel; with zoom_from (millions of tokens) a second zoomed panel is added."""
    ncols = 2 if zoom_from is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.5 * ncols if ncols > 1 else 7.5, 4.6), dpi=150)
    axes = np.atleast_1d(axes)
    _draw(axes[0], runs, key, rows_type, smooth_k)
    if ylim:
        axes[0].set_ylim(*ylim)
    axes[0].set_title(title, loc="left", fontsize=11)
    if zoom_from is not None:
        _draw(axes[1], runs, key, rows_type, smooth_k, x_min=zoom_from)
        axes[1].set_title(f"Zoom: last {int(300 - zoom_from)}M tokens", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xlabel("pretraining tokens (millions)"); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/pretrain")
    ap.add_argument("--out", default="outputs/pretrain_report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.runs)
    if not runs:
        raise SystemExit(f"no runs found under {a.runs}")

    plot(runs, "val_loss", "val", "frozen validation loss (nats/token)", "Pretraining validation loss",
         os.path.join(a.out, "val_loss_vs_tokens.png"), ylim=(None, None), zoom_from=200)
    plot(runs, "loss", "train", "training loss (running mean of 5 log points)", "Pretraining training loss",
         os.path.join(a.out, "train_loss_vs_tokens.png"), smooth_k=5, zoom_from=200)

    table, by_arm = [], defaultdict(list)
    for (arm, seed), (rows, done) in sorted(runs.items()):
        vals = [r for r in rows if r["type"] == "val"]
        trn = [r for r in rows if r["type"] == "train"]
        last = vals[-1]
        row = {"arm": arm, "seed": seed, "finished": done, "step": last["step"], "tokens": last["tokens"],
               "val_loss": round(last["val_loss"], 4), "train_flops": f"{last['train_flops']:.3e}",
               "train_gpu_h": round(trn[-1]["train_gpu_h"], 3) if trn else None,
               "mean_tok_s": round(float(np.mean([t["tok_s"] for t in trn])), 0) if trn else None}
        table.append(row)
        if done:
            by_arm[arm].append(last["val_loss"])
    with open(os.path.join(a.out, "pretrain_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    with open(os.path.join(a.out, "pretrain_arm_means.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "n_seeds", "mean_val_loss", "min", "max"])
        for arm, v in sorted(by_arm.items()):
            w.writerow([arm, len(v), round(float(np.mean(v)), 4), round(min(v), 4), round(max(v), 4)])
    for r in table:
        print(r)
    print(f"wrote figures and tables to {a.out}")


if __name__ == "__main__":
    main()
