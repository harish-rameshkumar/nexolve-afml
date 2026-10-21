"""Plot the distribution of sigmoid(lambda) per layer for one or more D models.

    python scripts/plot_lambda.py                                  # random-init D
    python scripts/plot_lambda.py --ckpt before_sft.pt after_sft.pt --labels before after

Left panel: box plot per layer of every sigmoid(lambda) value (all tokens, all heads) in the batch,
one colour per model. Other panels: heatmap of the per-head mean (rows = layers, columns = heads),
one per model, on a shared 0..1 scale, so before/after SFT can be compared directly.
Lambda depends on the input, so use the SAME ids for every model you compare (fixed seed by default).
"""
import argparse
import os

try:                      # imported as a package (tests) ...
    from scripts import _common as C
except ImportError:       # ... or run directly: python scripts/plot_*.py
    import _common as C   # also puts the repo root on sys.path
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from models import lambda_summary  # noqa: E402
from models.diagnostics import _device_of  # noqa: E402


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="*", default=[], help="D checkpoints (none = random init)")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--text")
    ap.add_argument("--ids")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default=os.path.join(C.DEFAULT_FIG_DIR, "lambda_distribution.png"))
    args = ap.parse_args(argv)

    dev = C.pick_device(args.device)
    ckpts = args.ckpt or [None]
    labels = args.labels or [("random init" if c is None else os.path.splitext(os.path.basename(c))[0]) for c in ckpts]
    assert len(labels) == len(ckpts), "need one label per checkpoint"

    per_model = []   # (label, lam (L, B*T*H) numpy, head_mean (L, H) numpy)
    for ck, lab in zip(ckpts, labels):
        model, _ = C.get_model("D", ck, args.tiny, args.seed, dev)
        ids, _ = C.make_ids(model.cfg.vocab_size, min(args.seq_len, model.cfg.context_len), args.batch, args.seed, args.text, args.ids)
        with torch.no_grad():
            model(ids.to(_device_of(model)))
            lam = model.lambda_values().float().cpu().flatten(1).numpy()
        per_model.append((lab, lam, lambda_summary(model, ids)["head_mean"].numpy()))

    n = len(per_model)
    L = per_model[0][1].shape[0]
    fig, axes = plt.subplots(1, 1 + n, figsize=(5 + 3.6 * n, 4.5), squeeze=False,
                             gridspec_kw={"width_ratios": [2.2] + [1] * n})
    ax = axes[0, 0]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    width = 0.8 / n
    for i, (lab, lam, _) in enumerate(per_model):
        pos = np.arange(L) + (i - (n - 1) / 2) * width
        bp = ax.boxplot(list(lam), positions=pos, widths=width * 0.9, patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set(facecolor=colors[i % len(colors)], alpha=0.6)
        ax.plot([], [], color=colors[i % len(colors)], lw=6, alpha=0.6, label=lab)
    ax.set_xticks(range(L), [f"L{l}" for l in range(L)])
    ax.set_ylim(0, 1), ax.set_xlabel("layer"), ax.set_ylabel("σ(λ)  (fraction of 2nd map subtracted)")
    ax.set_title("σ(λ) distribution per layer (all tokens, all heads)", fontsize=9)
    ax.axhline(0.5, color="grey", lw=0.6, ls=":")
    ax.legend(fontsize=8)
    for i, (lab, _, hm) in enumerate(per_model):
        a = axes[0, 1 + i]
        im = a.imshow(hm, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        a.set_title(f"{lab}: mean σ(λ) per head", fontsize=9)
        a.set_xlabel("head"), a.set_ylabel("layer")
        a.set_xticks(range(hm.shape[1])), a.set_yticks(range(hm.shape[0]))
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"saved {args.out}")
    return args.out


if __name__ == "__main__":
    main()
