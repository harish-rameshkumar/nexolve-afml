"""Plot attention maps of T and D side by side.

    python scripts/plot_attention.py                       # random-init models, random ids
    python scripts/plot_attention.py --ckpt-t t.pt --ckpt-d d.pt --layers 0 5 --heads 0 3
    python scripts/plot_attention.py --text "the cat sat on the mat"   # bytes as ids (see below)

Rows are (layer, head) pairs; left column T, right column D. Colour scales differ ON PURPOSE:
  T: ordinary softmax weights, all >= 0, sequential scale from 0.
  D: signed differential map A1 - sigmoid(lambda)*A2, which can be negative, so a diverging
     scale centred on 0 is used (red positive, blue negative, white zero).
Without a tokenizer, --text maps each UTF-8 byte to a token id; that is NOT the project
tokenizer, so a trained checkpoint will see unfamiliar input. Use --ids with real token ids instead.
"""
import argparse
import os

try:                      # imported as a package (tests) ...
    from scripts import _common as C
except ImportError:       # ... or run directly: python scripts/plot_*.py
    import _common as C   # also puts the repo root on sys.path
import matplotlib.pyplot as plt  # noqa: E402

from models import attention_maps  # noqa: E402


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-t")
    ap.add_argument("--ckpt-d")
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="default: first and last")
    ap.add_argument("--heads", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--text")
    ap.add_argument("--ids", help="comma-separated token ids")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tiny", action="store_true", help="tiny random config (smoke tests)")
    ap.add_argument("--out", default=os.path.join(C.DEFAULT_FIG_DIR, "attention_T_vs_D.png"))
    args = ap.parse_args(argv)

    dev = C.pick_device(args.device)
    t_model, t_name = C.get_model("T", args.ckpt_t, args.tiny, args.seed, dev)
    d_model, d_name = C.get_model("D", args.ckpt_d, args.tiny, args.seed, dev)
    vocab = t_model.cfg.vocab_size
    ids, labels = C.make_ids(vocab, min(args.seq_len, t_model.cfg.context_len, d_model.cfg.context_len), 1, args.seed, args.text, args.ids)
    n_layers, n_heads = t_model.cfg.n_layers, t_model.cfg.n_heads
    layers = args.layers or sorted({0, n_layers - 1})
    for l in layers:
        assert 0 <= l < n_layers, f"layer {l} out of range"
    for h in args.heads:
        assert 0 <= h < n_heads, f"head {h} out of range"

    maps_t = attention_maps(t_model, ids)[:, 0]   # (L, H, T, T)
    maps_d = attention_maps(d_model, ids)[:, 0]
    rows = [(l, h) for l in layers for h in args.heads]
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 4.2 * len(rows) + 0.6), squeeze=False)
    for r, (l, h) in enumerate(rows):
        a_t, a_d = maps_t[l, h].numpy(), maps_d[l, h].numpy()
        lim = max(abs(a_d).max(), 1e-8)           # symmetric limits => white is exactly 0
        im_t = axes[r, 0].imshow(a_t, cmap="Blues", vmin=0, vmax=max(a_t.max(), 1e-8))
        im_d = axes[r, 1].imshow(a_d, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[r, 0].set_title(f"T  layer {l} head {h}: softmax weights (>= 0)", fontsize=9)
        axes[r, 1].set_title(f"D  layer {l} head {h}: signed A1 - λ·A2 (diverging scale, 0 = white)", fontsize=9)
        fig.colorbar(im_t, ax=axes[r, 0], fraction=0.046), fig.colorbar(im_d, ax=axes[r, 1], fraction=0.046)
        for c in (0, 1):
            axes[r, c].set_xlabel("key token (attended to)"), axes[r, c].set_ylabel("query token")
            if len(labels) <= 40:
                axes[r, c].set_xticks(range(len(labels)), labels, fontsize=6, rotation=90)
                axes[r, c].set_yticks(range(len(labels)), labels, fontsize=6)
    fig.suptitle(f"{t_name}   vs   {d_name}\nD's maps can be negative: red = adds, blue = subtracts", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"saved {args.out}")
    return args.out


if __name__ == "__main__":
    main()
