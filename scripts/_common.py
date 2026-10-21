"""Helpers shared by the plotting scripts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # non-interactive: figures are only written to files

import torch  # noqa: E402

from configs.model_config import ModelConfig, default_config  # noqa: E402
from models import build_model, load_checkpoint  # noqa: E402

DEFAULT_FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")


def tiny_config() -> ModelConfig:
    """Small config for smoke tests and quick demos (--tiny)."""
    return default_config(vocab_size=128, context_len=64, d_model=64, n_layers=2, n_heads=4, ffn_hidden=176)


def pick_device(name: str = "cpu") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def get_model(arm: str, ckpt: str | None, tiny: bool, seed: int, device) -> tuple:
    """Load `ckpt` (checking its arm), or build a randomly initialised model with a fixed seed."""
    if ckpt:
        model, info = load_checkpoint(ckpt, device=device)
        if info["arm"] != arm:
            raise SystemExit(f"{ckpt} is an arm-{info['arm']} checkpoint, expected arm {arm}")
        return model.eval(), f"{arm} (trained: {os.path.basename(ckpt)})"
    torch.manual_seed(seed)
    cfg = tiny_config() if tiny else default_config()
    return build_model(arm, cfg).to(device).eval(), f"{arm} (random init, seed {seed})"


def make_ids(vocab_size: int, seq_len: int, batch: int, seed: int, text: str | None = None,
             ids: str | None = None):
    """Input ids plus per-position labels.

    Priority: --ids "1,2,3" (ids from the real tokenizer) > --text (UTF-8 bytes, a
    tokenizer-free stand-in: byte value = token id, NOT the project tokenizer) >
    random ids with a fixed seed.
    """
    if ids:
        seq = [int(t) for t in ids.split(",")]
        labels = [str(t) for t in seq]
    elif text:
        seq = list(text.encode("utf-8"))
        labels = [chr(b) if 32 < b < 127 else "·" for b in seq]
    else:
        g = torch.Generator().manual_seed(seed)
        seq = torch.randint(0, vocab_size, (seq_len,), generator=g).tolist()
        labels = [str(t) for t in seq]
    seq = [t % vocab_size for t in seq][:seq_len]   # truncate to seq_len (callers pass <= context_len)
    labels = labels[:len(seq)]
    return torch.tensor([seq] * batch), labels
