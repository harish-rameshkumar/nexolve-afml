"""Helpers shared by the exploratory experiment scripts (scripts/experiments/*.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from configs.model_config import ModelConfig, default_config  # noqa: E402
from models import build_model  # noqa: E402


def pick_device(name: str = "cpu") -> torch.device:
    """'cpu' (default), 'mps', 'cuda', or 'auto' (cuda > mps > cpu)."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def make_config(name: str) -> ModelConfig:
    """'default' = the project config (~28M params); 'tiny' = d_model 64, 2 layers (vocab stays 16384
    so that ln(vocab) is the same ~9.70 as in the default config)."""
    if name == "default":
        return default_config()
    return default_config(d_model=64, n_layers=2, n_heads=4, ffn_hidden=176, context_len=64)


def random_batch(vocab: int, batch: int, seq_len: int, seed: int, device):
    """(inputs, targets) of random token ids. Data is drawn on the CPU from a seeded generator, so the same
    seed gives the same tokens on cpu, mps and cuda."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (batch, seq_len + 1), generator=g)
    return ids[:, :-1].to(device), ids[:, 1:].to(device)


def build_seeded(arm: str, cfg: ModelConfig, seed: int, device) -> torch.nn.Module:
    """Build on the CPU under a fixed seed (identical initial weights on every device), then move."""
    torch.manual_seed(seed)
    return build_model(arm, cfg).to(device).train()


@torch.no_grad()
def eval_loss(model, x, y, vocab: int) -> float:
    model.eval()
    loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item()
    model.train()
    return loss


def train_step(model, opt, x, y, vocab: int) -> float:
    loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()
