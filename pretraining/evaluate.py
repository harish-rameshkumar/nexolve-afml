"""Frozen-validation loss. 

"""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from pretraining.dataset import PackedTokens


def autocast_ctx(device: str, amp: str):
    """fp16 on T4 (bf16 is emulated and slow there). Only used on CUDA."""
    if amp == "none" or not str(device).startswith("cuda"):
        return nullcontext()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[amp]
    return torch.autocast(device_type="cuda", dtype=dtype)


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Cross-entropy in float32. logits (B, T, V) vs targets (B, T) long."""
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction=reduction)


@torch.no_grad()
def val_loss(model, val: PackedTokens, device: str = "cuda", batch_size: int = 16, amp: str = "fp16",
             max_windows: int | None = None) -> float:
    """Mean per-token NLL (nats) over every window of the frozen val set, in a fixed order."""
    was_training = model.training
    model.eval()
    n = val.n_windows if max_windows is None else min(max_windows, val.n_windows)
    total, count = 0.0, 0
    try:
        for s in range(0, n, batch_size):
            batch = val.batch(range(s, min(s + batch_size, n))).to(device)
            with autocast_ctx(device, amp):
                logits = model(batch[:, :-1])
            total += next_token_loss(logits, batch[:, 1:], reduction="sum").item()
            count += batch[:, 1:].numel()
    finally:
        model.train(was_training)
    return total / count
