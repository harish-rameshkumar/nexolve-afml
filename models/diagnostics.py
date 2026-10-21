"""Diagnostics: lambda statistics (arm D) and attention maps (all arms).

Everything here runs under torch.no_grad, on whatever device the model lives on,
and restores the model's train/eval mode afterwards, so it can be called in the
middle of an SFT loop without side effects.
"""
from __future__ import annotations

import torch


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


def lambda_summary(model, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Statistics of sigmoid(lambda) over a batch, per layer (and per head for the mean).

    sigmoid(lambda) is the fraction of the second attention map that is subtracted
    for each (token, head). Returned tensors live on the CPU:
        "mean", "std", "min", "max": shape (n_layers,)  -- over all tokens and heads
        "head_mean":                 shape (n_layers, n_heads)  -- over tokens only
    Raises ValueError for arms without lambda (T and P).
    """
    arm = getattr(model, "arm", None)
    if arm != "D":
        raise ValueError(
            f"lambda_summary needs a D (DIFF V2) model, got arm={arm!r}: only D has lambda."
        )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(input_ids.to(_device_of(model)))
            lam = model.lambda_values().float().cpu()   # (L, B, T, H)
    finally:
        model.train(was_training)
    flat = lam.flatten(1)                                # (L, B*T*H)
    return {
        "mean": flat.mean(dim=1),
        "std": flat.std(dim=1, unbiased=False),          # unbiased=False: defined for 1 element
        "min": flat.min(dim=1).values,
        "max": flat.max(dim=1).values,
        "head_mean": lam.mean(dim=(1, 2)),               # (L, H)
    }


def lambda_log_dict(model, input_ids: torch.Tensor, prefix: str = "lambda") -> dict[str, float]:
    """Flat {name: float} version of lambda_summary, ready for any logger (wandb, tensorboard, print).

    Keys look like "lambda/layer0/mean", "lambda/layer0/std", ..., "lambda/layer0/head3/mean".
    """
    s = lambda_summary(model, input_ids)
    out: dict[str, float] = {}
    for layer in range(s["mean"].shape[0]):
        for stat in ("mean", "std", "min", "max"):
            out[f"{prefix}/layer{layer}/{stat}"] = float(s[stat][layer])
        for head in range(s["head_mean"].shape[1]):
            out[f"{prefix}/layer{layer}/head{head}/mean"] = float(s["head_mean"][layer, head])
    return out


def attention_maps(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Attention maps for any arm, stacked as (n_layers, B, n_heads, T, T) on the CPU.

    T and P: ordinary causal softmax maps (all entries >= 0, each row sums to 1).
    D:       signed differential maps A1 - sigmoid(lambda) * A2 (entries can be negative,
             each row sums to 1 - sigmoid(lambda)). Row = query token, column = key token.
    Uses the model's return_attn path, which always runs the explicit einsum attention.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _, maps = model(input_ids.to(_device_of(model)), return_attn=True)
    finally:
        model.train(was_training)
    return torch.stack([m.float().cpu() for m in maps])
