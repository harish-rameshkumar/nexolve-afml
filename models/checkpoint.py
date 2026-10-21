"""Checkpoint helpers. A checkpoint stores the arm and config next to the weights,
so it can be reloaded without remembering how the model was built."""
from __future__ import annotations

import torch

from configs.model_config import ModelConfig
from models.build import build_model


def save_checkpoint(path, model, optimizer=None, step: int | None = None, extra: dict | None = None):
    torch.save(
        {
            "arm": model.arm,
            "config": model.cfg.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, device="cpu", optimizer=None):
    """Rebuild the model from the stored arm+config and load weights.

    Returns (model, ckpt_dict). If `optimizer` is given its state is restored too.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arm"], ModelConfig.from_dict(ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return model, ckpt
