"""Single source of truth for model hyper-parameters (all three arms T / D / P).

Everything that defines an architecture lives here, so changing the default
config is a one-file edit. Teammates can also override fields:

    cfg = default_config(n_layers=4, d_model=256)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace


@dataclass
class ModelConfig:
    # ---- shape of the network -------------------------------------------
    vocab_size: int = 16384      # tokenizer vocabulary size
    context_len: int = 1024      # maximum sequence length
    d_model: int = 512           # residual-stream width
    n_layers: int = 6            # number of transformer blocks
    n_heads: int = 8             # attention heads (T); head_dim = d_model // n_heads
    ffn_hidden: int = 1408       # SwiGLU hidden size (~ 8/3 * d_model, rounded to /64)

    # ---- details ----------------------------------------------------------
    rope_theta: float = 10000.0  # RoPE base frequency
    norm_eps: float = 1e-6       # RMSNorm epsilon
    tie_embeddings: bool = True  # share input embedding and output projection
    init_std: float = 0.02       # std of the normal init for all weights
    # "sdpa": torch's fused scaled_dot_product_attention (fast; used for training)
    # "einsum": explicit einsum implementation (readable reference; also used
    #           whenever attention maps are requested). Both are tested equal.
    attn_impl: str = "sdpa"

    # ---- arm-specific (filled in by later steps) ---------------------------
    # P widens its FFN to absorb D's extra parameters. None -> build_model
    # computes the value that matches D's parameter count.
    ffn_hidden_p: int | None = None

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"
        assert self.attn_impl in ("sdpa", "einsum")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    # ---- (de)serialisation, so configs can be stored next to checkpoints ---
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def default_config(**overrides) -> ModelConfig:
    """The project default (about 28M params for T), optionally overridden."""
    return replace(ModelConfig(), **overrides)
