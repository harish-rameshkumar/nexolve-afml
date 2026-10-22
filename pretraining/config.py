
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field


@dataclass
class PretrainConfig:
    # ---- run identity -------------------------------------------------------
    arm: str = "T"                 # "T" | "D" | "P"
    seed: int = 0                  # controls model init AND data order (shared by all arms)

    # ---- data ---------------------------------------------------------------
    data_dir: str = "data/pretrain"   # holds train.bin, val.bin, meta.json, tokenizer.json
    seq_len: int = 1024               # must equal ModelConfig.context_len

    # ---- batch / budget -----------------------------------------------------
    # tokens per optimiser step = micro_batch * grad_accum * seq_len = 65,536
    # 4,600 steps * 65,536 = 301.5M tokens (the "300M tokens" budget in the plan)
    micro_batch: int = 16
    grad_accum: int = 4
    max_steps: int = 4600

    # ---- optimiser (AdamW) --------------------------------------------------
    lr: float = 2e-3
    min_lr_ratio: float = 0.1         # cosine decays to lr * min_lr_ratio
    warmup_steps: int = 200
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1         # only on >=2-D weights, never norms/embeddings
    grad_clip: float = 1.0

    # ---- precision ----------------------------------------------------------
    amp: str = "fp16"                 # "fp16" (T4: GradScaler) | "bf16" | "none"

    # ---- evaluation / checkpointing ----------------------------------------
    eval_every: int = 250             # frozen-val loss; also where intermediate ckpts go
    ckpt_every: int = 250             # intermediate (model-only) checkpoints
    ckpt_dtype: str = "fp16"          # storage dtype of intermediates (final is always fp32)
    resume_every: int = 100           # rolling latest.pt (model+optimizer+scaler) for crash resume
    log_every: int = 10
    out_dir: str = "outputs/pretrain"  # run goes to <out_dir>/<arm>_s<seed>/

    # ---- misc ---------------------------------------------------------------
    compile: bool = False             # torch.compile (off: untested on T4 for the D arm)
    device: str = "cuda"

    # ----------------------------------------------------------------------
    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch * self.grad_accum * self.seq_len

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_step * self.max_steps

    @property
    def run_name(self) -> str:
        return f"{self.arm}_s{self.seed}"

    def lr_at(self, step: int) -> float:
        """Linear warmup then cosine decay to min_lr_ratio * lr. `step` is 0-based."""
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PretrainConfig":
        d = dict(d)
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        return cls(**d)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
