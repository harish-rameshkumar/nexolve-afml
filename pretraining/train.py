"""Pretraining loop for all three arms (T / D / P): one recipe, only --arm and --seed change.

    python -m pretraining.train --arm T --seed 0
    python -m pretraining.train --arm D --seed 0
    python -m pretraining.train --arm P --seed 0

Built for free Kaggle/Colab sessions that die without warning:
  * a rolling `latest.pt` (model+optimizer+scaler) every --resume_every steps; re-running the
    SAME command resumes automatically, and the data order needs no state (see dataset.py);
  * a finished run writes done.json and is skipped if the command is re-run.

Outputs in <out_dir>/<arm>_s<seed>/ :
    run_config.json     everything needed to reproduce the run (+ data sha256, git commit, GPU)
    metrics.jsonl       "train" rows (loss, lr, tok/s, ...) and "val" rows (val loss, FLOPs, GPU-hours)
    step00250.pt ...    intermediate model-only checkpoints (fp16 weights) for the loss-matched analysis
    final.pt            final weights in fp32  <-- the hand-off to WS3 (load with models.load_checkpoint)
    done.json           summary of the finished run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from dataclasses import fields

import numpy as np
import torch

from configs.model_config import ModelConfig, default_config
from models import build_model, save_checkpoint
from pretraining.config import PretrainConfig
from pretraining.dataset import TrainStream, check_data_files, load_val_tokens
from pretraining.evaluate import autocast_ctx, next_token_loss, val_loss


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def flops_per_token(model, model_cfg: ModelConfig) -> float:
    """Training FLOPs per token, PaLM convention: 6*N + 12*L*H_q*hd*T.

    N = unique parameters (the tied embedding is used once as the output matmul, so the
    lookup costs nothing and counting it once is right). The attention term has no causal
    discount; D has 2H query heads, so its score/value matmuls cost twice T's.
    This is an ESTIMATE (it ignores norms, softmax, RoPE) -- good for comparing arms.
    """
    h_q = model_cfg.n_heads * (2 if model.arm == "D" else 1)
    attn = 12 * model_cfg.n_layers * h_q * model_cfg.head_dim * model_cfg.context_len
    return 6.0 * model.num_params() + attn


def build_optimizer(model, cfg: PretrainConfig) -> torch.optim.AdamW:
    """AdamW; weight decay only on >=2-D weights, excluding the (tied) embedding.
    Norm gains are 1-D so they are excluded automatically. Identical rule for every arm
    (D's lambda_proj is a 2-D Linear, so it is decayed like any other projection)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 and not name.startswith("embed.") else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    use_fused = cfg.device.startswith("cuda") and torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, fused=use_fused or None)


def _atomic_save(save_fn, path: str):
    tmp = path + ".tmp"
    save_fn(tmp)
    os.replace(tmp, path)


def save_intermediate(path, model, step, extra, dtype: str):
    """Model-only checkpoint in the SAME format as models.save_checkpoint (so models.load_checkpoint
    reads it), but with weights stored in fp16 to halve disk. Tied tensors stay shared."""
    td = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    memo: dict[int, torch.Tensor] = {}

    def conv(t):
        k = t.data_ptr()
        if k not in memo:
            memo[k] = t.detach().to(td).cpu()
        return memo[k]

    sd = {k: conv(v) for k, v in model.state_dict().items()}
    _atomic_save(lambda p: torch.save(
        {"arm": model.arm, "config": model.cfg.to_dict(), "model": sd, "optimizer": None,
         "step": step, "extra": extra}, p), path)


def truncate_metrics(path: str, step: int):
    """After a resume, drop log rows from steps we are about to redo."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    with open(path, "w") as f:
        for r in rows:
            if r["step"] <= step:
                f.write(json.dumps(r) + "\n")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
def train(cfg: PretrainConfig, model_cfg: ModelConfig | None = None) -> dict:
    assert cfg.arm in ("T", "D", "P"), f"unknown arm {cfg.arm!r}"
    model_cfg = model_cfg or default_config(context_len=cfg.seq_len)
    assert model_cfg.context_len == cfg.seq_len, "seq_len must equal ModelConfig.context_len"
    assert cfg.eval_every % cfg.log_every == 0 and cfg.resume_every % cfg.log_every == 0 \
        and cfg.max_steps % cfg.log_every == 0, "eval_every, resume_every and max_steps must be multiples of log_every"
    use_cuda = cfg.device.startswith("cuda")
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available (Colab: Runtime > Change runtime type > T4)")
    amp = cfg.amp if use_cuda else "none"

    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    done_path = os.path.join(run_dir, "done.json")
    if os.path.exists(done_path):
        print(f"[{cfg.run_name}] already finished ({done_path}); nothing to do")
        return json.load(open(done_path))

    meta = check_data_files(cfg.data_dir)
    assert meta["vocab_size"] == model_cfg.vocab_size, "tokenizer vocab != model vocab"

    # seeds: model init + (separately, inside TrainStream) data order
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    model = build_model(cfg.arm, model_cfg).to(cfg.device)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))
    fwd = torch.compile(model) if cfg.compile else model
    stream = TrainStream(cfg.data_dir, cfg.seq_len, cfg.seed, cfg.micro_batch * cfg.grad_accum, cfg.max_steps)
    val = load_val_tokens(cfg.data_dir, cfg.seq_len)

    fpt = flops_per_token(model, model_cfg)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    latest_path = os.path.join(run_dir, "latest.pt")

    # ---- resume ----------------------------------------------------------
    step, t_train, t_total = 0, 0.0, 0.0
    if os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=cfg.device, weights_only=False)
        assert ck["arm"] == cfg.arm and ck["extra"]["seed"] == cfg.seed, "latest.pt belongs to a different run"
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["extra"]["scaler"])
        step, t_train, t_total = ck["step"], ck["extra"]["t_train"], ck["extra"]["t_total"]
        del ck
        truncate_metrics(metrics_path, step)
        print(f"[{cfg.run_name}] RESUMED from step {step}")
    else:
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
        run_config = {
            "pretrain": cfg.to_dict(), "model": model_cfg.to_dict(), "n_params": model.num_params(),
            "flops_per_token": fpt, "data_meta": meta, "git_commit": git_commit(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        }
        json.dump(run_config, open(os.path.join(run_dir, "run_config.json"), "w"), indent=2)

    def log(row):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def evaluate_and_checkpoint(s: int):
        """Frozen-val loss, plus an intermediate checkpoint on the ckpt grid. Not timed as 'train' time."""
        nonlocal t_total
        t0 = time.time()
        vl = val_loss(model, val, cfg.device, cfg.micro_batch * 2, amp)
        row = {"type": "val", "step": s, "tokens": s * cfg.tokens_per_step, "val_loss": vl,
               "val_ppl": math.exp(vl), "train_flops": fpt * s * cfg.tokens_per_step,
               "train_gpu_h": t_train / 3600, "total_gpu_h": (t_total + time.time() - t0) / 3600}
        log(row)
        print(f"[{cfg.run_name}] step {s:5d} | val loss {vl:.4f} (ppl {math.exp(vl):.1f})", flush=True)
        if 0 < s < cfg.max_steps and s % cfg.ckpt_every == 0:
            save_intermediate(os.path.join(run_dir, f"step{s:05d}.pt"), model, s,
                              {"seed": cfg.seed, "val_loss": vl, "tokens": row["tokens"]}, cfg.ckpt_dtype)
        t_total += time.time() - t0

    def save_resume(s: int):
        nonlocal t_total
        t0 = time.time()
        extra = {"seed": cfg.seed, "scaler": scaler.state_dict(), "t_train": t_train, "t_total": t_total}
        _atomic_save(lambda p: save_checkpoint(p, model, optimizer, step=s, extra=extra), latest_path)
        t_total += time.time() - t0

    print(f"[{cfg.run_name}] {model.num_params():,} params | {cfg.tokens_per_step:,} tok/step x {cfg.max_steps} "
          f"= {cfg.total_tokens / 1e6:.1f}M tokens | amp={amp} | {fpt / 1e6:.1f} MFLOPs/token", flush=True)

    if step == 0:
        evaluate_and_checkpoint(0)          # sanity: should be about ln(16384) = 9.70

    # ---- training --------------------------------------------------------
    model.train()
    log_loss = torch.zeros((), device=cfg.device)
    log_n, log_t0, log_tokens = 0, time.time(), 0
    while step < cfg.max_steps:
        t_step = time.time()
        lr = cfg.lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=cfg.device)
        for m in range(cfg.grad_accum):
            batch = stream.get(step, m, cfg.micro_batch).to(cfg.device, non_blocking=True)
            with autocast_ctx(cfg.device, amp):
                logits = fwd(batch[:, :-1])
            loss = next_token_loss(logits, batch[:, 1:])
            scaler.scale(loss / cfg.grad_accum).backward()
            step_loss += loss.detach() / cfg.grad_accum
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        log_loss += step_loss; log_n += 1; log_tokens += cfg.tokens_per_step

        if step % cfg.log_every == 0 or step == cfg.max_steps:
            if use_cuda:
                torch.cuda.synchronize()
            dt = time.time() - log_t0
            t_train += dt; t_total += dt
            avg = (log_loss / log_n).item()
            if not math.isfinite(avg):
                raise RuntimeError(f"[{cfg.run_name}] non-finite training loss at step {step}")
            log({"type": "train", "step": step, "tokens": step * cfg.tokens_per_step, "loss": avg, "lr": lr,
                 "grad_norm": float(grad_norm), "tok_s": log_tokens / dt, "loss_scale": scaler.get_scale() if amp == "fp16" else 1.0,
                 "train_gpu_h": t_train / 3600})
            if step % (cfg.log_every * 10) == 0 or step == cfg.max_steps:
                print(f"[{cfg.run_name}] step {step:5d}/{cfg.max_steps} | loss {avg:.4f} | lr {lr:.2e} | "
                      f"gnorm {float(grad_norm):.2f} | {log_tokens / dt / 1e3:.1f}k tok/s", flush=True)
            log_loss.zero_(); log_n, log_t0, log_tokens = 0, time.time(), 0

        if step % cfg.eval_every == 0 and step < cfg.max_steps:
            # flush un-logged time so GPU-hours are right before the eval is timed separately
            evaluate_and_checkpoint(step)
            log_t0 = time.time()
        if step % cfg.resume_every == 0 and step < cfg.max_steps:
            save_resume(step)
            log_t0 = time.time()

    # ---- finish ----------------------------------------------------------
    evaluate_and_checkpoint(cfg.max_steps)
    final_path = os.path.join(run_dir, "final.pt")
    _atomic_save(lambda p: save_checkpoint(p, model, None, step=cfg.max_steps,
                                           extra={"seed": cfg.seed, "pretrain": cfg.to_dict()}), final_path)
    rows = [json.loads(l) for l in open(metrics_path) if l.strip()]
    last_val = [r for r in rows if r["type"] == "val"][-1]
    train_rows = [r for r in rows if r["type"] == "train"]
    summary = {
        "arm": cfg.arm, "seed": cfg.seed, "steps": cfg.max_steps, "tokens": cfg.total_tokens,
        "final_val_loss": last_val["val_loss"], "n_params": model.num_params(),
        "train_flops": last_val["train_flops"], "train_gpu_h": t_train / 3600, "total_gpu_h": t_total / 3600,
        "mean_tok_s": float(np.mean([r["tok_s"] for r in train_rows])) if train_rows else None,
        "final_checkpoint": final_path,
    }
    json.dump(summary, open(done_path, "w"), indent=2)
    if os.path.exists(latest_path):
        os.remove(latest_path)               # final.pt supersedes it (it also held optimizer state)
    print(f"[{cfg.run_name}] DONE | final val loss {summary['final_val_loss']:.4f} | "
          f"{summary['train_gpu_h']:.2f} train GPU-h", flush=True)
    return summary


def parse_args(argv=None) -> PretrainConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = PretrainConfig()
    for f in fields(PretrainConfig):
        if f.name == "betas":
            continue
        d = getattr(defaults, f.name)
        if isinstance(d, bool):
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=d)
        else:
            p.add_argument(f"--{f.name}", type=type(d), default=d)
    return PretrainConfig(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    train(parse_args())
