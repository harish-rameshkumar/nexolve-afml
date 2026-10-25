
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import load_checkpoint, save_checkpoint


DATA_DIR = Path(
    r"C:\Users\HARSHIT VAISH\projects\nexolve-data\sft"
)
PRETRAIN_DIR = Path(
    r"C:\Users\HARSHIT VAISH\projects\nexolve-data\outputs"
)
OUTPUT_DIR = Path(
    r"C:\Users\HARSHIT VAISH\projects\nexolve-data\sft_outputs"
)

DEFAULT_LR = 1e-5
DEFAULT_BUDGET = 100_000
DEFAULT_EVAL_EVERY = 10_000

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 1024
MAX_CONSECUTIVE_OVERFLOWS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        examples = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if not examples:
        raise ValueError(f"No examples found in {path}")

    return examples


def supervised_count(labels: list[int]) -> int:
    # Training predicts labels[:, 1:] from logits on input_ids[:, :-1].
    return sum(label != -100 for label in labels[1:])


def validate_examples(examples: list[dict], name: str) -> None:
    for i, ex in enumerate(examples):
        ids = ex["input_ids"]
        labels = ex["labels"]

        if len(ids) != len(labels):
            raise ValueError(
                f"{name} example {i}: input and label lengths differ"
            )

        if not 2 <= len(ids) <= MAX_SEQ_LEN:
            raise ValueError(
                f"{name} example {i}: invalid sequence length {len(ids)}"
            )

        if supervised_count(labels) == 0:
            raise ValueError(
                f"{name} example {i}: no supervised target tokens"
            )


def make_optimizer(model, lr: float):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if param.ndim >= 2 and not name.startswith("embed."):
            decay.append(param)
        else:
            no_decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )


def make_single_example(
    example: dict,
    device: str,
    target_budget: int,
):
    """Mask excess response targets to enforce an exact token budget."""
    ids_list = example["input_ids"]
    labels_list = list(example["labels"])
    remaining = target_budget

    for pos in range(1, len(labels_list)):
        if labels_list[pos] == -100:
            continue

        if remaining > 0:
            remaining -= 1
        else:
            labels_list[pos] = -100

    ids = torch.tensor(
        [ids_list],
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor(
        [labels_list],
        dtype=torch.long,
        device=device,
    )

    count = int((labels[:, 1:] != -100).sum().item())
    return ids, labels, count


@torch.no_grad()
def evaluate(
    model,
    examples: list[dict],
    device: str,
) -> tuple[float, int]:
    """Calculate mean response-token NLL on the fixed validation set."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    try:
        for example in examples:
            ids = torch.tensor(
                [example["input_ids"]],
                dtype=torch.long,
                device=device,
            )
            labels = torch.tensor(
                [example["labels"]],
                dtype=torch.long,
                device=device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(ids[:, :-1])

            targets = labels[:, 1:]
            logits = logits.float()

            if logits.shape[:2] != targets.shape:
                raise ValueError(
                    f"Evaluation shape mismatch: "
                    f"logits={logits.shape}, targets={targets.shape}"
                )

            count = int((targets != -100).sum().item())
            if count == 0:
                continue

            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

            if not torch.isfinite(loss_sum):
                raise RuntimeError(
                    "Validation loss became non-finite"
                )

            total_loss += loss_sum.item()
            total_tokens += count
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise RuntimeError("Validation has no supervised tokens")

    return total_loss / total_tokens, total_tokens


def truncate_metrics(path: Path, offset: int) -> None:
    """Remove log rows that were written after the saved checkpoint."""
    if offset < 0:
        raise ValueError(f"Invalid metrics offset: {offset}")

    if not path.exists():
        if offset != 0:
            raise RuntimeError(
                "Metrics file is missing but checkpoint expects "
                f"{offset} bytes."
            )
        path.touch()
        return

    size = path.stat().st_size
    if size < offset:
        raise RuntimeError(
            f"Metrics file is shorter than checkpoint offset: "
            f"{size} < {offset}"
        )

    with path.open("r+b") as f:
        f.truncate(offset)


def main():
    parser = argparse.ArgumentParser(
        description="SFT training for T, D, and P arms"
    )
    parser.add_argument(
        "--arm", choices=["T", "D", "P"], required=True
    )
    parser.add_argument(
        "--seed", type=int, choices=[0, 1], required=True
    )
    parser.add_argument(
        "--budget", type=int, default=DEFAULT_BUDGET
    )
    parser.add_argument(
        "--eval-every", type=int, default=DEFAULT_EVAL_EVERY
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR
    )
    args = parser.parse_args()

    if args.budget <= 0:
        raise ValueError("--budget must be positive")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = "cuda"
    run_name = f"{args.arm}_s{args.seed}"

    train_file = DATA_DIR / "pilot_train.jsonl"
    val_file = DATA_DIR / "pilot_val.jsonl"
    pretrained_file = PRETRAIN_DIR / run_name / "final.pt"

    for path in (train_file, val_file, pretrained_file):
        if not path.exists():
            raise FileNotFoundError(path)

    train_examples = read_jsonl(train_file)
    val_examples = read_jsonl(val_file)
    validate_examples(train_examples, "Training")
    validate_examples(val_examples, "Validation")

    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = run_dir / "metrics.jsonl"
    config_file = run_dir / "run_config.json"
    resume_file = run_dir / "latest.pt"
    final_file = run_dir / "final.pt"
    done_file = run_dir / "done.json"

    config = {
        "arm": args.arm,
        "seed": args.seed,
        "learning_rate": args.lr,
        "token_budget": args.budget,
        "eval_every_tokens": args.eval_every,
        "train_sha256": sha256_file(train_file),
        "validation_sha256": sha256_file(val_file),
        "pretrained_sha256": sha256_file(pretrained_file),
    }

    if done_file.exists():
        raise RuntimeError(
            f"{run_name} already completed in {run_dir}. "
            "Choose a fresh output directory for a new run."
        )

    if config_file.exists():
        saved_config = json.loads(
            config_file.read_text(encoding="utf-8")
        )
        if saved_config != config:
            raise RuntimeError(
                "Run configuration mismatch. Use a fresh output directory."
            )
    else:
        if resume_file.exists() or metrics_file.exists():
            raise RuntimeError(
                "Partial run files exist without a run configuration. "
                "Use a fresh output directory."
            )

        config_file.write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

    print(f"Run: {run_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Learning rate: {args.lr:g}")
    print(f"Supervised-token budget: {args.budget:,}")
    print(f"Output: {run_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, _ = load_checkpoint(
        str(pretrained_file),
        device=device,
    )
    optimizer = make_optimizer(model, args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    order = list(range(len(train_examples)))
    random.shuffle(order)

    cursor = 0
    epoch = 0
    step = 0
    tokens_seen = 0
    next_eval = args.eval_every
    started = time.time()
    consecutive_overflows = 0

    if resume_file.exists():
        state = torch.load(
            resume_file,
            map_location="cpu",
            weights_only=False,
        )

        if state["config"] != config:
            raise RuntimeError("Resume checkpoint configuration mismatch")
        if "metrics_offset" not in state:
            raise RuntimeError(
                "This checkpoint predates metrics-safe recovery. "
                "Use a fresh output directory."
            )

        truncate_metrics(metrics_file, state["metrics_offset"])

        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])

        order = state["order"]
        cursor = state["cursor"]
        epoch = state["epoch"]
        step = state["step"]
        tokens_seen = state["tokens_seen"]
        next_eval = state["next_eval"]

        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all(
            [rng.cpu() for rng in state["cuda_rng"]]
        )

        started = time.time() - state["elapsed_seconds"]

        print(
            f"Resuming at step {step}, "
            f"supervised tokens={tokens_seen:,}"
        )
    else:
        truncate_metrics(metrics_file, 0)

    def log(row: dict) -> None:
        with metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()

    def save_resume() -> None:
        # The checkpoint and this byte offset define the committed log state.
        offset = metrics_file.stat().st_size if metrics_file.exists() else 0

        state = {
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "order": order,
            "cursor": cursor,
            "epoch": epoch,
            "step": step,
            "tokens_seen": tokens_seen,
            "next_eval": next_eval,
            "metrics_offset": offset,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "elapsed_seconds": time.time() - started,
        }

        tmp_file = resume_file.with_suffix(".tmp")
        torch.save(state, tmp_file)
        tmp_file.replace(resume_file)

    if not resume_file.exists():
        baseline_nll, val_tokens = evaluate(
            model,
            val_examples,
            device,
        )

        log({
            "type": "val",
            "step": 0,
            "tokens_seen": 0,
            "val_nll": baseline_nll,
            "val_perplexity": math.exp(baseline_nll),
            "val_tokens": val_tokens,
        })

        print(f"Baseline validation NLL: {baseline_nll:.6f}")
        save_resume()

    model.train()

    while tokens_seen < args.budget:
        if cursor >= len(order):
            epoch += 1
            cursor = 0
            random.shuffle(order)

        remaining = args.budget - tokens_seen
        example = train_examples[order[cursor]]
        cursor += 1

        ids, labels, batch_tokens = make_single_example(
            example,
            device,
            remaining,
        )

        if batch_tokens <= 0:
            raise RuntimeError("Training example has no supervised targets")

        targets = labels[:, 1:]
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            logits = model(ids[:, :-1])

        logits = logits.float()

        if logits.shape[:2] != targets.shape:
            raise ValueError(
                f"Training shape mismatch: "
                f"logits={logits.shape}, targets={targets.shape}"
            )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )

        if not torch.isfinite(loss):
            save_resume()
            raise RuntimeError(
                f"Non-finite training loss at step {step}. "
                "Checkpoint saved for diagnosis."
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
            error_if_nonfinite=False,
        )

        if not torch.isfinite(grad_norm):
            # GradScaler records non-finite gradients during unscale_.
            # step() skips the unsafe update and update() reduces the scale.
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            tokens_seen += batch_tokens
            consecutive_overflows += 1

            log({
                "type": "train_skipped_overflow",
                "step": step,
                "tokens_seen": tokens_seen,
                "batch_supervised_tokens": batch_tokens,
                "loss": loss.item(),
                "grad_norm": None,
                "learning_rate": args.lr,
                "scaler_scale": scaler.get_scale(),
                "elapsed_seconds": time.time() - started,
            })

            print(
                f"[WARNING] {run_name}: non-finite gradients; "
                f"optimizer update skipped. "
                f"Consecutive overflows: {consecutive_overflows}/"
                f"{MAX_CONSECUTIVE_OVERFLOWS}; "
                f"scaler={scaler.get_scale():g}",
                flush=True,
            )

            should_evaluate = tokens_seen >= next_eval

            if should_evaluate:
                val_nll, val_tokens = evaluate(
                    model,
                    val_examples,
                    device,
                )
                log({
                    "type": "val",
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "val_nll": val_nll,
                    "val_perplexity": math.exp(val_nll),
                    "val_tokens": val_tokens,
                })

                while next_eval <= tokens_seen:
                    next_eval += args.eval_every

            save_resume()

            if consecutive_overflows >= MAX_CONSECUTIVE_OVERFLOWS:
                raise RuntimeError(
                    "Five consecutive non-finite gradient norms occurred. "
                    "Training stopped; inspect the log and scaler behavior "
                    "before continuing."
                )

            continue

        consecutive_overflows = 0
        scaler.step(optimizer)
        scaler.update()

        step += 1
        tokens_seen += batch_tokens

        log({
            "type": "train",
            "step": step,
            "tokens_seen": tokens_seen,
            "batch_supervised_tokens": batch_tokens,
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "learning_rate": args.lr,
            "scaler_scale": scaler.get_scale(),
            "elapsed_seconds": time.time() - started,
        })

        should_evaluate = tokens_seen >= next_eval

        if should_evaluate:
            val_nll, val_tokens = evaluate(
                model,
                val_examples,
                device,
            )

            log({
                "type": "val",
                "step": step,
                "tokens_seen": tokens_seen,
                "val_nll": val_nll,
                "val_perplexity": math.exp(val_nll),
                "val_tokens": val_tokens,
            })

            print(
                f"[{run_name}] step={step} "
                f"tokens={tokens_seen:,}/{args.budget:,} "
                f"train_loss={loss.item():.4f} "
                f"val_nll={val_nll:.4f}",
                flush=True,
            )

            while next_eval <= tokens_seen:
                next_eval += args.eval_every

        elif step % 10 == 0 or tokens_seen >= args.budget:
            print(
                f"[{run_name}] step={step} "
                f"tokens={tokens_seen:,}/{args.budget:,} "
                f"loss={loss.item():.4f}",
                flush=True,
            )

        if should_evaluate or step % 10 == 0 or tokens_seen >= args.budget:
            save_resume()

    final_nll, val_tokens = evaluate(
        model,
        val_examples,
        device,
    )

    log({
        "type": "val_final",
        "step": step,
        "tokens_seen": tokens_seen,
        "val_nll": final_nll,
        "val_perplexity": math.exp(final_nll),
        "val_tokens": val_tokens,
    })

    tmp_final = run_dir / "final.tmp"

    save_checkpoint(
        str(tmp_final),
        model,
        optimizer=None,
        step=step,
        extra={
            "seed": args.seed,
            "tokens_seen": tokens_seen,
            "sft_config": config,
            "final_val_nll": final_nll,
        },
    )

    tmp_final.replace(final_file)

    summary = {
        "run": run_name,
        "step": step,
        "tokens_seen": tokens_seen,
        "learning_rate": args.lr,
        "final_val_nll": final_nll,
        "final_val_perplexity": math.exp(final_nll),
        "elapsed_seconds": time.time() - started,
        "final_checkpoint": str(final_file),
    }

    done_file.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if resume_file.exists():
        resume_file.unlink()

    print("\nSFT RUN COMPLETE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
