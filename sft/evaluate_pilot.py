
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Make the repository root importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import load_checkpoint


DATA_DIR = Path(
    r"C:\Users\HARSHIT VAISH\projects\nexolve-data\sft"
)

CHECKPOINT = Path(
    r"C:\Users\HARSHIT VAISH\projects\nexolve-data\outputs\T_s0\final.pt"
)

VAL_FILE = DATA_DIR / "pilot_val.jsonl"


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    if not VAL_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VAL_FILE}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Activate the correct .venv "
            "and verify the PyTorch CUDA installation."
        )

    device = "cuda"

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Validation file: {VAL_FILE}")
    print("Loading checkpoint...")

    model, ckpt = load_checkpoint(
        str(CHECKPOINT),
        device=device,
        optimizer=None,
    )
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_examples = 0

    with VAL_FILE.open("r", encoding="utf-8") as f:
        with torch.no_grad():
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                ex = json.loads(line)

                ids = torch.tensor(
                    [ex["input_ids"]],
                    dtype=torch.long,
                    device=device,
                )

                labels = torch.tensor(
                    [ex["labels"]],
                    dtype=torch.long,
                    device=device,
                )

                if ids.shape != labels.shape:
                    raise ValueError(
                        f"Example {line_number}: input/label shape "
                        f"mismatch: {ids.shape} vs {labels.shape}"
                    )

                # Predict the next token from all preceding tokens.
                # Both tensors now have sequence length L - 1.
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = model(ids[:, :-1])

                shifted_logits = logits.float()
                shifted_labels = labels[:, 1:]

                if shifted_logits.shape[:2] != shifted_labels.shape:
                    raise ValueError(
                        f"Example {line_number}: shifted shape mismatch: "
                        f"logits={shifted_logits.shape}, "
                        f"labels={shifted_labels.shape}"
                    )

                supervised_mask = shifted_labels != -100
                token_count = int(supervised_mask.sum().item())

                if token_count == 0:
                    raise ValueError(
                        f"Example {line_number} has no supervised tokens."
                    )

                # Sum loss only over supervised assistant-response tokens.
                # Divide by the total number of supervised tokens at the end.
                loss_sum = F.cross_entropy(
                    shifted_logits.reshape(
                        -1, shifted_logits.shape[-1]
                    ),
                    shifted_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

                total_loss += loss_sum.item()
                total_tokens += token_count
                total_examples += 1

                if total_examples % 25 == 0:
                    print(
                        f"Evaluated {total_examples} examples..."
                    )

    if total_examples == 0 or total_tokens == 0:
        raise RuntimeError(
            "No valid validation examples or supervised tokens found."
        )

    mean_nll = total_loss / total_tokens

    print("\n" + "=" * 45)
    print("BASELINE VALIDATION RESULTS")
    print("=" * 45)
    print(f"Arm: T")
    print(f"Checkpoint: {CHECKPOINT.parent.name}")
    print(f"Validation examples: {total_examples}")
    print(f"Supervised response tokens: {total_tokens}")
    print(f"Response-token NLL: {mean_nll:.6f}")
    print(f"Perplexity: {torch.exp(torch.tensor(mean_nll)).item():.4f}")
    print("Evaluation: PASS")
    print("=" * 45)


if __name__ == "__main__":
    main()
