
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import load_checkpoint

DATA_DIR = Path(
    r"C:\Users\HARSHIT VAISH\projects\nexolve-data\sft"
)
CHECKPOINT = Path(
    r"C:\Users\HARSHIT VAISH\projects\nexolve-data\outputs\T_s0\final.pt"
)
TRAIN_FILE = DATA_DIR / "pilot_train.jsonl"

LEARNING_RATE = 0.002


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading T_s0 checkpoint...")

    model, _ = load_checkpoint(
        str(CHECKPOINT),
        device=device,
        optimizer=None,
    )
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Load one real training example.
    with TRAIN_FILE.open("r", encoding="utf-8") as f:
        example = json.loads(next(line for line in f if line.strip()))

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

    # Predict the next token. Masked prompt tokens do not contribute to loss.
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(ids[:, :-1])

    targets = labels[:, 1:]
    logits = logits.float()

    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"Shape mismatch: logits={logits.shape}, targets={targets.shape}"
        )

    supervised_tokens = int((targets != -100).sum().item())
    if supervised_tokens == 0:
        raise ValueError("No supervised response tokens in example.")

    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss.item()}")

    loss.backward()

    grad_params = [
        p for p in model.parameters()
        if p.grad is not None
    ]
    if not grad_params:
        raise RuntimeError("No gradients were produced.")

    for p in grad_params:
        if not torch.isfinite(p.grad).all():
            raise RuntimeError("Non-finite gradient detected.")

    # Record a parameter before the update to verify that it changes.
    first_param = next(p for p in model.parameters() if p.requires_grad)
    before = first_param.detach().clone()

    optimizer.step()

    changed = not torch.equal(before, first_param.detach())
    if not changed:
        raise RuntimeError("Optimizer step did not change the parameter.")

    print("\n" + "=" * 42)
    print("SFT TRAINING SMOKE TEST: PASS")
    print("=" * 42)
    print(f"Arm: T")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Sequence length: {ids.shape[1]}")
    print(f"Supervised response tokens: {supervised_tokens}")
    print(f"Loss before update: {loss.item():.6f}")
    print(f"Parameters with gradients: {len(grad_params)}")
    print(f"Optimizer changed parameters: {changed}")
    print(f"Allocated VRAM (MB): {torch.cuda.memory_allocated() / 1024**2:.1f}")
    print("Pretrained checkpoint file was not modified.")
    print("=" * 42)


if __name__ == "__main__":
    main()
