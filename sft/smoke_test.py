
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import load_checkpoint

DATA_ROOT = Path(r"C:\Users\HARSHIT VAISH\projects\nexolve-data")
CHECKPOINT = DATA_ROOT / "outputs" / "T_s0" / "final.pt"
PILOT = DATA_ROOT / "sft" / "pilot.jsonl"

device = "cuda"
torch.manual_seed(0)

# Load one real pilot example.
with PILOT.open(encoding="utf-8") as f:
    example = json.loads(f.readline())

input_ids = torch.tensor(
    [example["input_ids"]], dtype=torch.long, device=device
)
labels = torch.tensor(
    [example["labels"]], dtype=torch.long, device=device
)

# Load the actual pretrained checkpoint.
model, ckpt = load_checkpoint(str(CHECKPOINT), device=device)
model.train()

# Forward pass predicts next tokens.
logits = model(input_ids[:, :-1])
shift_labels = labels[:, 1:]

# Ensure at least one target contributes to loss.
valid = shift_labels != -100
assert valid.any(), "No supervised response targets"

loss = F.cross_entropy(
    logits.float().reshape(-1, logits.shape[-1]),
    shift_labels.reshape(-1),
    ignore_index=-100,
)

assert torch.isfinite(loss), "Non-finite SFT loss"

loss.backward()
torch.cuda.synchronize()

print("Checkpoint arm:", ckpt["arm"])
print("Sequence length:", input_ids.shape[1])
print("Supervised shifted tokens:", int(valid.sum().item()))
print("SFT loss:", round(loss.item(), 4))
print("Loss finite:", torch.isfinite(loss).item())
print("Gradients present:", any(
    p.grad is not None for p in model.parameters()
))
print("Allocated VRAM (MB):", round(
    torch.cuda.memory_allocated() / 1024**2, 1
))
print("Real SFT smoke test: PASS")
