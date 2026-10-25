
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import load_checkpoint
from models.diagnostics import lambda_log_dict

DATA_ROOT = Path(r"C:\Users\HARSHIT VAISH\projects\nexolve-data")
SFT_ROOT = DATA_ROOT / "sft_outputs"
VAL_FILE = DATA_ROOT / "sft" / "pilot_val.jsonl"
TOKENIZER_FILE = DATA_ROOT / "pretrain" / "tokenizer.json"
OUTPUT_FILE = SFT_ROOT / "lambda_post_training.json"


def main():
    from tokenizers import Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    # Use the same fixed validation prompts for both D checkpoints.
    examples = []
    with VAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids = row.get("input_ids")
            if ids is None:
                # If the JSONL stores text fields instead, encode the prompt.
                prompt = row.get("prompt", row.get("user", ""))
                ids = tokenizer.encode(prompt).ids
            if ids:
                examples.append(ids)

    if not examples:
        raise RuntimeError(f"No usable validation examples found in {VAL_FILE}")

    results = {
        "description": (
            "Post-training lambda diagnostics on fixed validation examples; "
            "these were not logged during SFT training."
        ),
        "device": device,
        "num_examples": len(examples),
        "runs": {},
    }

    for run in ("D_s0", "D_s1"):
        checkpoint = SFT_ROOT / run / "final.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        model, _ = load_checkpoint(str(checkpoint), device=device)
        model.eval()

        # Diagnostics are computed per example to avoid GPU memory spikes.
        rows = []
        for ids in examples:
            # Respect model context length.
            ids = ids[-1024:]
            if not ids:
                continue
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            rows.append(lambda_log_dict(model, input_ids))

        if not rows:
            raise RuntimeError(f"No diagnostic rows generated for {run}")

        keys = rows[0].keys()
        averaged = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
        results["runs"][run] = {
            "checkpoint": str(checkpoint),
            "examples_processed": len(rows),
            "mean_of_per_example_statistics": averaged,
        }
        print(f"\n{run}")
        for key, value in averaged.items():
            print(f"  {key}: {value:.6f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUTPUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved post-training diagnostics to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
