import hashlib
import json
import re
import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from pretraining.tokenizer import load_tokenizer

# ---------- Configuration ----------
DATASET_NAME = "HuggingFaceTB/smoltalk"
DATASET_CONFIG = "smol-magpie-ultra"

DATA_ROOT = Path(r"C:\Users\HARSHIT VAISH\projects\nexolve-data")
TOKENIZER_PATH = DATA_ROOT / "pretrain" / "tokenizer.json"
OUTPUT_DIR = DATA_ROOT / "sft"

MAX_RECORDS = 10_000
MAX_USER_TOKENS = 200
MAX_RESPONSE_TOKENS = 300
MAX_SEQUENCE_TOKENS = 1024

USER_ID = 3
ASSISTANT_ID = 4
END_ID = 5
PAD_ID = 1
IGNORE_INDEX = -100

# This is a pilot dataset, not the final experiment dataset.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_english_enough(text):
    """Conservative heuristic; this is not a language-identification model."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False

    ascii_letters = sum(c.isascii() and c.isalpha() for c in letters)
    return ascii_letters / len(letters) >= 0.95


def token_ids(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False).ids


def main():
    tokenizer = load_tokenizer(
        str(TOKENIZER_PATH),
        vocab_size=16384,
    )

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
    )

    examples = []
    seen = set()

    stats = {
        "scanned": 0,
        "invalid_structure": 0,
        "non_english": 0,
        "too_long": 0,
        "duplicates": 0,
        "kept": 0,
    }

    for record in dataset:
        if stats["scanned"] >= MAX_RECORDS:
            break

        stats["scanned"] += 1
        messages = record.get("messages", [])

        # Use only the first user-assistant pair.
        if len(messages) < 2:
            stats["invalid_structure"] += 1
            continue

        if (
            messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
        ):
            stats["invalid_structure"] += 1
            continue

        user_text = messages[0].get("content", "")
        assistant_text = messages[1].get("content", "")

        if not isinstance(user_text, str) or not isinstance(assistant_text, str):
            stats["invalid_structure"] += 1
            continue

        user_text = user_text.strip()
        assistant_text = assistant_text.strip()

        if not user_text or not assistant_text:
            stats["invalid_structure"] += 1
            continue

        if not is_english_enough(user_text) or not is_english_enough(assistant_text):
            stats["non_english"] += 1
            continue

        user_ids = token_ids(tokenizer, user_text)
        response_ids = token_ids(tokenizer, assistant_text)

        if (
            len(user_ids) > MAX_USER_TOKENS
            or len(response_ids) > MAX_RESPONSE_TOKENS
        ):
            stats["too_long"] += 1
            continue

        # Format: <|user|> prompt <|assistant|> response <|end|>
        input_ids = (
            [USER_ID]
            + user_ids
            + [ASSISTANT_ID]
            + response_ids
            + [END_ID]
        )

        if len(input_ids) > MAX_SEQUENCE_TOKENS:
            stats["too_long"] += 1
            continue

        # Mask prompt tokens. The assistant response and END token
        # are the only tokens whose next-token targets contribute to loss.
        assistant_start = len(user_ids) + 2
        labels = [IGNORE_INDEX] * assistant_start + response_ids + [END_ID]

        # Deduplicate by the normalized first-turn content.
        key = hashlib.sha256(
            (user_text + "\n" + assistant_text).encode("utf-8")
        ).hexdigest()

        if key in seen:
            stats["duplicates"] += 1
            continue

        seen.add(key)

        examples.append({
            "input_ids": input_ids,
            "labels": labels,
            "user_token_count": len(user_ids),
            "response_token_count": len(response_ids),
            "sequence_token_count": len(input_ids),
            "source_config": DATASET_CONFIG,
            "source_record_index": stats["scanned"] - 1,
            "example_hash": key,
        })
        stats["kept"] += 1

        if stats["scanned"] % 1000 == 0:
            print(
                f"Scanned {stats['scanned']:,} | "
                f"kept {stats['kept']:,}"
            )

    output_path = OUTPUT_DIR / "pilot.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "max_records": MAX_RECORDS,
        "max_user_tokens": MAX_USER_TOKENS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "special_token_ids": {
            "user": USER_ID,
            "assistant": ASSISTANT_ID,
            "end": END_ID,
            "pad": PAD_ID,
        },
        "ignore_index": IGNORE_INDEX,
        "stats": stats,
        "pilot_file": output_path.name,
        "pilot_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }

    manifest_path = OUTPUT_DIR / "pilot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n--- PREPARATION SUMMARY ---")
    print(json.dumps(stats, indent=2))
    print("Output:", output_path)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
