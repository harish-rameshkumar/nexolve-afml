
import hashlib
import json
import random
from pathlib import Path

DATA_DIR = Path(
    r"C:\Users\HARSHIT VAISH\projects\nexolve-data\sft"
)

INPUT_FILE = DATA_DIR / "pilot.jsonl"
TRAIN_FILE = DATA_DIR / "pilot_train.jsonl"
VAL_FILE = DATA_DIR / "pilot_val.jsonl"
SEED = 42
VAL_FRACTION = 0.10

with INPUT_FILE.open(encoding="utf-8") as f:
    examples = [json.loads(line) for line in f if line.strip()]

# Deterministic shuffle: same split every time.
indices = list(range(len(examples)))
random.Random(SEED).shuffle(indices)

n_val = max(1, round(len(examples) * VAL_FRACTION))
val_indices = set(indices[:n_val])

train = [ex for i, ex in enumerate(examples) if i not in val_indices]
val = [ex for i, ex in enumerate(examples) if i in val_indices]

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

write_jsonl(TRAIN_FILE, train)
write_jsonl(VAL_FILE, val)

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("Total examples:", len(examples))
print("Training examples:", len(train))
print("Validation examples:", len(val))
print("Train SHA256:", file_hash(TRAIN_FILE))
print("Validation SHA256:", file_hash(VAL_FILE))
print("Split seed:", SEED)
print("Pilot split: PASS")
