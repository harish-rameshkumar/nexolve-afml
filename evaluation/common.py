"""Stable hashes, explicit validation and atomic JSON."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

def require(condition, message):
    if not condition:
        raise ValueError(message)

def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name}: expected integer >= {minimum}")
    return value

def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"{name}: expected finite number")
    return float(value)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def verify_file(path, expected):
    require(isinstance(expected, str) and len(expected) == 64, f"Missing SHA-256 for {path}")
    require(sha256_file(path) == expected, f"SHA-256 mismatch: {path}")

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def resolve(base, name):
    path = Path(name)
    return path if path.is_absolute() else Path(base) / path
