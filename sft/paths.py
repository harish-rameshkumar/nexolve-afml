"""Paths shared by the pilot fine-tuning commands."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configured_path(variable, default):
    """Use a supplied path or a default anchored to the repository."""
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else PROJECT_ROOT / default


DATA_DIR = configured_path("NEXOLVE_SFT_DATA", "data/sft")
PRETRAIN_DIR = configured_path("NEXOLVE_PRETRAINED", "outputs/pretrain")
TOKENIZER_PATH = configured_path("NEXOLVE_TOKENIZER", "data/pretrain/tokenizer.json")
OUTPUT_DIR = configured_path("NEXOLVE_SFT_OUTPUT", "outputs/sft")
