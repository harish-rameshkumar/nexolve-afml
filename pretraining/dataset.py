"""Deterministic packed token stream + frozen validation loader.

Training data order is a pure function of (seed, step): the packed train.bin is cut
into windows of seq_len+1 tokens (stride seq_len, so every token is used once as a
target), the windows are permuted with numpy's default_rng(seed), and optimiser step
`s` consumes windows [s*B, (s+1)*B) of that permutation (B = micro_batch*grad_accum).

Consequences:
  * T, D and P with the same seed see exactly the same batches in the same order.
  * Resuming from step s needs no data-loader state: just ask for step s again.
  * Documents are packed back to back (separated by <|endoftext|>) with no
    cross-document attention mask, as in nanoGPT/GPT-style pretraining.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch


class PackedTokens:
    def __init__(self, path: str, seq_len: int):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n_windows = (len(self.tokens) - 1) // seq_len   # window = seq_len + 1 tokens

    def window(self, k: int) -> np.ndarray:
        s = k * self.seq_len
        return np.asarray(self.tokens[s: s + self.seq_len + 1], dtype=np.int64)

    def batch(self, idx) -> torch.Tensor:
        """(len(idx), seq_len+1) long tensor."""
        return torch.from_numpy(np.stack([self.window(int(k)) for k in idx]))


class TrainStream:
    """Seeded, step-addressable training batches."""

    def __init__(self, data_dir: str, seq_len: int, seed: int, batch_size: int, max_steps: int):
        self.data = PackedTokens(os.path.join(data_dir, "train.bin"), seq_len)
        self.batch_size = batch_size
        need = batch_size * max_steps
        assert self.data.n_windows >= need, (
            f"train.bin has {self.data.n_windows:,} windows but {need:,} are needed "
            f"({max_steps} steps x {batch_size}); re-run prepare_data with more --train_tokens "
            f"or lower --max_steps"
        )
        # Permute ALL windows (not just the first `need`) so the order does not depend on
        # max_steps: a short pilot run sees the same first batches as the full run.
        self.order = np.random.default_rng(seed).permutation(self.data.n_windows)

    def get(self, step: int, micro: int, micro_batch: int) -> torch.Tensor:
        """Micro-batch `micro` of optimiser step `step`: (micro_batch, seq_len+1)."""
        start = step * self.batch_size + micro * micro_batch
        return self.data.batch(self.order[start: start + micro_batch])


def load_val_tokens(data_dir: str, seq_len: int) -> PackedTokens:
    return PackedTokens(os.path.join(data_dir, "val.bin"), seq_len)


def load_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def check_data_files(data_dir: str, verify_sha: bool = False) -> dict:
    """Make sure this machine's data files are the ones described by meta.json."""
    meta = load_meta(data_dir)
    for name, info in meta["files"].items():
        path = os.path.join(data_dir, name)
        assert os.path.getsize(path) == info["bytes"], f"{name}: size differs from meta.json (wrong/partial copy?)"
        if verify_sha:
            from pretraining.prepare_data import sha256_file
            assert sha256_file(path) == info["sha256"], f"{name}: sha256 differs from meta.json"
    return meta
