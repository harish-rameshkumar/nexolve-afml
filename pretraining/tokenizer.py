"""Byte-level BPE tokenizer (vocab 16,384) shared by every workstream.

IMPORTANT FOR WS3 (SFT): the chat-format special tokens are reserved HERE, at
tokenizer-training time, with fixed ids. Pretraining never uses them (they have
no training signal and their embeddings stay near init), but SFT needs them, and
adding tokens later would change the vocab size and break the checkpoints.

    id 0  <|endoftext|>   document separator in pretraining; also end of an SFT response
    id 1  <|pad|>         padding (never trained on)
    id 2  <|system|>
    id 3  <|user|>
    id 4  <|assistant|>
    id 5  <|end|>         end of a turn
    id 6-7 <|reserved0|>, <|reserved1|>   spare, so WS3/WS4 can add a role without a vocab change
"""
from __future__ import annotations

from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 16384

SPECIAL_TOKENS = [
    "<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>",
    "<|assistant|>", "<|end|>", "<|reserved0|>", "<|reserved1|>",
]
EOS_ID = 0
PAD_ID = 1


def train_tokenizer(texts: Iterable[str], vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """Train byte-level BPE on an iterator of strings (GPT-2 style pre-tokenisation)."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,                       # includes the special tokens
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes: no <unk> ever
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    assert tok.get_vocab_size() == vocab_size, (
        f"vocab is {tok.get_vocab_size()}, expected {vocab_size}: corpus too small for this vocab size"
    )
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(t) == i, f"special token {t} has id {tok.token_to_id(t)}, expected {i}"
    return tok


def load_tokenizer(path: str, vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    tok = Tokenizer.from_file(path)
    assert tok.get_vocab_size() == vocab_size, f"tokenizer vocab {tok.get_vocab_size()} != {vocab_size}"
    return tok
