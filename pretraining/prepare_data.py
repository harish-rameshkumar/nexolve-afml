"""Build the frozen pretraining data: tokenizer + packed uint16 token files.

Run ONCE, on ONE machine, then share the outputs (data/pretrain/*) with the team
(Kaggle dataset / drive). Every run on every account must read the same files;
`meta.json` stores sizes and SHA-256 so this can be checked.

    python -m pretraining.prepare_data --out data/pretrain

Split rule (a pure function of the document's position in the stream, so it is
reproducible and no document can be in both splits):
    doc index i % val_every == 0  ->  validation candidate (kept until val_tokens is reached)
    everything else               ->  train (until train_tokens is reached)

Output files in --out:
    tokenizer.json   byte-level BPE, vocab 16,384 (see tokenizer.py for special-token ids)
    train.bin        uint16 tokens, documents joined by <|endoftext|>
    val.bin          uint16 tokens, same format  (the FROZEN held-out pretraining split)
    meta.json        counts, sizes, sha256, source dataset info

For an offline smoke test use --local_jsonl some.jsonl (one {"text": ...} per line).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from typing import Iterator

import numpy as np

from pretraining.tokenizer import EOS_ID, VOCAB_SIZE, load_tokenizer, train_tokenizer


def iter_docs(args) -> Iterator[tuple[int, str]]:
    """Yield (doc_index, text) in a deterministic order."""
    if args.local_jsonl:
        def gen():
            with open(args.local_jsonl) as f:
                for line in f:
                    yield json.loads(line)["text"]
    else:
        from datasets import load_dataset  # imported lazily: not needed for local tests

        ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=True)
        def gen():
            for row in ds:
                yield row["text"]
    return enumerate(gen())


def is_val_candidate(i: int, val_every: int) -> bool:
    return i % val_every == 0


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_tokenizer(args, out_dir: str):
    path = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(path) and not args.retrain_tokenizer:
        print(f"[tokenizer] found {path}, reusing (tokenizer is frozen; pass --retrain_tokenizer to rebuild)")
        return load_tokenizer(path, args.vocab_size)
    print(f"[tokenizer] training on the first {args.tok_docs:,} train-split docs ...")
    t0 = time.time()
    texts = (t for i, t in iter_docs(args) if not is_val_candidate(i, args.val_every))
    tok = train_tokenizer(itertools.islice(texts, args.tok_docs), args.vocab_size)
    tok.save(path)
    print(f"[tokenizer] done in {time.time() - t0:.0f}s -> {path}")
    return tok


def tokenize_corpus(args, tok, out_dir: str) -> dict:
    train_path, val_path = os.path.join(out_dir, "train.bin"), os.path.join(out_dir, "val.bin")
    counts = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    targets = {"train": args.train_tokens, "val": args.val_tokens}
    files = {"train": open(train_path, "wb"), "val": open(val_path, "wb")}
    t0, n_seen = time.time(), 0

    def flush(batch):
        """batch: list of (split, text). Tokenise together (multi-threaded in Rust)."""
        encs = tok.encode_batch([t for _, t in batch])
        for (split, _), enc in zip(batch, encs):
            if counts[split] >= targets[split]:
                continue
            ids = np.asarray(enc.ids + [EOS_ID], dtype=np.uint16)
            take = min(len(ids), targets[split] - counts[split])  # trim last doc to hit target exactly
            files[split].write(ids[:take].tobytes())
            counts[split] += take
            docs[split] += 1

    batch = []
    for i, text in iter_docs(args):
        n_seen = i + 1
        split = "val" if is_val_candidate(i, args.val_every) else "train"
        if counts[split] < targets[split]:
            batch.append((split, text))
        if len(batch) >= args.batch_docs:
            flush(batch); batch = []
            if (n_seen // args.batch_docs) % 20 == 0:
                el = time.time() - t0
                print(f"[tokenize] docs seen {n_seen:,} | train {counts['train']/1e6:.1f}M/{targets['train']/1e6:.0f}M "
                      f"| val {counts['val']/1e6:.2f}M/{targets['val']/1e6:.1f}M | {el:.0f}s", flush=True)
        if all(counts[s] >= targets[s] for s in counts):
            break
    if batch:
        flush(batch)
    for f in files.values():
        f.close()
    if any(counts[s] < targets[s] for s in counts):
        print(f"WARNING: source ended before targets were reached: {counts} vs {targets}", file=sys.stderr)
    return {"counts": counts, "docs": docs, "docs_seen": n_seen}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/pretrain")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--local_jsonl", default=None, help="offline source for smoke tests")
    # 4,600 steps x 65,536 tokens = 301.5M; keep a little spare so the budget can be raised slightly
    p.add_argument("--train_tokens", type=int, default=310_000_000)
    p.add_argument("--val_tokens", type=int, default=2_000_000)
    p.add_argument("--val_every", type=int, default=100)
    p.add_argument("--tok_docs", type=int, default=100_000, help="docs used to train the tokenizer")
    p.add_argument("--batch_docs", type=int, default=1000)
    p.add_argument("--vocab_size", type=int, default=VOCAB_SIZE,
                   help="must stay 16384 for the project; only tests use a smaller value")
    p.add_argument("--retrain_tokenizer", action="store_true")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    tok = build_tokenizer(args, args.out)
    stats = tokenize_corpus(args, tok, args.out)

    meta = {
        "source": args.local_jsonl or f"{args.dataset}/{args.subset}",
        "vocab_size": args.vocab_size,
        "val_every": args.val_every,
        "dtype": "uint16",
        "eos_id": EOS_ID,
        **stats,
        "files": {},
    }
    for name in ("tokenizer.json", "train.bin", "val.bin"):
        path = os.path.join(args.out, name)
        meta["files"][name] = {"bytes": os.path.getsize(path), "sha256": sha256_file(path)}
    try:
        import datasets
        meta["datasets_version"] = datasets.__version__
    except ImportError:
        pass
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in ("counts", "docs", "docs_seen")}, indent=2))
    print(f"wrote {args.out}/meta.json. Share this folder with the team; do NOT commit it to git.")


if __name__ == "__main__":
    main()
