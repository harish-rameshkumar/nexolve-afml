"""Adapters for the actual WS3 tokenized splits, final files and recovery files."""
from pathlib import Path
import random
from evaluation.common import canonical_hash, integer, read_json, require, sha256_file

TEMPLATE = "nexolve_ws3_v1"

def read_jsonl(path):
    rows = [__import__("json").loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    require(bool(rows), f"Empty JSONL: {path}")
    return rows

def validate_tokenized(rows, vocab_size, context_len):
    encoded = []
    for i, row in enumerate(rows):
        ids, labels = row["input_ids"], row["labels"]
        require(isinstance(ids, list) and isinstance(labels, list) and len(ids) == len(labels)
            and 4 <= len(ids) <= context_len, f"Invalid WS3 sequence {i}")
        require(all(type(t) is int and 0 <= t < vocab_size for t in ids), f"Out-of-vocabulary token in row {i}")
        require(all(type(t) is int for t in labels), f"Noninteger labels in row {i}")
        supervised = [j for j, value in enumerate(labels) if value != -100]
        require(bool(supervised), f"All-masked WS3 row {i}")
        start = supervised[0]
        require(start >= 2 and ids[0] == 3 and ids[start-1] == 4, f"Unexpected WS3 prompt format in row {i}")
        require(supervised == list(range(start, len(ids))) and labels[start:] == ids[start:],
            f"Noncontiguous/misaligned WS3 response in row {i}")
        require(ids[-1] == 5, f"WS3 END terminator missing in row {i}")
        encoded.append({"id": f"ws3-{i}", "ids": ids, "labels": labels, "assistant_start": start})
    require(bool(encoded), "Empty WS3 split")
    return encoded

def decoded_examples(encoded, tok):
    return [{"prompt": tok.decode(r["ids"][1:r["assistant_start"]-1], skip_special_tokens=False),
        "response": tok.decode(r["ids"][r["assistant_start"]:-1], skip_special_tokens=False)} for r in encoded]

def audit_training_log(config, logs, training_rows):
    """Replay example order/counters only, NOT optimizer updates or numerical training."""
    rows = [r for r in logs if r["type"] in ("train", "train_skipped_overflow")]
    require(bool(rows), "No training records")
    order = list(range(len(training_rows)))
    rng = random.Random(config["seed"]); rng.shuffle(order)
    cursor, attempted, successful, epoch = 0, 0, 0, 0
    skipped = []
    for step, row in enumerate(rows, 1):
        require(row["step"] == step, "Noncontiguous training log")
        if cursor == len(order):
            cursor = 0; epoch += 1; rng.shuffle(order)
        example = training_rows[order[cursor]]; cursor += 1
        count = min(sum(v != -100 for v in example["labels"][1:]), config["token_budget"]-attempted)
        require(count > 0 and row["batch_supervised_tokens"] == count, "Recorded response count differs from replayed stream")
        attempted += count
        require(row["tokens_seen"] == attempted, "Token counter mismatch")
        if row["type"] == "train":
            successful += count
        else:
            skipped.append({"step": step, "response_tokens": count})
    require(epoch == 0, "Repeated epochs: unique-token interpretation needs a separate positional audit")
    require(attempted == config["token_budget"], "Incomplete final run")
    return {"attempted_response_tokens": attempted, "successful_update_response_tokens": successful,
        "skipped_update_response_tokens": attempted-successful, "skipped_updates": skipped,
        "optimizer_attempts": len(rows), "successful_updates": len(rows)-len(skipped),
        "epoch": epoch, "stream_prefix_counter_replay": "passed",
        "numerical_training_reproduction": "not performed", "terminator_id": 5,
        "count_includes_end_token": True}

def load_ws3_checkpoint(path, parent_path, run_config, expected_kind="final", device="cpu"):
    """Rebuild recovery models from the actual verified WS2 parent, never a guessed config."""
    import torch
    from configs.model_config import ModelConfig
    from models import build_model
    require(sha256_file(parent_path) == run_config["pretrained_sha256"], "Pretrained parent hash mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    state = torch.load(path, map_location="cpu", weights_only=False)
    require(parent["arm"] == run_config["arm"], "Parent arm mismatch")
    require(parent["extra"]["seed"] == run_config["seed"], "Parent seed mismatch")
    if expected_kind == "final":
        require(state["arm"] == run_config["arm"] and state["extra"]["seed"] == run_config["seed"], "SFT identity mismatch")
        require(canonical_hash(state["extra"]["sft_config"]) == canonical_hash(run_config), "SFT config mismatch")
        require(state["config"] == parent["config"], "SFT architecture differs from parent")
        tokens = state["extra"]["tokens_seen"]
        require(tokens == run_config["token_budget"], "Final checkpoint token counter mismatch")
    else:
        require(expected_kind == "resume", "Unknown checkpoint kind")
        require(canonical_hash(state["config"]) == canonical_hash(run_config), "Recovery config mismatch")
        tokens = state["tokens_seen"]
    model = build_model(run_config["arm"], ModelConfig.from_dict(parent["config"]))
    model.load_state_dict(state["model"], strict=True)
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite model weights")
    model.to(device).eval()
    info = {"arm": run_config["arm"], "seed": run_config["seed"], "step": state["step"],
        "tokens_seen": tokens, "kind": expected_kind, "parent_model_config": parent["config"],
        "parameter_count": model.num_params(),
        "storage_dtypes": sorted({str(t.dtype) for t in state["model"].values()}),
        "logged_final_val_nll": state.get("extra", {}).get("final_val_nll")}
    del state, parent
    return model, info
