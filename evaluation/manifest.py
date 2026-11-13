"""Manifest validation independent of torch. Paths are relative to manifest.json."""
from evaluation.common import integer, require

def _hash(value, name):
    require(isinstance(value, str) and len(value) == 64
        and all(c in "0123456789abcdef" for c in value), f"Invalid SHA-256: {name}")

def validate_manifest(m, protocol):
    require(m.get("schema_version") == 1, "Unsupported manifest version")
    require(m.get("experiment") == protocol["experiment"], "Experiment mismatch")
    for name in ("tokenizer", "suite", "references", "val_tokens", "val_meta"):
        asset = m.get(name, {})
        require(isinstance(asset.get("path"), str) and bool(asset["path"]), f"Missing {name} path")
        _hash(asset.get("sha256"), name)
    if protocol["experiment"] != "synthetic":
        for name in ("sft_train", "development", "sft_recipe"):
            asset = m.get(name, {})
            require(isinstance(asset.get("path"), str), f"Missing {name} leakage-audit asset")
            _hash(asset.get("sha256"), name)
        require(m["sft_train"]["sha256"] == m.get("sft_stream_sha256"), "SFT stream must hash the ordered training audit asset")
        require(m.get("near_duplicate_review"), "Document near-duplicate review before real evaluation")
    require(m.get("unique_count_includes_eos") == protocol["unique_count_includes_eos"], "Unique-token EOS convention mismatch")
    rows = m.get("checkpoints")
    require(isinstance(rows, list) and bool(rows), "No checkpoints")
    identities, by_run = set(), {}
    for r in rows:
        require(r.get("arm") in ("T", "D", "P"), "Unknown arm")
        for name in ("pretrain_seed", "sft_seed", "step", "target_response_tokens",
                     "unique_response_tokens", "processed_response_tokens"):
            integer(r.get(name), name)
        require(r["pretrain_seed"] in protocol["paired_seeds"], "Unregistered seed")
        # One SFT recipe/seed per pretraining seed in v1; no pseudoreplication.
        require(r["processed_response_tokens"] == r["unique_response_tokens"],
            "v1 primary curve requires one pass, without repeated response exposures")
        require(r["target_response_tokens"] == r["unique_response_tokens"], "Exact budget grid required")
        require(r["unique_response_tokens"] == 0 or r["step"] > 0, "SFT checkpoint needs successful optimizer step")
        require(r["unique_response_tokens"] in protocol["budgets"], "Unregistered budget")
        require(r.get("phase") == ("pretrain" if r["unique_response_tokens"] == 0 else "sft"), "Invalid phase")
        for field in ("checkpoint_sha256", "parent_pretrain_sha256", "sft_stream_sha256", "chat_template_sha256"):
            _hash(r.get(field), field)
        require(r["chat_template_sha256"] == m.get("chat_template_sha256"), "Mixed templates")
        require(r["sft_stream_sha256"] == m.get("sft_stream_sha256"), "Mixed SFT streams")
        require(isinstance(r.get("checkpoint_path"), str) and bool(r["checkpoint_path"]), "Missing checkpoint path")
        identity = (r["arm"], r["pretrain_seed"], r["unique_response_tokens"])
        require(identity not in identities, "Duplicate checkpoint identity")
        identities.add(identity)
        by_run.setdefault(identity[:2], []).append(r)
    expected = {(a, s, b) for a in ("T", "D", "P") for s in protocol["paired_seeds"] for b in protocol["budgets"]}
    require(identities == expected, "Incomplete paired arm/seed/budget grid")
    for seed in protocol["paired_seeds"]:
        require(len({r["sft_seed"] for r in m["checkpoints"] if r["pretrain_seed"] == seed}) == 1,
            "Each pretraining seed needs one common SFT seed across arms and budgets")
    for rows in by_run.values():
        baseline = next(r for r in rows if r["unique_response_tokens"] == 0)
        require(baseline["checkpoint_sha256"] == baseline["parent_pretrain_sha256"], "Baseline must be own parent")
        require(all(r["parent_pretrain_sha256"] == baseline["checkpoint_sha256"] for r in rows), "Mixed parent checkpoints")
    return m
