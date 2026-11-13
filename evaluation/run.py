"""Evaluate a complete registered manifest with resumable per-prompt output."""
import argparse
import platform
from pathlib import Path
import time

from evaluation import __version__
from evaluation.checks import CHECKER_VERSION, score_response
from evaluation.common import (atomic_json, canonical_hash, read_json, require,
    resolve, sha256_file, verify_file)
from evaluation.config import prompt_ids, reference_ids, validate_protocol, validate_tokenizer
from evaluation.data import leakage_audit, load_assets
from evaluation.manifest import validate_manifest

def source_hash():
    root = Path(__file__).resolve().parents[1]
    files = [*sorted((root / "evaluation").glob("*.py")),
        *sorted((root / "models").glob("*.py")), *sorted((root / "configs").glob("*.py")),
        root / "pretraining/evaluate.py", root / "pretraining/dataset.py"]
    return canonical_hash({str(p.relative_to(root)): sha256_file(p) for p in files})

def preflight(protocol_path, manifest_path):
    protocol = validate_protocol(read_json(protocol_path))
    manifest = validate_manifest(read_json(manifest_path), protocol)
    base = Path(manifest_path).resolve().parent
    paths = {}
    for name in ("tokenizer", "suite", "references", "val_tokens", "val_meta", "sft_train", "development", "sft_recipe"):
        if name in manifest:
            paths[name] = resolve(base, manifest[name]["path"])
            verify_file(paths[name], manifest[name]["sha256"])
    require(manifest["chat_template_sha256"] == canonical_hash(protocol["template"]), "Protocol/template hash mismatch")
    suite, references = load_assets(paths["suite"], paths["references"])
    sets = {"strict": suite, "references": references}
    for name in ("sft_train", "development"):
        if name in paths:
            rows = read_json(paths[name])
            require(isinstance(rows, list) and bool(rows), f"Empty {name} audit set")
            for row in rows:
                require(isinstance(row.get("prompt"), str) and bool(row["prompt"]), "Audit row needs prompt")
            sets[name] = rows
    audit = leakage_audit(sets)
    require(not audit["overlaps"], "Data leakage audit found overlapping prompts/examples")
    meta = read_json(paths["val_meta"])
    require(meta.get("dtype") == "uint16", "Unsupported validation dtype")
    for name, artifact in (("tokenizer.json", "tokenizer"), ("val.bin", "val_tokens")):
        info = meta["files"][name]
        require(info["sha256"] == manifest[artifact]["sha256"], f"WS2 metadata mismatch for {name}")
        require(info["bytes"] == paths[artifact].stat().st_size, f"WS2 size mismatch for {name}")
    require(paths["val_tokens"].stat().st_size % 2 == 0, "Odd uint16 validation file length")
    for row in manifest["checkpoints"]:
        verify_file(resolve(base, row["checkpoint_path"]), row["checkpoint_sha256"])
        if row["phase"] == "pretrain" and protocol["experiment"] != "synthetic":
            asset = row.get("pretrain_run_config", {})
            require(asset.get("path") and asset.get("sha256"), "Parent run_config asset required")
            verify_file(resolve(base, asset["path"]), asset["sha256"])
    return protocol, manifest, paths, suite, references, audit

def validate_parent_contract(record, checkpoint, run_config, manifest, experiment):
    """Pure validation, including tuple/list normalization in saved WS2 recipes."""
    recipe = run_config["pretrain"]
    require(recipe["arm"] == record["arm"] and recipe["seed"] == record["pretrain_seed"], "Parent run identity mismatch")
    require(run_config["model"] == checkpoint["config"], "Parent run/model config mismatch")
    if experiment == "primary":
        require(checkpoint["step"] == recipe["max_steps"], "Primary baseline must be final equal-budget pretraining checkpoint")
        require(canonical_hash(checkpoint["extra"].get("pretrain")) == canonical_hash(recipe),
            "Parent checkpoint/pretraining recipe mismatch")
    comparable = {k: v for k, v in recipe.items()
        if k not in {"arm", "seed", "out_dir", "data_dir", "device", "compile"}}
    for filename, name in (("tokenizer.json", "tokenizer"), ("val.bin", "val_tokens")):
        require(run_config["data_meta"]["files"][filename]["sha256"] == manifest[name]["sha256"],
            f"Parent {name} mismatch")
    return comparable, canonical_hash(run_config["data_meta"])


def evaluate(protocol_path, manifest_path, out, device="cpu", allow_provisional=False):
    p, m, paths, suite, refs, audit = preflight(protocol_path, manifest_path)
    require(p["status"] == "frozen" or allow_provisional, "Provisional protocol: pass --allow-provisional for pilot only")
    import torch
    from tokenizers import Tokenizer
    from models import lambda_log_dict, load_checkpoint
    from evaluation.generate import decode_generated, greedy_generate
    from evaluation.metrics import response_nll, retention, summarize_predictions
    require(device == "cpu" or device == "mps" or device.startswith("cuda"), "Unsupported device")
    require(p["amp"] == "none" or device.startswith("cuda"), "Autocast precision requires CUDA")
    torch.manual_seed(0)
    if device.startswith("cuda"):
        require(torch.cuda.is_available(), "CUDA unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    tok = Tokenizer.from_file(str(paths["tokenizer"]))
    encoded_prompts = {r["id"]: prompt_ids(tok, r["prompt"], p["template"]["name"]) for r in suite}
    encoded_refs = [{"id": r["id"], "ids": ids, "labels": labels}
        for r in refs for ids, labels in [reference_ids(tok, r, p["include_eos_nll"], p["template"]["name"])]]
    probe_ids = encoded_prompts[suite[0]["id"]]
    environment = {"python": platform.python_version(), "torch": torch.__version__,
        "tokenizers": __import__("tokenizers").__version__, "device": device,
        "hardware": torch.cuda.get_device_name(device) if device.startswith("cuda") else platform.machine(),
        "cuda_version": torch.version.cuda, "deterministic_algorithms": True}
    provenance = {"version": __version__, "code_sha256": source_hash(),
        "checker_version": CHECKER_VERSION, "manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": sha256_file(protocol_path), "environment": environment,
        "execution_weight_dtype": "float32", "amp": p["amp"],
        "lambda_probe_sha256": canonical_hash(probe_ids), "leakage_audit": audit,
        "near_duplicate_review": m.get("near_duplicate_review", "not supplied; synthetic only")}
    evaluation_id = canonical_hash(provenance)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "results.json"
    if status_path.exists():
        require(read_json(status_path)["evaluation_id"] == evaluation_id, "Output belongs to another evaluation; choose a new --out")
    label = "SYNTHETIC MODEL SMOKE TEST" if p["experiment"] == "synthetic" else (
        "FROZEN EVALUATION" if p["status"] == "frozen" else "PROVISIONAL PILOT - NOT FINAL RESULTS")
    results = {"schema_version": 1, "evaluation_id": evaluation_id, "label": label,
        "protocol": p, "provenance": provenance, "complete": False, "rows": []}
    atomic_json(status_path, results)
    base = Path(manifest_path).resolve().parent
    shared_cfg = None
    baseline_nll = {}
    parent_recipe = None
    parent_data = None
    for r in sorted(m["checkpoints"], key=lambda r: (r["arm"], r["pretrain_seed"], r["unique_response_tokens"])):
        key = f'{r["arm"]}_s{r["pretrain_seed"]}_b{r["unique_response_tokens"]}'
        print(f"Evaluating {key}", flush=True)
        model, ck = load_checkpoint(resolve(base, r["checkpoint_path"]), device=device)
        require(ck["arm"] == r["arm"] and ck["step"] == r["step"], "Checkpoint identity mismatch")
        extra = ck.get("extra", {})
        seed = extra.get("pretrain_seed", extra.get("seed"))
        require(seed == r["pretrain_seed"], "Checkpoint seed mismatch")
        if r["phase"] == "sft":
            for field in ("sft_seed", "unique_response_tokens", "processed_response_tokens",
                          "parent_pretrain_sha256", "chat_template_sha256", "sft_stream_sha256"):
                require(extra.get(field) == r[field], f"SFT checkpoint metadata mismatch: {field}")
            if p["experiment"] != "synthetic":
                require(extra.get("sft_recipe_sha256") == m["sft_recipe"]["sha256"], "SFT recipe mismatch")
            require(extra.get("unique_count_includes_eos") == p["unique_count_includes_eos"], "Checkpoint EOS count convention mismatch")
            require(extra.get("tokenizer_sha256") == m["tokenizer"]["sha256"], "SFT tokenizer mismatch")
        if r["phase"] == "pretrain" and p["experiment"] != "synthetic":
            asset = r.get("pretrain_run_config", {})
            require(asset.get("path") and asset.get("sha256"), "Parent run_config asset required")
            config_path = resolve(base, asset["path"])
            verify_file(config_path, asset["sha256"])
            run_config = read_json(config_path)
            comparable_recipe, data_identity = validate_parent_contract(r, ck, run_config, m, p["experiment"])
            if parent_recipe is None:
                parent_recipe, parent_data = comparable_recipe, data_identity
            require(parent_recipe == comparable_recipe and parent_data == data_identity,
                "Pretraining recipe or ordered data differs across parent runs")
        cfg = model.cfg.to_dict()
        comparable = {k: v for k, v in cfg.items() if k != "ffn_hidden_p"}
        if shared_cfg is None:
            shared_cfg = comparable
        require(comparable == shared_cfg, "Incompatible model configs across arms/seeds")
        # P may widen FFN, but may not silently use an unmatched override.
        from models.build import parameter_summary
        summary = parameter_summary(model.cfg, verbose=False)
        require(abs(summary["P_vs_D_pct"]) <= 1, "P/D parameter mismatch exceeds one percent")
        validate_tokenizer(tok, model.cfg.vocab_size)
        require(model.cfg.vocab_size == read_json(paths["val_meta"])["vocab_size"], "Validation/model vocab mismatch")
        for ids in encoded_prompts.values():
            require(len(ids) + p["generation_cap"] <= model.cfg.context_len, "Overlong prompt; no silent truncation")
        for row in encoded_refs:
            require(len(row["ids"]) <= model.cfg.context_len, "Overlong reference; no silent truncation")
        progress_path = out / (key + ".predictions.json")
        predictions = []
        if progress_path.exists():
            cached = read_json(progress_path)
            require(cached["evaluation_id"] == evaluation_id and cached["checkpoint_sha256"] == r["checkpoint_sha256"],
                "Prediction cache identity mismatch")
            predictions = cached["predictions"]
            require(len(predictions) <= len(suite), "Too many cached predictions")
            for saved, row in zip(predictions, suite):
                require(saved["id"] == row["id"] and saved["family"] == row["family"], "Prediction cache order/family mismatch")
                generated_ids = saved["generated_ids"]
                require(0 < len(generated_ids) <= p["generation_cap"] and
                    all(type(i) is int and 0 <= i < model.cfg.vocab_size for i in generated_ids), "Invalid cached generation")
                require(saved["stop_reason"] in {"stop", "cap"}, "Invalid cached stop reason")
                require(saved["cap_hit"] == (saved["stop_reason"] == "cap") and
                    saved["generated_token_count"] == len(generated_ids), "Invalid cached generation counts")
                if saved["stop_reason"] == "stop":
                    require(generated_ids[-1] in p["stop_ids"] and not any(i in p["stop_ids"] for i in generated_ids[:-1]), "Invalid cached terminator")
                else:
                    require(len(generated_ids) == p["generation_cap"] and not any(i in p["stop_ids"] for i in generated_ids), "Invalid cached cap")
                require(saved["text"] == decode_generated(tok, saved, p["stop_ids"]), "Cached decoding mismatch")
                score = score_response(saved["text"], row["checks"])
                require(all(saved[k] == v for k, v in score.items()), "Cached score mismatch")
        t0 = time.perf_counter()
        for row in suite[len(predictions):]:
            generated = greedy_generate(model, encoded_prompts[row["id"]], p["generation_cap"], p["stop_ids"], p["amp"])
            text = decode_generated(tok, generated, p["stop_ids"])
            predictions.append({"id": row["id"], "family": row["family"], "text": text,
                **generated, **score_response(text, row["checks"])})
            atomic_json(progress_path, {"evaluation_id": evaluation_id,
                "checkpoint_sha256": r["checkpoint_sha256"], "predictions": predictions})
        gen_seconds = time.perf_counter() - t0
        nll = response_nll(model, encoded_refs, p["amp"])
        retained = retention(model, paths["val_tokens"], p["amp"])
        run_key = (r["arm"], r["pretrain_seed"])
        if r["unique_response_tokens"] == 0:
            baseline_nll[run_key] = retained["nll"]
        diagnostics = {}
        if model.arm == "D":
            diagnostics = lambda_log_dict(model, torch.tensor([probe_ids], dtype=torch.long, device=device))
        result = {**{k: r[k] for k in ("arm", "pretrain_seed", "sft_seed", "unique_response_tokens",
            "processed_response_tokens", "checkpoint_sha256", "parent_pretrain_sha256")},
            **summarize_predictions(predictions), "response_nll": nll, "retention": retained,
            "retention_delta_nll": retained["nll"] - baseline_nll[run_key],
            "lambda": diagnostics, "parameter_count": model.num_params(),
            "checkpoint_storage_dtypes": sorted({str(v.dtype) for v in ck["model"].values()}),
            "generation_seconds_this_invocation": gen_seconds,
            "generation_timing_note": "Resumed cached prompts excluded; not a throughput benchmark."}
        results["rows"].append(result)
        atomic_json(status_path, results)
        del model, ck
    results["complete"] = True
    atomic_json(status_path, results)
    return results

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-provisional", action="store_true")
    a = ap.parse_args(argv)
    evaluate(a.protocol, a.manifest, a.out, a.device, a.allow_provisional)
    print(Path(a.out) / "results.json")

if __name__ == "__main__":
    main()
