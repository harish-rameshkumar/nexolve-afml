"""Recompute frozen validation NLL for real saved pretraining candidates, then select."""
import argparse
from pathlib import Path
from evaluation.analysis import select_loss_matched
from evaluation.common import atomic_json, read_json, require, resolve, sha256_file, verify_file

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True, help="JSON list: path, sha256, arm, pretrain_seed, step")
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-sha256", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--tolerance", type=float, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    verify_file(a.val, a.val_sha256)
    from models import load_checkpoint
    from evaluation.metrics import retention
    candidates = []
    base = Path(a.candidates).resolve().parent
    for entry in read_json(a.candidates):
        if entry["pretrain_seed"] != a.seed: continue
        path = resolve(base, entry["path"])
        verify_file(path, entry["sha256"])
        model, ck = load_checkpoint(path, device=a.device)
        require(ck["arm"] == entry["arm"] and ck["step"] == entry["step"]
            and ck["extra"].get("seed", ck["extra"].get("pretrain_seed")) == a.seed, "Candidate identity mismatch")
        candidates.append({**entry, "path": str(path), "loaded_val_nll": retention(model, a.val, "none")["nll"],
            "evaluation_weight_dtype": "float32", "amp": "none", "val_sha256": a.val_sha256})
        del model, ck
    selected = select_loss_matched(candidates, a.seed, a.target, a.tolerance)
    atomic_json(a.out, {"selection": selected, "recomputed_candidates": candidates})
    print(a.out)

if __name__ == "__main__":
    main()
