"""Seed-paired AUC and censored thresholds. No prompt/checkpoint pseudoreplication."""
import argparse
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from evaluation.common import atomic_json, finite, read_json, require
from evaluation.config import validate_protocol

def validate_curve(points, budgets=None):
    require(len(points) >= 2, "At least two curve points needed")
    xs = []
    for x, y in points:
        require(type(x) is int and x >= 0, "Invalid token coordinate")
        require(0 <= finite(y, "strict score") <= 1, "Score outside [0,1]")
        xs.append(x)
    require(xs[0] == 0 and all(b > a for a, b in zip(xs, xs[1:])), "Nonincreasing/duplicate token grid")
    if budgets is not None:
        require(xs == budgets, "Incomplete or incompatible registered token grid")

def normalized_auc(points, budgets=None):
    validate_curve(points, budgets)
    return sum((b - a) * (ya + yb) / 2 for (a, ya), (b, yb) in zip(points, points[1:])) / points[-1][0]

def threshold_attainment(points, threshold):
    validate_curve(points)
    require(0 <= finite(threshold, "threshold") <= 1, "Invalid threshold")
    for i, (x, y) in enumerate(points):
        if y >= threshold:
            return {"reached": True, "observed_tokens": x, "crossing_interval": [points[max(0, i-1)][0], x],
                "right_censored": False, "remains_above": all(v >= threshold for _, v in points[i:])}
    return {"reached": False, "observed_tokens": None, "crossing_interval": None,
        "right_censored": True, "greater_than_tokens": points[-1][0], "remains_above": None}

def paired_summary(differences):
    require(len(differences) >= 2, "At least two seed differences needed")
    d = [finite(v, "paired difference") for v in differences]
    observed = abs(mean(d))
    values = [abs(mean(v * s for v, s in zip(d, signs))) for signs in product((-1, 1), repeat=len(d))]
    return {"differences": d, "mean": mean(d), "seed_sd": stdev(d), "min": min(d), "max": max(d),
        "n_pretrain_seeds": len(d),
        "descriptive_two_sided_sign_flip_p": sum(v >= observed - 1e-12 for v in values) / len(values),
        "inference_note": "Descriptive sign-flip assumes exchangeable symmetric seed differences; very low power with 2-3 seeds. No equivalence or significance claim."}

def analyze(results):
    p = validate_protocol(results["protocol"])
    require(results.get("complete") is True, "Cannot analyze incomplete evaluation")
    rows = results["rows"]
    by_key = {}
    for r in rows:
        key = (r["arm"], r["pretrain_seed"], r["unique_response_tokens"])
        require(key not in by_key, "Duplicate result identity")
        by_key[key] = r
    expected = {(a, s, b) for a in ("T", "D", "P") for s in p["paired_seeds"] for b in p["budgets"]}
    require(set(by_key) == expected, "Incomplete paired results")
    runs = []
    aucs = {}
    for arm in ("T", "D", "P"):
        for seed in p["paired_seeds"]:
            curve_rows = [by_key[(arm, seed, b)] for b in p["budgets"]]
            points = [(r["unique_response_tokens"], r["strict_score"]) for r in curve_rows]
            auc = normalized_auc(points, p["budgets"])
            aucs[(arm, seed)] = auc
            runs.append({"arm": arm, "pretrain_seed": seed, "normalized_auc": auc,
                "final_strict_score": points[-1][1], "threshold": threshold_attainment(points, p["threshold"]),
                "curve": curve_rows})
    comparisons = {f"D-{control}": paired_summary([aucs[("D", s)] - aucs[(control, s)]
        for s in p["paired_seeds"]]) for control in ("T", "P")}
    return {"schema_version": 1, "label": results["label"], "protocol": p,
        "evaluation_id": results["evaluation_id"], "provenance": results.get("provenance", {}),
        "runs": runs, "paired_auc": comparisons,
        "limitations": ["Two contrasts reported descriptively; no multiplicity-adjusted confirmatory inference.",
            "FFN-widened P controls total parameter count, not every architectural difference.",
            "Rule-based strict score measures constrained output compliance, not general helpfulness."]}

def select_loss_matched(candidates, seed, target, tolerance):
    """Select actual checkpoints only; candidate nll must be recomputed on loaded weights."""
    finite(target, "target loss"); require(finite(tolerance, "tolerance") >= 0, "Negative tolerance")
    selected = []
    for arm in ("T", "D", "P"):
        eligible = []
        for c in candidates:
            if c["arm"] == arm and c["pretrain_seed"] == seed:
                loss = finite(c["loaded_val_nll"], "loaded validation NLL")
                require(Path(c["path"]).is_file(), "Candidate checkpoint does not exist")
                if abs(loss - target) <= tolerance:
                    eligible.append(c)
        if not eligible:
            return {"available": False, "seed": seed, "target": target, "reason": f"No {arm} checkpoint within tolerance"}
        best = min(eligible, key=lambda c: (abs(c["loaded_val_nll"] - target), c["step"], c["path"]))
        selected.append(best)
    return {"available": True, "seed": seed, "target": target, "selected": selected,
        "max_pairwise_loss_gap": max(c["loaded_val_nll"] for c in selected) - min(c["loaded_val_nll"] for c in selected),
        "interpretation": "Requires fresh identical SFT from each selected parent; not causal isolation."}

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="completed results.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    output = analyze(read_json(args.results))
    atomic_json(Path(args.out) / "analysis.json", output)
    print(Path(args.out) / "analysis.json")

if __name__ == "__main__":
    main()
