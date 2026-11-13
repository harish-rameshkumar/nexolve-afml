"""Pure-Python aggregation shared by synthetic replay and actual model evaluation."""
from evaluation.common import require

def summarize_predictions(predictions):
    require(bool(predictions), "Empty predictions")
    require(len({p["id"] for p in predictions}) == len(predictions), "Duplicate prediction ID")
    families = {}
    for p in predictions:
        require(type(p["strict_pass"]) is bool, "Invalid strict pass")
        checks = p["constraint_passes"]
        require(bool(checks) and all(type(v) is bool for v in checks), "Invalid constraint passes")
        require(not p["strict_pass"] or all(checks), "Inconsistent strict pass")
        f = families.setdefault(p["family"], {"passed": 0, "count": 0})
        f["passed"] += int(p["strict_pass"]); f["count"] += 1
    n = len(predictions)
    return {"strict_score": sum(p["strict_pass"] for p in predictions) / n,
        "prompt_count": n,
        "constraint_score": sum(sum(p["constraint_passes"]) for p in predictions) /
            sum(len(p["constraint_passes"]) for p in predictions),
        "families": {k: {**v, "score": v["passed"] / v["count"]} for k, v in families.items()}}
