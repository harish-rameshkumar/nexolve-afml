"""Offline synthetic replay: tests scoring/analysis/report, does NOT execute a model."""
import argparse
from pathlib import Path
from evaluation.analysis import analyze
from evaluation.checks import score_response
from evaluation.common import atomic_json, canonical_hash, read_json, sha256_file
from evaluation.config import validate_protocol
from evaluation.data import load_assets
from evaluation.report import render
from evaluation.scoring import summarize_predictions

def demo(out):
    root = Path(__file__).resolve().parents[1]
    p = validate_protocol(read_json(root / "evaluation/protocol.example.json"))
    suite, _ = load_assets(root / "tests/fixtures/ws4/suite.json", root / "tests/fixtures/ws4/references.json")
    provenance = {"source": "Artificial responses selected from checker controls; no model run or training.",
        "suite_sha256": sha256_file(root / "tests/fixtures/ws4/suite.json")}
    results = {"schema_version": 1, "label": "SYNTHETIC REPLAY - NOT MODEL RESULTS",
        "protocol": p, "provenance": provenance, "complete": True,
        "evaluation_id": canonical_hash({"protocol": p, "provenance": provenance}), "rows": []}
    out = Path(out)
    for arm in ("T", "D", "P"):
        for seed in p["paired_seeds"]:
            for index, budget in enumerate(p["budgets"]):
                # Deliberately illustrative, never an estimate of architecture quality.
                pass_count = min(len(suite), 1 + index * 3 + seed)
                predictions = []
                for i, row in enumerate(suite):
                    text = row["passing_response"] if i < pass_count else row["failing_responses"][0]
                    predictions.append({"id": row["id"], "family": row["family"], "text": text,
                        **score_response(text, row["checks"])})
                atomic_json(out / f"{arm}_s{seed}_b{budget}.predictions.json",
                    {"label": results["label"], "predictions": predictions})
                results["rows"].append({"arm": arm, "pretrain_seed": seed,
                    "unique_response_tokens": budget, **summarize_predictions(predictions)})
    atomic_json(out / "results.json", results)
    analysis = analyze(results)
    atomic_json(out / "analysis.json", analysis)
    render(analysis, out)
    return results

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/ws4_demo")
    a = ap.parse_args(argv)
    demo(a.out)
    print(Path(a.out) / "report.html")

if __name__ == "__main__":
    main()
