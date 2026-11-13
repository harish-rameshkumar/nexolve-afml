"""Frozen assets and normalized exact-duplicate leakage audit."""
import unicodedata
from evaluation.checks import score_response, validate_check
from evaluation.common import canonical_hash, read_json, require

def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

def validate_assets(suite, references):
    require(isinstance(suite, list) and bool(suite), "Empty instruction suite")
    require(isinstance(references, list) and bool(references), "Empty NLL reference set")
    for rows, scored in ((suite, True), (references, False)):
        ids, prompts = set(), set()
        for row in rows:
            require(isinstance(row, dict), "Invalid example")
            require(isinstance(row.get("id"), str) and bool(row["id"]) and row["id"] not in ids,
                "Missing/duplicate example ID")
            require(isinstance(row.get("prompt"), str) and bool(row["prompt"]), "Empty prompt")
            key = normalized(row["prompt"])
            require(key not in prompts, "Duplicate normalized prompt")
            ids.add(row["id"]); prompts.add(key)
            if scored:
                require(isinstance(row.get("family"), str) and bool(row["family"]), "Missing checker family")
                require(isinstance(row.get("checks"), list) and bool(row["checks"]), "Missing checks")
                for check in row["checks"]: validate_check(check)
                require(isinstance(row.get("passing_response"), str), "Missing positive control")
                require(score_response(row["passing_response"], row["checks"])["strict_pass"],
                    f"Invalid positive control for {row['id']}")
                negatives = row.get("failing_responses")
                require(isinstance(negatives, list) and bool(negatives), "Missing negative controls")
                for answer in negatives:
                    require(not score_response(answer, row["checks"])["strict_pass"],
                        f"Invalid negative control for {row['id']}")
            else:
                require(isinstance(row.get("response"), str) and bool(row["response"]), "Empty reference")
    left = {normalized(r["prompt"]) for r in suite}
    require(not left.intersection(normalized(r["prompt"]) for r in references),
        "Strict and reference sets overlap")

def leakage_audit(named_sets):
    """Normalized exact matching only; NOT near-duplicate/web contamination detection."""
    seen_prompt, seen_example, overlaps = {}, {}, []
    for name, rows in named_sets.items():
        for row in rows:
            prompt = canonical_hash(normalized(row["prompt"]))
            example = canonical_hash([normalized(row["prompt"]), normalized(row.get("response", ""))])
            for kind, key, seen in (("prompt", prompt, seen_prompt), ("example", example, seen_example)):
                previous = seen.setdefault(key, set())
                for other in sorted(previous - {name}):
                    overlaps.append({"kind": kind, "sets": [other, name], "hash": key})
                previous.add(name)
    return {"method": "NFKC/casefold/whitespace normalized exact hashes",
        "near_duplicate_review": "not performed", "overlaps": overlaps}

def load_assets(suite_path, reference_path):
    suite, refs = read_json(suite_path), read_json(reference_path)
    validate_assets(suite, refs)
    return suite, refs
