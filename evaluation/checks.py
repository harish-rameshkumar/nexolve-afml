"""Strict versioned checks: no trimming, repair, or silent exceptions."""
import json
import re
from evaluation.common import require
CHECKER_VERSION = "2"
KINDS = {"exact", "contains", "forbids", "lowercase", "uppercase", "lines", "json_keys", "choice", "words", "prefix"}

def validate_check(check):
    require(isinstance(check, dict) and check.get("kind") in KINDS, "Unknown checker")
    kind = check["kind"]
    if kind in {"exact", "contains", "forbids", "prefix"}:
        require(set(check) == {"kind", "value"} and isinstance(check["value"], str), "Checker needs string value")
        require(kind == "exact" or bool(check["value"]), "Empty substring constraint")
    elif kind in {"lines", "words"}:
        require(set(check) == {"kind", "value"} and type(check["value"]) is int
            and check["value"] > 0, "lines needs positive integer value")
    elif kind == "choice":
        values = check.get("value")
        require(set(check) == {"kind", "value"} and isinstance(values,list) and len(values)>=2
            and all(isinstance(v,str) and bool(v) for v in values) and len(set(values))==len(values), "choice needs distinct nonempty strings")
    elif kind == "json_keys":
        keys = check.get("value")
        require(set(check) == {"kind", "value"} and isinstance(keys, list)
            and all(isinstance(k, str) for k in keys) and len(set(keys)) == len(keys),
            "json_keys needs unique string keys")
    else:
        require(set(check) == {"kind"}, "Case checker takes no arguments")

def _object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON key")
        obj[key] = value
    return obj

def _invalid_constant(value):
    raise ValueError(f"Nonstandard JSON constant: {value}")

def check_response(text, check):
    validate_check(check)
    require(isinstance(text, str), "Response must be text")
    kind, value = check["kind"], check.get("value")
    if kind == "exact": return text == value
    if kind == "choice": return text in value
    if kind == "prefix": return text.startswith(value)
    if kind == "words": return bool(re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+){"+str(value-1)+r"}",text))
    if kind == "contains": return value in text
    if kind == "forbids": return value not in text
    if kind == "lowercase": return text.islower()
    if kind == "uppercase": return text.isupper()
    if kind == "lines":
        return "\r" not in text and len(text.split("\n")) == value and all(text.split("\n"))
    try:
        obj = json.loads(text, object_pairs_hook=_object, parse_constant=_invalid_constant)
    except ValueError:
        return False
    return isinstance(obj, dict) and set(obj) == set(value)

def score_response(text, checks):
    require(bool(checks), "A prompt must have at least one checker")
    passed = [check_response(text, c) for c in checks]
    return {"strict_pass": bool(text) and all(passed), "constraint_passes": passed}
