"""Explicit protocol: examples are provisional, never silently production-ready."""
from evaluation.common import finite, integer, require
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>",
    "<|assistant|>", "<|end|>", "<|reserved0|>", "<|reserved1|>"]

def validate_protocol(p):
    require(p.get("schema_version") == 1, "Unsupported protocol version")
    require(p.get("status") in {"provisional", "frozen"}, "Protocol status required")
    require(p.get("experiment") in {"primary", "loss_matched", "synthetic"}, "Unknown experiment")
    seeds = p.get("paired_seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2, "At least two paired seeds required")
    for seed in seeds: integer(seed, "paired seed")
    require(len(set(seeds)) == len(seeds), "Duplicate seeds")
    budgets = p.get("budgets")
    require(isinstance(budgets, list) and len(budgets) >= 2 and budgets[0] == 0, "Budgets must start at zero")
    for b in budgets: integer(b, "budget")
    require(all(b > a for a, b in zip(budgets, budgets[1:])), "Budgets must strictly increase")
    integer(p.get("generation_cap"), "generation_cap", 1)
    require(p.get("stop_ids") in ([0], [0, 5], [5]), "Stop IDs must be [0], [5], or [0,5]")
    require(type(p.get("unique_count_includes_eos")) is bool, "Explicit unique-token EOS convention required")
    require(type(p.get("include_eos_nll")) is bool, "Explicit EOS NLL convention required")
    require(0 <= finite(p.get("threshold"), "threshold") <= 1, "Threshold outside [0,1]")
    require(p.get("amp") in {"none", "fp16", "bf16"}, "Unknown precision")
    require(p.get("grid_policy") == "exact", "This version requires exact shared token budgets")
    template = p.get("template", {})
    require(set(template) == {"name", "ws3_confirmed"} and template.get("name") in {"nexolve_single_turn_v1", "nexolve_ws3_v1"}
        and type(template.get("ws3_confirmed")) is bool, "Unsupported template: implement/test WS3 formatter")
    if p["status"] == "frozen":
        require(template["ws3_confirmed"], "Frozen protocol needs WS3 template confirmation")
        require(p.get("freeze_note"), "Frozen protocol needs dated freeze/pilot decision note")
    return p

def validate_tokenizer(tok, vocab_size):
    require(tok.get_vocab_size() == vocab_size, "Tokenizer/model vocabulary mismatch")
    for i, token in enumerate(SPECIAL_TOKENS):
        require(tok.token_to_id(token) == i, f"Special-token ID mismatch: {token}")

def prompt_ids(tok, text, template="nexolve_single_turn_v1"):
    """Provisional WS3 contract: no implicit whitespace or system prompt."""
    require(isinstance(text, str) and bool(text), "Empty prompt")
    ids = tok.encode(text, add_special_tokens=False).ids
    require(all(i >= len(SPECIAL_TOKENS) for i in ids), "Prompt contains reserved control tokens")
    require(template in {"nexolve_single_turn_v1", "nexolve_ws3_v1"}, "Unknown prompt template")
    return [3] + ids + ([4] if template == "nexolve_ws3_v1" else [5, 4])

def reference_ids(tok, row, include_eos, template="nexolve_single_turn_v1"):
    prompt = prompt_ids(tok, row["prompt"], template)
    response = tok.encode(row["response"], add_special_tokens=False).ids
    require(bool(response) and all(i >= len(SPECIAL_TOKENS) for i in response),
        "Empty reference or reserved control tokens")
    terminator = 5 if template == "nexolve_ws3_v1" else 0
    target = response + ([terminator] if include_eos else [])
    return prompt + target, [-100] * len(prompt) + target
