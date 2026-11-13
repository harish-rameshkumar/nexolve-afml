"""Native autoregressive greedy decoding; no padding, context cropping or KV cache."""
import torch
from evaluation.common import integer, require
from pretraining.evaluate import autocast_ctx

def greedy_generate(model, ids, max_new_tokens, stop_ids=(0,), amp="none"):
    integer(max_new_tokens, "max_new_tokens", 1)
    require(bool(ids), "Empty generation prefix")
    require(len(ids) + max_new_tokens <= model.cfg.context_len, "Prompt plus generation cap exceeds context")
    require(all(type(i) is int and 0 <= i < model.cfg.vocab_size for i in ids), "Invalid input IDs")
    require(bool(stop_ids) and all(0 <= i < model.cfg.vocab_size for i in stop_ids), "Invalid stop IDs")
    device = next(model.parameters()).device
    was_training = model.training
    generated = []
    reason = "cap"
    model.eval()
    try:
        with torch.inference_mode():
            sequence = torch.tensor([ids], dtype=torch.long, device=device)
            for _ in range(max_new_tokens):
                with autocast_ctx(str(device), amp):
                    logits = model(sequence)
                last = logits[0, -1].float()
                require(bool(torch.isfinite(last).all().item()), "Nonfinite generation logits")
                token = int(last.argmax().item())
                generated.append(token)
                if token in stop_ids:
                    reason = "stop"
                    break
                sequence = torch.cat((sequence, torch.tensor([[token]], device=device)), dim=1)
    finally:
        model.train(was_training)
    return {"generated_ids": generated, "stop_reason": reason,
        "generated_token_count": len(generated), "cap_hit": reason == "cap"}

def decode_generated(tok, result, stop_ids):
    ids = result["generated_ids"]
    content = ids[:-1] if result["stop_reason"] == "stop" and ids[-1] in stop_ids else ids
    # Keep any nonterminating special tokens visible to the strict checker.
    return tok.decode(content, skip_special_tokens=False)
