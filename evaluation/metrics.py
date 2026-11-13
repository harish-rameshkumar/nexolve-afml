"""Token-weighted response NLL, native retention and strict aggregation."""
import torch
import torch.nn.functional as F
from evaluation.common import finite, require
from pretraining.dataset import PackedTokens
from pretraining.evaluate import autocast_ctx, val_loss

from evaluation.scoring import summarize_predictions

def response_nll(model, encoded, amp="none", batch_size=1):
    """Right padding is safe here: causality prevents it influencing earlier real targets.

    Padding targets are -100. No left padding, packed unrelated examples or altered positions.
    """
    require(bool(encoded) and type(batch_size) is int and batch_size > 0, "Empty references or invalid batch size")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total, count, records = 0.0, 0, []
    ordered = sorted(encoded, key=lambda r: len(r["ids"]))
    try:
        with torch.inference_mode():
            for offset in range(0, len(ordered), batch_size):
                rows = ordered[offset:offset+batch_size]
                length = max(len(r["ids"])-1 for r in rows)
                inputs = torch.full((len(rows), length), 1, dtype=torch.long, device=device)
                targets = torch.full_like(inputs, -100)
                for i, row in enumerate(rows):
                    ids, labels = row["ids"], row["labels"]
                    require(len(ids) == len(labels) and 2 <= len(ids) <= model.cfg.context_len,
                        "Invalid or overlong reference sequence")
                    require(labels[0] == -100, "Reference must start with a masked prefix")
                    require(all(t == -100 or t == token for token,t in zip(ids,labels)), "Misaligned reference labels")
                    n = len(ids)-1
                    inputs[i,:n] = torch.tensor(ids[:-1], device=device)
                    targets[i,:n] = torch.tensor(labels[1:], device=device)
                    require(bool((targets[i] != -100).any()), "Reference has no supervised response targets")
                with autocast_ctx(str(device), amp):
                    logits = model(inputs)
                losses = F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]), targets.reshape(-1),
                    ignore_index=-100, reduction="none").reshape_as(targets)
                for i, row in enumerate(rows):
                    n = int((targets[i] != -100).sum().item())
                    loss = losses[i].sum().item()
                    finite(loss, "response NLL")
                    records.append({"id":row["id"], "nll_sum":loss, "target_tokens":n})
                    total += loss; count += n
    finally:
        model.train(was_training)
    require(count > 0, "Zero response-token denominator")
    return {"nll":total/count, "nll_sum":total, "target_tokens":count, "examples":records}

def retention(model, path, amp="none"):
    val = PackedTokens(str(path), model.cfg.context_len)
    require(val.n_windows > 0, "Frozen validation has no complete windows")
    result = val_loss(model, val, str(next(model.parameters()).device), batch_size=1, amp=amp)
    finite(result, "retention NLL")
    return {"nll": result, "target_tokens": val.n_windows * val.seq_len,
        "windows": val.n_windows, "coverage": "full_complete_windows"}
