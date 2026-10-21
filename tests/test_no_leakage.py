"""Leakage checks: the loss at position t must depend only on tokens up to t (plus its own target).

Setup used throughout: a sequence z of length T+1; inputs are z[:-1], targets are z[1:], so the loss at
position t depends on z[0..t] (inputs) and z[t+1] (its target). Changing z[k] for k >= c may therefore
change losses at positions >= c-1, but must leave positions 0..c-2 untouched.

The "positive control" tests remove the causal mask on purpose and require the same checks to FAIL,
which proves the checks can actually detect a leak.
"""
import math

import pytest
import torch
import torch.nn.functional as F

import models.attention as attention_mod
import models.diff_attention as diff_mod
from configs.model_config import default_config
from models import build_model

ARMS = ["T", "D", "P"]
IMPLS = ["sdpa", "einsum"]


def make(arm, cfg, impl="einsum"):
    return build_model(arm, default_config(**{**cfg.to_dict(), "attn_impl": impl}))


def per_position_loss(model, z):
    """(B, T) cross-entropy at every position; z is (B, T+1)."""
    logits = model(z[:, :-1])                                   # (B, T, V)
    return F.cross_entropy(logits.transpose(1, 2), z[:, 1:], reduction="none")


def brief_train(model, z, steps=10):
    """A few steps so the weights are not just the (near-symmetric) initialisation."""
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        loss = per_position_loss(model, z).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def remove_causal_mask(monkeypatch):
    allow_all = lambda n, device: torch.ones(n, n, dtype=torch.bool, device=device)
    monkeypatch.setattr(attention_mod, "causal_mask", allow_all)
    monkeypatch.setattr(diff_mod, "causal_mask", allow_all)


def future_perturbation_effect(model, z, c):
    """(max change in losses at positions < c-1, max change at positions >= c-1) after altering z[:, c:]."""
    z2 = z.clone()
    z2[:, c:] = (z2[:, c:] + 1 + torch.randint(0, 5, z2[:, c:].shape)) % model.cfg.vocab_size
    with torch.no_grad():
        a, b = per_position_loss(model, z), per_position_loss(model, z2)
    d = (a - b).abs()
    return d[:, : c - 1].max().item(), d[:, c - 1:].max().item()


# ---------------------------------------------------------------- 1. per-position losses (not just logits)
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("arm", ARMS)
def test_per_position_loss_ignores_later_tokens(tiny_cfg, arm, impl):
    torch.manual_seed(0)
    z = torch.randint(0, tiny_cfg.vocab_size, (3, 17))
    model = brief_train(make(arm, tiny_cfg, impl), z)
    early, late = future_perturbation_effect(model, z, c=9)
    assert early < 1e-5, f"{arm}/{impl}: losses at positions before c-1 changed by {early:.2e}"
    assert late > 1e-3, "sanity: the position whose target was changed should change"


# ---------------------------------------------------------------- 2. gradients: dL_t / d(embedding of token k>t) == 0
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("arm", ARMS)
def test_loss_gradient_does_not_reach_future_tokens(tiny_cfg, arm, impl):
    torch.manual_seed(0)
    z = torch.randint(0, tiny_cfg.vocab_size, (2, 13))
    model = brief_train(make(arm, tiny_cfg, impl), z)
    seen = {}

    def keep(module, inputs, output):
        output.retain_grad()
        seen["emb"] = output

    model.embed.register_forward_hook(keep)
    for t in (0, 4, 8):
        model.zero_grad()
        per_position_loss(model, z)[0, t].backward()
        g = seen["emb"].grad[0].abs().amax(dim=-1)              # (T,) gradient size per input position
        assert g[: t + 1].max() > 0, "sanity: the past/present does influence the loss"
        assert g[t + 1:].max() <= 1e-9 * g[: t + 1].max(), f"{arm}/{impl}: loss at t={t} depends on later inputs"


# ---------------------------------------------------------------- 3. no mixing across the batch dimension
@pytest.mark.parametrize("arm", ARMS)
def test_samples_in_a_batch_are_independent(tiny_cfg, arm):
    torch.manual_seed(0)
    z = torch.randint(0, tiny_cfg.vocab_size, (3, 17))
    model = brief_train(make(arm, tiny_cfg, "sdpa"), z)
    z2 = z.clone()
    z2[1] = torch.randint(0, tiny_cfg.vocab_size, (17,))        # change a different sample entirely
    with torch.no_grad():
        a, b = per_position_loss(model, z), per_position_loss(model, z2)
    assert torch.allclose(a[[0, 2]], b[[0, 2]], atol=1e-6)


# ---------------------------------------------------------------- 4. positive controls: a leaky model IS caught
@pytest.mark.parametrize("arm", ["T", "D"])
def test_checks_detect_a_leak_when_causal_mask_removed(tiny_cfg, monkeypatch, arm):
    remove_causal_mask(monkeypatch)
    torch.manual_seed(0)
    z = torch.randint(0, tiny_cfg.vocab_size, (3, 17))
    model = make(arm, tiny_cfg, "einsum").eval()
    early, _ = future_perturbation_effect(model, z, c=9)
    assert early > 1e-4, "the per-position check should notice that earlier losses now depend on later tokens"


# ---------------------------------------------------------------- 5. held-out random batch stays at ln(V)
def held_out_loss(arm, steps=100, V=64, T=24):
    """Train on a stream of fresh random batches, then measure on a held-out random batch.
    Random tokens carry no learnable signal, so a causal model must stay at ln(V); a model that can read
    the next token (a leak) quickly learns to copy it and drops far below ln(V)."""
    cfg = default_config(vocab_size=V, context_len=T, d_model=64, n_layers=2, n_heads=4, ffn_hidden=176,
                         attn_impl="einsum")
    torch.manual_seed(0)
    model = build_model(arm, cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        z = torch.randint(0, V, (32, T + 1), generator=g)
        loss = per_position_loss(model, z).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    hold = torch.randint(0, V, (64, T + 1), generator=torch.Generator().manual_seed(999))
    with torch.no_grad():
        return per_position_loss(model.eval(), hold).mean().item(), math.log(V)


@pytest.mark.parametrize("arm", ARMS)
def test_held_out_random_loss_stays_at_ln_vocab(arm):
    loss, ln_v = held_out_loss(arm)
    assert abs(loss - ln_v) < 0.05, f"{arm}: held-out loss {loss:.3f} vs ln(V) {ln_v:.3f}"


def test_held_out_check_detects_leaky_d(monkeypatch):
    remove_causal_mask(monkeypatch)
    loss, ln_v = held_out_loss("D")
    assert loss < ln_v - 1.0, f"leaky D should collapse far below ln(V); got {loss:.3f} vs {ln_v:.3f}"
