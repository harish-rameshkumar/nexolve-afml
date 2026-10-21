import pytest
import torch

from configs.model_config import default_config
from models import build_model, load_checkpoint, save_checkpoint
from models.diff_attention import DiffAttentionV2
from tests.reference_diff import naive_diff_attention


def cfg_with(base, **kw):
    return default_config(**{**base.to_dict(), **kw})


def test_output_shape(tiny_cfg):
    m = build_model("D", tiny_cfg)
    ids = torch.randint(0, tiny_cfg.vocab_size, (3, 20))
    assert m(ids).shape == (3, 20, tiny_cfg.vocab_size)


def test_param_count_vs_t():
    cfg = default_config()
    t, d = build_model("T", cfg).num_params(), build_model("D", cfg).num_params()
    print(f"T={t:,}  D={d:,}  D-T={d - t:,} ({(d - t) / t:.2%})")
    # Extra per layer: Q2 projection (d*H*hd) + lambda_proj (d*H).
    per_layer = cfg.d_model * cfg.n_heads * cfg.head_dim + cfg.d_model * cfg.n_heads
    assert d - t == cfg.n_layers * per_layer == 1_597_440


def test_only_attention_differs_from_t(tiny_cfg):
    """Everything outside the attention module has identical names and shapes."""
    ts = {n: p.shape for n, p in build_model("T", tiny_cfg).named_parameters() if ".attn." not in n}
    ds = {n: p.shape for n, p in build_model("D", tiny_cfg).named_parameters() if ".attn." not in n}
    assert ts == ds


def test_q_k_v_head_counts(tiny_cfg):
    a = build_model("D", tiny_cfg).blocks[0].attn
    H, hd, d = tiny_cfg.n_heads, tiny_cfg.head_dim, tiny_cfg.d_model
    assert a.q_proj.weight.shape == (2 * H * hd, d)      # 2H query heads
    assert a.k_proj.weight.shape == (H * hd, d)          # H key heads
    assert a.v_proj.weight.shape == (H * hd, d)          # H value heads
    assert a.o_proj.weight.shape == (d, H * hd)          # same as T
    assert a.lambda_proj.weight.shape == (H, d)          # one lambda per token per head


@pytest.mark.parametrize("impl", ["sdpa", "einsum"])
def test_causality(tiny_cfg, impl):
    m = build_model("D", cfg_with(tiny_cfg, attn_impl=impl)).eval()
    a = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    b = a.clone()
    b[:, 10:] = (b[:, 10:] + 1) % tiny_cfg.vocab_size
    with torch.no_grad():
        la, lb = m(a), m(b)
    assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-5)
    assert not torch.allclose(la[:, 10:], lb[:, 10:], atol=1e-5)


def test_sdpa_matches_einsum(tiny_cfg):
    m = build_model("D", tiny_cfg).eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 24))
    with torch.no_grad():
        fast = m(ids)
        slow, maps = m(ids, return_attn=True)   # return_attn forces the einsum path
    assert torch.allclose(fast, slow, atol=1e-5)
    assert len(maps) == tiny_cfg.n_layers


def _small_attn(tiny_cfg):
    cfg = cfg_with(tiny_cfg, d_model=16, n_heads=2, ffn_hidden=32, context_len=8)
    torch.manual_seed(0)
    mod = DiffAttentionV2(cfg).eval()
    # Larger lambda weights so lambda is clearly not constant 0.5.
    torch.nn.init.normal_(mod.lambda_proj.weight, std=1.0)
    return mod, torch.randn(2, 6, 16)


@pytest.mark.parametrize("impl", ["sdpa", "einsum"])
def test_matches_naive_reference(tiny_cfg, impl):
    mod, x = _small_attn(tiny_cfg)
    mod.impl = impl
    with torch.no_grad():
        out, _ = mod(x)
        ref_out, ref_map = naive_diff_attention(mod, x)
        _, my_map = mod(x, return_attn=True)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.allclose(my_map, ref_map, atol=1e-5)   # signed A1 - lam*A2 map


def test_wrong_half_half_pairing_differs(tiny_cfg):
    """Locks in the 0::2 / 1::2 pairing: the blog's Ablation 1 split gives other outputs."""
    mod, x = _small_attn(tiny_cfg)
    with torch.no_grad():
        out, _ = mod(x)
        good, _ = naive_diff_attention(mod, x, pairing="interleaved")
        bad, _ = naive_diff_attention(mod, x, pairing="halves")
    assert torch.allclose(out, good, atol=1e-5)
    assert not torch.allclose(out, bad, atol=1e-3)


def test_diff_map_rows_sum_to_one_minus_lambda(tiny_cfg):
    mod, x = _small_attn(tiny_cfg)
    with torch.no_grad():
        _, amap = mod(x, return_attn=True)               # (B, H, T, T)
    lam = mod.last_lambda.permute(0, 2, 1)               # (B, H, T)
    assert torch.allclose(amap.sum(-1), 1 - lam, atol=1e-5)
    assert torch.all(torch.triu(amap, diagonal=1) == 0)  # nothing from the future


def test_lambda_exposed_per_layer_and_head(tiny_cfg):
    m = build_model("D", tiny_cfg).eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 12))
    with torch.no_grad():
        m(ids)
    lam = m.lambda_values()
    assert lam.shape == (tiny_cfg.n_layers, 2, 12, tiny_cfg.n_heads)
    assert lam.min() > 0 and lam.max() < 1
    assert lam.mean(dim=(1, 2)).shape == (tiny_cfg.n_layers, tiny_cfg.n_heads)
    # at init sigmoid(~0) ~ 0.5
    assert abs(lam.mean().item() - 0.5) < 0.05


def test_t_has_no_lambda(tiny_cfg):
    m = build_model("T", tiny_cfg)
    m(torch.randint(0, tiny_cfg.vocab_size, (1, 4)))
    with pytest.raises(RuntimeError):
        m.lambda_values()


def test_overfit_tiny_batch(tiny_cfg):
    torch.manual_seed(0)
    m = build_model("D", tiny_cfg)
    ids = torch.randint(0, tiny_cfg.vocab_size, (4, 17))
    x, y = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, tiny_cfg.vocab_size), y.reshape(-1))
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1 * first


@pytest.mark.parametrize("impl", ["einsum", "sdpa"])
def test_gradcheck_double(tiny_cfg, impl):
    """Analytic vs numeric gradients of a small float64 differential-attention module."""
    cfg = cfg_with(tiny_cfg, d_model=8, n_heads=2, ffn_hidden=16, context_len=5, attn_impl=impl)
    torch.manual_seed(0)
    mod = DiffAttentionV2(cfg).double()
    x = torch.randn(1, 4, 8, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: mod(inp)[0], (x,), eps=1e-6, atol=1e-5)
    # and w.r.t. a parameter (lambda_proj), the piece unique to D
    xd = x.detach()

    def f(w):
        return torch.func.functional_call(mod, {"lambda_proj.weight": w}, (xd,))[0]
    w0 = mod.lambda_proj.weight.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (w0,), eps=1e-6, atol=1e-5)


def test_checkpoint_roundtrip(tiny_cfg, tmp_path):
    m = build_model("D", tiny_cfg).eval()
    save_checkpoint(tmp_path / "d.pt", m, step=3)
    m2, ck = load_checkpoint(tmp_path / "d.pt")
    m2.eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 10))
    assert ck["arm"] == "D"
    assert torch.equal(m(ids), m2(ids))
