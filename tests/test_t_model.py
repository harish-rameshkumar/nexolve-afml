import torch

from configs.model_config import default_config
from models import build_model, load_checkpoint, save_checkpoint


def test_output_shape(tiny_cfg):
    m = build_model("T", tiny_cfg)
    ids = torch.randint(0, tiny_cfg.vocab_size, (3, 20))
    assert m(ids).shape == (3, 20, tiny_cfg.vocab_size)


def test_default_param_count_about_28m():
    n = build_model("T", default_config()).num_params()
    print(f"T params: {n:,}")
    assert 26e6 < n < 30e6


def test_tied_embeddings(tiny_cfg):
    m = build_model("T", tiny_cfg)
    assert m.lm_head.weight is m.embed.weight


def test_no_biases(tiny_cfg):
    m = build_model("T", tiny_cfg)
    assert not any(n.endswith("bias") for n, _ in m.named_parameters())


def test_causality(tiny_cfg):
    """Changing a future token must not change earlier logits."""
    for impl in ("sdpa", "einsum"):
        cfg = default_config(**{**tiny_cfg.to_dict(), "attn_impl": impl})
        m = build_model("T", cfg).eval()
        a = torch.randint(0, cfg.vocab_size, (2, 16))
        b = a.clone()
        b[:, 10:] = (b[:, 10:] + 1) % cfg.vocab_size
        with torch.no_grad():
            la, lb = m(a), m(b)
        assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-5)
        assert not torch.allclose(la[:, 10:], lb[:, 10:], atol=1e-5)


def test_sdpa_matches_einsum(tiny_cfg):
    m = build_model("T", tiny_cfg).eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 24))
    with torch.no_grad():
        fast = m(ids)
        slow, maps = m(ids, return_attn=True)  # return_attn forces the einsum path
    assert torch.allclose(fast, slow, atol=1e-5)
    assert len(maps) == tiny_cfg.n_layers


def test_attention_maps_valid(tiny_cfg):
    m = build_model("T", tiny_cfg).eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 12))
    with torch.no_grad():
        _, maps = m(ids, return_attn=True)
    for a in maps:
        assert a.shape == (2, tiny_cfg.n_heads, 12, 12)
        assert torch.allclose(a.sum(-1), torch.ones(2, tiny_cfg.n_heads, 12), atol=1e-5)
        assert torch.all(torch.triu(a, diagonal=1) == 0)  # no attention to the future


def test_overfit_tiny_batch(tiny_cfg):
    torch.manual_seed(0)
    m = build_model("T", tiny_cfg)
    ids = torch.randint(0, tiny_cfg.vocab_size, (4, 17))
    x, y = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    loss_fn = torch.nn.functional.cross_entropy
    first = None
    for _ in range(150):
        loss = loss_fn(m(x).reshape(-1, tiny_cfg.vocab_size), y.reshape(-1))
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1 * first


def test_checkpoint_roundtrip(tiny_cfg, tmp_path):
    m = build_model("T", tiny_cfg).eval()
    save_checkpoint(tmp_path / "c.pt", m, step=7)
    m2, ck = load_checkpoint(tmp_path / "c.pt")
    m2.eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 10))
    assert ck["step"] == 7 and ck["arm"] == "T"
    assert torch.equal(m(ids), m2(ids))
