import pytest
import torch

from configs.model_config import default_config
from models import build_model, load_checkpoint, parameter_summary, save_checkpoint
from models.attention import StandardAttention
from models.build import diff_extra_params, matched_ffn_hidden


def test_p_matches_d_within_one_percent():
    cfg = default_config()
    s = parameter_summary(cfg, verbose=False)
    print(f"T={s['T']:,} D={s['D']:,} P={s['P']:,}  P-D={s['P_minus_D']:+,} ({s['P_vs_D_pct']:+.4f}%)")
    assert abs(s["P_vs_D_pct"]) < 1.0
    # Default config: 1408 + 173.33 = 1581.33 -> nearest multiple of 16 = 1584,
    # which is 24,576 params (+0.0840%) above D.
    assert s["ffn_hidden"]["P"] == 1584
    assert s["P_minus_D"] == 24_576
    assert s["P"] == 29_284_864 and s["D"] == 29_260_288
    assert round(s["P_vs_D_pct"], 4) == 0.0840
    # The summary (meta device) agrees with really-built models.
    assert build_model("P", cfg).num_params() == s["P"]
    assert build_model("D", cfg).num_params() == s["D"]


def test_matched_hidden_formula():
    cfg = default_config()
    assert diff_extra_params(cfg) == 1_597_440
    assert matched_ffn_hidden(cfg, multiple=1) == 1581   # exact integer
    assert matched_ffn_hidden(cfg) == 1584               # default: multiple of 16
    # Generalises to another config: rounding to a multiple of 16 is off by at most 8 hidden units.
    small = default_config(d_model=128, n_heads=4, n_layers=3, ffn_hidden=352, vocab_size=512)
    d = build_model("D", small).num_params()
    p = build_model("P", small).num_params()
    assert abs(p - d) <= 8 * 3 * small.d_model * small.n_layers


def test_p_attention_identical_to_t(tiny_cfg):
    t, p = build_model("T", tiny_cfg), build_model("P", tiny_cfg)
    for bt, bp in zip(t.blocks, p.blocks):
        assert type(bp.attn) is StandardAttention and type(bt.attn) is type(bp.attn)
        st = {n: v.shape for n, v in bt.attn.state_dict().items()}
        sp = {n: v.shape for n, v in bp.attn.state_dict().items()}
        assert st == sp


def test_only_ffn_width_differs_from_t(tiny_cfg):
    t = dict(build_model("T", tiny_cfg).named_parameters())
    p = dict(build_model("P", tiny_cfg).named_parameters())
    assert t.keys() == p.keys()
    differing = {n for n in t if t[n].shape != p[n].shape}
    assert differing and all(".ffn." in n for n in differing)
    assert p["blocks.0.ffn.gate.weight"].shape[0] == matched_ffn_hidden(tiny_cfg)


def test_ffn_hidden_p_override(tiny_cfg):
    cfg = default_config(**{**tiny_cfg.to_dict(), "ffn_hidden_p": 256})
    assert build_model("P", cfg).blocks[0].ffn.gate.out_features == 256


def test_all_arms_from_same_config_object(tiny_cfg):
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 10))
    for arm in ("T", "D", "P"):
        m = build_model(arm, tiny_cfg)
        assert m.arm == arm and m.cfg is tiny_cfg
        assert m(ids).shape == (2, 10, tiny_cfg.vocab_size)


def test_unknown_arm_raises(tiny_cfg):
    with pytest.raises(ValueError):
        build_model("X", tiny_cfg)


def test_output_shape(tiny_cfg):
    m = build_model("P", tiny_cfg)
    assert m(torch.randint(0, tiny_cfg.vocab_size, (3, 20))).shape == (3, 20, tiny_cfg.vocab_size)


@pytest.mark.parametrize("impl", ["sdpa", "einsum"])
def test_causality(tiny_cfg, impl):
    cfg = default_config(**{**tiny_cfg.to_dict(), "attn_impl": impl})
    m = build_model("P", cfg).eval()
    a = torch.randint(0, cfg.vocab_size, (2, 16))
    b = a.clone()
    b[:, 10:] = (b[:, 10:] + 1) % cfg.vocab_size
    with torch.no_grad():
        la, lb = m(a), m(b)
    assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-5)
    assert not torch.allclose(la[:, 10:], lb[:, 10:], atol=1e-5)


def test_overfit_tiny_batch(tiny_cfg):
    torch.manual_seed(0)
    m = build_model("P", tiny_cfg)
    ids = torch.randint(0, tiny_cfg.vocab_size, (4, 17))
    x, y = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, tiny_cfg.vocab_size), y.reshape(-1))
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1 * first


def test_checkpoint_roundtrip(tiny_cfg, tmp_path):
    m = build_model("P", tiny_cfg).eval()
    save_checkpoint(tmp_path / "p.pt", m, step=5)
    m2, ck = load_checkpoint(tmp_path / "p.pt")
    m2.eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 10))
    assert ck["arm"] == "P"
    assert m2.blocks[0].ffn.gate.out_features == m.blocks[0].ffn.gate.out_features
    assert torch.equal(m(ids), m2(ids))
