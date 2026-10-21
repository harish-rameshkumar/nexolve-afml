import pytest
import torch

import scripts.plot_attention as plot_attention
import scripts.plot_lambda as plot_lambda
from models import attention_maps, build_model, lambda_log_dict, lambda_summary, save_checkpoint


def test_lambda_summary_keys_and_ranges(tiny_cfg):
    m = build_model("D", tiny_cfg)
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 12))
    s = lambda_summary(m, ids)
    L, H = tiny_cfg.n_layers, tiny_cfg.n_heads
    assert set(s) == {"mean", "std", "min", "max", "head_mean"}
    for k in ("mean", "std", "min", "max"):
        assert s[k].shape == (L,)
    assert s["head_mean"].shape == (L, H)
    for k in ("mean", "min", "max", "head_mean"):
        assert torch.all((s[k] > 0) & (s[k] < 1))
    assert torch.all(s["min"] <= s["mean"]) and torch.all(s["mean"] <= s["max"])
    assert torch.all(s["std"] >= 0)


def test_lambda_log_dict_is_flat_floats(tiny_cfg):
    m = build_model("D", tiny_cfg)
    d = lambda_log_dict(m, torch.randint(0, tiny_cfg.vocab_size, (2, 8)))
    assert all(isinstance(k, str) and type(v) is float for k, v in d.items())
    assert "lambda/layer0/mean" in d and f"lambda/layer{tiny_cfg.n_layers - 1}/head{tiny_cfg.n_heads - 1}/mean" in d
    # 4 stats + n_heads head means, per layer
    assert len(d) == tiny_cfg.n_layers * (4 + tiny_cfg.n_heads)
    assert "myrun/layer0/std" in lambda_log_dict(m, torch.zeros(1, 4, dtype=torch.long), prefix="myrun")


def test_lambda_summary_no_grad_and_mode_preserved(tiny_cfg):
    m = build_model("D", tiny_cfg).train()
    ids = torch.randint(0, tiny_cfg.vocab_size, (1, 6))
    s = lambda_summary(m, ids)
    assert m.training                      # mode restored
    assert not s["mean"].requires_grad     # no graph kept
    with torch.no_grad():                  # also fine when the caller is already in no_grad
        lambda_summary(m, ids)


@pytest.mark.parametrize("arm", ["T", "P"])
def test_lambda_summary_errors_for_t_and_p(tiny_cfg, arm):
    m = build_model(arm, tiny_cfg)
    with pytest.raises(ValueError, match="only D has lambda"):
        lambda_summary(m, torch.zeros(1, 4, dtype=torch.long))
    with pytest.raises(ValueError):
        lambda_log_dict(m, torch.zeros(1, 4, dtype=torch.long))


def test_attention_maps_all_arms(tiny_cfg):
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 10))
    shape = (tiny_cfg.n_layers, 2, tiny_cfg.n_heads, 10, 10)
    for arm in ("T", "P", "D"):
        a = attention_maps(build_model(arm, tiny_cfg), ids)
        assert a.shape == shape and a.device.type == "cpu"
        assert torch.all(torch.triu(a, diagonal=1) == 0)
    # T/P: ordinary softmax (non-negative, rows sum to 1). D: signed, rows sum to 1 - lambda.
    assert attention_maps(build_model("T", tiny_cfg), ids).min() >= 0
    d = build_model("D", tiny_cfg)
    amap = attention_maps(d, ids)
    lam = d.lambda_values().permute(0, 1, 3, 2)          # (L, B, H, T)
    assert torch.allclose(amap.sum(-1), 1 - lam, atol=1e-5)


def test_plot_attention_end_to_end(tmp_path):
    out = tmp_path / "att.png"
    assert plot_attention.main(["--tiny", "--out", str(out), "--seq-len", "12"]) == str(out)
    assert out.stat().st_size > 1000
    out2 = tmp_path / "att_text.png"     # byte-level text fallback
    plot_attention.main(["--tiny", "--text", "hello world", "--heads", "0", "--layers", "1", "--out", str(out2)])
    assert out2.exists()


def test_plot_attention_with_checkpoints(tiny_cfg, tmp_path):
    save_checkpoint(tmp_path / "t.pt", build_model("T", tiny_cfg))
    save_checkpoint(tmp_path / "d.pt", build_model("D", tiny_cfg))
    out = tmp_path / "a.png"
    plot_attention.main(["--ckpt-t", str(tmp_path / "t.pt"), "--ckpt-d", str(tmp_path / "d.pt"),
                         "--ids", "1,2,3,4,5,6", "--out", str(out)])
    assert out.exists()
    with pytest.raises(SystemExit):       # wrong arm is rejected, not silently plotted
        plot_attention.main(["--ckpt-t", str(tmp_path / "d.pt"), "--out", str(out)])


def test_plot_lambda_end_to_end(tiny_cfg, tmp_path):
    out = tmp_path / "lam.png"
    plot_lambda.main(["--tiny", "--out", str(out), "--seq-len", "16"])
    assert out.stat().st_size > 1000
    save_checkpoint(tmp_path / "before.pt", build_model("D", tiny_cfg))
    save_checkpoint(tmp_path / "after.pt", build_model("D", tiny_cfg))
    out2 = tmp_path / "cmp.png"
    plot_lambda.main(["--ckpt", str(tmp_path / "before.pt"), str(tmp_path / "after.pt"),
                      "--labels", "before", "after", "--out", str(out2)])
    assert out2.exists()
