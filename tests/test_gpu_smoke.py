import json

import pytest
import torch

import scripts.gpu_smoke_test as smoke
from configs.model_config import default_config


def test_cpu_dry_run_end_to_end(tmp_path):
    out = tmp_path / "res.json"
    assert smoke.main(["--cpu-dry-run", "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["all_passed"] and res["dry_run"]
    assert {r["arm"] for r in res["runs"]} == {"T", "D", "P"}
    assert all(r["status"] == "ok" for r in res["runs"])


def test_no_gpu_exits_cleanly_with_message(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert smoke.main(["--out", "unused.json"]) == 2
    assert "No CUDA GPU found" in capsys.readouterr().out


def test_oom_retry_halves_batch(monkeypatch):
    cfg = default_config(vocab_size=64, context_len=16, d_model=32, n_layers=1, n_heads=2, ffn_hidden=96)
    real = smoke.run_training

    def fake(arm, cfg, mode, batch, seq_len, steps, dev):
        if batch > 2:
            raise torch.OutOfMemoryError("simulated")
        return real(arm, cfg, mode, batch, seq_len, steps, dev)

    monkeypatch.setattr(smoke, "run_training", fake)
    res = smoke.run_with_oom_retry("T", cfg, ("fp32", None, False, None), 8, 16, 5, torch.device("cpu"))
    assert res["status"] == "ok" and res["batch"] == 2
    assert len(res["notes"]) == 2 and "retrying with batch 4" in res["notes"][0]


def test_nonfinite_loss_is_reported_as_failure(monkeypatch):
    cfg = default_config(vocab_size=64, context_len=16, d_model=32, n_layers=1, n_heads=2, ffn_hidden=96)
    real_ce = torch.nn.functional.cross_entropy
    monkeypatch.setattr(smoke.F, "cross_entropy", lambda *a, **k: real_ce(*a, **k) * float("nan"))
    res = smoke.run_training("T", cfg, ("fp32", None, False, None), 2, 16, 5, torch.device("cpu"))
    assert not res["all_finite"] and res["first_nonfinite_step"] == 0 and not res["loss_decreased"]
