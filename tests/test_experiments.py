"""Smoke tests: the exploratory scripts in scripts/experiments run, are deterministic, and behave as documented."""
import math

import torch

from scripts.experiments import fixed_batch_sweep, heldout_leak_control, lambda_ablation
from scripts.experiments import _common as C


def test_fixed_batch_sweep_runs_and_is_deterministic(capsys):
    args = ["--config", "tiny", "--arms", "T", "D", "--seeds", "0", "--lrs", "3e-4", "--steps", "10"]
    a = fixed_batch_sweep.main(args)
    out = capsys.readouterr().out
    b = fixed_batch_sweep.main(args)
    assert a == b                                           # same seed -> identical numbers
    assert "no sign of leakage" in out
    ln_v = math.log(16384)
    for rec in a.values():
        assert rec[-1][1] < rec[0][1]                       # fixed-batch loss goes down
        assert all(x[2] > ln_v - 0.05 for x in rec)         # fresh-batch loss stays at/above ln V


def test_lambda_ablation_runs_and_frozen_lambda_has_no_parameters():
    res = lambda_ablation.main(["--config", "tiny", "--seeds", "0", "--steps", "6"])
    assert set(res) == {"T", "D", "D-lam0", "D-lam1"} and all(len(v) == 1 and math.isfinite(v[0]) for v in res.values())
    frozen = lambda_ablation.FrozenLambdaLogit(4, -30.0)
    assert sum(p.numel() for p in frozen.parameters()) == 0
    lam = torch.sigmoid(frozen(torch.zeros(2, 5, 64)))
    assert lam.shape == (2, 5, 4) and lam.max() < 1e-6


def test_heldout_leak_control_detects_removed_mask():
    rows = heldout_leak_control.main(["--arms", "D", "--steps", "100"])
    intact = next(r for r in rows if not r[2])
    leaky = next(r for r in rows if r[2])
    assert abs(intact[3] - intact[4]) < 0.05 and leaky[3] < intact[4] - 1.0
    # the patched mask must be restored afterwards
    import models.diff_attention as diff_mod
    assert diff_mod.causal_mask(3, torch.device("cpu")).sum().item() == 6   # lower triangle again


def test_random_batch_is_seeded_and_device_independent():
    x1, y1 = C.random_batch(100, 2, 8, 5, torch.device("cpu"))
    x2, y2 = C.random_batch(100, 2, 8, 5, torch.device("cpu"))
    assert torch.equal(x1, x2) and torch.equal(y1[:, :-1], x1[:, 1:])      # targets are inputs shifted by one
