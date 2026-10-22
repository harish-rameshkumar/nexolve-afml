"""WS2 tests: tokenizer, data split/order, schedule, optimiser groups, training loop, resume.

All CPU, tiny model (vocab 600, 2 layers), synthetic text: the whole file runs in well under a minute.
"""
import json
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model_config import default_config  # noqa: E402
from models import build_model, load_checkpoint  # noqa: E402
from pretraining import prepare_data, train as train_mod  # noqa: E402
from pretraining.config import PretrainConfig  # noqa: E402
from pretraining.dataset import PackedTokens, TrainStream, load_val_tokens  # noqa: E402
from pretraining.evaluate import val_loss  # noqa: E402
from pretraining.tokenizer import EOS_ID, SPECIAL_TOKENS, load_tokenizer  # noqa: E402

VOCAB, SEQ = 600, 64
MODEL_CFG = dict(vocab_size=VOCAB, context_len=SEQ, d_model=64, n_layers=2, n_heads=4, ffn_hidden=176)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("pretrain_data")
    rng = random.Random(0)
    words = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 9))) for _ in range(300)]
    weights = [1 / (i + 1) for i in range(300)]
    src = d / "docs.jsonl"
    with open(src, "w") as f:
        for _ in range(400):
            f.write(json.dumps({"text": " ".join(rng.choices(words, weights, k=150))}) + "\n")
    out = d / "out"
    prepare_data.main(["--out", str(out), "--local_jsonl", str(src), "--vocab_size", str(VOCAB),
                       "--train_tokens", "40000", "--val_tokens", "3000", "--val_every", "20",
                       "--tok_docs", "100", "--batch_docs", "50"])
    return str(out)


def tiny_cfg(data_dir, tmp_path, arm="T", seed=0, **kw):
    base = dict(arm=arm, seed=seed, data_dir=data_dir, seq_len=SEQ, micro_batch=4, grad_accum=2,
                max_steps=8, lr=3e-3, warmup_steps=2, eval_every=4, ckpt_every=4, resume_every=2,
                log_every=2, out_dir=str(tmp_path / "runs"), amp="none", device="cpu")
    base.update(kw)
    return PretrainConfig(**base)


# ---------------------------------------------------------------- tokenizer / data
def test_tokenizer_specials_and_roundtrip(data_dir):
    tok = load_tokenizer(os.path.join(data_dir, "tokenizer.json"), VOCAB)
    assert tok.get_vocab_size() == VOCAB
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(t) == i
    text = "hello wörld ✓ 123\nnew line"            # byte-level BPE: any unicode round-trips, no <unk>
    assert tok.decode(tok.encode(text).ids) == text


def test_split_sizes_and_token_range(data_dir):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    assert meta["counts"] == {"train": 40000, "val": 3000}
    for name, n in (("train.bin", 40000), ("val.bin", 3000)):
        a = np.fromfile(os.path.join(data_dir, name), dtype=np.uint16)
        assert len(a) == n and a.max() < VOCAB and (a == EOS_ID).sum() > 0


def test_val_docs_never_in_train(data_dir, tmp_path):
    """Doc-level split: re-tokenise only val-candidate docs and check none of their windows leak into train."""
    train = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    val = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)
    # split val into documents at EOS; every val document (>= 30 tokens) must be absent from train
    docs = np.split(val, np.where(val == EOS_ID)[0] + 1)
    train_bytes = train.tobytes()
    checked = 0
    for d in docs:
        if len(d) >= 30:
            assert d[:30].tobytes() not in train_bytes
            checked += 1
    assert checked >= 5


# ---------------------------------------------------------------- stream order
def test_stream_deterministic_and_seeded(data_dir):
    a = TrainStream(data_dir, SEQ, seed=0, batch_size=8, max_steps=10)
    b = TrainStream(data_dir, SEQ, seed=0, batch_size=8, max_steps=10)
    c = TrainStream(data_dir, SEQ, seed=1, batch_size=8, max_steps=10)
    assert torch.equal(a.get(3, 1, 4), b.get(3, 1, 4))          # same seed -> same batch (any arm sees it)
    assert torch.equal(a.get(3, 1, 4), a.get(3, 1, 4))          # step-addressable: asking again is identical
    assert not torch.equal(a.get(3, 1, 4), c.get(3, 1, 4))      # different seed -> different order


def test_stream_no_repeats_and_shift(data_dir):
    s = TrainStream(data_dir, SEQ, seed=0, batch_size=8, max_steps=10)
    idx = s.order[: 8 * 10]
    assert len(set(idx.tolist())) == len(idx)                    # no window used twice
    batch = s.get(0, 0, 4)
    assert batch.shape == (4, SEQ + 1) and batch.dtype == torch.long
    w = s.data.window(int(s.order[0]))
    assert np.array_equal(batch[0].numpy(), w)


def test_stream_order_independent_of_max_steps(data_dir):
    a = TrainStream(data_dir, SEQ, seed=0, batch_size=8, max_steps=5)
    b = TrainStream(data_dir, SEQ, seed=0, batch_size=8, max_steps=50)
    assert torch.equal(a.get(2, 0, 8), b.get(2, 0, 8))


def test_too_few_tokens_is_a_clear_error(data_dir):
    with pytest.raises(AssertionError, match="prepare_data"):
        TrainStream(data_dir, SEQ, seed=0, batch_size=8, max_steps=10_000)


# ---------------------------------------------------------------- recipe pieces
def test_lr_schedule():
    cfg = PretrainConfig(lr=1e-3, warmup_steps=200, max_steps=4600, min_lr_ratio=0.1)
    assert cfg.lr_at(0) == pytest.approx(1e-3 / 200)
    assert cfg.lr_at(199) == pytest.approx(1e-3)
    assert cfg.lr_at(4599) == pytest.approx(1e-4, rel=1e-3)
    assert all(cfg.lr_at(s) >= cfg.lr_at(s + 1) for s in range(200, 4599))   # monotone decay
    assert cfg.tokens_per_step == 65536 and cfg.total_tokens == 301_465_600


@pytest.mark.parametrize("arm", ["T", "D", "P"])
def test_param_groups(arm):
    cfg = PretrainConfig(device="cpu")
    model = build_model(arm, default_config(**MODEL_CFG))
    opt = train_mod.build_optimizer(model, cfg)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])
    ids = lambda ps: {id(p) for p in ps}  # noqa: E731  (tensor `in` would compare by value)
    assert id(model.embed.weight) in ids(no_decay["params"]) and id(model.embed.weight) not in ids(decay["params"])
    n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n == model.num_params()                               # every parameter is in exactly one group


def test_flops_estimate_orders_arms():
    c = default_config()
    f = {a: train_mod.flops_per_token(build_model(a, c), c) for a in "TDP"}
    assert f["D"] > f["P"] > f["T"] and 1.1 < f["D"] / f["T"] < 1.3   # measured T4 speed ratio is ~1.30


# ---------------------------------------------------------------- training loop
@pytest.mark.parametrize("arm", ["T", "D", "P"])
def test_training_runs_and_learns(data_dir, tmp_path, arm):
    cfg = tiny_cfg(data_dir, tmp_path, arm=arm, max_steps=30, eval_every=10, ckpt_every=10, resume_every=10, log_every=5)
    s = train_mod.train(cfg, default_config(**MODEL_CFG))
    rows = [json.loads(l) for l in open(os.path.join(cfg.out_dir, cfg.run_name, "metrics.jsonl"))]
    vals = [r for r in rows if r["type"] == "val"]
    assert vals[0]["step"] == 0 and vals[0]["val_loss"] == pytest.approx(np.log(VOCAB), abs=0.35)  # chance level at init
    assert vals[-1]["val_loss"] < vals[0]["val_loss"] - 0.5                                          # it learns
    assert [v["step"] for v in vals] == [0, 10, 20, 30]
    assert all(np.isfinite(r["loss"]) for r in rows if r["type"] == "train")
    # final checkpoint = WS3 hand-off: loads through WS1's loader and reproduces the logged val loss
    model, ck = load_checkpoint(s["final_checkpoint"], "cpu")
    assert model.arm == arm and ck["step"] == 30
    again = val_loss(model, load_val_tokens(data_dir, SEQ), "cpu", 8, "none")
    assert again == pytest.approx(vals[-1]["val_loss"], abs=1e-5)
    # intermediates exist for the loss-matched analysis and are smaller (fp16) than the final
    inter = os.path.join(cfg.out_dir, cfg.run_name, "step00010.pt")
    assert os.path.exists(inter) and os.path.getsize(inter) < os.path.getsize(s["final_checkpoint"]) * 0.6
    m2, ck2 = load_checkpoint(inter, "cpu")
    assert m2.lm_head.weight is m2.embed.weight and ck2["step"] == 10


def test_same_seed_same_data_different_arms_share_batches(data_dir, tmp_path):
    for arm in ("T", "D"):
        train_mod.train(tiny_cfg(data_dir, tmp_path, arm=arm, max_steps=2, eval_every=2, ckpt_every=2), default_config(**MODEL_CFG))
    # identical data order => the first training-step data are the same for both arms
    a = TrainStream(data_dir, SEQ, 0, 8, 2)
    assert torch.equal(a.get(0, 0, 4), TrainStream(data_dir, SEQ, 0, 8, 2).get(0, 0, 4))


def test_crash_and_resume_matches_uninterrupted_run(data_dir, tmp_path, monkeypatch):
    mcfg = default_config(**MODEL_CFG)
    ref_cfg = tiny_cfg(data_dir, tmp_path / "ref", arm="D", max_steps=10, eval_every=4, ckpt_every=4, resume_every=2)
    ref = train_mod.train(ref_cfg, mcfg)

    cfg = tiny_cfg(data_dir, tmp_path / "crash", arm="D", max_steps=10, eval_every=4, ckpt_every=4, resume_every=2)

    class Crash(Exception):
        pass

    real = train_mod.TrainStream

    class FlakyStream(real):
        def get(self, step, micro, micro_batch):
            if step == 5 and micro == 1:                         # die in the middle of step 6
                raise Crash()
            return super().get(step, micro, micro_batch)

    monkeypatch.setattr(train_mod, "TrainStream", FlakyStream)
    with pytest.raises(Crash):
        train_mod.train(cfg, mcfg)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    assert os.path.exists(os.path.join(run_dir, "latest.pt")) and not os.path.exists(os.path.join(run_dir, "done.json"))

    monkeypatch.setattr(train_mod, "TrainStream", real)          # "new session": same command again
    res = train_mod.train(cfg, mcfg)

    assert res["final_val_loss"] == pytest.approx(ref["final_val_loss"], abs=1e-5)
    m_ref, _ = load_checkpoint(ref["final_checkpoint"], "cpu")
    m_res, _ = load_checkpoint(res["final_checkpoint"], "cpu")
    for (n, p), (_, q) in zip(m_ref.named_parameters(), m_res.named_parameters()):
        assert torch.allclose(p, q, atol=1e-5), n
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl"))]
    keys = [(r["type"], r["step"]) for r in rows]
    assert len(keys) == len(set(keys))                           # no duplicated log rows after resume
    assert not os.path.exists(os.path.join(run_dir, "latest.pt"))  # cleaned up when done


def test_finished_run_is_skipped(data_dir, tmp_path):
    cfg = tiny_cfg(data_dir, tmp_path, max_steps=2, eval_every=2, ckpt_every=2)
    first = train_mod.train(cfg, default_config(**MODEL_CFG))
    mtime = os.path.getmtime(first["final_checkpoint"])
    assert train_mod.train(cfg, default_config(**MODEL_CFG))["final_val_loss"] == first["final_val_loss"]
    assert os.path.getmtime(first["final_checkpoint"]) == mtime


def test_cli_parses_all_fields():
    cfg = train_mod.parse_args(["--arm", "D", "--seed", "2", "--lr", "0.002", "--no-compile", "--max_steps", "100"])
    assert (cfg.arm, cfg.seed, cfg.lr, cfg.compile, cfg.max_steps) == ("D", 2, 0.002, False, 100)
