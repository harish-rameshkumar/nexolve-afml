"""GPU smoke test for the T / D / P models.

    python scripts/gpu_smoke_test.py                  # on a CUDA GPU (e.g. Colab T4)
    python scripts/gpu_smoke_test.py --cpu-dry-run    # logic check on CPU: tiny config, no CUDA-only parts

For each arm and each available precision (fp32, bf16 autocast, fp16 autocast + GradScaler) it runs 20 AdamW
training steps on one fixed random batch and checks that the loss is finite at every step and goes down,
and reports peak GPU memory and tokens/second. It also checks, on the device, that the fused (SDPA) and
explicit (einsum) attention paths agree, that causality holds, and that the lambda / attention-map
diagnostics run. Results go to a JSON file and a summary table; the exit code is 1 if any check fails.
"""
import argparse
import gc
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from configs.model_config import default_config  # noqa: E402
from models import attention_maps, build_model, lambda_log_dict  # noqa: E402

ARMS = ("T", "D", "P")
WARMUP_STEPS = 3
LR = 3e-4
OOM_ERRORS = tuple(e for e in (getattr(torch, "OutOfMemoryError", None), torch.cuda.OutOfMemoryError) if e)


# ----------------------------------------------------------------------------- helpers
def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def cleanup(dev):
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def environment(dev):
    info = {"torch": torch.__version__, "device": str(dev), "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda}
    if dev.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(dev)
        info["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(dev)))
        info["total_memory_gb"] = round(torch.cuda.get_device_properties(dev).total_memory / 2**30, 2)
        info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        try:   # newer torch can tell native bf16 from emulated bf16 (e.g. T4 emulates it, slowly)
            info["bf16_native"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:
            info["bf16_native"] = None
    return info


def precision_modes(dev, dry_run):
    """[(name, autocast dtype or None, use GradScaler, skip reason or None)]"""
    modes = [("fp32", None, False, None)]
    if dev.type == "cuda":
        ok = torch.cuda.is_bf16_supported()
        modes.append(("bf16", torch.bfloat16, False, None if ok else "bf16 not supported on this GPU"))
        modes.append(("fp16", torch.float16, True, None))
    elif dry_run:   # exercise the same code paths on CPU where PyTorch allows it
        modes.append(("bf16", torch.bfloat16, False, None))
        modes.append(("fp16", torch.float16, True, None))
    else:
        modes.append(("bf16", torch.bfloat16, False, f"autocast modes are only run on CUDA (device={dev.type})"))
        modes.append(("fp16", torch.float16, True, f"autocast modes are only run on CUDA (device={dev.type})"))
    return modes


# ----------------------------------------------------------------------------- training run
def run_training(arm, cfg, mode, batch, seq_len, steps, dev):
    name, amp_dtype, use_scaler, _ = mode
    torch.manual_seed(0)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = build_model(arm, cfg).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(dev.type, enabled=use_scaler)
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, cfg.vocab_size, (batch, seq_len + 1), generator=g).to(dev)  # ONE fixed batch
    x, y = data[:, :-1], data[:, 1:]

    losses, t0 = [], None
    for step in range(steps):
        if step == WARMUP_STEPS:
            sync(dev)
            t0 = time.perf_counter()
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.detach().float())          # no .item(): keeps the timing free of extra syncs
    sync(dev)
    elapsed = time.perf_counter() - t0
    losses = torch.stack(losses).cpu().tolist()

    finite = [math.isfinite(v) for v in losses]
    first_bad = None if all(finite) else finite.index(False)
    final = sum(losses[-3:]) / 3
    return {
        "arm": arm, "precision": name, "batch": batch, "seq_len": seq_len, "steps": steps,
        "params": model.num_params(),
        "loss_first": losses[0], "loss_final_mean_last3": final, "losses": losses,
        "all_finite": all(finite), "first_nonfinite_step": first_bad,
        "loss_decreased": all(finite) and final < losses[0],
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3) if dev.type == "cuda" else None,
        "tokens_per_sec": round(batch * seq_len * (steps - WARMUP_STEPS) / elapsed, 1),
        "status": "ok",
    }


def run_with_oom_retry(arm, cfg, mode, batch, seq_len, steps, dev):
    notes, b = [], batch
    while True:
        oom = False
        try:
            res = run_training(arm, cfg, mode, b, seq_len, steps, dev)
            res["notes"] = notes
            return res
        except OOM_ERRORS:
            oom = True
        except RuntimeError as e:   # e.g. an unsupported op for this precision on this GPU
            cleanup(dev)
            return {"arm": arm, "precision": mode[0], "batch": b, "status": "error",
                    "error": f"{type(e).__name__}: {str(e)[:300]}", "notes": notes}
        if oom:
            cleanup(dev)
            if b == 1:
                return {"arm": arm, "precision": mode[0], "batch": b, "status": "error",
                        "error": "out of memory even at batch size 1", "notes": notes}
            msg = f"out of memory at batch {b}; retrying with batch {b // 2}"
            print(f"    [{arm}/{mode[0]}] {msg}")
            notes.append(msg)
            b //= 2


# ----------------------------------------------------------------------------- correctness checks
def check(name, passed, detail):
    print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")
    return {"name": name, "passed": bool(passed), "detail": detail}


def correctness_checks(cfg, dev):
    results = []
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (2, min(128, cfg.context_len))).to(dev)
    T = ids.shape[1]
    cut = T // 2
    ids2 = ids.clone()
    ids2[:, cut:] = (ids2[:, cut:] + 1) % cfg.vocab_size   # change the "future"

    for arm in ARMS:
        model = build_model(arm, cfg).to(dev).eval()
        with torch.no_grad():
            fused = model(ids)                         # default attn_impl="sdpa" path
            explicit, _ = model(ids, return_attn=True)  # forces the einsum path
        diff = (fused - explicit).abs().max().item()
        results.append(check(f"{arm}: SDPA vs einsum logits (fp32)", diff < 1e-4, f"max |diff| = {diff:.2e} (tol 1e-4)"))

        for impl in ("sdpa", "einsum"):
            for m in model.modules():   # per-module switch only; never mutate the shared cfg
                if hasattr(m, "impl"):
                    m.impl = impl
            with torch.no_grad():
                a, b = model(ids), model(ids2)
            early = (a[:, :cut] - b[:, :cut]).abs().max().item()
            late = (a[:, cut:] - b[:, cut:]).abs().max().item()
            results.append(check(f"{arm}: causality on device ({impl})", early < 1e-4 and late > 1e-4,
                                 f"earlier logits changed by {early:.2e} (want ~0), later by {late:.2e} (want > 0)"))
        del model
        cleanup(dev)

    # diagnostics on the device (D only; T must refuse to give lambda)
    d = build_model("D", cfg).to(dev)
    probe = ids[:, :64]
    log = lambda_log_dict(d, probe)
    vals = list(log.values())
    ok = (len(log) == cfg.n_layers * (4 + cfg.n_heads) and all(math.isfinite(v) and 0 < v < 1 for v in vals)
          and all(type(v) is float for v in vals))
    results.append(check("D: lambda_log_dict on device", ok, f"{len(log)} finite float values in (0,1)"))
    maps = attention_maps(d, probe)
    expect = (cfg.n_layers, 2, cfg.n_heads, probe.shape[1], probe.shape[1])
    ok = tuple(maps.shape) == expect and bool(torch.isfinite(maps).all()) and maps.device.type == "cpu"
    results.append(check("D: attention_maps on device", ok, f"shape {tuple(maps.shape)}, finite, returned on CPU"))
    t = build_model("T", cfg).to(dev)
    try:
        lambda_log_dict(t, probe)
        results.append(check("T: lambda_log_dict refuses", False, "no error raised"))
    except ValueError:
        results.append(check("T: lambda_log_dict refuses", True, "ValueError raised"))
    return results


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-dry-run", action="store_true",
                    help="logic check on CPU: tiny config, small batch, skips CUDA-only parts (memory stats, GPU info)")
    ap.add_argument("--device", default=None, help="cuda (default), or cpu/mps for a rough check")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="gpu_smoke_results.json")
    args = ap.parse_args(argv)
    assert args.steps > WARMUP_STEPS, f"--steps must exceed {WARMUP_STEPS} warmup steps"

    if args.cpu_dry_run:
        dev = torch.device("cpu")
        cfg = default_config(vocab_size=256, context_len=64, d_model=64, n_layers=2, n_heads=4, ffn_hidden=176)
        args.batch, args.seq_len = min(args.batch, 2), min(args.seq_len, 32)
    else:
        dev = torch.device(args.device or "cuda")
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("No CUDA GPU found. On Colab: Runtime > Change runtime type > choose a GPU, then re-run.")
            return 2
        cfg = default_config()

    env = environment(dev)
    print("=" * 72)
    print(f"mode: {'CPU DRY RUN (tiny config; not a GPU result)' if args.cpu_dry_run else 'full'}")
    for k, v in env.items():
        print(f"  {k}: {v}")
    if dev.type == "cuda":
        print(f"  bf16: {'native' if env['bf16_native'] else ('EMULATED (slow)' if env['bf16_supported'] else 'not supported')}")
    print(f"  config: d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} vocab={cfg.vocab_size} "
          f"ctx={cfg.context_len}; batch={args.batch} seq_len={args.seq_len} steps={args.steps}")
    print("=" * 72)

    print("Correctness checks")
    checks = correctness_checks(cfg, dev)

    print("\nTraining runs")
    runs = []
    for arm in ARMS:
        for mode in precision_modes(dev, args.cpu_dry_run):
            if mode[3]:
                print(f"  SKIP  {arm}/{mode[0]}: {mode[3]}")
                runs.append({"arm": arm, "precision": mode[0], "status": "skipped", "reason": mode[3]})
                continue
            res = run_with_oom_retry(arm, cfg, mode, args.batch, args.seq_len, args.steps, dev)
            runs.append(res)
            if res["status"] == "ok":
                print(f"  done  {arm}/{mode[0]}: loss {res['loss_first']:.3f} -> {res['loss_final_mean_last3']:.3f}, "
                      f"{res['tokens_per_sec']:,.0f} tok/s")
            else:
                print(f"  ERROR {arm}/{mode[0]}: {res['error']}")
            cleanup(dev)

    for r in runs:
        if r["status"] != "ok":
            continue
        tag = f"{r['arm']}/{r['precision']}"
        checks.append(check(f"{tag}: loss finite on all {r['steps']} steps", r["all_finite"],
                            "all finite" if r["all_finite"] else f"NON-FINITE loss first seen at step {r['first_nonfinite_step']}"))
        checks.append(check(f"{tag}: loss decreased on fixed batch", r["loss_decreased"],
                            f"{r['loss_first']:.3f} -> {r['loss_final_mean_last3']:.3f} (mean of last 3)"))
    for r in runs:
        if r["status"] == "error":
            checks.append(check(f"{r['arm']}/{r['precision']}: run completed", False, r["error"]))

    passed = all(c["passed"] for c in checks)
    results = {"environment": env, "dry_run": args.cpu_dry_run, "batch": args.batch, "seq_len": args.seq_len,
               "steps": args.steps, "lr": LR, "runs": runs, "checks": checks, "all_passed": passed}
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72 + "\nSUMMARY" + (" (CPU DRY RUN: not a GPU result)" if args.cpu_dry_run else ""))
    print(f"{'arm':<4}{'prec':<6}{'batch':>6}{'loss first':>12}{'loss last3':>12}{'peak mem GB':>13}{'tokens/s':>11}  status")
    for r in runs:
        if r["status"] == "ok":
            mem = f"{r['peak_mem_gb']:.2f}" if r["peak_mem_gb"] is not None else "n/a"
            note = "  (batch reduced after OOM)" if r["notes"] else ""
            print(f"{r['arm']:<4}{r['precision']:<6}{r['batch']:>6}{r['loss_first']:>12.3f}{r['loss_final_mean_last3']:>12.3f}"
                  f"{mem:>13}{r['tokens_per_sec']:>11,.0f}  ok{note}")
        else:
            print(f"{r['arm']:<4}{r['precision']:<6}{'':>6}{'':>12}{'':>12}{'':>13}{'':>11}  {r['status'].upper()}: "
                  f"{r.get('reason') or r.get('error')}")
    print()
    for c in checks:
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['name']}")
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}   (results written to {args.out})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
