"""Measure two cheap candidate speedups of the ConfidenceLUT train read (F.embedding_bag, mode=sum, per-sample
weights) at the locked d24 geometry, against the current form, fwd and fwd+bwd:

  int64   current: int64 flat cell indices, every table gathered (dropped tables only get weight 0)
  int32   int32 flat cell indices (smaller radix-sort keys in the weight-grad path, half the saved index memory)
  skipdrop  table-dropout-dropped entries routed to padding_idx, which embedding_bag skips in fwd and bwd
  int32+skipdrop  both

Also checks the gradients of the variants against the current form (exact-sum semantics must not change).

Run:  python experiments/lut_nanochat/profiling/read_path_probe.py --tokens 8192 32768 [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch
import torch.nn.functional as F

G, TPH, K, D = 16, 64, 256, 48        # locked geometry: groups(=heads), tables/head, cells/table, d_out
DROP = 0.2                            # base_train --lut-ffn-table-dropout default


def timeit(fn, warmup=5, repeats=20, setup=None):
    for _ in range(warmup):
        if setup:
            setup()
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        if setup:
            setup()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("No CUDA device; stopping.")
        return 2
    torch.manual_seed(0)
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "results": {}}
    for N in args.tokens:
        # Table rows + one extra all-zero padding row at the end (index PAD) for the skip variants.
        W = (torch.randn(G * TPH * K + 1, D, device="cuda") * 1e-3)
        PAD = G * TPH * K
        W[PAD] = 0
        W.requires_grad_(True)
        c = torch.randint(0, K, (N, G, TPH), device="cuda")
        base = ((torch.arange(G, device="cuda").view(G, 1) * TPH + torch.arange(TPH, device="cuda").view(1, TPH)) * K)
        gc = (c + base).reshape(N * G, TPH)
        keep = (torch.rand(N, G, TPH, device="cuda") >= DROP).reshape(N * G, TPH)
        s = torch.rand(N * G, TPH, device="cuda")
        psw_masked = (s * keep / (1 - DROP)).detach()          # what the cartridge does today: weight 0, still read
        go = torch.randn(N * G, D, device="cuda")
        variants = {
            "int64": (gc, None),
            "int32": (gc.to(torch.int32), None),
            "skipdrop": (torch.where(keep, gc, torch.full_like(gc, PAD)), PAD),
            "int32+skipdrop": (torch.where(keep, gc, torch.full_like(gc, PAD)).to(torch.int32), PAD),
        }
        r, ref_grads = {}, None
        for name, (idx, pad) in variants.items():
            psw = psw_masked.clone().requires_grad_(True)

            def fwd():
                return F.embedding_bag(idx, W, per_sample_weights=psw, mode="sum", padding_idx=pad)

            def fb():
                fwd().backward(go)

            def zero():
                W.grad = None
                psw.grad = None
            tf = timeit(fwd, repeats=args.repeats)
            tb = timeit(fb, repeats=args.repeats, setup=zero)
            zero()
            fb()
            gW = W.grad[:PAD].clone()
            gs = (psw.grad * keep).clone()                     # psw grad only matters where the table is kept
            if ref_grads is None:
                ref_grads = (gW, gs)
                err = (0.0, 0.0)
            else:
                err = ((gW - ref_grads[0]).abs().max().item(), (gs - ref_grads[1]).abs().max().item())
            r[name] = {"fwd_ms": tf, "fwdbwd_ms": tb, "max_abs_diff_gradW": err[0], "max_abs_diff_grad_psw_kept": err[1]}
            print(f"N={N:6d} {name:15s} fwd {tf:7.3f} ms  fwd+bwd {tb:7.3f} ms   |dgW|max {err[0]:.2e}  "
                  f"|dpsw|max {err[1]:.2e}", flush=True)
        res["results"][str(N)] = r
        del W, c, gc, keep, s, go
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
