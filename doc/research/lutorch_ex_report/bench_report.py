"""Benchmark table for the lutorch_ex report, measured with the library's own harness (bench.py).

Champion geometry LUTSpec(8, 8, tph=64, nap=8, d_in=48, d_out=48), one cartridge per generation plus the
quantised one. CPU rows use the pure cartridges (the pure-torch path); GPU rows use the fused twins the
sweep ran (and the two Gen-3 cartridges, which have no twin). For each cell bench.benchmark() times
forward_eval, forward_train and backward separately (CUDA events via synchronize, median of `repeats`
after `warmup`); the report quotes step = forward_train + backward. Peak memory is a separate probe of one
training step after a reset of the CUDA peak counter.

    .venv/bin/python bench_report.py --device cuda --batch 24576 --json bench_cuda.json
    .venv/bin/python bench_report.py --device cpu  --batch 1024  --json bench_cpu.json
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from spiky.lutorch_ex import (  # noqa: E402
    ConfidenceLUT, FusedManifestoHardLUT, FusedManifestoSoftLUT, FusedSoftSignHardLUT,
    FusedSoftSignSmoothLUT, LUTSpec, ManifestoHardLUT, ManifestoSoftLUT, QuantisedConfidenceLUT,
    SoftSignHardLUT, SoftSignSmoothLUT,
)
from spiky.lutorch_ex import bench  # noqa: E402

SPEC = LUTSpec(h_in=8, h_out=8, tph=64, nap=8, d_in=48, d_out=48)
GEN3 = dict(read_top_n=2, beta_init=2.0, gamma_init=1.0, read_tau_init=0.5)

# The seven optimised configurations of the sweep: the four fused twins (on CPU they take their tier-1
# embedding_bag path; the native kernels need CUDA) and the three Gen-3 configurations, whose
# embedding_bag + compiled forward IS the optimised path (they have no separate twin). The pure reference
# cartridges are deliberately not benchmarked here; `--pure` adds them for a cross-check.
FUSED = [
        ("FusedManifestoHardLUT", lambda: FusedManifestoHardLUT(SPEC, seed=1)),
        ("FusedManifestoSoftLUT", lambda: FusedManifestoSoftLUT(SPEC, seed=1)),
        ("FusedSoftSignHardLUT", lambda: FusedSoftSignHardLUT(SPEC, seed=1)),
        ("FusedSoftSignSmoothLUT", lambda: FusedSoftSignSmoothLUT(SPEC, seed=1)),
        ("ConfidenceLUT n=1", lambda: ConfidenceLUT(SPEC, seed=1, read_top_n=1)),
        ("ConfidenceLUT n=2", lambda: ConfidenceLUT(SPEC, seed=1, **GEN3)),
        ("QuantisedConfidenceLUT n=2", lambda: QuantisedConfidenceLUT(SPEC, seed=1, quant_mode="p2_int8", **GEN3)),
]
PURE = [
        ("ManifestoHardLUT", lambda: ManifestoHardLUT(SPEC, seed=1)),
        ("ManifestoSoftLUT", lambda: ManifestoSoftLUT(SPEC, seed=1)),
        ("SoftSignHardLUT", lambda: SoftSignHardLUT(SPEC, seed=1)),
        ("SoftSignSmoothLUT", lambda: SoftSignSmoothLUT(SPEC, seed=1)),
]
CARTRIDGES = {
    "cpu": FUSED,
    "cuda": [
        ("FusedManifestoHardLUT", lambda: FusedManifestoHardLUT(SPEC, seed=1)),
        ("FusedManifestoSoftLUT", lambda: FusedManifestoSoftLUT(SPEC, seed=1)),
        ("FusedSoftSignHardLUT", lambda: FusedSoftSignHardLUT(SPEC, seed=1)),
        ("FusedSoftSignSmoothLUT", lambda: FusedSoftSignSmoothLUT(SPEC, seed=1)),
        ("ConfidenceLUT n=1", lambda: ConfidenceLUT(SPEC, seed=1, read_top_n=1)),
        ("ConfidenceLUT n=2", lambda: ConfidenceLUT(SPEC, seed=1, **GEN3)),
        ("QuantisedConfidenceLUT n=2", lambda: QuantisedConfidenceLUT(SPEC, seed=1, quant_mode="p2_int8", **GEN3)),
    ],
}


def peak_train_step(factory, batch, device):
    """Peak allocated memory (GB) over one forward_train + backward, after a warm step (compile etc.)."""
    if device.type != "cuda":
        return None
    cart = factory().to(device).train()
    x = torch.randn(batch, SPEC.h_in, SPEC.d_in, device=device, requires_grad=True)
    for _ in range(2):                     # warm: compile, JIT, allocator
        cart(x).sum().backward(); cart.zero_grad(set_to_none=True); x.grad = None
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    cart(x).sum().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)
    del cart, x; torch.cuda.empty_cache()
    return (peak - base) / 1e9, peak / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--batch", type=int, nargs="+", default=[24576])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--json", required=True)
    p.add_argument("--pure", action="store_true", help="also benchmark the pure reference cartridges")
    a = p.parse_args()
    if a.pure:
        CARTRIDGES[a.device] = list(CARTRIDGES[a.device]) + PURE
    if a.device == "cuda":
        name = torch.cuda.get_device_name(0)
        label = "cuda:RTX5090" if "5090" in name else "cuda:H100" if "H100" in name else "cuda:A100"
        dev = torch.device("cuda")
    else:
        name, label, dev = "cpu", "cpu", torch.device("cpu")
    print(f"device: {name}  torch {torch.__version__}  threads {torch.get_num_threads()}")
    out = {"device_name": name, "spec": str(SPEC), "batches": a.batch, "warmup": a.warmup,
           "repeats": a.repeats, "torch": torch.__version__, "rows": []}
    for nm, factory in CARTRIDGES[a.device]:
        t0 = time.time()
        rows = bench.benchmark(factory, name=nm, batch_sizes=a.batch, devices=[label],
                               warmup=a.warmup, repeats=a.repeats, measure_cuda=(a.device == "cuda"))
        rec = {"cartridge": nm, "cells": [r.to_dict() for r in rows], "peak": {}}
        for b in a.batch:
            try:
                pk = peak_train_step(factory, b, dev)
                rec["peak"][str(b)] = None if pk is None else {"step_gb": pk[0], "total_gb": pk[1]}
            except Exception as e:
                rec["peak"][str(b)] = f"error: {type(e).__name__}: {e}"
        out["rows"].append(rec)
        print(bench.format_table(rows)); print("peak:", rec["peak"], f"({time.time()-t0:.0f}s)\n", flush=True)
        json.dump(out, open(a.json, "w"), indent=1)   # partial results survive an abort
    print("wrote", a.json)


if __name__ == "__main__":
    main()
