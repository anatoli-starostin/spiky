"""Manifesto vs Confidence cartridges, cartridge-only train step, RTX 5090. Two geometries at 32,768 tokens (the nanochat
micro-batch): the Stage 2a reference h8 d8 tph64 nap8 (r = 64) and the d24 LUT-FFN champion h16 d48 tph64 nap8.
Table dropout 0.2, pairs anchors, train mode. Forward (with grad enabled, no backward) and fwd+bwd, CUDA-event median
of 3 x 20 iters after 5 warm-up; peak memory above the inputs. Every arm is tried; OOM / errors are reported.

    python bench_manifesto.py [--rounds 3]   (rounds = interleaved sessions; median over them)
"""
import argparse
import statistics

import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex.cartridges.fused_confidence import FusedConfidenceLUT
from spiky.lutorch_ex.lut_spec import LUTSpec

import os
GEOMS = {"ref r64 (h8 d8 tph64 nap8)": (8, 8, 64, 8), "d24 champion (h16 d48 tph64 nap8)": (16, 48, 64, 8),
         "h16 r128 (h16 d8 tph64 nap8)": (16, 8, 64, 8)}
if os.environ.get("GEOMS"):
    GEOMS = {k: v for k, v in GEOMS.items() if any(s in k for s in os.environ["GEOMS"].split(","))}
KW = dict(seed=1, table_dropout_rate=0.2)


def arms(spec):
    return {
        "ManifestoHard pure": lambda: lx.ManifestoHardLUT(spec, **KW),
        "ManifestoSoft pure": lambda: lx.ManifestoSoftLUT(spec, **KW),
        "FusedManifestoHard auto(=native)": lambda: lx.FusedManifestoHardLUT(spec, **KW),
        "FusedManifestoHard tier1": lambda: lx.FusedManifestoHardLUT(spec, backend="tier1", **KW),
        "FusedManifestoSoft auto(=tier1)": lambda: lx.FusedManifestoSoftLUT(spec, **KW),
        "FusedManifestoSoft native": lambda: lx.FusedManifestoSoftLUT(spec, backend="native", **KW),
        "ConfidenceLUT n=1 (compiled)": lambda: lx.ConfidenceLUT(spec, read_top_n=1, index_dtype=torch.int32, **KW),
        "FusedConfidenceLUT n=1 (CUDA)": lambda: FusedConfidenceLUT(spec, read_top_n=1, **KW),
        "FusedConfidenceLUT n=2 (CUDA)": lambda: FusedConfidenceLUT(spec, read_top_n=2, **KW),
    }


def time_mod(mod, x, go, fwd_only):
    def step():
        y = mod(x)
        if not fwd_only:
            y.backward(go.to(y.dtype))
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    meds = []
    for _ in range(3):
        ts = []
        for _ in range(20):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); step(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        meds.append(statistics.median(ts))
    return statistics.median(meds)


def peak(mod, x, go):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    y = mod(x)
    y.backward(go.to(y.dtype))
    torch.cuda.synchronize()
    mod.zero_grad(set_to_none=True)
    return (torch.cuda.max_memory_allocated() - base) / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=32768)
    a = ap.parse_args()
    print(f"device {torch.cuda.get_device_name()}  torch {torch.__version__}  tokens {a.tokens}  rounds {a.rounds}")
    res = {}
    for r in range(a.rounds):
        for gname, (h, d, tph, nap) in GEOMS.items():
            spec = LUTSpec(h_in=h, h_out=h, tph=tph, nap=nap, d_in=d, d_out=d, anchor_mode="pairs")
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.randn(a.tokens, h, d, device="cuda", dtype=dtype, requires_grad=True)
                go = torch.randn(a.tokens, h, d, device="cuda")
                for aname, mk in arms(spec).items():
                    key = (gname, str(dtype).split(".")[-1], aname)
                    if res.get(key) == "skip":
                        continue
                    try:
                        torch._dynamo.reset()
                        mod = mk().cuda().train().to(dtype)
                        f = time_mod(mod, x, go, True)
                        fb = time_mod(mod, x, go, False)
                        pk = peak(mod, x, go) if r == 0 else None
                        res.setdefault(key, {"f": [], "fb": [], "pk": None})
                        res[key]["f"].append(f)
                        res[key]["fb"].append(fb)
                        if pk is not None:
                            res[key]["pk"] = pk
                    except Exception as e:
                        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
                        res[key] = "skip"
                        print(f"  [{gname} {key[1]}] {aname}: {msg}")
                    finally:
                        mod = None
                        torch.cuda.empty_cache()
    for gname, (h, d, tph, nap) in GEOMS.items():
        print(f"\n== {gname}, {a.tokens} tokens: median ms over {a.rounds} interleaved rounds; "
              f"gather GB/s = forward table-row bytes / forward time")
        for dt in ("float32", "bfloat16"):
            for aname in arms(LUTSpec(h_in=h, h_out=h, tph=tph, nap=nap, d_in=d, d_out=d)).keys():
                v = res.get((gname, dt, aname))
                if v is None or v == "skip":
                    continue
                f, fb = statistics.median(v["f"]), statistics.median(v["fb"])
                cells = 2 if ("Soft" in aname or "n=2" in aname) else 1
                elt = 2 if dt == "bfloat16" else 4
                gbytes = a.tokens * h * tph * cells * d * elt / 1e9
                print(f"  {dt:8s} {aname:33s} fwd {f:7.3f}  fwd+bwd {fb:8.3f}  bwd {fb - f:8.3f}  "
                      f"peak {v['pk']:5.2f} GiB  gather {gbytes / (f / 1e3):6.0f} GB/s")


if __name__ == "__main__":
    main()
