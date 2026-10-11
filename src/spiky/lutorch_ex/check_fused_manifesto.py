"""Numerics of the FusedManifesto 'cuda' backend at the canonical geometry (h16 d48 tph64 nap8, 32,768 vectors, table
dropout 0.2): agreement with the existing backends, and run-to-run variation (the backward uses fp32 atomics).

    python -m spiky.lutorch_ex.check_fused_manifesto [--repeats 5]

Prints max-abs-relative differences (max |a - b| / max |b|) of the value, the input gradient and the table gradient.
"""
import argparse

import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex.lut_spec import LUTSpec

SPEC = LUTSpec(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48, anchor_mode="pairs")


def run(cls, backend, x, go, dtype, seed=11):
    m = cls(SPEC, seed=1, backend=backend, table_dropout_rate=0.2).cuda().train().to(dtype)
    xx = x.to(dtype).detach().clone().requires_grad_(True)
    torch.manual_seed(seed)                    # same table-dropout draw on every backend
    y = m(xx)
    y.backward(go.to(y.dtype))
    return y.detach().float(), xx.grad.float(), m.weights.grad.float()


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=32768)
    a = ap.parse_args()
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(a.tokens, 16, 48, device="cuda", generator=g)
    go = torch.randn(a.tokens, 16, 48, device="cuda", generator=g)
    print(f"device {torch.cuda.get_device_name()}  h16 d48 tph64 nap8  tokens {a.tokens}  table dropout 0.2")
    for cls in (lx.FusedManifestoHardLUT, lx.FusedManifestoSoftLUT):
        for dtype in (torch.float32, torch.bfloat16):
            dn = str(dtype).split(".")[-1]
            ref = run(cls, "cuda", x, go, dtype)
            reps = [run(cls, "cuda", x, go, dtype) for _ in range(a.repeats - 1)]
            fwd_bitwise = all(torch.equal(r[0], ref[0]) for r in reps)
            vx = max(rel(r[1], ref[1]) for r in reps)
            vw = max(rel(r[2], ref[2]) for r in reps)
            bw = all(torch.equal(r[1], ref[1]) and torch.equal(r[2], ref[2]) for r in reps)
            print(f"{cls.__name__:22s} {dn:8s} run-to-run over {a.repeats} runs: value bit-identical {fwd_bitwise}; "
                  f"grads bit-identical {bw}; max rel diff grad x {vx:.2e}, grad W {vw:.2e}")
            for other in ("native", "tier1"):
                try:
                    o = run(cls, other, x, go, dtype)
                except Exception as e:  # e.g. OOM of tier1 at bf16
                    print(f"{'':22s} {dn:8s} vs {other:6s}: not run ({type(e).__name__}: {str(e).splitlines()[0][:80]})")
                    continue
                print(f"{'':22s} {dn:8s} vs {other:6s}: value {rel(ref[0], o[0]):.2e}  grad x {rel(ref[1], o[1]):.2e}  "
                      f"grad W {rel(ref[2], o[2]):.2e}")
                del o
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
