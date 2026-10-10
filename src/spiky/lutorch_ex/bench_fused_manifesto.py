"""Manifesto cartridges, cartridge-only training step, at the canonical geometry h16 d48 tph64 nap8, 32,768 vectors.

    python -m spiky.lutorch_ex.bench_fused_manifesto [--rounds 3] [--tokens 32768] [--dtypes float32,bfloat16]

Every arm is a cartridge with table dropout 0.2 (the nanochat LUT-FFN setting), pairs anchors, train mode. Per arm and
dtype (table and input in that dtype): forward with grad enabled, and forward + backward, CUDA-event median over
``rounds`` interleaved rounds of (3 x 20 iterations, median) after 5 warm-up steps; peak memory above the inputs
(first round). An arm whose backend is not offered by the installed cartridge class is skipped and said so; an arm
that fails (OOM, dtype refusal) is reported with its error, never silently dropped.

SPIKY_LUTORCH_REQUIRE_NATIVE=1 is set by default, so an involuntary fallback raises instead of being timed; every row
carries the backend that actually ran (``_fallback.backend_of``).
"""
import argparse
import os
import statistics

os.environ.setdefault("SPIKY_LUTORCH_REQUIRE_NATIVE", "1")

import torch  # noqa: E402

import spiky.lutorch_ex as lx  # noqa: E402
from spiky.lutorch_ex.cartridges._fallback import backend_of  # noqa: E402
from spiky.lutorch_ex.cartridges.fused_confidence import FusedConfidenceLUT  # noqa: E402
from spiky.lutorch_ex.lut_spec import LUTSpec  # noqa: E402

GEOM = dict(h=16, d=48, tph=64, nap=8)
KW = dict(seed=1, table_dropout_rate=0.2)

# (label, class, backend or None for the class's only implementation, extra kwargs)
ARMS = [
    ("FusedManifestoHard auto", lx.FusedManifestoHardLUT, "auto", {}),
    ("FusedManifestoHard cuda", lx.FusedManifestoHardLUT, "cuda", {}),
    ("FusedManifestoHard native", lx.FusedManifestoHardLUT, "native", {}),
    ("FusedManifestoHard tier1", lx.FusedManifestoHardLUT, "tier1", {}),
    ("FusedManifestoSoft auto", lx.FusedManifestoSoftLUT, "auto", {}),
    ("FusedManifestoSoft cuda", lx.FusedManifestoSoftLUT, "cuda", {}),
    ("FusedManifestoSoft native", lx.FusedManifestoSoftLUT, "native", {}),
    ("FusedManifestoSoft tier1", lx.FusedManifestoSoftLUT, "tier1", {}),
    ("ManifestoHard (pure)", lx.ManifestoHardLUT, None, {}),
    ("ManifestoSoft (pure)", lx.ManifestoSoftLUT, None, {}),
    ("FusedConfidenceLUT n=1 (reference)", FusedConfidenceLUT, "cuda", {"read_top_n": 1}),
    ("FusedConfidenceLUT n=2 (reference)", FusedConfidenceLUT, "cuda", {"read_top_n": 2}),
]


def make(cls, backend, extra, spec):
    if backend is not None and backend not in getattr(cls, "_BACKENDS", ()):
        return None
    kw = dict(KW, **extra)
    if backend is not None:
        kw["backend"] = backend
    return cls(spec, **kw)


def time_step(mod, x, go, fwd_only):
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
            a.record()
            step()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        meds.append(statistics.median(ts))
    return statistics.median(meds)


def peak_gib(mod, x, go):
    mod.zero_grad(set_to_none=True)
    x.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    y = mod(x)
    y.backward(go.to(y.dtype))
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--only", default="", help="comma-separated substrings of arm labels to run")
    a = ap.parse_args()
    h, d, tph, nap = GEOM["h"], GEOM["d"], GEOM["tph"], GEOM["nap"]
    spec = LUTSpec(h_in=h, h_out=h, tph=tph, nap=nap, d_in=d, d_out=d, anchor_mode="pairs")
    arms = [r for r in ARMS if not a.only or any(s in r[0] for s in a.only.split(","))]
    dtypes = [getattr(torch, s) for s in a.dtypes.split(",")]
    print(f"device {torch.cuda.get_device_name()}  torch {torch.__version__}  geometry h{h} d{d} tph{tph} nap{nap}  "
          f"tokens {a.tokens}  rounds {a.rounds}  table dropout {KW['table_dropout_rate']}  "
          f"SPIKY_LUTORCH_REQUIRE_NATIVE={os.environ.get('SPIKY_LUTORCH_REQUIRE_NATIVE')}")
    res, notes = {}, {}
    for r in range(a.rounds):
        for dt in dtypes:
            x = torch.randn(a.tokens, h, d, device="cuda", dtype=dt, requires_grad=True)
            go = torch.randn(a.tokens, h, d, device="cuda")
            for label, cls, backend, extra in arms:
                key = (str(dt).split(".")[-1], label)
                if key in notes:
                    continue
                mod = None
                try:
                    mod = make(cls, backend, extra, spec)
                    if mod is None:
                        notes[key] = f"skipped: {cls.__name__} has no backend {backend!r} in this version"
                        continue
                    mod = mod.cuda().train().to(dt)
                    torch._dynamo.reset()
                    f = time_step(mod, x, go, True)
                    fb = time_step(mod, x, go, False)
                    pk = peak_gib(mod, x, go) if r == 0 else None
                    e = res.setdefault(key, {"f": [], "fb": [], "pk": None, "be": set()})
                    e["be"].add(backend_of(mod))
                    e["f"].append(f)
                    e["fb"].append(fb)
                    if pk is not None:
                        e["pk"] = pk
                except Exception as ex:  # OOM, dtype refusal, a fallback under strict mode: reported, not dropped
                    notes[key] = f"{type(ex).__name__}: {str(ex).strip().splitlines()[0][:110]}"
                    res.pop(key, None)
                finally:
                    mod = None
                    x.grad = None
                    torch.cuda.empty_cache()
    print(f"\nmedian ms over {a.rounds} interleaved rounds (each the median of 3 x 20 iterations); peak GiB above the "
          f"inputs, forward+backward")
    for dt in dtypes:
        dn = str(dt).split(".")[-1]
        for label, *_ in arms:
            key = (dn, label)
            if key in res:
                v = res[key]
                f, fb = statistics.median(v["f"]), statistics.median(v["fb"])
                print(f"  {dn:8s} {label:36s} fwd {f:8.3f}  fwd+bwd {fb:8.3f}  bwd {fb - f:8.3f}  "
                      f"peak {v['pk']:6.2f} GiB  [backend: {'/'.join(sorted(v['be']))}]")
            else:
                print(f"  {dn:8s} {label:36s} {notes.get(key, 'no result')}")


if __name__ == "__main__":
    main()
