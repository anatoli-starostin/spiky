"""index_dtype int64 vs int32, per cartridge, at the d24 LUT geometry: eval forward, train forward, train fwd+bwd.

Every cartridge (pure Manifesto / SoftSign, Confidence n=1/n=2, Quantised n=1/n=2, the fused twins on their tier1 and
native paths, and the deploy-only int8 n=2 object, eval only), one cartridge alone (no projections), CUDA-event
medians. Needs a lutorch_ex with index_dtype (LUTORCH_EX_SRC=<repo>/src to point at one).

  python experiments/lut_nanochat/profiling/index_dtype_cartridges.py --tokens 32768 --out /tmp/idx.json
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LIB_SRC = Path(os.environ.get("LUTORCH_EX_SRC") or REPO / "src").resolve()
sys.path.insert(0, str(LIB_SRC))


def lib_git(*args):
    try:
        out = subprocess.run(["git", "-C", str(LIB_SRC), *args], capture_output=True, text=True, timeout=30)
        return (out.stdout.rstrip() or None) if out.returncode == 0 else None
    except Exception:
        return None


def lib_dirty_files():
    """Modified, staged and untracked (non-ignored) files under the profiled src/."""
    out = lib_git("status", "--porcelain", "--untracked-files=all", "--", ".")
    return out.splitlines() if out else []

import torch  # noqa: E402

import spiky.lutorch_ex as lx  # noqa: E402
from spiky.lutorch_ex import LUTSpec  # noqa: E402

GEOM = dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48)
CARTS = [
    ("ManifestoHardLUT", {}), ("ManifestoSoftLUT", {}), ("SoftSignHardLUT", {}), ("SoftSignSmoothLUT", {}),
    ("ConfidenceLUT", dict(read_top_n=1)), ("ConfidenceLUT", dict(read_top_n=2)),
    ("QuantisedConfidenceLUT", dict(read_top_n=1)), ("QuantisedConfidenceLUT", dict(read_top_n=2)),
    ("FusedManifestoHardLUT", dict(backend="tier1")), ("FusedManifestoHardLUT", dict(backend="native")),
    ("FusedManifestoSoftLUT", dict(backend="tier1")), ("FusedManifestoSoftLUT", dict(backend="native")),
    ("FusedSoftSignHardLUT", dict(backend="tier1")), ("FusedSoftSignHardLUT", dict(backend="native")),
    ("FusedSoftSignSmoothLUT", dict(backend="tier1")), ("FusedSoftSignSmoothLUT", dict(backend="native")),
    ("DeployedQuantisedConfidenceLUT", dict(read_top_n=2)),
]


def t_ms(fn, warm, rep, setup=None):
    for _ in range(warm):
        if setup:
            setup()
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        if setup:
            setup()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def build(name, kw, dt, spec):
    torch.manual_seed(0)
    if name == "DeployedQuantisedConfidenceLUT":
        from spiky.lutorch_ex.cartridges.quantised_confidence import DeployedQuantisedConfidenceLUT
        p = lx.QuantisedConfidenceLUT(spec, seed=1, **kw).cuda().to_deployment()
        return DeployedQuantisedConfidenceLUT(spec, p["tensors"], p["meta"], device="cuda", index_dtype=dt)
    return getattr(lx, name)(spec, seed=1, table_dropout_rate=0.2, index_dtype=dt, **kw).cuda()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-dirty-lib", action="store_true",
                    help="profile a lutorch_ex src/ with uncommitted or untracked files (recorded, not refused)")
    a = ap.parse_args(argv)
    dirty = lib_dirty_files()
    if dirty and not a.allow_dirty_lib:
        print(f"Refusing to run: the profiled lutorch_ex src/ ({LIB_SRC}) has uncommitted or untracked files, so the "
              "numbers would not be attributable to any commit:\n  " + "\n  ".join(dirty)
              + "\nCommit them, or pass --allow-dirty-lib to profile anyway (the dirty files are recorded in --out).")
        return 3
    if not torch.cuda.is_available():
        print("No CUDA device; stopping.")
        return 2
    torch.set_float32_matmul_precision("high")
    spec = LUTSpec(**GEOM)
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "tokens": a.tokens,
           "lutorch_ex_src_commit": lib_git("rev-parse", "HEAD"),
           "lutorch_ex_src_describe": lib_git("describe", "--always", "--dirty"),
           "lutorch_ex_src_dirty_files": dirty, "rows": []}
    for name, kw in CARTS:
        label = name + "".join(f" {k}={v}" for k, v in kw.items())
        row = {"cartridge": label}
        for dt in (torch.int64, torch.int32):
            tag = str(dt).replace("torch.", "")
            try:
                m = build(name, kw, dt, spec)
                xe = torch.randn(a.tokens, 16, 48, device="cuda")
                m.eval()
                with torch.no_grad():
                    ev = t_ms(lambda: m(xe), a.warmup, a.repeats)
                r = {"eval_ms": ev}
                if name != "DeployedQuantisedConfidenceLUT":
                    m.train()
                    x = torch.randn(a.tokens, 16, 48, device="cuda", requires_grad=True)
                    g = torch.randn(a.tokens, 16, 48, device="cuda")
                    tf = t_ms(lambda: m(x), a.warmup, a.repeats)

                    def zero():
                        for p in m.parameters():
                            p.grad = None
                        x.grad = None
                    tb = t_ms(lambda: m(x).backward(g), a.warmup, a.repeats, setup=zero)
                    r.update(train_fwd_ms=tf, train_fwdbwd_ms=tb, train_bwd_ms=tb - tf)
                row[tag] = r
                del m
            except Exception as e:
                row[tag] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            torch.cuda.empty_cache()
        if "error" not in row["int64"] and "error" not in row["int32"]:
            row["delta_pct"] = {k: 100 * (row["int32"][k] / row["int64"][k] - 1) for k in row["int64"]}
        res["rows"].append(row)
        d = row.get("delta_pct", {})
        print(f"{label:52s} " + "  ".join(
            f"{k.replace('_ms', '')}: {row['int64'][k]:7.2f}->{row['int32'][k]:7.2f} ({d[k]:+5.1f}%)" for k in d)
              + ("" if d else f"  {row}"), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
