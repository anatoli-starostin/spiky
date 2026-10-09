"""Per-op profile: ConfidenceLUT vs QuantisedConfidenceLUT (p2_int8) at the locked d24 geometry, one cartridge.

Library-level comparison of the two cartridges (the ProjectionMHL compress/decompress GEMMs are identical for both
and are profiled by profile_lut_ffn.py). Headline timings come from the library's own harness,
spiky.lutorch_ex.bench.benchmark() - the tool behind the lutorch_ex report's benchmark table (driven there by
doc/research/lutorch_ex_report/bench_report.py on branch feature/lutorch_ex_report) - and are cross-checked here
with CUDA-event median + IQR. On top of that:

  * per-op breakdown: torch.profiler, forward (train mode) and fwd+bwd sessions; backward = the per-kernel difference
    (compiled backwards donate their buffers, so a retained graph cannot be re-run). Profiler warm-up session burned
    first; the "## Call CompiledFxGraph" wrapper excluded; an empty GPU trace is detected and recorded, never
    reported as zeros.
  * manual breakdown: each library building block timed alone, fwd and fwd+bwd (addressing, confidence score / its
    power-of-two STE, fake-quant tables, embedding_bag read fwd / psw grad / weight grad, table-dropout mask, cell-TV,
    casts).
  * bytes moved per token by the read, analytic, per path.
  * saved-for-backward inventory and peak memory per cartridge.
  * EVAL (no-grad) forward for every cartridge, and the DEPLOY-ONLY packed-int8 object
    (DeployedQuantisedConfidenceLUT: int8 rows, int32 shift-add, no fp32 master). Deployment exists for
    read_top_n=2 only (the library refuses n=1 export), so the deploy comparison is run at n=2 and labelled so.

Cartridges (--cartridges), all at h_in=h_out=16, d_in=d_out=48, tph=64, nap=8, pairs, beta/gamma 2/1, learnable score,
table dropout 0.2, seed 1:
  conf-n1      ConfidenceLUT read_top_n=1 (fp32)              - the live d24 status quo
  quant-n1     QuantisedConfidenceLUT p2_int8 read_top_n=1    - its quantised twin (train-only: not exportable)
  conf-n2      ConfidenceLUT read_top_n=2                      - reference for the deploy comparison
  quant-n2     QuantisedConfidenceLUT p2_int8 read_top_n=2
  deploy-n2    DeployedQuantisedConfidenceLUT from quant-n2.to_deployment()   [DEPLOY-ONLY, eval only]

Run:  experiments/lut_nanochat/profiling/profile_cartridges.sh [--batch 4 16] [--out-dir DIR] [--passes ...]
      (or python experiments/lut_nanochat/profiling/profile_cartridges.py --help)
Outputs in --out-dir: cartridges.json (environment block + every number), profile_cart_<name>_<phase>.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import profile_lut_ffn as plf                                    # noqa: E402  (also puts src/ + nanochat on sys.path)

import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

GEOM = dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48, anchor_mode="pairs")
COMMON = dict(seed=1, beta_init=2.0, gamma_init=1.0, learnable_score=True, table_dropout_rate=0.2)
TRAIN_CARTS = ["conf-n1", "quant-n1"]
ALL_CARTS = ["conf-n1", "quant-n1", "conf-n2", "quant-n2", "deploy-n2"]
ALL_PASSES = ["env", "ext", "bench", "timing", "components", "profile", "memory"]


def spec():
    from spiky.lutorch_ex import LUTSpec
    return LUTSpec(**GEOM)


def build(name: str):
    """Fresh cartridge on CUDA, train mode."""
    from spiky.lutorch_ex import ConfidenceLUT, QuantisedConfidenceLUT
    n = 2 if name.endswith("n2") else 1
    if name.startswith("conf"):
        return ConfidenceLUT(spec(), read_top_n=n, **COMMON).cuda().train()
    q = QuantisedConfidenceLUT(spec(), quant_mode="p2_int8", read_top_n=n, **COMMON).cuda().train()
    if name.startswith("quant"):
        return q
    from spiky.lutorch_ex.cartridges.quantised_confidence import DeployedQuantisedConfidenceLUT
    payload = q.to_deployment()
    assert payload["format"] == "p2_int8", payload["format"]
    d = DeployedQuantisedConfidenceLUT(spec(), payload["tensors"], payload["meta"], device="cuda").eval()
    return d


def make_x(tokens: int, requires_grad: bool):
    s = spec()
    return torch.randn(tokens, s.h_in, s.d_in, device="cuda", requires_grad=requires_grad)


def read_bytes_per_token(name: str) -> dict:
    """Analytic bytes the READ touches per token (per layer), HW-independent. G*tph (token, table) entries; n cells each."""
    s = spec()
    G, tph, d = s.n_groups, s.tph, s.d_out
    n = 2 if name.endswith("n2") else 1
    entries = G * tph * n
    if name.startswith("deploy"):
        row, idx, w = d * 1, 8, 0          # int8 row; int64 flat index; the shift group (int) is computed, not read
        note = "int8 rows (packed), int32 shift-add; no fp32 master; scale folded into decompress"
    else:
        row, idx, w = d * 4, 8, 4          # fp32 row (quant train/eval: the fp32 FAKE-QUANT copy), int64 index, fp32 weight
        note = ("fp32 rows gathered from the per-call fp32 fake-quant table copy (value W_hat*2^e)"
                if name.startswith("quant") else "fp32 rows from the fp32 master table")
    out = {"rows_bytes": entries * row, "index_bytes": entries * idx, "weight_bytes": entries * w,
           "total_bytes": entries * (row + idx + w), "row_dtype": "int8" if name.startswith("deploy") else "float32",
           "cells_per_table": n, "note": note}
    if name.startswith("quant"):
        n_tab = G * tph * s.n_cells * d
        # per CALL (not per token): ste_tables reads the 4-byte master for amax, reads it again to quantise and
        # writes the 4-byte fake-quant copy
        out["per_call_fake_quant_table_pass_bytes"] = n_tab * 4 * 3
    return out


# ------------------------------------------------------------------------------------------------ passes

def pass_ext() -> dict:
    """Does the quantised cartridge's JIT int8 extension build/load here, and at what cost?"""
    from spiky.lutorch_ex.cartridges import _pow2_int8
    t0 = time.time()
    ok = _pow2_int8.ensure_registered()
    dt = time.time() - t0
    r = {"ensure_registered": bool(ok), "seconds": round(dt, 2), "enabled": bool(_pow2_int8._enabled),
         "error": _pow2_int8._error, "validated_arches": [list(a) for a in getattr(_pow2_int8, "VALIDATED_ARCHES", [])],
         "torch_extensions_dir": "set by lutorch_ex (_native_ops.py) unless TORCH_EXTENSIONS_DIR is exported"}
    plf.log(f"  [ext] p2_int8 native op: registered={r['ensure_registered']} enabled={r['enabled']} "
            f"in {r['seconds']} s; error={r['error']}")
    return r


def pass_bench(args, carts) -> dict:
    """The library's own harness (spiky.lutorch_ex.bench.benchmark), as used for the report's benchmark table."""
    from spiky.lutorch_ex import bench
    label = "cuda:" + torch.cuda.get_device_name(0).replace(" ", "")
    out = {}
    for name in carts:
        ops = ("forward_eval",) if name.startswith("deploy") else bench.OPERATIONS
        rows = bench.benchmark(lambda: build(name), name=name, batch_sizes=args.tokens, operations=ops,
                               devices=[label], warmup=args.warmup, repeats=args.repeats, measure_cuda=True)
        out[name] = [r.to_dict() for r in rows]
        for r in rows:
            plf.log(f"  [bench] {name:9s} {r.operation:13s} N={r.batch_size:6d} {r.status:6s} "
                    f"median {r.median_ms if r.median_ms is not None else float('nan'):8.3f} ms  {r.note[:80]}")
        plf.free_cuda()
    return out


def pass_timing(args, carts) -> dict:
    """CUDA-event median + IQR for eval forward, train forward, train fwd+bwd (backward = difference)."""
    out = {}
    for name in carts:
        out[name] = {}
        for N in args.tokens:
            plf.free_cuda()
            cart = build(name)
            r = {}
            xe = make_x(N, False)
            cart.eval()
            with torch.no_grad():
                t = plf.cuda_time_ms(lambda: cart(xe), args.warmup, args.repeats)
            r["eval_fwd"] = t
            if not name.startswith("deploy"):
                cart.train()
                x = make_x(N, True)
                g = torch.randn(N, spec().h_out, spec().d_out, device="cuda")
                tf = plf.cuda_time_ms(lambda: cart(x), args.warmup, args.repeats)

                def fb():
                    cart(x).backward(g)

                def zero():
                    for p in cart.parameters():
                        p.grad = None
                    x.grad = None
                tb = plf.cuda_time_ms(fb, args.warmup, args.repeats, setup=zero)
                r["train_fwd"], r["train_fwdbwd"] = tf, tb
                r["train_bwd_by_difference_ms"] = tb["median_ms"] - tf["median_ms"]
            r["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
            out[name][str(N)] = r
            plf.log(f"  [timing] {name:9s} N={N:6d} eval {r['eval_fwd']['median_ms']:7.3f} (IQR {r['eval_fwd']['iqr_ms']:.3f})"
                    + (f"  train fwd {r['train_fwd']['median_ms']:7.3f} (IQR {r['train_fwd']['iqr_ms']:.3f})"
                       f"  fwd+bwd {r['train_fwdbwd']['median_ms']:7.3f} (IQR {r['train_fwdbwd']['iqr_ms']:.3f})"
                       f"  bwd {r['train_bwd_by_difference_ms']:7.3f}" if "train_fwd" in r else "") + " ms")
            del cart, xe
    return out


def pass_components(args) -> dict:
    """Manual breakdown: each library building block of the TRAIN read alone, fwd and fwd+bwd (median ms)."""
    from spiky.lutorch_ex.cartridges import _pow2
    from spiky.lutorch_ex.cartridges._fused_ops import _global_cells
    N = max(args.tokens)
    out = {"tokens": N}
    for name in TRAIN_CARTS:
        plf.free_cuda()
        cart = build(name)
        G, tph, K, d = cart.weights.shape
        x = make_x(N, True)
        r = {}

        def t(label, fwd, leaves):
            tf = plf.cuda_time_ms(fwd, args.warmup, args.repeats)

            def fb():
                y = fwd()
                y.float().sum().backward() if not isinstance(y, tuple) else y[0].float().sum().backward()

            def zero():
                for p in list(leaves) + list(cart.parameters()):
                    p.grad = None
            tb = plf.cuda_time_ms(fb, args.warmup, args.repeats, setup=zero) if leaves else None
            r[label] = {"fwd_ms": tf["median_ms"], "fwdbwd_ms": tb["median_ms"] if tb else None}
            plf.log(f"  [components] {name:8s} {label:40s} fwd {tf['median_ms']:8.3f}"
                    + (f"  fwd+bwd {tb['median_ms']:8.3f}" if tb else "") + " ms")

        addr = torch.compile(cart._addresses, dynamic=True)
        t("addressing (margins + sign-bit pack, compiled)", lambda: addr(x)[1], [x])
        _, u, c, *_ = addr(x.detach())
        u = u.detach().requires_grad_(True)
        score = torch.compile(cart._score, dynamic=True)
        t("confidence score s (compiled)", lambda: score(u), [u])
        if name.startswith("quant"):
            cfg = cart._quant

            def ste_weight():
                m = u.abs()
                beta, gamma = cart._betagamma(u.dtype)
                msum = m.sum(-1)
                lse = F.logsigmoid(beta * m).sum(-1)
                s = msum * torch.exp(gamma * lse)
                k1 = _pow2.round_half_up(torch.log2(msum) + gamma * lse / _pow2.LN2)
                skip = k1 < cfg["lo"]
                return cart._ste_score_weight(s, torch.clamp(k1, cfg["lo"], cfg["hi"]), skip)
            stew = torch.compile(ste_weight, dynamic=True)
            t("power-of-two STE score weight (compiled)", stew, [u])
            fq = torch.compile(cart._fake_quant_tables, dynamic=True)
            t("fake-quant tables (STE, full table, compiled)", fq, [cart.weights])
            fq_eager = cart._fake_quant_tables
            t("fake-quant tables (eager)", fq_eager, [cart.weights])
        t("table-dropout mask", lambda: cart._table_dropout_mask(N, x.device, torch.float32), [])
        W2 = (cart._fake_quant_tables() if name.startswith("quant") else cart.weights.reshape(G * tph * K, d)).detach()
        W2 = W2.clone().requires_grad_(True)
        gc = _global_cells(c, G, tph, K).reshape(N * G, tph)
        s = cart._score(u.detach()).reshape(N * G, tph).detach().clone().requires_grad_(True)
        t("embedding_bag read (fp32 rows)", lambda: F.embedding_bag(gc, W2, per_sample_weights=s, mode="sum"), [s, W2])
        t("cell-TV penalty (per optimizer step)", lambda: 10.0 * cart.cell_tv(), [cart.weights])
        xb = torch.randn(N, G * 48, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
        t("cast bf16->fp32 [N,768]", lambda: xb.float(), [xb])
        out[name] = r
        del cart, x, u, W2, gc, s
    return out


def pass_profile(args, carts) -> dict:
    out = {}
    N = args.profile_tokens or max(args.tokens)
    a = torch.randn(1024, 1024, device="cuda")
    plf._profile_session(args, "cart_warmup_discard", lambda: a @ a)
    for name in carts:
        plf.free_cuda()
        cart = build(name)
        r = {"tokens": N}
        xe = make_x(N, False)
        cart.eval()
        with torch.no_grad():
            for _ in range(3):
                cart(xe)

            def ev():
                cart(xe)
            k_e, ops_e = plf._profile_session(args, f"cart_{name}_eval", ev)
        sessions = {"eval": k_e}
        if not name.startswith("deploy"):
            cart.train()
            x = make_x(N, True)
            g = torch.randn(N, spec().h_out, spec().d_out, device="cuda")
            for _ in range(3):
                cart(x).backward(g)

            def fwd():
                cart(x)

            def fwdbwd():
                for p in cart.parameters():
                    p.grad = None
                cart(x).backward(g)
            k_f, _ = plf._profile_session(args, f"cart_{name}_trainfwd", fwd)
            k_fb, _ = plf._profile_session(args, f"cart_{name}_trainfwdbwd", fwdbwd)
            sessions["train_fwd"] = k_f
            sessions["train_bwd"] = {k: k_fb.get(k, 0.0) - k_f.get(k, 0.0) for k in k_fb} if k_fb else {}
        for phase, kern in sessions.items():
            if not kern:
                r[phase] = {"cuda_trace": "EMPTY on this host (no CUDA kernel records) - see the components pass"}
                continue
            top = sorted(kern.items(), key=lambda kv: -kv[1])
            r[phase] = {"cuda_time_per_iter_ms": sum(kern.values()) / 1e3,
                        "n_kernels": sum(1 for _, v in top if v > 0.5),
                        "kernels_ms": [{"kernel": k[:200], "ms": round(v / 1e3, 4), "bucket": plf.bucket_of(k)}
                                       for k, v in top if abs(v) >= 1.0]}
            plf.log(f"  [profile] {name:9s} {phase:9s} {r[phase]['cuda_time_per_iter_ms']:8.3f} ms GPU/iter")
            for kk in r[phase]["kernels_ms"][:8]:
                plf.log(f"        {kk['ms']:8.3f}  {kk['kernel'][:110]}")
        out[name] = r
        del cart
    return out


def pass_memory(args, carts) -> dict:
    out = {}
    N = max(args.tokens)
    for name in [c for c in carts if not c.startswith("deploy")]:
        plf.free_cuda()
        cart = build(name)
        x = make_x(N, True)
        g = torch.randn(N, spec().h_out, spec().d_out, device="cuda")
        cart(x).backward(g)
        for p in cart.parameters():
            p.grad = None
        plf.free_cuda()
        base = torch.cuda.memory_allocated()
        saved = []

        def pack(t):
            saved.append((tuple(t.shape), str(t.dtype).replace("torch.", ""), t.numel() * t.element_size()))
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = cart(x)
        held = torch.cuda.memory_allocated() - base
        y.backward(g)
        torch.cuda.synchronize()
        uniq, seen = [], set()
        for s_ in sorted(saved, key=lambda s_: -s_[2]):
            if s_ not in seen:
                seen.add(s_)
                uniq.append({"shape": s_[0], "dtype": s_[1], "MiB": round(s_[2] / 2 ** 20, 1)})
        out[name] = {"tokens": N, "held_after_fwd_gib": held / 2 ** 30,
                     "peak_alloc_fwdbwd_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                     "peak_reserved_fwdbwd_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
                     "saved_total_gib": sum(s_[2] for s_ in saved) / 2 ** 30, "n_saved": len(saved),
                     "saved_top": uniq[:12]}
        plf.log(f"  [memory] {name:9s} N={N}: held {held / 2 ** 30:.2f} GiB, peak {out[name]['peak_alloc_fwdbwd_gib']:.2f}"
                f" GiB, saved {out[name]['saved_total_gib']:.2f} GiB in {len(saved)} tensors")
        del cart, x, y
    # Parameter/buffer memory of the cartridge itself (fp32 master vs packed int8)
    for name in ("conf-n2", "quant-n2", "deploy-n2"):
        cart = build(name)
        nbytes = sum(t.numel() * t.element_size() for t in list(cart.parameters()) + list(cart.buffers()))
        out.setdefault("param_and_buffer_MiB", {})[name] = round(nbytes / 2 ** 20, 1)
        del cart
    plf.log(f"  [memory] cartridge params+buffers MiB: {out['param_and_buffer_MiB']}")
    plf.free_cuda()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "profile_cartridges")
    ap.add_argument("--batch", type=int, nargs="+", default=[4, 16], help="sequences; tokens = batch x --seq")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--cartridges", nargs="+", default=ALL_CARTS, choices=ALL_CARTS)
    ap.add_argument("--passes", nargs="+", default=ALL_PASSES, choices=ALL_PASSES)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--profile-iters", type=int, default=3)
    ap.add_argument("--profile-tokens", type=int, default=None)
    ap.add_argument("--with-stack", action="store_true")
    ap.add_argument("--fp32-matmul-precision", default="high", choices=["highest", "high", "medium"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.tokens = sorted(b * args.seq for b in args.batch)
    if not torch.cuda.is_available():
        plf.log("No usable CUDA device: this profile only produces GPU numbers. Stopping.")
        return 2
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision(args.fp32_matmul_precision)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "geometry": GEOM, "common": COMMON, "env": plf.env_block()}
    res["read_bytes_per_token"] = {c: read_bytes_per_token(c) for c in ALL_CARTS}
    plf.log(json.dumps(res["env"], indent=1))

    def save():
        (args.out_dir / "cartridges.json").write_text(json.dumps(res, indent=1, default=str))
    carts = list(args.cartridges)
    for p in ALL_PASSES[1:]:
        if p not in args.passes:
            continue
        plf.log(f"== {p}")
        res[p] = {"ext": lambda: pass_ext(), "bench": lambda: pass_bench(args, carts),
                  "timing": lambda: pass_timing(args, carts), "components": lambda: pass_components(args),
                  "profile": lambda: pass_profile(args, carts), "memory": lambda: pass_memory(args, carts)}[p]()
        save()
    res["env"]["fa3_loader_imported"] = "nanochat.flash_attention" in sys.modules
    save()
    plf.log(f"done -> {args.out_dir / 'cartridges.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
