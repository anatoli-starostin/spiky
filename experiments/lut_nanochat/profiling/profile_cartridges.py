"""Per-op profile of every lutorch_ex cartridge (35 arms) at the locked d24 geometry, cartridge level.

Library-level comparison of the cartridges (the ProjectionMHL compress/decompress GEMMs are identical for all and are
profiled by profile_lut_ffn.py). Headline timings are the CUDA-event median + IQR of the timing pass; the library's own
harness, spiky.lutorch_ex.bench.benchmark() (the tool behind the lutorch_ex report's benchmark table), cross-checks the
fp32 arms (it builds fp32 inputs only). On top of that:

  * per-op breakdown: torch.profiler, forward (train mode) and fwd+bwd sessions; backward = the per-kernel difference
    (compiled backwards donate their buffers, so a retained graph cannot be re-run). Profiler warm-up session burned
    first; the "## Call CompiledFxGraph" wrapper excluded; an empty GPU trace is detected and recorded, never
    reported as zeros.
  * manual breakdown (conf-n1 / quant-n1 only): each library building block timed alone, fwd and fwd+bwd.
  * static, HARDWARE-INDEPENDENT facts per arm (the "static" pass): analytic read MACs/token and read bytes/token,
    the index dtype the cartridge actually uses, parameter/buffer bytes. check_hw_indep.py compares these (and the
    saved-for-backward inventory) between two hosts and FAILS on any mismatch.
  * saved-for-backward inventory and peak memory per arm.

Arms (--cartridges; default = all 35), all at h_in=h_out=16, d_in=d_out=48, tph=64, nap=8, pairs, table dropout 0.2,
seed 1 (Confidence family also beta/gamma 2/1, learnable score). "-bf16" = the cartridge AND its input in bf16 (only
the Fused* cartridges accept bf16; the others reject it by design). Default index dtype = the library's (int32 when
the table fits, on a lutorch_ex with index_dtype; int64 on one without).
  pure oracles       manifesto-hard  manifesto-soft  softsign-hard  softsign-smooth
  FusedManifestoHard fmh-tier1  fmh-native                       (+ -bf16)   auto -> native
  FusedManifestoSoft fms-pure  fms-tier1  fms-native              (+ -bf16)   auto -> tier1
  FusedSoftSignHard  fsh-tier1  fsh-native  fsh-native-eager      (+ -bf16)   auto -> native
  FusedSoftSignSmooth fss-tier1  fss-native  fss-native-eager     (+ -bf16)   auto -> tier1
  ConfidenceLUT      conf-n1  conf-n2  conf-n1-fused  conf-n2-fused (fused = fused_read=True,the fused read)
                     conf-n1-i64  conf-n1-fused-i64                (int64 index tie-in controls)
  Quantised          quant-n1  quant-n2                            (p2_int8)
  Deploy             deploy-n2   DeployedQuantisedConfidenceLUT from quant-n2.to_deployment()   [eval only]
fmh-auto / fms-auto / fsh-auto / fss-auto (fp32): backend="auto", i.e. what the library itself picks; the static pass
records what it resolved to (train / eval), including whether a soft-sign native pick ran the kernel or the eager tail.
"-native" soft-sign arms run the softsign_surrogate_grad CUDA kernel; "-native-eager" force the eager surrogate tail
IN-PROCESS (--softsign-tail overrides both). LUTORCH_EX_NO_CUDA_EXT=1 cannot be used for that: it also disables the
lprojection extension the native backward needs.

An arm whose feature the profiled library lacks (the fused read, index_dtype, a soft-sign kernel that does not build) is
recorded as {"status": "n/a", "reason": ...} - never run on a substitute path. An OOM (or other error) in one arm is
recorded and the session continues.

Run:  experiments/lut_nanochat/profiling/profile_cartridges.sh [--batch 4 16] [--out-dir DIR] [--passes ...]
      (or python experiments/lut_nanochat/profiling/profile_cartridges.py --help)
Outputs in --out-dir: cartridges.json (environment block + every number), profile_cart_<name>_<phase>.txt.
"""
from __future__ import annotations

import argparse
import inspect
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
BASE = dict(seed=1, table_dropout_rate=0.2)
CONF = dict(beta_init=2.0, gamma_init=1.0, learnable_score=True)
COMMON = {**BASE, **CONF}                                        # the Confidence family's full kwargs (recorded)
TRAIN_CARTS = ["conf-n1", "quant-n1"]                             # the components pass (Confidence internals)
ALL_PASSES = ["env", "ext", "static", "bench", "timing", "components", "profile", "memory"]


def _arms() -> dict:
    """name -> {cls, kw, dtype, tail, eval_only, needs, value_cells, weighted}. value_cells = cells in the VALUE the
    model computes per (token, table) (hard 1, blend/two-cell 2): path-independent, so it defines read MACs/token."""
    a = {}
    for name, cls, cells, w in (("manifesto-hard", "ManifestoHardLUT", 1, False),
                                ("manifesto-soft", "ManifestoSoftLUT", 2, True),
                                ("softsign-hard", "SoftSignHardLUT", 1, False),
                                ("softsign-smooth", "SoftSignSmoothLUT", 2, True)):
        a[name] = dict(cls=cls, kw={}, dtype="float32", tail=None, eval_only=False, needs=[], value_cells=cells,
                       weighted=w)
    fused = [("fmh", "FusedManifestoHardLUT", ("tier1", "native"), 1, False),
             ("fms", "FusedManifestoSoftLUT", ("pure", "tier1", "native"), 2, True),
             ("fsh", "FusedSoftSignHardLUT", ("tier1", "native", "native-eager"), 1, False),
             ("fss", "FusedSoftSignSmoothLUT", ("tier1", "native", "native-eager"), 2, True)]
    for dt, suf in (("float32", ""), ("bfloat16", "-bf16")):
        for short, cls, bes, cells, w in fused:
            for be in bes:
                soft_sign = short in ("fsh", "fss")
                tail = ("eager" if be.endswith("eager") else "kernel") if soft_sign and be.startswith("native") else None
                a[f"{short}-{be}{suf}"] = dict(cls=cls, kw=dict(backend=be.replace("-eager", "")), dtype=dt, tail=tail,
                                               eval_only=False, needs=["ss_kernel"] if tail == "kernel" else [],
                                               value_cells=cells, weighted=w)
    # backend="auto": whatever the library itself picks (recorded by the static pass as "auto_resolves_to"); for the
    # soft-sign cartridges that includes whether the surrogate kernel or the eager fallback runs (no override).
    for short, cls, cells, w in [(f[0], f[1], f[3], f[4]) for f in fused]:
        a[f"{short}-auto"] = dict(cls=cls, kw=dict(backend="auto"), dtype="float32", tail=None, eval_only=False,
                                  needs=[], value_cells=cells, weighted=w)
    for n in (1, 2):
        a[f"conf-n{n}"] = dict(cls="ConfidenceLUT", kw=dict(read_top_n=n, **CONF), dtype="float32", tail=None,
                               eval_only=False, needs=[], value_cells=n, weighted=True)
    for n in (1, 2):
        a[f"conf-n{n}-fused"] = dict(cls="ConfidenceLUT", kw=dict(read_top_n=n, fused_read=True, **CONF),
                                     dtype="float32", tail=None, eval_only=False, needs=["fused_read"],
                                     value_cells=n, weighted=True)
    a["conf-n1-i64"] = dict(cls="ConfidenceLUT", kw=dict(read_top_n=1, index_dtype="int64", **CONF), dtype="float32",
                            tail=None, eval_only=False, needs=["index_dtype"], value_cells=1, weighted=True)
    a["conf-n1-fused-i64"] = dict(cls="ConfidenceLUT", kw=dict(read_top_n=1, index_dtype="int64",
                                                               fused_read=True, **CONF),
                                  dtype="float32", tail=None, eval_only=False, needs=["index_dtype", "fused_read"],
                                  value_cells=1, weighted=True)
    for n in (1, 2):
        a[f"quant-n{n}"] = dict(cls="QuantisedConfidenceLUT", kw=dict(read_top_n=n, quant_mode="p2_int8", **CONF),
                                dtype="float32", tail=None, eval_only=False, needs=[], value_cells=n, weighted=True)
    a["deploy-n2"] = dict(cls="DeployedQuantisedConfidenceLUT", kw={}, dtype="float32", tail=None, eval_only=True,
                          needs=[], value_cells=2, weighted=True)
    return a


ARMS = _arms()
ALL_CARTS = list(ARMS)
_DT = {"float32": torch.float32, "bfloat16": torch.bfloat16, "int64": torch.int64, "int32": torch.int32}
_FEATURES: dict = {}
_SS = {"ext": None, "override": "per-arm"}


def spec():
    from spiky.lutorch_ex import LUTSpec
    return LUTSpec(**GEOM)


def lib_features() -> dict:
    """What the profiled lutorch_ex offers (signature checks + whether the soft-sign kernel builds). Cached."""
    if _FEATURES:
        return _FEATURES
    import spiky.lutorch_ex as lx
    from spiky.lutorch_ex.cartridges import _native_softsign as ns
    import warnings
    conf_params = inspect.signature(lx.ConfidenceLUT.__init__).parameters
    # the fp32 fused read: ConfidenceLUT(fused_read=True), or table_dtype=float32 in the lutorch_ex that had narrow tables
    _FEATURES["fused_read"] = "fused_read" in conf_params or "table_dtype" in conf_params
    _FEATURES["index_dtype"] = "index_dtype" in inspect.signature(lx.ManifestoLUT.__init__).parameters
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        _SS["ext"] = ns._ss_ext() if torch.cuda.is_available() else None
    _FEATURES["ss_kernel"] = _SS["ext"] is not None
    _FEATURES["ss_kernel_warning"] = [str(w.message)[:300] for w in wl] or None
    return _FEATURES


def unmet(name: str) -> str | None:
    f = lib_features()
    miss = [n for n in ARMS[name]["needs"] if not f.get(n)]
    if not miss:
        return None
    why = {"fused_read": "this lutorch_ex has no ConfidenceLUT fused read (neither fused_read= nor table_dtype=)",
           "index_dtype": "this lutorch_ex has no index_dtype option (every row is int64 there)",
           "ss_kernel": "the softsign_surrogate_grad CUDA kernel does not build/load in this lutorch_ex "
                        "(it falls back to the eager tail: see the -native-eager arm)"}
    return "; ".join(why[m] for m in miss)


def _set_tail(name: str):
    """Point the soft-sign tail at the kernel or the eager fallback for this arm (in-process)."""
    tail = ARMS[name]["tail"]
    if tail is None:
        return
    if _SS["override"] != "per-arm":
        tail = _SS["override"]
    from spiky.lutorch_ex.cartridges import _native_softsign as ns
    ns._SS_TRIED = True
    ns._SS_EXT = _SS["ext"] if tail == "kernel" else None


def build(name: str):
    """Fresh cartridge on CUDA, train mode (deploy: eval), in the arm's dtype; soft-sign tail set for the arm."""
    import spiky.lutorch_ex as lx
    a = ARMS[name]
    _set_tail(name)
    kw = {k: (_DT[v] if k == "index_dtype" else v) for k, v in a["kw"].items()}
    if kw.get("fused_read") and "fused_read" not in inspect.signature(lx.ConfidenceLUT.__init__).parameters:
        del kw["fused_read"]
        kw["table_dtype"] = torch.float32       # the same fp32 fused read in the lutorch_ex that had narrow tables
    if a["cls"] == "DeployedQuantisedConfidenceLUT":
        from spiky.lutorch_ex.cartridges.quantised_confidence import DeployedQuantisedConfidenceLUT
        q = lx.QuantisedConfidenceLUT(spec(), quant_mode="p2_int8", read_top_n=2, **COMMON).cuda().train()
        payload = q.to_deployment()
        assert payload["format"] == "p2_int8", payload["format"]
        return DeployedQuantisedConfidenceLUT(spec(), payload["tensors"], payload["meta"], device="cuda").eval()
    m = getattr(lx, a["cls"])(spec(), **kw, **BASE).cuda().train()
    return m.to(_DT[a["dtype"]]) if a["dtype"] != "float32" else m


def make_x(name: str, tokens: int, requires_grad: bool):
    s = spec()
    return torch.randn(tokens, s.h_in, s.d_in, device="cuda", dtype=_DT[ARMS[name]["dtype"]],
                       requires_grad=requires_grad)


def make_g(name: str, tokens: int):
    s = spec()
    return torch.randn(tokens, s.h_out, s.d_out, device="cuda", dtype=_DT[ARMS[name]["dtype"]])


def guarded(name: str, fn):
    """Run one arm's measurement; an unmet feature -> n/a, an OOM / other error -> recorded, the session goes on."""
    why = unmet(name)
    if why:
        return {"status": "n/a", "reason": why}
    try:
        r = fn()
        if isinstance(r, dict):
            r.setdefault("status", "ok")
        return r
    except Exception as e:
        status = "OOM" if plf.is_oom(e) else "error"
        plf.log(f"  [{status}] {name}: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
        return {"status": status, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"}
    finally:
        plf.free_cuda()


def static_facts(name: str, cart, tokens=(32768,)) -> dict:
    """HARDWARE-INDEPENDENT facts for one arm (analytic; must be identical on every host)."""
    s = spec()
    a = ARMS[name]
    G, tph, d = s.n_groups, s.tph, s.d_out
    entries = G * tph * a["value_cells"]
    idt = getattr(cart, "index_dtype", None)
    idx_bytes = (torch.empty((), dtype=idt).element_size() if isinstance(idt, torch.dtype) else 8)
    if name.startswith("deploy"):
        row, w, row_dtype = d * 1, 0, "int8"
        note = "int8 rows (packed), int32 shift-add; no fp32 master; scale folded into decompress"
    else:
        tdt = getattr(cart, "table_dtype", None)
        rdt = tdt if isinstance(tdt, torch.dtype) else cart.weights.dtype
        row, w, row_dtype = d * torch.empty((), dtype=rdt).element_size(), (4 if a["weighted"] else 0), str(rdt)[6:]
        note = ("fp32 rows gathered from the per-call fp32 fake-quant table copy" if name.startswith("quant")
                else f"{row_dtype} rows")
    nbytes = sum(t.numel() * t.element_size() for t in list(cart.parameters()) + list(cart.buffers()))
    out = {"value_cells_per_table": a["value_cells"],
           "read_macs_per_token": entries * d,
           "read_bytes_per_token": {"rows_bytes": entries * row, "index_bytes": entries * idx_bytes,
                                    "weight_bytes": entries * w, "total_bytes": entries * (row + idx_bytes + w),
                                    "row_dtype": row_dtype, "note": note},
           "index_dtype": str(idt).replace("torch.", "") if idt is not None else "int64 (no index_dtype option)",
           "param_dtype": a["dtype"],
           "n_params": sum(p.numel() for p in cart.parameters()),
           "param_and_buffer_bytes": nbytes,
           "softsign_tail": a["tail"] if _SS["override"] == "per-arm" or a["tail"] is None else _SS["override"]}
    if a["kw"].get("backend") == "auto":
        # _pick can depend on the batch (FusedManifestoSoftLUT: tier1 from 4096 rows), so ask it at each token count
        # of this run, with a zero-stride stand-in of the real shape (no allocation).
        res = {}
        for n in tokens:
            xs = torch.empty(1, s.h_in, s.d_in, device="cuda", dtype=_DT[a["dtype"]]).expand(n, s.h_in, s.d_in)
            cart.train()
            be = cart._pick(xs)
            if be == "native" and "SoftSign" in a["cls"]:
                be += (" (soft-sign surrogate kernel)" if lib_features()["ss_kernel"]
                       else " (eager soft-sign tail: the kernel does not build in this lutorch_ex)")
            cart.eval()
            res[str(n)] = {"train": be, "eval": cart._pick(xs)}
        out["auto_resolves_to"] = res
    if name.startswith("quant"):
        # per CALL (not per token): ste_tables reads the 4-byte master for amax, reads it again to quantise and
        # writes the 4-byte fake-quant copy
        out["per_call_fake_quant_table_pass_bytes"] = G * tph * s.n_cells * d * 4 * 3
    return out


# ------------------------------------------------------------------------------------------------ passes

def pass_ext() -> dict:
    """Do the cartridges' JIT extensions build/load here, and at what cost?"""
    from spiky.lutorch_ex.cartridges import _native_ops, _pow2_int8
    t0 = time.time()
    ok = _pow2_int8.ensure_registered()
    dt = time.time() - t0
    r = {"ensure_registered": bool(ok), "seconds": round(dt, 2), "enabled": bool(_pow2_int8._enabled),
         "error": _pow2_int8._error, "validated_arches": [list(a) for a in getattr(_pow2_int8, "VALIDATED_ARCHES", [])],
         "torch_extensions_dir": "set by lutorch_ex (_native_ops.py) unless TORCH_EXTENSIONS_DIR is exported"}
    f = lib_features()
    r["lprojection_native"] = _native_ops.native_manager() is not None
    r["softsign_surrogate_grad"] = f["ss_kernel"]
    r["softsign_surrogate_grad_warning"] = f["ss_kernel_warning"]
    sig = getattr(_native_ops, "_single_ig_ext", None)
    r["single_anchor_input_grad"] = (sig() is not None) if sig else None
    r["features"] = {k: v for k, v in f.items() if not k.endswith("warning")}
    plf.log(f"  [ext] p2_int8 native op: registered={r['ensure_registered']} enabled={r['enabled']} "
            f"in {r['seconds']} s; error={r['error']}")
    plf.log(f"  [ext] lprojection={r['lprojection_native']} softsign_surrogate_grad={r['softsign_surrogate_grad']} "
            f"single_anchor_input_grad={r['single_anchor_input_grad']} features={r['features']}")
    return r


def pass_static(args, carts) -> dict:
    out = {}
    for name in carts:
        out[name] = guarded(name, lambda: static_facts(name, build(name), args.tokens))
        if out[name].get("status") == "ok":
            plf.log(f"  [static] {name:20s} MACs/tok {out[name]['read_macs_per_token']:7d}  read B/tok "
                    f"{out[name]['read_bytes_per_token']['total_bytes']:7d}  index {out[name]['index_dtype']}")
    return out


def pass_bench(args, carts) -> dict:
    """The library's own harness (spiky.lutorch_ex.bench.benchmark), as used for the report's benchmark table.
    It builds fp32 inputs, so bf16 arms are recorded n/a here (the timing pass covers them)."""
    from spiky.lutorch_ex import bench
    label = "cuda:" + torch.cuda.get_device_name(0).replace(" ", "")
    out = {}
    for name in carts:
        if ARMS[name]["dtype"] != "float32":
            out[name] = {"status": "n/a", "reason": "bench.benchmark builds fp32 inputs; see the timing pass"}
            continue

        def run(name=name):
            ops = ("forward_eval",) if ARMS[name]["eval_only"] else bench.OPERATIONS
            rows = bench.benchmark(lambda: build(name), name=name, batch_sizes=args.tokens, operations=ops,
                                   devices=[label], warmup=args.warmup, repeats=args.repeats, measure_cuda=True)
            for r in rows:
                plf.log(f"  [bench] {name:20s} {r.operation:13s} N={r.batch_size:6d} {r.status:6s} "
                        f"median {r.median_ms if r.median_ms is not None else float('nan'):8.3f} ms  {r.note[:80]}")
            return {"rows": [r.to_dict() for r in rows]}
        out[name] = guarded(name, run)
    return out


def pass_timing(args, carts) -> dict:
    """CUDA-event median + IQR for eval forward, train forward, train fwd+bwd (backward = difference), per arm and
    token count; peak allocated memory per (arm, tokens). Each (arm, tokens) cell is guarded separately."""
    out = {}
    for name in carts:
        out[name] = {}
        for N in args.tokens:
            def run(name=name, N=N):
                cart = build(name)
                r = {}
                xe = make_x(name, N, False)
                cart.eval()
                with torch.no_grad():
                    t0 = time.time()
                    cart(xe)
                    torch.cuda.synchronize()
                    r["eval_first_call_s"] = round(time.time() - t0, 3)
                    t = plf.cuda_time_ms(lambda: cart(xe), args.warmup, args.repeats)
                r["eval_fwd"] = t
                r["eval_peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
                if not ARMS[name]["eval_only"]:
                    cart.train()
                    torch.cuda.reset_peak_memory_stats()
                    x = make_x(name, N, True)
                    g = make_g(name, N)
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
                plf.log(f"  [timing] {name:20s} N={N:6d} eval {r['eval_fwd']['median_ms']:8.3f} "
                        f"(IQR {r['eval_fwd']['iqr_ms']:.3f})"
                        + (f"  train fwd {r['train_fwd']['median_ms']:8.3f}  fwd+bwd {r['train_fwdbwd']['median_ms']:8.3f}"
                           f" (IQR {r['train_fwdbwd']['iqr_ms']:.3f})" if "train_fwd" in r else "")
                        + f" ms  peak {r['peak_alloc_gib']:.2f} GiB")
                return r
            out[name][str(N)] = guarded(name, run)
    return out


def pass_components(args, carts) -> dict:
    """Manual breakdown: each library building block of the TRAIN read alone, fwd and fwd+bwd (median ms)."""
    from spiky.lutorch_ex.cartridges import _pow2
    from spiky.lutorch_ex.cartridges._fused_ops import _global_cells
    N = max(args.tokens)
    out = {"tokens": N}
    for name in [c for c in TRAIN_CARTS if c in carts]:
        def run(name=name):
            cart = build(name)
            G, tph, K, d = cart.weights.shape
            x = make_x(name, N, True)
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
                t("fake-quant tables (eager)", cart._fake_quant_tables, [cart.weights])
            t("table-dropout mask", lambda: cart._table_dropout_mask(N, x.device, torch.float32), [])
            W2 = (cart._fake_quant_tables() if name.startswith("quant")
                  else cart.weights.reshape(G * tph * K, d)).detach()
            W2 = W2.clone().requires_grad_(True)
            gc = _global_cells(c, G, tph, K).reshape(N * G, tph)
            s = cart._score(u.detach()).reshape(N * G, tph).detach().clone().requires_grad_(True)
            t("embedding_bag read (fp32 rows)", lambda: F.embedding_bag(gc, W2, per_sample_weights=s, mode="sum"),
              [s, W2])
            t("cell-TV penalty (per optimizer step)", lambda: 10.0 * cart.cell_tv(), [cart.weights])
            xb = torch.randn(N, G * 48, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
            t("cast bf16->fp32 [N,768]", lambda: xb.float(), [xb])
            return r
        out[name] = guarded(name, run)
    return out


def pass_profile(args, carts) -> dict:
    out = {}
    N = args.profile_tokens or max(args.tokens)
    a = torch.randn(1024, 1024, device="cuda")
    plf._profile_session(args, "cart_warmup_discard", lambda: a @ a)
    for name in carts:
        def run(name=name):
            cart = build(name)
            r = {"tokens": N}
            xe = make_x(name, N, False)
            cart.eval()
            with torch.no_grad():
                for _ in range(3):
                    cart(xe)

                def ev():
                    cart(xe)
                k_e, _ = plf._profile_session(args, f"cart_{name}_eval", ev)
            sessions = {"eval": k_e}
            if not ARMS[name]["eval_only"]:
                cart.train()
                x = make_x(name, N, True)
                g = make_g(name, N)
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
                plf.log(f"  [profile] {name:20s} {phase:9s} {r[phase]['cuda_time_per_iter_ms']:8.3f} ms GPU/iter")
                for kk in r[phase]["kernels_ms"][:8]:
                    plf.log(f"        {kk['ms']:8.3f}  {kk['kernel'][:110]}")
            return r
        out[name] = guarded(name, run)
    return out


def pass_memory(args, carts) -> dict:
    out = {}
    N = max(args.tokens)
    for name in [c for c in carts if not ARMS[c]["eval_only"]]:
        def run(name=name):
            cart = build(name)
            x = make_x(name, N, True)
            g = make_g(name, N)
            cart(x).backward(g)
            for p in cart.parameters():
                p.grad = None
            x.grad = None
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
            for s_ in sorted(saved, key=lambda s_: (-s_[2], s_[0], s_[1])):
                if s_ not in seen:
                    seen.add(s_)
                    uniq.append({"shape": s_[0], "dtype": s_[1], "MiB": round(s_[2] / 2 ** 20, 1)})
            r = {"tokens": N, "held_after_fwd_gib": held / 2 ** 30,
                 "peak_alloc_fwdbwd_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                 "peak_reserved_fwdbwd_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
                 "saved_total_bytes": sum(s_[2] for s_ in saved), "n_saved": len(saved), "saved_top": uniq[:12]}
            plf.log(f"  [memory] {name:20s} N={N}: held {held / 2 ** 30:.2f} GiB, peak {r['peak_alloc_fwdbwd_gib']:.2f}"
                    f" GiB, saved {r['saved_total_bytes'] / 2 ** 30:.2f} GiB in {len(saved)} tensors")
            return r
        out[name] = guarded(name, run)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "profile_cartridges")
    ap.add_argument("--batch", type=int, nargs="+", default=[4, 16], help="sequences; tokens = batch x --seq")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--cartridges", nargs="+", default=ALL_CARTS, choices=ALL_CARTS)
    ap.add_argument("--passes", nargs="+", default=ALL_PASSES, choices=ALL_PASSES)
    ap.add_argument("--softsign-tail", default="per-arm", choices=["per-arm", "kernel", "eager"],
                    help="soft-sign surrogate tail for the FusedSoftSign native arms: per-arm (-native = kernel, "
                         "-native-eager = eager), or force one for all of them (in-process switch)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--profile-iters", type=int, default=3)
    ap.add_argument("--profile-tokens", type=int, default=None)
    ap.add_argument("--with-stack", action="store_true")
    ap.add_argument("--fp32-matmul-precision", default="high", choices=["highest", "high", "medium"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-dirty-lib", action="store_true",
                    help="profile a lutorch_ex src/ with uncommitted or untracked files (recorded, not refused)")
    args = ap.parse_args(argv)
    dirty_lib = plf.lib_src_dirty_files()
    if dirty_lib and not args.allow_dirty_lib:
        plf.log("Refusing to run: the profiled lutorch_ex src/ has uncommitted or untracked files, so the numbers "
                "would not be attributable to any commit:\n  " + "\n  ".join(dirty_lib)
                + "\nCommit them, or pass --allow-dirty-lib to profile anyway (the dirty files are recorded in env).")
        return 3
    args.tokens = sorted(b * args.seq for b in args.batch)
    _SS["override"] = args.softsign_tail
    if not torch.cuda.is_available():
        plf.log("No usable CUDA device: this profile only produces GPU numbers. Stopping.")
        return 2
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision(args.fp32_matmul_precision)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "geometry": GEOM, "common": COMMON, "arms": {c: ARMS[c] for c in args.cartridges}, "env": plf.env_block()}
    res["env"]["lib_features"] = {k: v for k, v in lib_features().items() if not k.endswith("warning")}
    plf.log(json.dumps(res["env"], indent=1))

    def save():
        (args.out_dir / "cartridges.json").write_text(json.dumps(res, indent=1, default=str))
    carts = list(args.cartridges)
    for p in ALL_PASSES[1:]:
        if p not in args.passes:
            continue
        plf.log(f"== {p}")
        res[p] = {"ext": lambda: pass_ext(), "static": lambda: pass_static(args, carts),
                  "bench": lambda: pass_bench(args, carts), "timing": lambda: pass_timing(args, carts),
                  "components": lambda: pass_components(args, carts), "profile": lambda: pass_profile(args, carts),
                  "memory": lambda: pass_memory(args, carts)}[p]()
        save()
    res["env"]["fa3_loader_imported"] = "nanochat.flash_attention" in sys.modules
    save()
    plf.log(f"done -> {args.out_dir / 'cartridges.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
