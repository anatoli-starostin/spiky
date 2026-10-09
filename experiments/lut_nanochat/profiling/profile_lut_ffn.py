"""Profile the d24 LUT-FFN block (ProjectionMHL + ConfidenceLUT, read_top_n=1) against the dense d24 FFN.

Re-runnable on any CUDA host (gpustar's RTX 5090, the Nebius H100, ...): results are partly hardware-dependent, so
every number lands in a JSON keyed identically on every host, for a field-by-field diff between machines.

Run (from anywhere; paths resolve relative to this file):
  experiments/lut_nanochat/profiling/profile_lut_ffn.sh                        # all passes, defaults below
  experiments/lut_nanochat/profiling/profile_lut_ffn.sh --batch 16 --passes block profile --out-dir /tmp/x
  python experiments/lut_nanochat/profiling/profile_lut_ffn.py --help
Needs a CUDA GPU (exits immediately with a message otherwise), torch, and this repo: src/ (spiky.lutorch_ex) and the
vendored nanochat package are put on sys.path here. Nothing is installed; nothing is written outside --out-dir
(torch/Triton/extension caches aside). If $HOME is read-only, export TRITON_CACHE_DIR to a writable dir first
(the shell wrapper does that only when it is unset). Attention / FA3 is NOT involved: the harness never imports
nanochat.gpt (whose import fetches FA3 kernels from the HF Hub on Hopper) - it re-declares the 3-line dense MLP and
records in the JSON that nanochat.flash_attention was never imported.

Variants (--variants):
  dense-fp32   nanochat d24 MLP: c_fc 1536->6144, relu^2, c_proj 6144->1536, fp32 activations (TF32 GEMMs under
               --fp32-matmul-precision high, which is what base_train sets).
  dense-bf16   same MLP, bf16 activations (nanochat's Linear casts the fp32 master weight to the input dtype).
  dense-fp8    same MLP through nanochat's Float8Linear (what the d24 baseline trains with, --fp8).
  lut-fp32     LUTFeedForward exactly as --lut-ffn wires it: ProjectionMHL(ConfidenceLUT).float(), the whole block
               an fp32 island (the status quo).
  lut-bf16     EXPERIMENTAL narrowed island, harness-only (lutorch_ex and the training path are untouched):
               compress/decompress GEMMs in bf16, fp32 only around addressing + scoring + gather.
  lut-fp8      EXPERIMENTAL narrowed island with compress/decompress through nanochat's fp8 matmul.
fp8 variants are SKIPPED with the reason printed and recorded when a functional torch._scaled_mm probe fails here
(never assumed from the GPU name).

Passes (--passes):
  env         GPU name, compute capability, VRAM, SM count, driver, CUDA, torch, Triton, repo commit, fp8 probe.
  peak        measured GEMM throughput per dtype and device copy bandwidth (the roofline references).
  block       one block, fwd and fwd+bwd: median ms + IQR, tokens/s, analytic MACs/token, achieved GEMM TFLOP/s,
              peak allocated/reserved, scaled to 24 layers per 2^20-token optimizer step.
  components  manual sub-module breakdown of the LUT block (casts, compress, addressing, score, embedding_bag gather,
              a cost model of the embedding_bag backward, decompress) per projection dtype, + per-step fixed costs
              (cell-TV, AdamW over the LUT params). Cross-checks the profiler.
  closeness   narrowed-island variants vs lut-fp32 on identical inputs/params/dropout: max abs/rel deviation of
              the output and of the input gradient, and the fraction of LUT cell addresses that flip.
  compile     whole-module torch.compile as base_train does it: graph count, graph breaks + reasons, recompiles.
  memory      peak allocated/reserved for fwd+bwd, and the saved-for-backward tensor inventory.
  profile     torch.profiler per-kernel breakdown (fwd and fwd+bwd sessions; bwd = difference), kernels bucketed.

Outputs in --out-dir: results.json (everything), summary.md, profile_<variant>_<fwd|fwdbwd>.txt, trace_*.json.
"""
from __future__ import annotations

import argparse
import contextlib
import subprocess
import datetime as _dt
import json
import math
import os
import platform
import re
import statistics
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]                  # <repo>/experiments/lut_nanochat/profiling/this.py
NANOCHAT_DIR = REPO_ROOT / "experiments" / "lut_nanochat" / "nanochat"
# LUTORCH_EX_SRC=<path to a repo's src/> profiles a different lutorch_ex (e.g. a feature branch's worktree) with this
# harness; recorded in the env block.
_LIB_SRC = Path(os.environ.get("LUTORCH_EX_SRC") or (REPO_ROOT / "src")).resolve()
for p in (NANOCHAT_DIR, _LIB_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _lib_src_git(*args):
    try:
        out = subprocess.run(["git", "-C", str(_LIB_SRC), *args], capture_output=True, text=True, timeout=30)
        return (out.stdout.rstrip() or None) if out.returncode == 0 else None
    except Exception:
        return None


def lib_src_dirty_files() -> list[str]:
    """Modified, staged AND untracked (non-ignored) files under the profiled src/ - an untracked module there is code
    that runs but is in no commit, so it counts as dirty."""
    out = _lib_src_git("status", "--porcelain", "--untracked-files=all", "--", ".")
    return out.splitlines() if out else []

import torch                                                      # noqa: E402
import torch.nn as nn                                             # noqa: E402
import torch.nn.functional as F                                   # noqa: E402

D_MODEL, N_LAYER, SEQ_LEN = 1536, 24, 2048
STEP_TOKENS = 2 ** 20                                              # d24 global batch (tokens / optimizer step)
LUT_GEOM = dict(h=16, d=48, tph=64, nap=8)                         # locked --lut-ffn geometry (r = h*d = 768)
ALL_VARIANTS = ["dense-fp32", "dense-bf16", "dense-fp8", "lut-fp32", "lut-bf16", "lut-fp8",
                "lut-fp8lib-c32", "lut-fp8lib-c16", "lut-fp8lib-d", "lut-bf16lib",
                "lut-fp32-i32", "lut-bf16lib-i32",
                "lut-fp32-i32-tbf16", "lut-bf16lib-i32-tfp32", "lut-bf16lib-i32-tbf16", "lut-bf16lib-i32-tfp8"]
# "-i32": ConfidenceLUT(index_dtype=torch.int32) (needs a lutorch_ex with that option; skipped with a reason otherwise)
I32_VARIANTS = {"lut-fp32-i32": "lut-fp32", "lut-bf16lib-i32": "lut-bf16lib"}
# "-t<dtype>": ConfidenceLUT(table_dtype=...) narrow-table read (needs a lutorch_ex with that option)
TABLE_VARIANTS = {"lut-fp32-i32-tbf16": ("lut-fp32-i32", "bfloat16"),
                  "lut-bf16lib-i32-tfp32": ("lut-bf16lib-i32", "float32"),     # control: fused read, fp32 rows
                  "lut-bf16lib-i32-tbf16": ("lut-bf16lib-i32", "bfloat16"),
                  "lut-bf16lib-i32-tfp8": ("lut-bf16lib-i32", "float8_e4m3fn")}
DEFAULT_VARIANTS = ALL_VARIANTS[:6]
# Library-level fp8 projections (ProjectionMHL(fp8_projections=..., compress_fp8_out_dtype=...)): only where the
# profiled lutorch_ex has that option; otherwise skipped with a reason.
FP8LIB_VARIANTS = {
    "lut-fp8lib-c32": dict(fp8_projections=("compress", "decompress"), compress_fp8_out_dtype="float32"),
    "lut-fp8lib-c16": dict(fp8_projections=("compress", "decompress"), compress_fp8_out_dtype="bfloat16"),
    "lut-fp8lib-d": dict(fp8_projections=("decompress",)),
    # library bf16 knob: ProjectionMHL(projection_dtype=torch.bfloat16), cartridge fp32
    "lut-bf16lib": dict(fp8_projections=(), projection_dtype="bfloat16"),
}
ALL_PASSES = ["env", "peak", "block", "components", "closeness", "fp8proj", "compile", "memory", "profile"]
DEFAULT_PASSES = [p for p in ALL_PASSES if p != "fp8proj"]
_R = LUT_GEOM["h"] * LUT_GEOM["d"]
# Analytic multiply-accumulates per token (hardware-independent).
MACS_PER_TOKEN = {
    "dense": 2 * D_MODEL * 4 * D_MODEL,                                            # 18,874,368
    "lut_projections": 2 * D_MODEL * _R,                                           # 2,359,296
    "lut_read": LUT_GEOM["h"] * LUT_GEOM["tph"] * LUT_GEOM["d"],                   # 49,152 (score-weighted sum)
}
MACS_PER_TOKEN["lut"] = MACS_PER_TOKEN["lut_projections"] + MACS_PER_TOKEN["lut_read"]


# ------------------------------------------------------------------------------------------------ utils

def log(msg: str) -> None:
    print(msg, flush=True)


def cuda_time_ms(fn, warmup: int, repeats: int, setup=None) -> dict:
    """Median/min/max ms of fn() over `repeats` timed calls (CUDA events, synchronised), after `warmup` calls.
    `setup` (untimed) runs before every call, e.g. to zero grads."""
    for _ in range(warmup):
        if setup:
            setup()
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if setup:
            setup()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    q1, _, q3 = statistics.quantiles(times, n=4) if len(times) >= 2 else (times[0], None, times[0])
    return {"median_ms": statistics.median(times), "q1_ms": q1, "q3_ms": q3, "iqr_ms": q3 - q1,
            "min_ms": min(times), "max_ms": max(times), "n": repeats}


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def free_cuda():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ------------------------------------------------------------------------------------------------ env

def fp8_probe() -> tuple[bool, str]:
    """Functional check: can this GPU/torch run the tensorwise torch._scaled_mm that nanochat's Float8Linear uses?"""
    try:
        a = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn).t()
        one = torch.tensor(1.0, device="cuda")
        torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16, use_fast_accum=True)
        torch.cuda.synchronize()
        return True, "torch._scaled_mm (e4m3, tensorwise) ran on this device"
    except Exception as e:                                         # unsupported arch / build
        return False, f"torch._scaled_mm failed here: {type(e).__name__}: {str(e).splitlines()[0][:200]}"


def env_block() -> dict:
    props = torch.cuda.get_device_properties(0)
    try:
        import triton
        triton_ver = triton.__version__
    except Exception:
        triton_ver = None
    ok8, why8 = fp8_probe()

    def _run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=REPO_ROOT).stdout.strip() or None
        except Exception:
            return None
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "repo_commit": _run(["git", "rev-parse", "HEAD"]),
        "lutorch_ex_src": None if _LIB_SRC == (REPO_ROOT / "src").resolve() else "LUTORCH_EX_SRC override",
        "lutorch_ex_src_commit": _lib_src_git("rev-parse", "HEAD"),
        "lutorch_ex_src_describe": _lib_src_git("describe", "--always", "--dirty"),
        "lutorch_ex_src_dirty_files": lib_src_dirty_files(),           # includes untracked; [] = clean
        "repo_tracked_changes": bool(dirty),
        "driver": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "host_platform": platform.platform(),
        "python": platform.python_version(),
        "gpu_name": props.name,
        "gpu_vram_gib": round(props.total_memory / 2 ** 30, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": triton_ver,
        "fp32_matmul_precision": torch.get_float32_matmul_precision(),
        "fp8_usable": ok8,
        "fp8_probe": why8,
        "macs_per_token": MACS_PER_TOKEN,
    }


# ------------------------------------------------------------------------------------------------ peak

def measure_peak(fp8_ok: bool, warmup: int, repeats: int) -> dict:
    """Measured (not datasheet) dense GEMM throughput per dtype at 8192^3, and device copy bandwidth."""
    out = {}
    n = 8192
    flops = 2 * n ** 3
    a32, b32 = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
    t = cuda_time_ms(lambda: a32 @ b32, warmup, repeats)
    out[f"gemm_{torch.get_float32_matmul_precision()}_fp32_tflops"] = flops / t["median_ms"] / 1e9
    a16, b16 = a32.bfloat16(), b32.bfloat16()
    t = cuda_time_ms(lambda: a16 @ b16, warmup, repeats)
    out["gemm_bf16_tflops"] = flops / t["median_ms"] / 1e9
    if fp8_ok:
        a8 = a16.to(torch.float8_e4m3fn)
        b8 = b16.to(torch.float8_e4m3fn).t().contiguous().t()
        one = torch.tensor(1.0, device="cuda")
        t = cuda_time_ms(lambda: torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16,
                                                  use_fast_accum=True), warmup, repeats)
        out["gemm_fp8_tflops"] = flops / t["median_ms"] / 1e9
    del a32, b32, a16, b16
    src = torch.empty(2 ** 28, dtype=torch.float32, device="cuda")        # 1 GiB
    dst = torch.empty_like(src)
    t = cuda_time_ms(lambda: dst.copy_(src), warmup, repeats)
    out["copy_bandwidth_gbs"] = 2 * src.numel() * 4 / t["median_ms"] / 1e6  # read + write
    del src, dst
    free_cuda()
    for k, v in out.items():
        log(f"  [peak] {k}: {v:.1f}")
    return out


# ------------------------------------------------------------------------------------------------ modules

def _lut_cfg():
    from nanochat.lut_ffn import LUTFFNConfig
    return LUTFFNConfig(enabled=True, **LUT_GEOM)                   # all other fields = base_train defaults


class LUTProjVariant(nn.Module):
    """LUTFeedForward with its compress/decompress GEMMs in bf16 or fp8; addressing/score/gather stay fp32.
    Reuses the wrapped module's parameters (same shapes, same init)."""

    def __init__(self, lut, proj_dtype: str):
        super().__init__()
        self.lut, self.proj_dtype = lut, proj_dtype
        self.mhl = lut.mhl

    def _proj(self, x2, lin):
        if self.proj_dtype == "bf16":
            return F.linear(x2.to(torch.bfloat16), lin.weight.to(torch.bfloat16), lin.bias.to(torch.bfloat16))
        from nanochat.fp8 import _Float8Matmul
        return _Float8Matmul.apply(x2.to(torch.bfloat16), lin.weight) + lin.bias.to(torch.bfloat16)

    def forward(self, x):
        B, T, C = x.shape
        spec = self.mhl.cartridge.spec
        z = self._proj(x.reshape(B * T, C), self.mhl.compress).float()
        y = self.mhl.cartridge(z.reshape(B * T, spec.h_in, spec.d_in)).reshape(B * T, spec.out_features)
        return self._proj(y, self.mhl.decompress).reshape(B, T, C).to(x.dtype)


class _CastLinear(nn.Linear):
    """nanochat.gpt.Linear: fp32 master weight cast to the activation dtype in forward."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class DenseFFN(nn.Module):
    """nanochat.gpt.MLP, re-declared verbatim (c_fc, relu^2, c_proj) so the harness never imports nanochat.gpt,
    whose import tries to fetch FA3 kernels from the HF Hub on Hopper."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = _CastLinear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = _CastLinear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


def input_dtype(name: str):
    return torch.float32 if name == "dense-fp32" else torch.bfloat16    # the d24 residual stream is bf16


def i32_supported() -> tuple[bool, str]:
    import inspect
    from spiky.lutorch_ex.cartridges.manifesto_base import ManifestoLUT
    if "index_dtype" not in inspect.signature(ManifestoLUT.__init__).parameters:
        return False, f"the profiled lutorch_ex ({_LIB_SRC}) has no ManifestoLUT(index_dtype=...)"
    return True, ""


def build_variant(name: str, decompress_std: float):
    """Fresh module for `name`, on CUDA, train mode, params fp32 (as in training: master weights fp32)."""
    if name in TABLE_VARIANTS:
        base, tdt = TABLE_VARIANTS[name]
        mod = build_variant(base, decompress_std)
        lut = mod if hasattr(mod, "mhl") else mod.lut
        lut.mhl.cartridge.table_dtype = getattr(torch, tdt)   # read in forward; the cache exists from __init__
        return mod
    if name in I32_VARIANTS:
        mod = build_variant(I32_VARIANTS[name], decompress_std)
        lut = mod if hasattr(mod, "mhl") else mod.lut
        lut.mhl.cartridge.index_dtype = torch.int32      # read at trace time; same as passing index_dtype=int32
        return mod
    if name.startswith("dense"):
        mod = DenseFFN(D_MODEL).cuda()
        with torch.no_grad():                                      # base_train init: c_fc uniform, c_proj zeros
            s = 3 ** 0.5 * D_MODEL ** -0.5
            mod.c_fc.weight.uniform_(-s * 0.4, s * 0.4)
            mod.c_proj.weight.normal_(0, decompress_std)           # not zero, so backward does real work
        if name == "dense-fp8":
            from nanochat.fp8 import convert_to_float8_training
            convert_to_float8_training(mod)
        return mod.train()
    from nanochat.lut_ffn import LUTFeedForward
    lut = LUTFeedForward(D_MODEL, _lut_cfg(), device="cuda")
    with torch.no_grad():                                          # zero-init decompress would make bwd trivial
        lut.mhl.decompress.weight.normal_(0, decompress_std)
    if name in FP8LIB_VARIANTS:
        lut.mhl = fp8lib_projection(lut.mhl, **FP8LIB_VARIANTS[name])
        return lut.train()
    mod = lut if name == "lut-fp32" else LUTProjVariant(lut, name.split("-")[1])
    return mod.train()


def fp8lib_supported(needs=("fp8_projections",)) -> tuple[bool, str]:
    import inspect
    from spiky.lutorch_ex import ProjectionMHL
    params = inspect.signature(ProjectionMHL.__init__).parameters
    missing = [n for n in needs if n not in params]
    if missing:
        return False, f"the profiled lutorch_ex ({_LIB_SRC}) has no ProjectionMHL({', '.join(missing)}=...)"
    return True, ""


def fp8lib_projection(mhl, fp8_projections, compress_fp8_out_dtype="float32", projection_dtype=None):
    """A ProjectionMHL with library fp8 / bf16 projections sharing `mhl`'s cartridge and (copied) compress/decompress."""
    from spiky.lutorch_ex import ProjectionMHL
    kw = {} if projection_dtype is None else {"projection_dtype": getattr(torch, projection_dtype)}
    new = ProjectionMHL(mhl.cartridge, d_model=mhl.input_dim, fp8_projections=fp8_projections,
                        compress_fp8_out_dtype=getattr(torch, compress_fp8_out_dtype), **kw).float().cuda()
    new.compress.load_state_dict(mhl.compress.state_dict())
    new.decompress.load_state_dict(mhl.decompress.state_dict())
    return new


def configure_dynamo_like_base_train():
    import torch._dynamo
    torch._dynamo.config.force_parameter_static_shapes = False
    torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, "cache_size_limit", 8), 256)


def maybe_compile(mod, mode: str):
    """mode 'model' = what base_train does (whole module, dynamic=False); 'none' = no outer compile (the
    ConfidenceLUT cartridge still self-compiles its train forward on CUDA, as lutorch_ex always does)."""
    return torch.compile(mod, dynamic=False) if mode == "model" else mod


def make_input(tokens: int, requires_grad=True, dtype=torch.bfloat16):
    """[B, T, d_model] activations, T = --seq (tokens = --batch x --seq)."""
    T = min(SEQ_LEN, tokens)
    x = torch.randn(tokens // T, T, D_MODEL, device="cuda", dtype=dtype)
    return x.requires_grad_(requires_grad)


def ffn_flops(name: str, tokens: int) -> float:
    """Forward GEMM FLOPs of the block (the LUT read itself is ~0 FLOPs; gather bytes are reported separately)."""
    if name.startswith("dense"):
        return 2 * tokens * D_MODEL * 4 * D_MODEL * 2
    r = LUT_GEOM["h"] * LUT_GEOM["d"]
    return 2 * tokens * D_MODEL * r * 2


def gather_bytes_fwd(tokens: int) -> int:
    """Bytes the forward read touches at minimum: one fp32 row of d_out per (token, group, table), + int64 index
    and fp32 per-sample weight per (token, group, table)."""
    G, tph, d = LUT_GEOM["h"], LUT_GEOM["tph"], LUT_GEOM["d"]
    return tokens * G * tph * (d * 4 + 8 + 4)


# ------------------------------------------------------------------------------------------------ passes

def pass_block(args, variants, peak) -> dict:
    res = {}
    for name in variants:
        res[name] = {}
        for tokens in args.tokens:
            free_cuda()
            try:
                mod = maybe_compile(build_variant(name, args.decompress_std), args.compile_mode)
                x = make_input(tokens, dtype=input_dtype(name))
                g = torch.randn_like(x)

                def fwd():
                    return mod(x)

                def fwdbwd():
                    mod(x).backward(g)

                def zero():
                    for p in mod.parameters():
                        p.grad = None
                    x.grad = None

                t0 = time.time()
                fwdbwd()                                               # first call = compile
                torch.cuda.synchronize()
                first_call_s = time.time() - t0
                tf = cuda_time_ms(fwd, args.warmup, args.repeats)
                tb = cuda_time_ms(fwdbwd, args.warmup, args.repeats, setup=zero)
                fl = ffn_flops(name, tokens)
                r = {
                    "tokens": tokens, "first_call_incl_compile_s": round(first_call_s, 2),
                    "fwd_ms": tf["median_ms"], "fwdbwd_ms": tb["median_ms"],
                    "fwd_iqr_ms": tf["iqr_ms"], "fwdbwd_iqr_ms": tb["iqr_ms"],
                    "fwd_q1_q3_ms": [tf["q1_ms"], tf["q3_ms"]], "fwdbwd_q1_q3_ms": [tb["q1_ms"], tb["q3_ms"]],
                    "macs_per_token": MACS_PER_TOKEN["dense" if name.startswith("dense") else "lut"],
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
                    "fwd_tokens_per_s": tokens / tf["median_ms"] * 1e3,
                    "fwdbwd_tokens_per_s": tokens / tb["median_ms"] * 1e3,
                    "gemm_flops_fwd": fl,
                    "achieved_gemm_tflops_fwdbwd": 3 * fl / tb["median_ms"] / 1e9,
                    # per optimizer step: 24 layers x (2^20 / tokens) micro-batches, FFN only
                    "ffn24_per_step_s": N_LAYER * tb["median_ms"] * (STEP_TOKENS / tokens) / 1e3,
                    "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                }
                if name.startswith("lut"):
                    gb = gather_bytes_fwd(tokens)
                    r["gather_bytes_fwd"] = gb
                    r["gather_gbs_if_fwd_were_pure_read"] = gb / tf["median_ms"] / 1e6
                res[name][str(tokens)] = r
                log(f"  [block] {name:11s} tokens={tokens:6d}  fwd {tf['median_ms']:8.2f} ms  fwd+bwd "
                    f"{tb['median_ms']:8.2f} ms  ({r['fwdbwd_tokens_per_s'] / 1e6:6.2f} Mtok/s)  "
                    f"24L/step {r['ffn24_per_step_s']:6.2f} s  peak {r['peak_alloc_gib']:.1f} GiB")
                del mod, x, g
            except Exception as e:
                if not is_oom(e):
                    raise
                res[name][str(tokens)] = {"tokens": tokens, "oom": True}
                log(f"  [block] {name:11s} tokens={tokens:6d}  OOM")
                free_cuda()
    return res


def pass_components(args, proj_dtypes) -> dict:
    """Split the LUT block into its pieces at each token count; time each piece fwd and fwd+bwd in isolation."""
    from spiky.lutorch_ex.cartridges._fused_ops import _global_cells
    out = {}
    for tokens in args.tokens:
        free_cuda()
        r = {}
        try:
            lut = build_variant("lut-fp32", args.decompress_std)
            mhl, cart = lut.mhl, lut.mhl.cartridge
            spec = cart.spec
            G, tph, K, d_out = cart.weights.shape
            x = make_input(tokens).reshape(tokens, D_MODEL).detach()

            def timed(label, fwd_fn, inputs_req_grad):
                tf = cuda_time_ms(lambda: fwd_fn(), args.warmup, args.repeats)

                def fb():
                    y = fwd_fn()
                    y.backward(torch.ones_like(y))

                def zero():
                    for t in list(inputs_req_grad) + list(lut.parameters()):
                        t.grad = None
                tb = cuda_time_ms(fb, args.warmup, args.repeats, setup=zero) if inputs_req_grad else None
                r[label] = {"fwd_ms": tf["median_ms"], "fwdbwd_ms": tb["median_ms"] if tb else None}
                log(f"  [components] tokens={tokens:6d} {label:34s} fwd {tf['median_ms']:8.3f} ms"
                    + (f"  fwd+bwd {tb['median_ms']:8.3f} ms" if tb else ""))

            xr = x.clone().requires_grad_(True)
            timed("cast_in bf16->fp32", lambda: xr.float(), [xr])
            for pd in proj_dtypes:
                xin = (x.float() if pd == "fp32" else x).clone().requires_grad_(True)
                if pd == "fp32":
                    fn = lambda: mhl.compress(xin)                            # noqa: E731
                elif pd == "bf16":
                    fn = lambda: F.linear(xin, mhl.compress.weight.bfloat16(), mhl.compress.bias.bfloat16())  # noqa
                else:
                    from nanochat.fp8 import _Float8Matmul
                    fn = lambda: _Float8Matmul.apply(xin, mhl.compress.weight) + mhl.compress.bias.bfloat16()  # noqa
                timed(f"compress_{pd}", fn, [xin])
            z = mhl.compress(x.float()).detach().reshape(tokens, spec.h_in, spec.d_in)
            zr = z.clone().requires_grad_(True)
            timed("cartridge_whole (compiled, train)", lambda: cart(zr), [zr])
            addr = torch.compile(cart._addresses, dynamic=True)
            timed("  addressing (compiled)", lambda: addr(zr)[1], [zr])
            _, u, c, *_ = addr(z)
            u = u.detach().clone().requires_grad_(True)
            timed("  score (eager)", lambda: cart._score(u), [u])
            score_c = torch.compile(cart._score, dynamic=True)
            timed("  score (compiled)", lambda: score_c(u), [u])
            s = cart._score(u).detach().clone().requires_grad_(True)
            W2 = cart.weights.reshape(G * tph * K, d_out)
            gc = _global_cells(c, G, tph, K).reshape(tokens * G, tph)
            timed("  embedding_bag read", lambda: F.embedding_bag(gc, W2, per_sample_weights=s.reshape(tokens * G, tph),
                                                                    mode="sum"), [s])
            r["embedding_bag_bytes_fwd"] = gather_bytes_fwd(tokens)
            # Building blocks of the embedding_bag backward, timed in isolation (a cost model that does not need a
            # kernel trace): sorting the N*G*tph flat cell indices (how CUDA embedding backward groups duplicates),
            # the scatter-add of N*G*tph gradient rows of d_out into grad_W, and the per-sample-weight gradient
            # (one gathered row dotted with the output grad per (token, group, table)).
            flat = gc.reshape(-1)
            rows = torch.randn(flat.numel(), d_out, device="cuda")
            gW = torch.zeros_like(W2)
            go = torch.randn(tokens * G, d_out, device="cuda")
            for label, fn in (
                ("  bwd-model: sort indices", lambda: torch.sort(flat)),
                ("  bwd-model: index_add_ grad rows", lambda: gW.index_add_(0, flat, rows)),
                ("  bwd-model: psw grad (gather.dot)", lambda: (W2[gc] * go.unsqueeze(1)).sum(-1)),
            ):
                t = cuda_time_ms(fn, args.warmup, args.repeats)
                r[label.strip()] = {"fwd_ms": t["median_ms"], "fwdbwd_ms": None}
                log(f"  [components] tokens={tokens:6d} {label:34s} {t['median_ms']:8.3f} ms")
            del rows, gW, go, flat
            y = cart(z).detach().reshape(tokens, spec.out_features)
            for pd in proj_dtypes:
                yin = (y if pd == "fp32" else y.bfloat16()).clone().requires_grad_(True)
                if pd == "fp32":
                    fn = lambda: mhl.decompress(yin)                          # noqa: E731
                elif pd == "bf16":
                    fn = lambda: F.linear(yin, mhl.decompress.weight.bfloat16(), mhl.decompress.bias.bfloat16())  # noqa
                else:
                    from nanochat.fp8 import _Float8Matmul
                    fn = lambda: _Float8Matmul.apply(yin, mhl.decompress.weight) + mhl.decompress.bias.bfloat16()  # noqa
                timed(f"decompress_{pd}", fn, [yin])
            o = torch.randn(tokens, D_MODEL, device="cuda").requires_grad_(True)
            timed("cast_out fp32->bf16", lambda: o.to(torch.bfloat16), [o])
            out[str(tokens)] = r
            del lut, x, z, zr, u, s, y, o
        except Exception as e:
            if not is_oom(e):
                raise
            out[str(tokens)] = {"oom": True}
            log(f"  [components] tokens={tokens} OOM")
        free_cuda()

    # per-optimizer-step fixed costs (independent of tokens): cell-TV x24 and AdamW over 24 layers' LUT params
    fixed = {}
    lut = build_variant("lut-fp32", args.decompress_std)
    cart = lut.mhl.cartridge

    def tv():
        (10.0 * cart.cell_tv()).backward()
    t = cuda_time_ms(tv, args.warmup, args.repeats, setup=lambda: setattr(cart.weights, "grad", None))
    fixed["cell_tv_fwdbwd_ms_per_layer"] = t["median_ms"]
    fixed["cell_tv_fwdbwd_ms_24_layers"] = N_LAYER * t["median_ms"]
    log(f"  [components] cell-TV fwd+bwd: {t['median_ms']:.2f} ms/layer -> {N_LAYER * t['median_ms']:.1f} ms/step")
    shapes = [p.shape for p in lut.parameters()]
    del lut
    free_cuda()
    try:
        params = [torch.zeros(s, device="cuda", requires_grad=True) for _ in range(N_LAYER) for s in shapes]
        for p in params:
            p.grad = torch.randn_like(p) * 1e-3
        n_params = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.0, fused=True)
        t = cuda_time_ms(opt.step, args.warmup, max(3, args.repeats // 2))
        fixed["adamw_fused_step_ms_lut_params"] = t["median_ms"]
        fixed["lut_params_24_layers"] = n_params
        log(f"  [components] AdamW(fused) step over {n_params / 1e6:.1f}M LUT params: {t['median_ms']:.1f} ms/step")
        del params, opt
    except Exception as e:
        if not is_oom(e):
            raise
        fixed["adamw_fused_step_ms_lut_params"] = "OOM"
    free_cuda()
    out["per_step_fixed"] = fixed
    return out


def pass_closeness(args, variants) -> dict:
    """Narrowed-island variants vs lut-fp32: SAME parameters, SAME input, SAME table-dropout mask (the global RNG is
    reseeded before each forward), so any difference is the projection precision. Reports max abs / max rel
    deviation of the block output and of the input gradient, and the fraction of (token, group, table) LUT cell
    addresses that change - the compress output decides sign bits, so low precision can flip addresses."""
    out = {}
    tokens = min(args.tokens)
    lut = build_variant("lut-fp32", args.decompress_std)
    x = make_input(tokens, requires_grad=False)
    g = torch.randn_like(x)
    cart = lut.mhl.cartridge
    spec = cart.spec

    def run(mod):
        xi = x.clone().requires_grad_(True)
        torch.manual_seed(1234)
        y = mod(xi)
        y.backward(g)
        return y.detach().float(), xi.grad.detach().float()

    def addresses(z):
        return cart._addresses(z.float().reshape(tokens, spec.h_in, spec.d_in))[2]

    with torch.no_grad():
        x2 = x.reshape(tokens, D_MODEL)
        c_ref = addresses(lut.mhl.compress(x2.float()))
        # Baseline for the flip fractions: the status quo itself runs compress in TF32 ("high"); how many addresses
        # does that already flip relative to true-fp32 ("highest") compress?
        prev = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        c_exact = addresses(lut.mhl.compress(x2.float()))
        torch.set_float32_matmul_precision(prev)
    out["status_quo_tf32_vs_exact_fp32_address_flip_fraction"] = (c_ref != c_exact).float().mean().item()
    log(f"  [closeness] baseline: TF32 compress (status quo, precision={prev}) vs exact fp32 flips "
        f"{100 * out['status_quo_tf32_vs_exact_fp32_address_flip_fraction']:.3f}% of addresses")
    y_ref, gx_ref = run(lut)
    for name in [v for v in variants if v in ("lut-bf16", "lut-fp8")]:
        mod = LUTProjVariant(lut, name.split("-")[1]).train()
        for p in lut.parameters():
            p.grad = None
        y, gx = run(mod)
        with torch.no_grad():
            c = addresses(mod._proj(x2, lut.mhl.compress))

        def dev(a, ref):
            d = (a - ref).abs()
            return {"max_abs": d.max().item(), "max_rel_to_max_ref": (d.max() / ref.abs().max().clamp_min(1e-30)).item(),
                    "mean_abs": d.mean().item(), "ref_max_abs": ref.abs().max().item()}
        out[name] = {"tokens": tokens, "output": dev(y, y_ref), "input_grad": dev(gx, gx_ref),
                     "address_flip_fraction": (c != c_ref).float().mean().item()}
        log(f"  [closeness] {name} vs lut-fp32: out max|d| {out[name]['output']['max_abs']:.3e} "
            f"(rel {out[name]['output']['max_rel_to_max_ref']:.2e}, mean|d| {out[name]['output']['mean_abs']:.2e} vs "
            f"max|ref| {out[name]['output']['ref_max_abs']:.2e}), dx max|d| {out[name]['input_grad']['max_abs']:.3e} "
            f"(rel {out[name]['input_grad']['max_rel_to_max_ref']:.2e}), addresses flipped "
            f"{100 * out[name]['address_flip_fraction']:.3f}%")
    del lut, x, g
    free_cuda()
    return out


def pass_fp8proj(args) -> dict:
    """Library fp8 projections (ProjectionMHL fp8_projections): (1) sign-bit flip rates of the compress output vs an
    exact-fp32 compress, per variant, plus an emulation that attributes the flips to operand vs output precision;
    (2) GEMM-only timings per direction (compress K=1536 -> 768, decompress K=768 -> 1536), fp32(TF32) / bf16 / fp8,
    with the raw _scaled_mm cost separated from the amax/scale/cast overhead. Block-level timings of the same variants
    come from the `block` pass (--variants lut-fp8lib-*)."""
    ok, why = fp8lib_supported()
    out = {"supported": ok, "reason": why}
    if not ok:
        log(f"  [fp8proj] skipped: {why}")
        return out
    from spiky.lutorch_ex.fp8 import _to_fp8, fp8_linear
    N = max(args.tokens)
    out["tokens"] = N
    lut = build_variant("lut-fp32", args.decompress_std)
    mhl, cart = lut.mhl, lut.mhl.cartridge
    spec = cart.spec
    W, b = mhl.compress.weight.detach(), mhl.compress.bias.detach()
    x = make_input(N, requires_grad=False).reshape(N, D_MODEL).float()   # the island's input: bf16 stream cast up
    prev = torch.get_float32_matmul_precision()

    def linear_at(precision, *a):
        torch.set_float32_matmul_precision(precision)
        try:
            return F.linear(*a)
        finally:
            torch.set_float32_matmul_precision(prev)

    def bits_of(z):
        u = cart._addresses(z.float().reshape(N, spec.h_in, spec.d_in))[1]          # [N, G, tph, nap] margins
        return u > cart.cmp_eps

    def q8(t):                                                                       # e4m3 quantise-dequantise
        tq, inv = _to_fp8(t, torch.float8_e4m3fn)
        return tq.float() * inv

    with torch.no_grad():
        z_exact = linear_at("highest", x, W, b)
        bits_ref = bits_of(z_exact)
        rms_ref = z_exact.pow(2).mean().sqrt()
        cases = {
            "tf32 operands, fp32 out (status quo)": lambda: linear_at("high", x, W, b),
            "bf16 operands, bf16 out (harness lut-bf16)": lambda: F.linear(x.bfloat16(), W.bfloat16(), b.bfloat16()),
            "fp8 lib: e4m3 operands, fp32 acc, fp32 out": lambda: fp8_linear(x, mhl.compress, torch.float32),
            "fp8 lib: e4m3 operands, fp32 acc, bf16 out": lambda: fp8_linear(x, mhl.compress, torch.bfloat16),
            "fp8 lib: decompress only (compress untouched = status quo)": lambda: linear_at("high", x, W, b),
            "emu: exact operands, output rounded to bf16": lambda: z_exact.bfloat16().float(),
            "emu: exact operands, output rounded to e4m3": lambda: q8(z_exact),
            "emu: e4m3 operands (both), exact math, fp32 out": lambda: linear_at("highest", q8(x), q8(W), b),
            "emu: e4m3 weight only, exact math": lambda: linear_at("highest", x, q8(W), b),
            "emu: e4m3 activation only, exact math": lambda: linear_at("highest", q8(x), W, b),
        }
        flips = {}
        for label, fn in cases.items():
            z = fn().float()
            d = bits_of(z) != bits_ref
            flips[label] = {"bit_flip_fraction": d.float().mean().item(),
                            "cell_flip_fraction": d.any(-1).float().mean().item(),
                            "compress_out_rel_rms_err": ((z - z_exact).pow(2).mean().sqrt() / rms_ref).item()}
            log(f"  [fp8proj] flips {label:58s} bits {100 * flips[label]['bit_flip_fraction']:7.3f}%  "
                f"cells {100 * flips[label]['cell_flip_fraction']:7.3f}%  rel-rms(z) {flips[label]['compress_out_rel_rms_err']:.2e}")
        # Library-faithful rows: the real ProjectionMHL variant (same weights), cartridge input captured by a hook.
        for name, cfg in FP8LIB_VARIANTS.items():
            needs = ("fp8_projections",) + (("projection_dtype",) if "projection_dtype" in cfg else ())
            if not fp8lib_supported(needs)[0]:
                continue
            pm = fp8lib_projection(mhl, **cfg).eval()
            seen = {}
            hk = pm.cartridge.register_forward_pre_hook(lambda mod, a: seen.setdefault("z", a[0].detach()))
            try:
                pm(x)
            finally:
                hk.remove()
            z = seen["z"]
            assert z.dtype == torch.float32, z.dtype                                # cartridge input must be fp32
            d = bits_of(z) != bits_ref
            label = f"library {name} ({cfg})"
            flips[label] = {"bit_flip_fraction": d.float().mean().item(),
                            "cell_flip_fraction": d.any(-1).float().mean().item(),
                            "compress_out_rel_rms_err": ((z.reshape(N, -1) - z_exact).pow(2).mean().sqrt()
                                                         / rms_ref).item(),
                            "cartridge_input_dtype": str(z.dtype)}
            log(f"  [fp8proj] flips {label[:58]:58s} bits {100 * flips[label]['bit_flip_fraction']:7.3f}%  "
                f"cells {100 * flips[label]['cell_flip_fraction']:7.3f}%  (cartridge input {z.dtype})")
            del pm, z
        u_abs = (z_exact.reshape(N, spec.h_in, spec.d_in))
        u_ref = cart._addresses(u_abs)[1].abs()
        out["margin_quantiles_abs"] = {f"q{int(p * 100)}": u_ref.flatten()[:2 ** 24].float().quantile(p).item()
                                       for p in (0.01, 0.05, 0.5)}
        out["compress_out_rms"] = rms_ref.item()
    out["flips_vs_exact_fp32"] = flips
    out["flip_caveat"] = ("measured at init (compress ~ N(0, 0.02), random bf16 inputs); trained margins differ")
    del z_exact, bits_ref, x

    # ---- GEMM-only timings per direction (eager and torch.compile'd, as base_train compiles the model)
    gemm = {}
    for direction, (k, n) in (("compress", (D_MODEL, _R)), ("decompress", (_R, D_MODEL))):
        lin = nn.Linear(k, n).cuda()
        nn.init.normal_(lin.weight, std=0.02)
        xin = torch.randn(N, k, device="cuda")
        r = {"M": N, "K": k, "N": n, "flops_fwd": 2 * N * k * n}
        fwds = {
            "tf32": lambda xx: lin(xx),
            "bf16": lambda xx: F.linear(xx.bfloat16(), lin.weight.bfloat16(), lin.bias.bfloat16()),
            # what ProjectionMHL(projection_dtype=bf16) runs: the bf16 GEMM + the cast back to fp32
            "bf16 (lib, +cast to fp32)": lambda xx: F.linear(xx.bfloat16(), lin.weight.bfloat16(),
                                                             lin.bias.bfloat16()).float(),
            "fp8 (lib, fp32 out)": lambda xx: fp8_linear(xx, lin, torch.float32),
            "fp8 (lib, bf16 out)": lambda xx: fp8_linear(xx, lin, torch.bfloat16),
        }
        for mode in ("eager", "compiled"):
            for label, f in fwds.items():
                fn = torch.compile(f, dynamic=False) if mode == "compiled" else f
                xr = xin.clone().requires_grad_(True)
                tf = cuda_time_ms(lambda: fn(xr), args.warmup, args.repeats)

                def fb():
                    fn(xr).float().sum().backward()

                def zero():
                    lin.weight.grad = lin.bias.grad = xr.grad = None
                tb = cuda_time_ms(fb, args.warmup, args.repeats, setup=zero)
                r[f"{mode}: {label}"] = {"fwd_ms": tf["median_ms"], "fwdbwd_ms": tb["median_ms"],
                                         "fwd_iqr_ms": tf["iqr_ms"], "fwdbwd_iqr_ms": tb["iqr_ms"]}
                log(f"  [fp8proj] GEMM {direction:10s} {mode:8s} {label:20s} fwd {tf['median_ms']:7.3f}  "
                    f"fwd+bwd {tb['median_ms']:7.3f} ms")
        # Raw fp8 GEMMs on pre-quantised operands: the floor the overhead (amax, scale, cast, layout copies) sits on.
        with torch.no_grad():
            xq, xi = _to_fp8(xin, torch.float8_e4m3fn)
            wq, wi = _to_fp8(lin.weight, torch.float8_e4m3fn)
            gq, gi = _to_fp8(torch.randn(N, n, device="cuda"), torch.float8_e5m2)
            w_col = wq.t().contiguous().t()
            g_t = gq.t().contiguous()
            x_col = xq.t().contiguous().t()
            raw_f = cuda_time_ms(lambda: torch._scaled_mm(xq, wq.t(), scale_a=xi, scale_b=wi, out_dtype=torch.float32),
                                 args.warmup, args.repeats)
            raw_dx = cuda_time_ms(lambda: torch._scaled_mm(gq, w_col, scale_a=gi, scale_b=wi, out_dtype=torch.float32),
                                  args.warmup, args.repeats)
            raw_dw = cuda_time_ms(lambda: torch._scaled_mm(g_t, x_col, scale_a=gi, scale_b=xi, out_dtype=torch.float32),
                                  args.warmup, args.repeats)
        with torch.no_grad():
            xb, wb = xin.bfloat16(), lin.weight.bfloat16()
            gb = torch.randn(N, n, device="cuda", dtype=torch.bfloat16)
            rb_f = cuda_time_ms(lambda: xb @ wb.t(), args.warmup, args.repeats)
            rb_dx = cuda_time_ms(lambda: gb @ wb, args.warmup, args.repeats)
            rb_dw = cuda_time_ms(lambda: gb.t() @ xb, args.warmup, args.repeats)
        r["raw bf16 mm (pre-cast operands)"] = {
            "fwd_ms": rb_f["median_ms"], "fwdbwd_ms": rb_f["median_ms"] + rb_dx["median_ms"] + rb_dw["median_ms"]}
        log(f"  [fp8proj] GEMM {direction:10s} raw bf16 mm only: fwd {rb_f['median_ms']:.3f}  "
            f"3 GEMMs {r['raw bf16 mm (pre-cast operands)']['fwdbwd_ms']:.3f} ms")
        r["raw fp8 _scaled_mm (pre-quantised)"] = {
            "fwd_ms": raw_f["median_ms"], "fwdbwd_ms": raw_f["median_ms"] + raw_dx["median_ms"] + raw_dw["median_ms"]}
        log(f"  [fp8proj] GEMM {direction:10s} raw fp8 _scaled_mm only: fwd {raw_f['median_ms']:.3f}  "
            f"3 GEMMs {r['raw fp8 _scaled_mm (pre-quantised)']['fwdbwd_ms']:.3f} ms")
        gemm[direction] = r
        del lin, xin
        free_cuda()
    out["gemm_only"] = gemm
    return out


def pass_compile(args) -> dict:
    """How base_train's whole-module torch.compile sees the LUT block: graphs, breaks, recompiles."""
    import torch._dynamo
    from torch._dynamo.utils import counters
    out = {}
    tokens = min(args.tokens)
    for name in [v for v in ("lut-fp32", "dense-bf16") if v in args.variants]:
        torch._dynamo.reset()
        counters.clear()
        mod = build_variant(name, args.decompress_std)
        x = make_input(tokens)
        expl = torch._dynamo.explain(mod)(x)
        r = {
            "explain_graph_count": expl.graph_count,
            "explain_graph_break_count": expl.graph_break_count,
            "explain_op_count": expl.op_count,
            "break_reasons": [str(b.reason)[:300] for b in expl.break_reasons],
        }
        torch._dynamo.reset()
        counters.clear()
        cm = torch.compile(mod, dynamic=False)
        for _ in range(3):
            cm(x).backward(torch.randn_like(x))
        torch.cuda.synchronize()
        r["counters_after_3_fwdbwd"] = {k: dict(v) for k, v in counters.items() if v}
        out[name] = r
        log(f"  [compile] {name}: graphs={expl.graph_count} breaks={expl.graph_break_count} ops={expl.op_count}"
            f" stats={dict(counters.get('stats', {}))}")
        for b in r["break_reasons"]:
            log(f"      break: {b}")
        del mod, cm, x
        free_cuda()
    return out


def pass_memory(args, variants) -> dict:
    """Peak allocated per fwd+bwd, and the saved-for-backward inventory (eager outer, so hooks see the tensors)."""
    out = {}
    tokens = max(t for t in args.tokens)
    for name in variants:
        free_cuda()
        try:
            mod = build_variant(name, args.decompress_std)
            x = make_input(tokens, dtype=input_dtype(name))
            mod(x).backward(torch.randn_like(x))                       # warm (inner compiles)
            for p in mod.parameters():
                p.grad = None
            free_cuda()
            base = torch.cuda.memory_allocated()
            saved = []

            def pack(t):
                saved.append((tuple(t.shape), str(t.dtype).replace("torch.", ""), t.numel() * t.element_size()))
                return t
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                y = mod(x)
            after_fwd = torch.cuda.memory_allocated()
            y.backward(torch.randn_like(y))
            torch.cuda.synchronize()
            seen, uniq = set(), []
            for s in sorted(saved, key=lambda s: -s[2]):
                if s not in seen:
                    seen.add(s)
                    uniq.append({"shape": s[0], "dtype": s[1], "MiB": round(s[2] / 2 ** 20, 1)})
            out[name] = {
                "tokens": tokens,
                "activations_held_after_fwd_gib": (after_fwd - base) / 2 ** 30,
                "peak_alloc_fwdbwd_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                "peak_reserved_fwdbwd_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
                "saved_for_backward_total_gib": sum(s[2] for s in saved) / 2 ** 30,
                "saved_for_backward_top": uniq[:12],
            }
            log(f"  [memory] {name:11s} tokens={tokens}: held after fwd {out[name]['activations_held_after_fwd_gib']:.2f}"
                f" GiB, peak {out[name]['peak_alloc_fwdbwd_gib']:.2f} GiB")
            del mod, x, y
        except Exception as e:
            if not is_oom(e):
                raise
            out[name] = {"tokens": tokens, "oom": True}
        free_cuda()
    return out


BUCKETS = [   # first match wins; fused Triton kernels are matched before the gather patterns their names contain
    ("gemm", r"gemm|cutlass|xmma|nvjet|cublas|scaled_mm|_tensorop_"),
    ("fused_triton (addressing/score/casts)", r"^triton_"),
    ("embedding_bag fwd", r"EmbeddingBag_updateOutputKernel|embedding_bag_forward"),
    ("embedding_bag bwd: per_sample_weights grad", r"per_sample_weights_backward"),
    ("embedding_bag bwd: weight grad (sort/unique/scatter)", r"radix|RadixSort|DeviceRadix|onesweep|unique_by_key|"
                                                             r"compute_grad_weight|sum_and_scatter|krn_partial|"
                                                             r"segment_offsets|embedding_backward|cub::"),
    ("other gather/scatter/index", r"index_select|gather|scatter|indexing_backward|index_put|index_add"),
    ("copy/cast", r"copy|cast|direct_copy|to_copy|CatArrayBatched"),
    ("reduce", r"reduce_kernel|reduce"),
    ("elementwise/other", r".*"),
]


def bucket_of(kernel: str) -> str:
    for b, pat in BUCKETS:
        if re.search(pat, kernel):
            return b
    return "elementwise/other"


def _profile_session(args, label, step_fn):
    """Profile args.profile_iters calls of step_fn; return per-iteration CUDA kernel times (us) and the CPU op table.
    Kernel times are empty if the profiler gets no CUDA activity records (e.g. CUPTI blocked in a sandbox)."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                 with_stack=args.with_stack) as prof:
        for _ in range(args.profile_iters):
            step_fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.out_dir / f"trace_{label}.json"))
    (args.out_dir / f"profile_{label}.txt").write_text(
        prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40, max_name_column_width=90))
    kern = {}
    for ev in prof.events():
        # "## Call CompiledFxGraph ... ##" is a wrapper whose device time is the sum of its own kernels: skip it,
        # or every compiled region is counted twice.
        if ev.device_type == torch.autograd.DeviceType.CUDA and not ev.name.startswith("## "):
            us = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0.0)
            kern[ev.name] = kern.get(ev.name, 0.0) + us / args.profile_iters
    ops = {}
    for ev in prof.key_averages():
        if ev.key.startswith(("aten::", "triton_", "cudaLaunch", "cuLaunch")):
            ops[ev.key] = ev.count / args.profile_iters
    return kern, ops


def pass_profile(args, variants) -> dict:
    """Profile fwd and fwd+bwd separately; backward = per-kernel difference (compiled backwards donate their
    buffers, so a retained graph cannot be re-run for a backward-only session)."""
    out = {}
    tokens = args.profile_tokens or max(args.tokens)
    # The first profiler session in a process can come back with an empty GPU trace (CUPTI initialises lazily);
    # burn one on a trivial matmul so the measured sessions below are complete.
    a = torch.randn(1024, 1024, device="cuda")
    _profile_session(args, "warmup_discard", lambda: a @ a)
    for name in variants:
        free_cuda()
        mod = maybe_compile(build_variant(name, args.decompress_std), args.compile_mode)
        x = make_input(tokens)
        g = torch.randn_like(x)
        for _ in range(3):                                            # warm: compile + autotune
            mod(x).backward(g)

        def fwd():
            mod(x)

        def fwdbwd():
            for p in mod.parameters():
                p.grad = None
            mod(x).backward(g)
        k_f, ops_f = _profile_session(args, f"{name}_fwd", fwd)
        k_fb, ops_fb = _profile_session(args, f"{name}_fwdbwd", fwdbwd)
        r = {"tokens": tokens, "op_calls_per_iter_fwd": ops_f, "op_calls_per_iter_fwdbwd": ops_fb}
        if not k_fb:
            r["cuda_trace"] = ("EMPTY: the profiler received no CUDA kernel records on this host (CUPTI activity "
                               "tracing unavailable, e.g. inside a sandbox) - kernel tables omitted; see the "
                               "components pass for CUDA-event timings of each piece")
            log(f"  [profile] {name}: CUDA trace empty on this host - kernel breakdown unavailable (CPU op tables saved)")
        else:
            k_b = {k: k_fb.get(k, 0.0) - k_f.get(k, 0.0) for k in k_fb}
            for phase, kern in (("fwd", k_f), ("bwd", k_b)):
                buckets = {}
                for k, v in kern.items():
                    buckets[bucket_of(k)] = buckets.get(bucket_of(k), 0.0) + v
                top = sorted(kern.items(), key=lambda kv: -kv[1])[:20]
                r[phase] = {
                    "cuda_time_per_iter_ms": sum(kern.values()) / 1e3,
                    "buckets_ms": {k: round(v / 1e3, 3) for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])},
                    "top_kernels_ms": [{"kernel": k[:160], "ms": round(v / 1e3, 3), "bucket": bucket_of(k)}
                                       for k, v in top],
                }
                log(f"  [profile] {name} {phase}: {r[phase]['cuda_time_per_iter_ms']:.2f} ms GPU/iter; "
                    f"buckets {r[phase]['buckets_ms']}")
        out[name] = r
        del mod, x, g
        free_cuda()
    return out


# ------------------------------------------------------------------------------------------------ main

def write_summary(res: dict, path: Path) -> None:
    """Human-readable mirror of results.json, each line labelled hardware-dependent or -independent."""
    L = []
    e = res.get("env", {})
    L.append(f"# LUT-FFN d24 block benchmark ({e.get('gpu_name')}, cc {e.get('compute_capability')})\n")
    L.append(f"torch {e.get('torch')} / CUDA {e.get('cuda_runtime')} / Triton {e.get('triton')}; "
             f"fp32 matmul precision {e.get('fp32_matmul_precision')}; fp8: {e.get('fp8_probe')}\n")
    if "peak" in res:
        L.append("## Measured peaks [HW-dep]\n")
        L += [f"- {k}: {v:.1f}" for k, v in res["peak"].items()]
    if "block" in res:
        L.append("\n## Single block, median ms [HW-dep]\n")
        L.append("| variant | tokens | fwd ms (IQR) | fwd+bwd ms (IQR) | Mtok/s (f+b) | MACs/token | GEMM TFLOP/s (f+b) "
                 "| 24 layers / 2^20-tok step (s) | peak alloc / reserved GiB |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for v, per in res["block"].items():
            for t, r in per.items():
                if r.get("oom"):
                    L.append(f"| {v} | {t} | OOM | | | | | | |")
                    continue
                L.append(f"| {v} | {t} | {r['fwd_ms']:.2f} ({r['fwd_iqr_ms']:.2f}) | {r['fwdbwd_ms']:.2f} ({r['fwdbwd_iqr_ms']:.2f})"
                         f" | {r['fwdbwd_tokens_per_s'] / 1e6:.2f} | {r['macs_per_token'] / 1e6:.2f}M"
                         f" | {r['achieved_gemm_tflops_fwdbwd']:.1f} | {r['ffn24_per_step_s']:.2f}"
                         f" | {r['peak_alloc_gib']:.1f} / {r['peak_reserved_gib']:.1f} |")
    for key in ("components", "compile", "memory", "profile"):
        if key in res:
            label = "[HW-indep]" if key == "compile" else "[HW-dep]"
            L.append(f"\n## {key} {label}\n\n```json\n{json.dumps(res[key], indent=1, default=str)[:20000]}\n```")
    path.write_text("\n".join(L) + "\n")


def main(argv=None):
    global SEQ_LEN
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "profile_lut_ffn")
    ap.add_argument("--batch", type=int, nargs="+", default=[4, 16],
                    help="batch sizes (sequences); tokens per call = batch x --seq. d24 trains at DBS=16 (16x2048)")
    ap.add_argument("--seq", type=int, default=SEQ_LEN, help="sequence length (d24: 2048)")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=ALL_VARIANTS,
                    help="lut-fp8lib-* need a lutorch_ex with ProjectionMHL(fp8_projections=...) (see LUTORCH_EX_SRC)")
    ap.add_argument("--proj-dtypes", nargs="+", default=["fp32", "bf16", "fp8"], choices=["fp32", "bf16", "fp8"],
                    help="projection dtypes for the components pass")
    ap.add_argument("--passes", nargs="+", default=DEFAULT_PASSES, choices=ALL_PASSES)
    ap.add_argument("--compile-mode", default="model", choices=["model", "none"],
                    help="'model' = whole-module torch.compile as base_train does; 'none' = eager outer")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--profile-iters", type=int, default=3)
    ap.add_argument("--profile-tokens", type=int, default=None, help="tokens for the profile pass (default: largest)")
    ap.add_argument("--profile-variants", nargs="+", default=None, choices=ALL_VARIANTS,
                    help="variants for the profile pass (default: lut-fp32 + the dense-fp8/bf16 reference)")
    ap.add_argument("--with-stack", action="store_true", help="profiler with_stack (slower, larger traces)")
    ap.add_argument("--decompress-std", type=float, default=0.02,
                    help="init std for decompress / c_proj so backward does real work (training zero-inits them)")
    ap.add_argument("--fp32-matmul-precision", default="high", choices=["highest", "high", "medium"],
                    help="base_train sets 'high' (TF32 for fp32 GEMMs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-dirty-lib", action="store_true",
                    help="profile a lutorch_ex src/ with uncommitted or untracked files (recorded, not refused)")
    args = ap.parse_args(argv)
    SEQ_LEN = args.seq

    dirty_lib = lib_src_dirty_files()
    if dirty_lib and not args.allow_dirty_lib:
        log(f"Refusing to run: the profiled lutorch_ex src/ ({_LIB_SRC}) has uncommitted or untracked files, so the "
            "numbers would not be attributable to any commit:\n  " + "\n  ".join(dirty_lib)
            + "\nCommit them, or pass --allow-dirty-lib to profile anyway (the dirty files are recorded in env).")
        return 3
    args.tokens = sorted(b * args.seq for b in args.batch)

    if not torch.cuda.is_available():
        log("No usable CUDA device: this benchmark only produces GPU numbers. Stopping.")
        return 2
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision(args.fp32_matmul_precision)
    configure_dynamo_like_base_train()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    res = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    res["env"] = env_block()
    log(json.dumps(res["env"], indent=1))
    variants = list(args.variants)
    if not res["env"]["fp8_usable"]:
        skipped = [v for v in variants if "fp8" in v]
        variants = [v for v in variants if "fp8" not in v]
        args.proj_dtypes = [d for d in args.proj_dtypes if d != "fp8"]
        args.passes = [p for p in args.passes if p != "fp8proj"]
        res["skipped"] = {"variants": skipped, "passes": ["fp8proj"], "reason": res["env"]["fp8_probe"]}
        log(f"fp8 skipped ({res['env']['fp8_probe']}): {skipped} + fp8proj pass")
    for v in [v for v in variants if v in TABLE_VARIANTS]:
        import inspect
        from spiky.lutorch_ex import ConfidenceLUT
        if "table_dtype" not in inspect.signature(ConfidenceLUT.__init__).parameters:
            variants.remove(v)
            res.setdefault("skipped_lib_variants", {})[v] = "no ConfidenceLUT(table_dtype=...)"
            log(f"{v} skipped: the profiled lutorch_ex has no ConfidenceLUT(table_dtype=...)")
    for v in [v for v in variants if v in I32_VARIANTS or v in TABLE_VARIANTS]:
        ok, why = i32_supported()
        if not ok:
            variants.remove(v)
            res.setdefault("skipped_lib_variants", {})[v] = why
            log(f"{v} skipped: {why}")
    def _root(v):
        v = TABLE_VARIANTS[v][0] if v in TABLE_VARIANTS else v
        return I32_VARIANTS.get(v, v)
    for v in [v for v in variants if _root(v) in FP8LIB_VARIANTS]:
        base = _root(v)
        needs = ("fp8_projections",) + (("projection_dtype",) if "projection_dtype" in FP8LIB_VARIANTS[base] else ())
        lib_ok, lib_why = fp8lib_supported(needs)
        if not lib_ok:
            variants.remove(v)
            res.setdefault("skipped_lib_variants", {})[v] = lib_why
            log(f"{v} skipped: {lib_why}")

    def save():
        (args.out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str))
        write_summary(res, args.out_dir / "summary.md")

    if "peak" in args.passes:
        log("== peak");          res["peak"] = measure_peak(res["env"]["fp8_usable"], args.warmup, args.repeats); save()
    if "compile" in args.passes:
        log("== compile");       res["compile"] = pass_compile(args); save()
    if "block" in args.passes:
        log("== block");         res["block"] = pass_block(args, variants, res.get("peak")); save()
    if "components" in args.passes:
        log("== components");    res["components"] = pass_components(args, args.proj_dtypes); save()
    if "closeness" in args.passes:
        log("== closeness");     res["closeness"] = pass_closeness(args, variants); save()
    if "fp8proj" in args.passes:
        log("== fp8proj");       res["fp8proj"] = pass_fp8proj(args); save()
    if "memory" in args.passes:
        log("== memory");        res["memory"] = pass_memory(args, variants); save()
    if "profile" in args.passes:
        dense_ref = "dense-fp8" if "dense-fp8" in variants else "dense-bf16"
        pv = args.profile_variants or [v for v in ("lut-fp32", dense_ref) if v in variants]
        log("== profile");       res["profile"] = pass_profile(args, [v for v in pv if v in variants]); save()
    # Attention is not part of this benchmark: record that nothing pulled in nanochat's FA3 loader.
    res["env"]["fa3_loader_imported"] = "nanochat.flash_attention" in sys.modules
    if res["env"]["fa3_loader_imported"]:
        log("WARNING: nanochat.flash_attention was imported - something pulled in attention/FA3; not intended here")
    save()
    log(f"done -> {args.out_dir / 'results.json'}, {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
