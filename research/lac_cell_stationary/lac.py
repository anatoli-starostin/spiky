"""Build and wrap the LAC kernels; reference implementation; timing helper.

The extension is built once per process into ~/.cache (writable inside the cage) and
compiled with `-Xptxas -v` so the register/spill/SMEM report lands in the build log,
which `report_ptxas()` parses. `LAC_VEC_ATOMIC` is defined because the 16-byte
atomicAdd(float4*) form compiles on sm_120 (probed separately); flip it off to compare
against four scalar atomics.
"""
import os
import re
import statistics
import subprocess

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.expanduser('~/.cache/lac_cell_stationary_build')
ARCH = '12.0'                     # RTX 5090, sm_120
_MOD = None


def mod(vec_atomic: bool = True, verbose: bool = False):
    global _MOD
    if _MOD is None:
        os.makedirs(BUILD, exist_ok=True)
        os.environ['TORCH_CUDA_ARCH_LIST'] = ARCH
        # -std=c++20 is load-bearing: GCC 15 rejects torch 2.9's ATen/core/List_inl.h
        # under C++17 ("need 'typename' before ... dependent scope"). The repo's own
        # gather_cuda.py carries the same flag for the same reason.
        flags = ['-O3', '-std=c++20', '--use_fast_math', '-lineinfo', '-Xptxas', '-v',
                 '-gencode=arch=compute_120,code=sm_120']
        if vec_atomic:
            flags.append('-DLAC_VEC_ATOMIC')
        _MOD = load(name='lac_cs', sources=[os.path.join(HERE, 'lac_kernels.cu')],
                    extra_cuda_cflags=flags, extra_cflags=['-O3', '-std=c++20'],
                    build_directory=BUILD, verbose=verbose)
    return _MOD


def report_ptxas():
    """Parse the nvcc -Xptxas -v output saved in the ninja build log.

    Returns {kernel_signature: {'regs': int, 'spill_stores': int, 'spill_loads': int,
                                'smem': int}}.
    """
    log = os.path.join(BUILD, 'ptxas.log')
    if not os.path.exists(log):
        return {}
    txt = open(log).read()
    out = {}
    cur = None
    for line in txt.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            cur = m.group(1)
            out[cur] = {'regs': 0, 'spill_stores': 0, 'spill_loads': 0, 'smem': 0}
            continue
        if cur is None:
            continue
        m = re.search(r'Used (\d+) registers', line)
        if m:
            out[cur]['regs'] = int(m.group(1))
        m = re.search(r'(\d+) bytes spill stores', line)
        if m:
            out[cur]['spill_stores'] = int(m.group(1))
        m = re.search(r'(\d+) bytes spill loads', line)
        if m:
            out[cur]['spill_loads'] = int(m.group(1))
        m = re.search(r'(\d+) bytes smem', line)
        if m:
            out[cur]['smem'] = int(m.group(1))
    return out


# ------------------------------------------------------------------ reference

@torch.no_grad()
def reference(T, J, C, use_coef=True, chunk=256):
    """y[b,k] = sum_t c[b,t] * T[t, j[b,t], k], in fp32, chunked over the batch.

    Sums over tables in table order, which is the same order the gather kernel uses,
    so the two agree to fp32 reassociation only (they do not reassociate at all at
    n=1: both walk t ascending).
    """
    NT, R, N = T.shape
    B = J.shape[0]
    Tf = T.to(torch.float32)
    out = torch.empty(B, N, device=T.device, dtype=torch.float32)
    tix = torch.arange(NT, device=T.device)
    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)
        idx = J[b0:b1].long()                                  # [b, NT]
        rows = Tf[tix.unsqueeze(0), idx]                       # [b, NT, N]
        if use_coef:
            rows = rows * C[b0:b1].unsqueeze(-1)
        out[b0:b1] = rows.sum(1)
    return out


# ------------------------------------------------------------------ launchers

def run_gather(T, J, C, tsplit=1, use_coef=True, tok_per_blk=1, y=None):
    N = T.shape[2]
    B = J.shape[0]
    if y is None:
        y = torch.zeros(B, N, device=T.device, dtype=torch.float32)
    elif tsplit > 1:
        y.zero_()
    mod().gather(T, J, C, y, tsplit, int(use_coef), tok_per_blk)
    return y


def run_v0(T, J, C, K=4, M=48, use_coef=True, y=None):
    N = T.shape[2]
    B = J.shape[0]
    if y is None:
        y = torch.zeros(B, N, device=T.device, dtype=torch.float32)
    else:
        y.zero_()
    mod().cs_v0(T, J, C, y, K, M, int(use_coef))
    return y


def transpose_tables(T):
    """[NT, R, N] -> [R, NT, N], done once offline. The cell-stationary kernels take
    this with trans=1 and derive NT/R from the permuted shape."""
    return T.permute(1, 0, 2).contiguous()


def run_v2(T, J, C, K=32, M=32, TB=4, trans=0, use_coef=True, y=None):
    N = T.shape[2]
    B = J.shape[0]
    if y is None:
        y = torch.zeros(B, N, device=T.device, dtype=torch.float32)
    else:
        y.zero_()
    mod().cs_v2(T, J, C, y, K, M, TB, trans, int(use_coef))
    return y


def run_v3(T, J, C, K=32, M=32, G=1, trans=0, use_coef=True, y=None):
    N = T.shape[2]
    B = J.shape[0]
    if y is None:
        y = torch.zeros(B, N, device=T.device, dtype=torch.float32)
    else:
        y.zero_()
    mod().cs_v3(T, J, C, y, K, M, G, trans, int(use_coef))
    return y


def run_v1(T, J, C, K=32, M=1, TB=4, CLU=1, use_coef=True, y=None):
    N = T.shape[2]
    B = J.shape[0]
    if y is None:
        y = torch.zeros(B, N, device=T.device, dtype=torch.float32)
    else:
        y.zero_()
    mod().cs_v1(T, J, C, y, K, M, TB, CLU, int(use_coef))
    return y


# ------------------------------------------------------------------ timing

def burn_in(fn, seconds=2.0):
    """The 5090 idles at 1627 MHz and boosts to 3210. Timing before the ramp measures
    the ramp -- this is the same rule as the FFN benchmark harness's burn_in()."""
    import time
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        fn()
        n += 1
    torch.cuda.synchronize()
    return n


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return {'median_ms': statistics.median(ts), 'min_ms': ts[0], 'max_ms': ts[-1],
            'iters': iters}


def clocks():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,power.draw',
                              '--format=csv,noheader'], capture_output=True, text=True).stdout
        return out.strip()
    except Exception:
        return 'n/a'
