"""Build and wrap the direct-scatter kernels; budgets and traffic accounting."""
import os
import shutil
import sys

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.expanduser('~/.cache/lac_scatter_build')
_MOD = None

SMEM_LIMIT = 101376
SMEM_2BLOCK = SMEM_LIMIT // 2        # <= this allows two blocks per SM on shared memory
LS = [1, 2, 4, 8, 16]
BTS = [4, 8, 12, 16, 24]
D = 1024


def mod(verbose=False, fresh=False):
    global _MOD
    if fresh and os.path.isdir(BUILD):
        shutil.rmtree(BUILD)
        _MOD = None
    if _MOD is None:
        os.makedirs(BUILD, exist_ok=True)
        os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
        _MOD = load(name='lac_scatter', sources=[os.path.join(HERE, 'scatter_kernel.cu')],
                    extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-Xptxas', '-v',
                                       '-gencode=arch=compute_120,code=sm_120'],
                    extra_cflags=['-O3', '-std=c++20'],
                    build_directory=BUILD, verbose=verbose)
    return _MOD


def run_a(W, cells, D_, L, threads=256):
    return mod().scatter_global(W, cells, D_, L, threads)


def run_b(W, cells, D_, L, Bt, pad=0, threads=256):
    return mod().scatter_shared(W, cells, D_, L, Bt, pad, threads)


def budget_b(Bt, pad=0):
    smem = Bt * (D + pad) * 4
    return {'Bt': Bt, 'pad': pad, 'smem': smem, 'fits': smem <= SMEM_LIMIT,
            'blocks_per_sm_by_smem': (SMEM_LIMIT // smem) if smem <= SMEM_LIMIT else 0}


def traffic(N, T, K, fetched_per_table, L, arm, Bt=None):
    """Derived from the launch geometry and the MEASURED discard rate, not from counters
    (ncu is unavailable: RmProfilingAdminOnly=1)."""
    lane_updates = N * T * fetched_per_table * D
    # table bytes: one int8 row segment per fetched cell, at 32 B sector granularity when L < 32
    amp = 32 / min(L, 32)
    table = N * T * fetched_per_table * D * amp
    cells = N * T * 3
    if arm == 'A':
        atomic = lane_updates * 8          # int32 RMW: read + write
        out_w = 0                          # atomics ARE the write
        zero = N * D * 4                   # the mandatory pre-zero
        shared_ops = 0
    else:
        atomic = 0
        out_w = N * D * 4                  # one plain store per element
        zero = 0                           # not needed: the block is the unique writer
        shared_ops = lane_updates          # shared-atomic op count
    total = table + cells + atomic + out_w + zero
    return {'lane_updates': lane_updates, 'table': table, 'table_amp': amp, 'cells': cells,
            'atomic_bytes': atomic, 'out_bytes': out_w, 'zero_bytes': zero,
            'shared_atomic_ops': shared_ops, 'total_bytes': total,
            'floor_ms_at_peak': total / 1792e9 * 1e3}
