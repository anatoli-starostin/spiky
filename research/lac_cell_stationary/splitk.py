"""Build and wrap the split-K arm."""
import os
import re
import shutil

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.expanduser('~/.cache/lac_splitk_build')
_MOD = None
SS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BLOCK_N, T_DEFAULT = 16, 256
SMS = 170


def mod(verbose=False, fresh=False):
    global _MOD
    if fresh and os.path.isdir(BUILD):
        shutil.rmtree(BUILD)
        _MOD = None
    if _MOD is None:
        os.makedirs(BUILD, exist_ok=True)
        os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
        _MOD = load(name='lac_splitk', sources=[os.path.join(HERE, 'splitk_kernel.cu')],
                    extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-Xptxas', '-v',
                                       '-gencode=arch=compute_120,code=sm_120'],
                    extra_cflags=['-O3', '-std=c++20'],
                    build_directory=BUILD, verbose=verbose)
    return _MOD


def run(W, cells, D, S):
    """int32 [N, H, D]; the output is allocated zeroed inside the extension (atomics need it)."""
    return mod().splitk(W, cells, D, S)


def grid(B, S, block_n=BLOCK_N, H=1):
    return ((B + block_n - 1) // block_n) * H * S


def smem(S, T=T_DEFAULT, block_n=BLOCK_N):
    return (T // S) * block_n * 3


def saturation_B(S, blocks_per_sm, block_n=BLOCK_N):
    """Smallest B whose grid fills the GPU: ceil(B/block_n)*S >= SMS*blocks_per_sm."""
    need = SMS * blocks_per_sm
    tiles = (need + S - 1) // S
    return tiles * block_n


def ptxas(log):
    rows, cur = {}, None
    for line in open(log, errors='replace'):
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            nm = m.group(1)
            nums = re.findall(r'Li(\d+)E', nm)
            cur = {'regs': 0, 'spill_st': 0, 'spill_ld': 0, 'smem': 0, 'stack': 0}
            rows[f'splitk<block_n={nums[0]},UPR={nums[1]}>' if len(nums) >= 2 else nm] = cur
            continue
        if cur is None:
            continue
        m = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, '
                      r'(\d+) bytes spill loads', line)
        if m:
            cur['stack'], cur['spill_st'], cur['spill_ld'] = (int(x) for x in m.groups())
        m = re.search(r'Used (\d+) registers', line)
        if m:
            cur['regs'] = int(m.group(1))
        m = re.search(r'(\d+) bytes smem', line)
        if m:
            cur['smem'] = int(m.group(1))
    return rows
