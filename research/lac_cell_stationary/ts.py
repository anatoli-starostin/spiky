"""Build and wrap the table-stationary kernel, and report its ptxas numbers."""
import os
import re

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.expanduser('~/.cache/lac_ts_build')
LOG = os.path.join(BUILD, 'ptxas.log')
_MOD = None

# Every (G, L, Bt) the .cu instantiates, with its budget. Kept here so the report and the
# dispatch cannot drift apart.
CONFIGS = [(8, 96, 256), (32, 24, 1024), (16, 48, 512),          # the three named in the task
           (8, 64, 256), (8, 32, 256), (16, 32, 512),
           (32, 16, 512), (32, 16, 256), (8, 64, 512), (16, 32, 256),
           (8, 16, 256), (16, 16, 256), (16, 16, 512), (32, 8, 512), (8, 16, 1024),
           (4, 32, 256), (4, 32, 512), (16, 8, 512), (32, 4, 512),
           (2, 64, 256), (4, 64, 256), (2, 128, 192), (1, 128, 256)]
SMEM_LIMIT = 101376


def budget(G, L, Bt):
    acc, ce = Bt * L * 4, Bt * G * 3
    return {'G': G, 'L': L, 'Bt': Bt, 'acc_bytes': acc, 'cells_bytes': ce,
            'smem_bytes': acc + ce, 'fits_smem': acc + ce <= SMEM_LIMIT,
            'table_regs': G * L // 4}


def mod(verbose=False):
    global _MOD
    if _MOD is None:
        os.makedirs(BUILD, exist_ok=True)
        os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
        _MOD = load(name='lac_ts',
                    sources=[os.path.join(HERE, 'ts_kernel.cu')],
                    extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-Xptxas', '-v',
                                       '-gencode=arch=compute_120,code=sm_120'],
                    extra_cflags=['-O3', '-std=c++20'],
                    build_directory=BUILD, verbose=verbose)
    return _MOD


def run(W, cells, D, G, L, Bt, tlay=0, clay=0, N=None, T=None):
    """int32 [N, 1, D]. The output is allocated zeroed inside the extension.

    tlay/clay select the layout variants; W and cells must already be in that layout
    (see blocked_tables / transposed_cells, both host-side one-offs)."""
    if N is None:
        N = cells.shape[1] if clay else cells.shape[0]
    if T is None:
        T = cells.shape[0] if clay else cells.shape[2]
    return mod().ts_read(W, cells, D, G, L, Bt, tlay, clay, N, T)


def blocked_tables(W, T, K, D, G, L):
    """[T*K, D] -> flat [T/G][D/L][K][G][L]. HOST-SIDE ONE-OFF, never inside a timed region.

    Each block's (K cells x G tables x L lanes) tile becomes fully contiguous: 32 KiB at
    G*L = 128. Thread r then reads G*L contiguous bytes for its own cell.
    """
    return (W[:, :D].contiguous().view(T // G, G, K, D // L, L)
            .permute(0, 3, 2, 1, 4).contiguous().view(-1))


def transposed_cells(cells):
    """[N, 1, T, 3] -> [T, N, 3]. HOST-SIDE ONE-OFF, never inside a timed region."""
    return cells.squeeze(1).permute(1, 0, 2).contiguous()


def ptxas(log_path):
    """{pretty name: {regs, spill_st, spill_ld, smem, stack}} from a -Xptxas -v log."""
    rows, cur = {}, None
    for line in open(log_path, errors='replace'):
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            nums = [int(x) for x in re.findall(r'Li(\d+)E', m.group(1))]
            name = (f'ts<G={nums[0]},L={nums[1]},Bt={nums[2]}>' if len(nums) >= 3
                    else m.group(1))
            cur = {'regs': 0, 'spill_st': 0, 'spill_ld': 0, 'smem': 0, 'stack': 0}
            rows[name] = cur
            continue
        if cur is None:
            continue
        m = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, '
                      r'(\d+) bytes spill loads', line)
        if m:
            cur['stack'], cur['spill_st'], cur['spill_ld'] = (int(g) for g in m.groups())
        m = re.search(r'Used (\d+) registers', line)
        if m:
            cur['regs'] = int(m.group(1))
        m = re.search(r'(\d+) bytes smem', line)
        if m:
            cur['smem'] = int(m.group(1))
    return rows
