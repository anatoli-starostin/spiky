"""Bit-exactness gate: the PyTorch Act1+Act2 producer against the fused kernel's own cells.

The fused kernel exposes its internally computed integers through the optional CELLS_OUT
argument (`pow2_int8_read.cu:143-148`, reached via `read_fused(..., cells_out=...)`), so
the comparison is byte for byte on the exact tensor the gather would consume -- no
end-to-end proxy needed.

Three producers are compared against it:
  kernel      the fused kernel's own CELLS_OUT      (the reference)
  ours        cells_producer.cells, eager
  ours+compile cells_producer, torch.compile
  repo        pow2_int8.table_integers + pack_cells, the path already in the tree

The repo's own header says its torch path is NOT bit-identical to the CUDA definition
(pow2_scalars.cuh:7-9), so that row is expected to disagree somewhere; the point is to
measure by how much rather than assume.

The cells tensor does not depend on D at all -- D lives only in the tables -- so this
gate runs at a D the kernel already supports and needs none of the wide-row edits.
"""
import os
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402
from spiky.lutorch import pow2_int8                                # noqa: E402
import cells_producer                                              # noqa: E402

DIN, D, T, NAP, H = 256, 64, 256, 8, 1
N = 4096


def compare(name, got, ref):
    eq = (got == ref)
    per_byte = eq.reshape(-1, 3)
    allsame = per_byte.all(dim=1)
    n = allsame.numel()
    bad = int((~allsame).sum())
    cols = [int((~per_byte[:, i]).sum()) for i in range(3)]
    print(f'  {name:<16} tables differing {bad:>9,} / {n:,}  ({100*bad/n:.6f}%)   '
          f'c1 {cols[0]:,}  c2 {cols[1]:,}  sh {cols[2]:,}')
    return bad, cols


@torch.no_grad()
def main():
    ok = pow2_int8.ensure_registered()
    print(f'pow2_int8 extension: {ok} / {pow2_int8.available()[1]}')
    torch.manual_seed(7)
    lut = LightMultiHeadLUT(input_dim=DIN, output_dim=D, n_tables=T, n_anchor_pairs=NAP,
                            n_heads=H, multi_head_input=True, read_top_n=2,
                            confidence_form='learned_margin', learned_margin_freeze_g=True,
                            forward_mode='scored', quant_mode='p2_int8',
                            initial_weights_noise=0.001, random_seed=1234,
                            device=torch.device('cuda')).cuda().eval()
    cfg = lut._quant
    lo, hi, Q = cfg['lo'], cfg['hi'], cfg['Q']
    raw = lut._quant_scalars()                       # tensors, for the repo's own op
    tau, g, beta, gamma = (float(v) for v in raw)
    print(f'config: N={N} H={H} T={T} nap={NAP} din={DIN} D={D}; '
          f'lo={lo} hi={hi} Q={Q}')
    print(f'scalars: tau={tau:.6g} g={g:.6g} beta={beta:.6g} gamma={gamma:.6g}')

    _, packed = lut.quantised_tables()
    tab = pow2_int8.stride_tables(packed, D)
    a32 = lut.anchor_a.to(torch.int32).contiguous()
    b32 = lut.anchor_b.to(torch.int32).contiguous()
    scal = tuple(torch.as_tensor([v], device='cuda', dtype=torch.float32)
                 for v in (tau, g, beta, gamma))

    z = torch.randn(N, H, DIN, device='cuda')
    # a second draw scaled down, to push tables into the skip / drop corners
    z = torch.cat([z, torch.randn(N, H, DIN, device='cuda') * 0.02], dim=0)
    NN = z.shape[0]

    # block_n=16 is required here, and the requirement is itself informative: the fused
    # path's shared memory is cells (T*BLOCK_N*3) + Z (BLOCK_N*din*4) + anchors
    # (2*T*nap), which at BLOCK_N=64 with T=256 and din=256 is 118,784 B -- over the
    # 5090's 101,376 B per-block opt-in limit, and the launch fails with
    # "invalid argument". At BLOCK_N=16 it is 32,768 B. This also exercises the
    # newly added block_n=16 dispatch arm.
    BN = 16
    ref = torch.empty(NN, H, T, 3, device='cuda', dtype=torch.uint8)
    pow2_int8.read_fused(z, a32, b32, tab, scal, NAP, D, lo, hi, Q,
                         block_n=BN, cells_out=ref)
    torch.cuda.synchronize()

    sh1 = ref[..., 2] & 15
    sh2 = ref[..., 2] >> 4
    print(f'\nreference cells from the kernel: {tuple(ref.shape)} {ref.dtype}')
    print(f'  skipped tables (sh1 == DISCARD): {100*float((sh1 == 15).float().mean()):.3f}%')
    print(f'  dropped second cells (sh2 == DISCARD, sh1 valid): '
          f'{100*float(((sh2 == 15) & (sh1 != 15)).float().mean()):.3f}%')
    print(f'  distinct shift codes in sh1: {sorted(set(sh1.unique().tolist()))}')

    print('\nbit-exactness against the kernel\'s own cells:')
    ours = cells_producer.cells(z, lut.anchor_a, lut.anchor_b, tau, g, beta, gamma, lo, hi, Q)
    compare('ours (eager)', ours, ref)

    try:
        comp = cells_producer.cells_compiled()
        ourc = comp(z, lut.anchor_a, lut.anchor_b, tau, g, beta, gamma, lo, hi, Q)
        compare('ours (compiled)', ourc, ref)
    except Exception as e:
        print(f'  ours (compiled)  FAILED: {type(e).__name__}: {str(e).splitlines()[0][:120]}')

    d = cells_producer.margins(z, lut.anchor_a, lut.anchor_b)
    index = ((d > 0).to(torch.int64) * lut.powers.view(1, 1, 1, -1)).sum(-1)
    idx, q, k, skip, drop = pow2_int8.table_integers(
        d, index, lut.powers, *raw, cfg)
    repo = pow2_int8.pack_cells(idx, q, k, skip, drop)
    compare('repo (via op)', repo, ref)
    print('    ^ note: table_integers dispatches to torch.ops.spiky_lutorch.p2_scalars when the')
    print('      extension is loaded, so that row re-runs the SAME CUDA function and is exact')
    print('      trivially. The row below forces the pure-torch fallback, which is the one the')
    print('      header calls "not bit-identical" (pow2_scalars.cuh:7-9).')
    pow2_int8.set_enabled(False)
    try:
        idx2, q2, k2, s2, d2 = pow2_int8.table_integers(
            d, index, lut.powers, *raw, cfg)
        compare('repo (torch)', pow2_int8.pack_cells(idx2, q2, k2, s2, d2), ref)
    finally:
        pow2_int8.set_enabled(True)

    # traffic accounting asked for in the task, at the study config
    Tb, Db = 256, 1024
    cells_b, table_b = Tb * 3, 2 * Tb * Db
    print(f'\ncells-tensor traffic at the study config (T={Tb}, D={Db}, two cells):')
    print(f'  cells read per token  {cells_b:,} B')
    print(f'  table rows per token  {table_b:,} B')
    print(f'  cells / table rows    {100*cells_b/table_b:.4f}%   '
          f'cells / total          {100*cells_b/(cells_b+table_b):.4f}%')


if __name__ == '__main__':
    main()
