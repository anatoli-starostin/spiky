"""Norm statistics at every tap point inside a CompressionMHL block.

THE TAP POINTS, enumerated from autoencoder.py's block() and compression_mhl.py's forward(), for the
config in use (pre-norm, no residual, inner_in_dim=128, inner_out_dim=-1, a_i = 1):

    h                         T0  block input, BEFORE the pre-norm        <- 'in'
    lns[i](h)                 T1  what the compress projection reads
    compress(T1) = z          T2  what the LUT ADDRESSES; margins and scores are taken here
    lut_light(z) = y          T3  the LUT's raw output
    decompress(y)             T4  Identity at inner_out_dim=-1, so T4 == T3
    a_i * T4                  T5  block output, a_i = 1 without a residual, so T5 == T4 == T3  <- 'out'

So at THIS config T3, T4 and T5 are the same tensor and "the readout" is unambiguous: it is T3/T5, what
the next block or the decoder consumes. With a residual or inner_out_dim>0 they would separate, which is
why all of them are computed rather than assumed equal.

THERE IS NO TOKEN DIMENSION in this model. The stream is [batch, width]; it is an autoencoder over flat
784-dim images, not a sequence. "Per-token norm" is therefore the per-SAMPLE L2 norm, and the spread is
across the batch. Stated explicitly because the request asked for per-token statistics.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))

TAPS = ('t0_in', 't1_postnorm', 't2_postcompress', 't3_lut_out', 't5_block_out')


@torch.no_grad()
def readout_norms(model, x):
    """Per-block per-sample L2 norm statistics at each tap. Returns {block: {tap: {stat: value}}}."""
    from autoencoder import inner_lut
    out = []
    h = model.enc(x)
    for i in range(model.n_blocks):
        blk = model.blocks[i]
        taps = {'t0_in': h}
        z = h
        if model.block_norm == 'layernorm' and model.norm_position == 'pre':
            z = model.lns[i](z)
        taps['t1_postnorm'] = z
        if hasattr(blk, 'compress'):
            z = blk.compress(z)
        taps['t2_postcompress'] = z
        # the LUT applied to exactly what it addresses. For a CompressionMHL block inner_lut() is the
        # wrapped LightMHL and z is post-compress; for a bare LightMHL block inner_lut() IS the block
        # and z is post-norm, which is what that block reads. Either way this is the LUT's own output.
        taps['t3_lut_out'] = inner_lut(blk)(z)
        h = model.block(i, h)
        taps['t5_block_out'] = h
        rec = {}
        for k, t in taps.items():
            n = t.norm(dim=-1)
            rec[k] = dict(mean=float(n.mean()), median=float(n.median()), max=float(n.max()),
                          std=float(n.std()), min=float(n.min()))
        rec['gain_out_over_in'] = rec['t5_block_out']['mean'] / max(rec['t0_in']['mean'], 1e-12)
        out.append(rec)
    return out


def flatten(recs):
    """One flat dict per probe row, so the stats ride in run.json beside everything else."""
    flat = {}
    for b, rec in enumerate(recs):
        for tap in TAPS:
            for stat, v in rec[tap].items():
                flat[f'norm/{tap}_{stat}_b{b}'] = v
        flat[f'norm/gain_b{b}'] = rec['gain_out_over_in']
    return flat
