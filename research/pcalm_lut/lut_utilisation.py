"""Codebook utilisation and table-geometry diagnostics for a LightMHL stack.

DEFINITIONS, stated because each has a choice in it:

usage distribution   For one table, the empirical distribution over its 2^nap rows of WHICH ROW the
                     batch addresses. p_r = (times row r was the addressed row) / batch size. This is
                     the HARD address (the argmax cell), not the read_top_n=2 blend weights -- the
                     blend's second cell is a neighbour of the first, so counting it would smear the
                     utilisation measure across adjacent rows rather than report which cells the data
                     actually selects.
participation ratio  PR = 1 / sum_r p_r^2, the effective number of rows in use. Ranges 1 (one row for
                     every sample) to 2^nap (uniform). Reported both raw and as a fraction of 2^nap.
entropy              H = -sum_r p_r log p_r, in nats, against a log(2^nap) maximum.
dead fraction        rows never addressed by the batch. NOTE the batch bound: with batch 512 and 256
                     rows per table, at most 512 distinct rows can be touched, so "dead" here means
                     "not used by these 512 samples", not "never used ever". Stated because a naive
                     read of a high dead fraction would overstate the case.
row norm             mean L2 norm of a table row, over all rows.
row-mean norm        the norm of the MEAN row. If the rows share a large common component, this is
                     close to the row norm and the sqrt(k) independent-sum argument does not apply;
                     if the rows are spread around zero, this is small relative to the row norm.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))


@torch.no_grad()
def lut_utilisation(model, x):
    """Per-block utilisation and table geometry. Returns a list of dicts, one per block."""
    from autoencoder import inner_lut
    out = []
    h = model.enc(x)
    for i in range(model.n_blocks):
        blk = model.blocks[i]
        lut = inner_lut(blk)
        z = h
        if model.block_norm == 'layernorm' and model.norm_position == 'pre':
            z = model.lns[i](z)
        if hasattr(blk, 'compress'):
            z = blk.compress(z)
        d = z[:, lut.anchor_a] - z[:, lut.anchor_b]                     # [B, T, NAP]
        addr = ((d > 0).to(torch.int64) * lut.powers.view(1, 1, -1)).sum(-1)   # [B, T]
        n_rows = lut.table_size
        b, t = addr.shape
        # counts[table, row]
        counts = torch.zeros(t, n_rows, device=addr.device)
        counts.scatter_add_(1, addr.t(), torch.ones(t, b, device=addr.device))
        p = counts / b
        pr = 1.0 / p.pow(2).sum(-1).clamp_min(1e-12)                    # [T]
        ent = -(p.clamp_min(1e-12).log() * p).sum(-1)                   # [T]
        dead = (counts == 0).float().mean(-1)                           # [T]
        tab = lut.tables                                                # [T, rows, out]
        row_norm = tab.norm(dim=-1)                                     # [T, rows]
        mean_row_norm = tab.mean(dim=1).norm(dim=-1)                    # [T]
        out.append(dict(
            pr_mean=float(pr.mean()), pr_frac=float(pr.mean()) / n_rows,
            entropy_mean=float(ent.mean()), entropy_max=float(torch.log(torch.tensor(float(n_rows)))),
            dead_frac=float(dead.mean()),
            top1_share=float(p.max(-1).values.mean()),                  # the single most-used row
            row_norm_mean=float(row_norm.mean()), row_norm_max=float(row_norm.max()),
            row_mean_norm=float(mean_row_norm.mean()),
            common_ratio=float(mean_row_norm.mean()) / max(float(row_norm.mean()), 1e-12),
            n_rows=n_rows, batch=b))
        h = model.block(i, h)
    return out


def grad_norms(model):
    """Gradient norms at the two taps the prediction is about: the encoder, and block-0's compress."""
    g = {}
    if model.enc.weight.grad is not None:
        g['grad/enc_weight'] = float(model.enc.weight.grad.norm())
    if model.dec.weight.grad is not None:
        g['grad/dec_weight'] = float(model.dec.weight.grad.norm())
    b0 = model.blocks[0] if len(model.blocks) else None
    if b0 is not None and hasattr(b0, 'compress') and hasattr(b0.compress, 'weight') \
            and b0.compress.weight.grad is not None:
        g['grad/b0_compress'] = float(b0.compress.weight.grad.norm())
    from autoencoder import inner_lut
    for i, blk in enumerate(model.blocks):
        t = inner_lut(blk).tables
        if t.grad is not None:
            g[f'grad/tables_b{i}'] = float(t.grad.norm())
    return g


def flatten(recs):
    flat = {}
    for b, r in enumerate(recs):
        for k, v in r.items():
            flat[f'util/{k}_b{b}'] = v
    return flat
