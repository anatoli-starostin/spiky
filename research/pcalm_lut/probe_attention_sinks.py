"""Are there attention sinks in the ViT autoencoder? Measured from saved checkpoints, no retraining.

A sink is a key position that receives a large share of the attention mass from ESSENTIALLY EVERY query.
High mass alone is not enough -- a key that some queries attend to strongly and others ignore is an
informative token, not a sink -- so every high-mass position here is also scored for query-invariance:
what fraction of queries rank it top-1, and how much its weight varies across queries.

This model has no CLS and no BOS, so any sink has to be an ordinary spatial patch. Token index t maps to
patch (row, col) = divmod(t, 14) on the 14x14 grid of 2x2 patches, matching patchify()'s row-major
unfold, and the report names the border patches because in Fashion-MNIST those are near-constant black.

THE INSTRUMENT. nn.MultiheadAttention averages over heads by default, which would hide a per-head sink
completely, so every call here passes average_attn_weights=False and the returned shape is asserted to
carry the head dimension. Attention rows are asserted to sum to 1. The forward pass is re-implemented
rather than hooked, so that what is measured is exactly the tensor the block consumes; the reimplementation
is checked against the model's own forward() to 1e-5 before any number below is believed.

Run:  python probe_attention_sinks.py
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from data import load  # noqa: E402
from vit_autoencoder import ViTAutoencoder, patchify, unpatchify  # noqa: E402

R = os.path.join(HERE, 'runs_vit')

GRID = 14
N_IMG, CHUNK = 512, 128
RUNS = [('d128+aug', 'vit-p2-k8-e4d4-d128h4-lat128-nowarm-full-aug-s60000', None),
        ('d64 plain', 'vit-p2-k8-e4d4-nowarm-full-s60000', ['model.pt'])]


def load_model(run, weights, dev):
    """Rebuild from the run's own recorded cfg and load one checkpoint, in eval mode."""
    d = json.load(open(os.path.join(R, run, 'run.json')))
    c, s = d['cfg'], d['summary']
    m = ViTAutoencoder(s['n_tokens'], s['patch_dim'], c['d_model'], c['n_heads'], c['enc_layers'],
                       c['dec_layers'], c['latent'], c['latent_tokens'], c['ffn_mult'], c['ffn'],
                       c['tables'], dev, c['seed'], c.get('dropout', 0.0))
    m.load_state_dict(torch.load(os.path.join(R, run, weights), map_location=dev))
    m.eval()
    return m, d


def value_norms(attn, src):
    """Mean L2 norm per key position of the VALUE vectors this attention module computes from `src`."""
    d = attn.embed_dim
    w, b = attn.in_proj_weight[2 * d:], attn.in_proj_bias[2 * d:]
    return F.linear(src, w, b).norm(dim=-1)                      # [B, S]


class Stats:
    """Streaming accumulators, so a few hundred images never materialise as one big weight tensor."""

    def __init__(self, n_heads, n_keys, dev):
        self.mass = torch.zeros(n_heads, n_keys, device=dev)     # sum over queries+images of attn
        self.sq = torch.zeros(n_heads, n_keys, device=dev)       # sum of attn^2, for the variance
        self.top1 = torch.zeros(n_heads, n_keys, device=dev)     # times this key was a query's argmax
        self.ent = torch.zeros(n_heads, device=dev)
        self.maxw = torch.zeros(n_heads, device=dev)
        self.n = 0

    def add(self, w):                                            # w [B, H, Q, K]
        b, h, q, k = w.shape
        f = w.transpose(0, 1).reshape(h, b * q, k)               # [H, B*Q, K]
        self.mass += f.sum(1)
        self.sq += f.pow(2).sum(1)
        self.top1 += F.one_hot(f.argmax(-1), k).float().sum(1)
        self.ent += -(f.clamp_min(1e-12).log() * f).sum(-1).sum(-1)
        self.maxw += f.max(-1).values.sum(-1)
        self.n += b * q

    def done(self, n_keys):
        mean = self.mass / self.n
        var = self.sq / self.n - mean.pow(2)
        return dict(mean=mean, var=var, top1_frac=self.top1 / self.n,
                    entropy=(self.ent / self.n), max_w=(self.maxw / self.n), uniform=1.0 / n_keys)


@torch.no_grad()
def analyse(fwd_model, x):
    """Re-run the ViT forward, collecting per-head attention stats for every attention module."""
    m = fwd_model
    dev = x.device
    st, vnorm, hnorm = {}, {}, {}

    def key(kind, i):
        return f'{kind}[{i}]'

    for start in range(0, x.shape[0], CHUNK):
        xb = patchify(x[start:start + CHUNK], 28, 2)
        h = m.embed(xb) + m.pos_enc
        # encoder self-attention
        for i, blk in enumerate(m.enc):
            k = key('enc', i)
            hn = h.norm(dim=-1)
            q = blk.ln1(h)
            out, w = blk.attn(q, q, q, need_weights=True, average_attn_weights=False)
            st.setdefault(k, Stats(w.shape[1], w.shape[3], dev)).add(w)
            vnorm[k] = vnorm.get(k, 0) + value_norms(blk.attn, q).mean(0)
            hnorm[k] = hnorm.get(k, 0) + hn.mean(0)
            h = h + blk.drop1(out)
            h = h + blk.drop2(blk.ffn(blk.ln2(h)))
        # encoder cross-attention: 8 learned queries over the 196 encoded tokens
        ca, b = m.enc_cross, h.shape[0]
        qq, kk = ca.lnq(m.lat_q.expand(b, -1, -1)), ca.lnk(h)
        out, w = ca.attn(qq, kk, kk, need_weights=True, average_attn_weights=False)
        st.setdefault('enc_cross', Stats(w.shape[1], w.shape[3], dev)).add(w)
        vnorm['enc_cross'] = vnorm.get('enc_cross', 0) + value_norms(ca.attn, kk).mean(0)
        lat = m.lat_q.expand(b, -1, -1) + ca.drop(out)
        mem = m.from_latent(m.to_latent(lat))
        # decoder cross-attention: 196 positional queries over the 8 latent tokens
        ca = m.dec_cross
        qq, kk = ca.lnq(m.pos_dec.expand(b, -1, -1)), ca.lnk(mem)
        out, w = ca.attn(qq, kk, kk, need_weights=True, average_attn_weights=False)
        st.setdefault('dec_cross', Stats(w.shape[1], w.shape[3], dev)).add(w)
        vnorm['dec_cross'] = vnorm.get('dec_cross', 0) + value_norms(ca.attn, kk).mean(0)
        h = m.pos_dec.expand(b, -1, -1) + ca.drop(out)
        for i, blk in enumerate(m.dec):
            k = key('dec', i)
            hn = h.norm(dim=-1)
            q = blk.ln1(h)
            out, w = blk.attn(q, q, q, need_weights=True, average_attn_weights=False)
            st.setdefault(k, Stats(w.shape[1], w.shape[3], dev)).add(w)
            vnorm[k] = vnorm.get(k, 0) + value_norms(blk.attn, q).mean(0)
            hnorm[k] = hnorm.get(k, 0) + hn.mean(0)
            h = h + blk.drop1(out)
            h = h + blk.drop2(blk.ffn(blk.ln2(h)))
        recon = unpatchify(m.readout(m.ln_out(h)), 28, 2)

    n_chunks = (x.shape[0] + CHUNK - 1) // CHUNK
    out = {}
    for k, s in st.items():
        d = s.done(s.mass.shape[1])
        d['value_norm'] = vnorm[k] / n_chunks
        if k in hnorm:
            d['hidden_norm'] = hnorm[k] / n_chunks
        out[k] = d
    return out, recon


def sanity(model, x):
    """Before believing anything: per-head weights, rows summing to 1, forward reproduced."""
    xb = patchify(x[:8], 28, 2)
    blk = model.enc[0]
    q = blk.ln1(model.embed(xb) + model.pos_enc)
    _, w_avg = blk.attn(q, q, q, need_weights=True)
    _, w = blk.attn(q, q, q, need_weights=True, average_attn_weights=False)
    heads = model.enc[0].attn.num_heads
    print(f'  head-averaged shape {tuple(w_avg.shape)} vs per-head {tuple(w.shape)}; '
          f'num_heads {heads} -> per-head dim present: {w.dim() == 4 and w.shape[1] == heads}')
    rows = w.sum(-1)
    print(f'  attention rows sum to 1: max |sum-1| = {float((rows - 1).abs().max()):.2e}')
    assert w.dim() == 4 and w.shape[1] == heads, 'per-head weights not returned'
    assert float((rows - 1).abs().max()) < 1e-5, 'attention rows do not sum to 1'
    return heads


@torch.no_grad()
def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xte, _ = load('fashion', train=False, device=dev)
    x = xte[:N_IMG]
    report = {}

    for tag, run, only in RUNS:
        d = json.load(open(os.path.join(R, run, 'run.json')))
        files = only or [c['file'] for c in d['summary']['checkpoints']]
        steps = {c['file']: c['step'] for c in d['summary']['checkpoints']}
        for wf in files:
            model, _ = load_model(run, wf, dev)
            label = f'{tag} @ {steps.get(wf, 60000)//1000}K'
            print(f'\n=== {label} ({run}/{wf}) ===')
            heads = sanity(model, x)
            # the re-implemented forward must match the model's own, or none of this means anything
            ref = unpatchify(model(patchify(x[:64], 28, 2)), 28, 2)
            got = analyse(model, x[:64])[1]
            print(f'  re-implemented forward matches model.forward: '
                  f'max |diff| = {float((ref - got).abs().max()):.2e}')
            assert float((ref - got).abs().max()) < 1e-4, 'forward re-implementation diverges'
            stats, _ = analyse(model, x)
            report[label] = {k: {kk: vv.cpu().tolist() if torch.is_tensor(vv) else vv
                                 for kk, vv in v.items()} for k, v in stats.items()}
            report[label]['_heads'] = heads
    path = os.path.join(HERE, 'runs_vit', 'attention_sinks.json')
    json.dump(report, open(path, 'w'))
    print('\nwrote', path)


if __name__ == '__main__':
    main()
