"""ViT autoencoder on 14x14 Fashion-MNIST, plain backprop, MSE reconstruction.

A new experiment file; autoencoder.py is untouched. One pixel is one token (196 tokens, patch size 1,
scalar per token), embedded to d_model with learned positional embeddings, through pre-LN transformer
blocks, a mean-pooled 64-dim bottleneck, and back.

    embed 1 -> d          + learned pos emb [196, d]
    encoder x N           LN -> MHSA -> residual,  LN -> FFN -> residual
    bottleneck            mean-pool over tokens -> d -> 64 -> d -> broadcast -> re-add pos emb
    decoder x N           same block
    readout d -> 1        -> 196 -> 14x14

UNITS AND DOWNSAMPLING. data.py returns pixels already standardised as (x/255 - 0.1307)/0.3081 (the
branch's hard-coded constants, kept for comparability and flagged because they set the MSE scale, and
they are MNIST's applied to Fashion-MNIST). The 2x2 average pool is applied AFTER that normalisation.
For an affine normalisation this is identical to pooling first and then normalising with the same
constants -- mean((x-m)/s) = (mean(x)-m)/s -- so the choice is free; it is stated because consistency
between train and test is not, and both go through the identical path here.

THE 14x14 NUMBERS ARE NOT COMPARABLE WITH THE 784-DIM ONES. A 64-dim bottleneck out of 196 inputs is a
3.06x compression where the old one was 12.25x, so this file re-runs its own linear and mean baselines.

The FFN sub-block is pluggable behind --ffn {mlp,lut}; mlp is the default and the lut seam is wired but
not exercised here.

Usage: python3 vit_autoencoder.py --arch vit --enc-layers 2 --dec-layers 2
"""
import argparse
import json
import math
import os
import statistics as st
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from data import TensorLoader, load  # noqa: E402
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402

PIX_STD = 0.3081
LUT_KW = dict(n_anchor_pairs=8, read_top_n=2, read_tau=0.5, read_tau_learnable=True,
              confidence_form='margin', cell_mode='constant', forward_mode='scored',
              multi_head_input=False, initial_weights_noise=1e-3, head_dropout_rate=0.0)


def downsample(x, side=28, factor=2):
    """28x28 -> 14x14 by 2x2 average pooling, on the already-normalised tensor."""
    b = x.shape[0]
    return F.avg_pool2d(x.view(b, 1, side, side), factor).reshape(b, -1)


class LutFFN(nn.Module):
    """The LUT seam: a LightMHL standing in for the FFN, applied per token exactly as the
    ffn_replacement work does (tokens folded into the batch, d_model in and out). Wired, not run."""

    def __init__(self, d_model, n_tables, device, seed):
        super().__init__()
        self.lut = LightMultiHeadLUT(input_dim=d_model, n_tables=n_tables, output_dim=d_model,
                                     random_seed=seed, device=torch.device(device), **LUT_KW)

    def forward(self, x):
        b, t, d = x.shape
        return self.lut(x.reshape(b * t, d)).reshape(b, t, d)


class Block(nn.Module):
    """Pre-LN transformer block: LN -> MHSA -> residual, LN -> FFN -> residual."""

    def __init__(self, d_model, n_heads, ffn_mult, ffn_kind, n_tables, device, seed):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        if ffn_kind == 'lut':
            self.ffn = LutFFN(d_model, n_tables, device, seed)
        else:
            self.ffn = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
                                     nn.Linear(ffn_mult * d_model, d_model))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ffn(self.ln2(x))


class ViTAutoencoder(nn.Module):
    def __init__(self, n_tokens=196, d_model=64, n_heads=4, enc_layers=2, dec_layers=2, latent=64,
                 ffn_mult=4, ffn_kind='mlp', n_tables=64, device='cuda', seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_tokens, self.latent = n_tokens, latent
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        mk = lambda i: Block(d_model, n_heads, ffn_mult, ffn_kind, n_tables, device, seed + 17 * i)  # noqa: E731
        self.enc = nn.ModuleList([mk(i) for i in range(enc_layers)])
        self.dec = nn.ModuleList([mk(100 + i) for i in range(dec_layers)])
        self.to_latent = nn.Linear(d_model, latent)
        self.from_latent = nn.Linear(latent, d_model)
        self.ln_out = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, 1)
        self.to(device)

    def forward(self, x):                                  # x [B, n_tokens]
        h = self.embed(x.unsqueeze(-1)) + self.pos
        for b in self.enc:
            h = b(h)
        z = self.to_latent(h.mean(1))                      # bottleneck: pool over tokens
        h = self.from_latent(z).unsqueeze(1).expand(-1, self.n_tokens, -1) + self.pos
        for b in self.dec:
            h = b(h)
        return self.readout(self.ln_out(h)).squeeze(-1)


class LinearAE(nn.Module):
    """The reference at this resolution: 196 -> 64 -> 196."""

    def __init__(self, n_in=196, latent=64, device='cuda', seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc, self.dec = nn.Linear(n_in, latent), nn.Linear(latent, n_in)
        self.to(device)

    def forward(self, x):
        return self.dec(self.enc(x))


def evaluate(model, x, bs=4096):
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            b = x[i:i + bs]
            tot += float((model(b) - b).pow(2).sum())
            n += b.numel()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='vit', choices=['vit', 'linear'])
    ap.add_argument('--ffn', default='mlp', choices=['mlp', 'lut'],
                    help='FFN sub-block. lut is wired to LightMHL but is not the arm being run here.')
    ap.add_argument('--enc-layers', type=int, default=2)
    ap.add_argument('--dec-layers', type=int, default=2)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--ffn-mult', type=int, default=4)
    ap.add_argument('--latent', type=int, default=64)
    ap.add_argument('--tables', type=int, default=64)
    ap.add_argument('--steps', type=int, default=500)          # matches autoencoder.py
    ap.add_argument('--batch', type=int, default=128)          # matches autoencoder.py
    ap.add_argument('--lr', type=float, default=1e-3)          # matches autoencoder.py
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--probe-every', type=int, default=25)     # matches autoencoder.py
    ap.add_argument('--out-dir', default='runs_vit_ae')
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xtr = downsample(load('fashion', train=True, device=dev)[0])
    xte = downsample(load('fashion', train=False, device=dev)[0])
    n_tok = xtr.shape[1]

    model = (LinearAE(n_tok, a.latent, dev, a.seed) if a.arch == 'linear'
             else ViTAutoencoder(n_tok, a.d_model, a.n_heads, a.enc_layers, a.dec_layers, a.latent,
                                 a.ffn_mult, a.ffn, a.tables, dev, a.seed))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loader = TensorLoader(xtr, xtr, batch_size=a.batch, seed=a.seed)
    name = a.name or (f'{a.arch}-{a.ffn}-e{a.enc_layers}d{a.dec_layers}' if a.arch == 'vit'
                      else f'linear-{n_tok}-{a.latent}')
    nparam = sum(p.numel() for p in model.parameters())
    print(f'{name} | {a.arch} ffn={a.ffn} tokens={n_tok} d_model={a.d_model} heads={a.n_heads} '
          f'enc={a.enc_layers} dec={a.dec_layers} latent={a.latent} | Adam lr {a.lr} batch {a.batch} | '
          f'{nparam/1e6:.3f}M params', flush=True)

    with torch.no_grad():
        mu = xtr.mean(0, keepdim=True)
        mean_tr = float((xtr - mu).pow(2).mean())
        mean_te = float((xte - mu).pow(2).mean())
    print(f'  per-pixel mean baseline at {n_tok} px: train {mean_tr:.4f}, test {mean_te:.4f}', flush=True)

    hist, it, t0 = [], iter(loader), time.time()
    for step in range(1, a.steps + 1):
        try:
            bx, _ = next(it)
        except StopIteration:
            it = iter(loader)
            bx, _ = next(it)
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        loss = (model(bx) - bx).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dt = time.time() - ts
        if step % a.probe_every == 0 or step == 1:
            row = {'step': step, 'train/mse_batch': float(loss.detach()), 'train/s_per_step': dt,
                   'eval/train_mse': evaluate(model, xtr[:10000]), 'eval/test_mse': evaluate(model, xte)}
            hist.append(row)
            print(f'  step {step:>4d}  train {row["eval/train_mse"]:.5f}  test {row["eval/test_mse"]:.5f}'
                  f'  {dt*1e3:.1f} ms/step', flush=True)
            if not math.isfinite(row['eval/train_mse']):
                print('  DIVERGED (non-finite) -- stopping', flush=True)
                break

    tail = [r['eval/test_mse'] for r in hist[-3:]]
    prev = [r['eval/test_mse'] for r in hist[-6:-3]] or tail
    summary = {'wall_s': time.time() - t0, 'params': nparam, 'n_blocks': a.enc_layers + a.dec_layers,
               'n_tokens': n_tok, 'train_mse': evaluate(model, xtr[:10000]), 'test_mse': evaluate(model, xte),
               'mean_baseline_train': mean_tr, 'mean_baseline_test': mean_te,
               's_per_step': st.median([r['train/s_per_step'] for r in hist]),
               'still_improving_pct': 100.0 * (st.mean(prev) - st.mean(tail)) / max(st.mean(prev), 1e-12),
               'improve_window_steps': 3 * a.probe_every, 'steps_done': hist[-1]['step'],
               'pix_std': PIX_STD}
    out_dir = os.path.join(HERE, a.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump({'cfg': dict(vars(a), exp_name=name), 'hist': hist, 'summary': summary},
              open(os.path.join(out_dir, 'run.json'), 'w'), indent=1)
    print(f'{name} done: {summary["wall_s"]:.1f}s, train {summary["train_mse"]:.5f}, '
          f'test {summary["test_mse"]:.5f} (mean baseline {mean_te:.5f}), '
          f'still improving {summary["still_improving_pct"]:.2f}%/{3*a.probe_every} steps')


if __name__ == '__main__':
    main()
