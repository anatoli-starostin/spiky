"""ViT autoencoder on 14x14 Fashion-MNIST, plain backprop, MSE reconstruction.

autoencoder.py is untouched; this is the transformer line. Defaults reproduce the first (500-step)
result exactly -- patch 1, one pooled global latent, no warmup, no weight decay -- so the earlier runs
stay reproducible, and every addition below is opt-in behind a flag.

    embed patch -> d      + learned positional embeddings
    encoder x N           pre-LN: LN -> MHSA -> residual,  LN -> FFN -> residual
    bottleneck            see --latent-tokens
    decoder x N           same block
    readout d -> patch    per token, shared weights

BOTTLENECK SHAPES (--latent-tokens):
  1  the original: mean-pool over tokens -> Linear d -> latent -> Linear back -> ONE vector broadcast to
     every position, plus the positional embedding. The decoder's only per-position information is the
     positional embedding.
  N  N latent tokens of latent/N dims each, formed by learned queries cross-attending to the encoded
     tokens; the decoder cross-attends positional queries to those N latent tokens instead of receiving
     a single broadcast vector. Same total latent budget.

UNITS. data.py standardises as (x/255 - 0.1307)/0.3081 (the branch's hard-coded constants, MNIST's
applied to Fashion-MNIST, kept for comparability -- they set the MSE scale). The 2x2 average pool to
14x14 is applied after that; for an affine normalisation this is identical to pooling first, and train
and test take the same path.

The 14x14 numbers are NOT comparable with the 784-dim ones: 196 -> 64 is 3.06x compression where the
old line was 12.25x, so this file re-runs its own linear and per-pixel-mean baselines.

The FFN sub-block is pluggable behind --ffn {mlp,lut}; mlp is the default and the lut seam is wired but
not exercised here.
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
    b = x.shape[0]
    return F.avg_pool2d(x.view(b, 1, side, side), factor).reshape(b, -1)


def patchify(x, side=14, patch=1):
    """[B, side*side] -> [B, n_tokens, patch*patch]. patch=1 is one pixel per token."""
    b = x.shape[0]
    if patch == 1:
        return x.unsqueeze(-1)
    g = side // patch
    return (x.view(b, 1, side, side)
             .unfold(2, patch, patch).unfold(3, patch, patch)
             .reshape(b, g * g, patch * patch))


def unpatchify(t, side=14, patch=1):
    """[B, n_tokens, patch*patch] -> [B, side*side]."""
    b = t.shape[0]
    if patch == 1:
        return t.squeeze(-1)
    g = side // patch
    return (t.view(b, g, g, patch, patch).permute(0, 1, 3, 2, 4).reshape(b, side * side))


class LutFFN(nn.Module):
    """The LUT seam: LightMHL standing in for the FFN, per token, as the ffn_replacement work does."""

    def __init__(self, d_model, n_tables, device, seed):
        super().__init__()
        self.lut = LightMultiHeadLUT(input_dim=d_model, n_tables=n_tables, output_dim=d_model,
                                     random_seed=seed, device=torch.device(device), **LUT_KW)

    def forward(self, x):
        b, t, d = x.shape
        return self.lut(x.reshape(b * t, d)).reshape(b, t, d)


class Block(nn.Module):
    """Pre-LN: LN -> MHSA -> residual, LN -> FFN -> residual.

    DROPOUT, when p > 0, sits in the three conventional places for a pre-LN transformer: on the attention
    weights (inside MultiheadAttention), on each sub-block's output just before it joins the residual
    stream (drop1, drop2), and inside the MLP after the GELU. Nothing is dropped on the residual branch
    itself, on the embeddings, or on the readout. p = 0 builds nn.Dropout(0)/attention dropout 0, which
    are exact identities, so every pre-existing run is bit-unchanged.
    """

    def __init__(self, d_model, n_heads, ffn_mult, ffn_kind, n_tables, device, seed, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = (LutFFN(d_model, n_tables, device, seed) if ffn_kind == 'lut'
                    else nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(ffn_mult * d_model, d_model)))
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.drop1(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop2(self.ffn(self.ln2(x)))


class CrossAttn(nn.Module):
    """Pre-LN cross-attention: queries attend to a memory sequence."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.lnq, self.lnk = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, mem):
        return q + self.drop(self.attn(self.lnq(q), self.lnk(mem), self.lnk(mem), need_weights=False)[0])


class ViTAutoencoder(nn.Module):
    def __init__(self, n_tokens=196, patch_dim=1, d_model=64, n_heads=4, enc_layers=2, dec_layers=2,
                 latent=64, latent_tokens=1, ffn_mult=4, ffn_kind='mlp', n_tables=64, device='cuda',
                 seed=0, dropout=0.0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_tokens, self.latent, self.latent_tokens = n_tokens, latent, latent_tokens
        if latent % latent_tokens:
            raise ValueError(f'latent {latent} must divide by latent_tokens {latent_tokens}')
        self.lat_dim = latent // latent_tokens
        self.embed = nn.Linear(patch_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        # a SEPARATE decoder positional embedding: the encoder's positions say "where this pixel came
        # from", the decoder's say "where this query is asking about", and sharing one tensor makes the
        # two roles fight. The original run shared them; that is now the non-default path.
        self.pos_dec = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        mk = lambda i: Block(d_model, n_heads, ffn_mult, ffn_kind, n_tables, device, seed + 17 * i,  # noqa: E731
                             dropout)
        self.enc = nn.ModuleList([mk(i) for i in range(enc_layers)])
        self.dec = nn.ModuleList([mk(100 + i) for i in range(dec_layers)])
        if latent_tokens == 1:
            self.to_latent = nn.Linear(d_model, latent)
            self.from_latent = nn.Linear(latent, d_model)
        else:
            self.lat_q = nn.Parameter(torch.randn(1, latent_tokens, d_model) * 0.02)
            self.enc_cross = CrossAttn(d_model, n_heads, dropout)
            self.dec_cross = CrossAttn(d_model, n_heads, dropout)
            self.to_latent = nn.Linear(d_model, self.lat_dim)
            self.from_latent = nn.Linear(self.lat_dim, d_model)
        self.ln_out = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, patch_dim)
        self.to(device)

    def forward(self, x):                                  # x [B, n_tokens, patch_dim]
        h = self.embed(x) + self.pos_enc
        for b in self.enc:
            h = b(h)
        if self.latent_tokens == 1:
            z = self.to_latent(h.mean(1))                                        # [B, latent]
            h = self.from_latent(z).unsqueeze(1).expand(-1, self.n_tokens, -1) + self.pos_dec
        else:
            lat = self.enc_cross(self.lat_q.expand(h.shape[0], -1, -1), h)       # [B, K, d]
            z = self.to_latent(lat)                                              # [B, K, lat_dim]
            mem = self.from_latent(z)                                            # [B, K, d]
            h = self.dec_cross(self.pos_dec.expand(h.shape[0], -1, -1), mem)     # [B, T, d]
        for b in self.dec:
            h = b(h)
        return self.readout(self.ln_out(h))


class LinearAE(nn.Module):
    def __init__(self, n_in=196, latent=64, device='cuda', seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc, self.dec = nn.Linear(n_in, latent), nn.Linear(latent, n_in)
        self.to(device)

    def forward(self, x):
        return self.dec(self.enc(x))


def make_eval(model, arch, side, patch):
    """One callable mapping flat pixels to flat reconstruction, whatever the arch."""
    if arch == 'linear':
        return lambda b: model(b)
    return lambda b: unpatchify(model(patchify(b, side, patch)), side, patch)


def evaluate(fwd, x, bs=4096, model=None):
    """Reconstruction MSE over x, ALWAYS with the model in eval mode.

    Passing the model matters as soon as --dropout is on: a probe run in train mode would measure a
    randomly-thinned network, so the reported curve would be noisier and biased upward relative to the
    model one actually keeps. Train mode is restored afterwards, so this is invisible to the loop.
    """
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            b = x[i:i + bs]
            tot += float((fwd(b) - b).pow(2).sum())
            n += b.numel()
    if was_training:
        model.train()
    return tot / n


def lr_at(step, total, base, warmup, sched):
    if warmup and step <= warmup:
        return base * step / max(warmup, 1)
    if sched == 'cosine':
        t = (step - warmup) / max(total - warmup, 1)
        return base * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='vit', choices=['vit', 'linear'])
    ap.add_argument('--ffn', default='mlp', choices=['mlp', 'lut'])
    ap.add_argument('--patch', type=int, default=1, help='1 -> 196 tokens of 1 dim; 2 -> 49 tokens of 4')
    ap.add_argument('--enc-layers', type=int, default=2)
    ap.add_argument('--dec-layers', type=int, default=2)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--ffn-mult', type=int, default=4)
    ap.add_argument('--latent', type=int, default=64, help='TOTAL latent budget in dims')
    ap.add_argument('--latent-tokens', type=int, default=1,
                    help='1 = pooled global vector broadcast back; K>1 = K latent tokens of latent/K '
                         'dims, formed by learned queries and decoded by cross-attention')
    ap.add_argument('--tables', type=int, default=64)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=0, help='linear warmup steps; 0 = none (the original)')
    ap.add_argument('--sched', default='none', choices=['none', 'cosine'])
    ap.add_argument('--wd', type=float, default=0.0, help='>0 switches Adam to AdamW')
    ap.add_argument('--dropout', type=float, default=0.0,
                    help='dropout p inside every transformer block: attention weights, each sub-block '
                         'output before the residual add, and inside the MLP after the GELU. Also the '
                         'two cross-attention modules. 0 = off, exactly the previous behaviour.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--probe-every', type=int, default=25)
    ap.add_argument('--ckpt-every', type=int, default=0,
                    help='also save model_s<step>.pt every N steps, each kept separately, so a long run '
                         'can be rendered at several points along training. 0 = only the final model.pt.')
    ap.add_argument('--no-downsample', action='store_true',
                    help='train on the full 28x28 source images instead of average-pooling them to '
                         '14x14. Default is OFF, i.e. the 14x14 behaviour every existing run used.')
    ap.add_argument('--out-dir', default='runs_vit')
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # the raw Fashion-MNIST tensors are 28x28 (verified: ds.data.shape = (60000, 28, 28)); data.py only
    # flattens and standardises them, so the 14x14 in every previous run came from the average pool here
    side = 28 if a.no_downsample else 14
    raw_tr, raw_te = load('fashion', train=True, device=dev)[0], load('fashion', train=False, device=dev)[0]
    xtr = raw_tr if a.no_downsample else downsample(raw_tr)
    xte = raw_te if a.no_downsample else downsample(raw_te)
    n_pix = xtr.shape[1]
    n_tok, patch_dim = (n_pix // (a.patch ** 2), a.patch ** 2)

    model = (LinearAE(n_pix, a.latent, dev, a.seed) if a.arch == 'linear'
             else ViTAutoencoder(n_tok, patch_dim, a.d_model, a.n_heads, a.enc_layers, a.dec_layers,
                                 a.latent, a.latent_tokens, a.ffn_mult, a.ffn, a.tables, dev, a.seed,
                                 a.dropout))
    opt = (torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd) if a.wd > 0
           else torch.optim.Adam(model.parameters(), lr=a.lr))
    fwd = make_eval(model, a.arch, side, a.patch)
    loader = TensorLoader(xtr, xtr, batch_size=a.batch, seed=a.seed)
    name = a.name or f'{a.arch}-{a.ffn}-p{a.patch}-k{a.latent_tokens}-d{a.d_model}'
    nparam = sum(p.numel() for p in model.parameters())
    print(f'{name} | {a.arch} {side}x{side} ({n_pix} px, {n_pix/a.latent:.2f}x compression) '
          f'patch {a.patch} -> {n_tok} tokens x {patch_dim} | d_model {a.d_model} '
          f'heads {a.n_heads} enc {a.enc_layers} dec {a.dec_layers} | latent {a.latent} in '
          f'{a.latent_tokens} token(s) | {"AdamW wd %.3g" % a.wd if a.wd > 0 else "Adam"} lr {a.lr} '
          f'warmup {a.warmup} sched {a.sched} | {nparam/1e6:.3f}M params', flush=True)
    out_dir = os.path.join(HERE, a.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        mu = xtr.mean(0, keepdim=True)
        mean_tr, mean_te = float((xtr - mu).pow(2).mean()), float((xte - mu).pow(2).mean())

    hist, ckpts, it, t0 = [], [], iter(loader), time.time()
    for step in range(1, a.steps + 1):
        try:
            bx, _ = next(it)
        except StopIteration:
            it = iter(loader)
            bx, _ = next(it)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, a.steps, a.lr, a.warmup, a.sched)
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        loss = (fwd(bx) - bx).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dt = time.time() - ts
        if step % a.probe_every == 0 or step == 1:
            row = {'step': step, 'train/mse_batch': float(loss.detach()), 'train/s_per_step': dt,
                   'train/lr': opt.param_groups[0]['lr'],
                   'eval/train_mse': evaluate(fwd, xtr[:10000], model=model), 'eval/test_mse': evaluate(fwd, xte, model=model)}
            hist.append(row)
            if step == 1 or step % (a.probe_every * 20) == 0 or step == a.steps:
                print(f'  step {step:>5d}  train {row["eval/train_mse"]:.5f}  '
                      f'test {row["eval/test_mse"]:.5f}  lr {row["train/lr"]:.2e}  '
                      f'{dt*1e3:.1f} ms/step', flush=True)
            if not math.isfinite(row['eval/train_mse']):
                print('  DIVERGED (non-finite) -- stopping', flush=True)
                break
        if a.ckpt_every and step % a.ckpt_every == 0:
            # a separate file per checkpoint, so a whole training trajectory can be reconstructed later
            # from one run. Each carries the eval it was taken at, so the figure never has to re-measure.
            torch.save(model.state_dict(), os.path.join(out_dir, f'model_s{step}.pt'))
            ck = {'step': step, 'file': f'model_s{step}.pt',
                  'train_mse': evaluate(fwd, xtr[:10000], model=model), 'test_mse': evaluate(fwd, xte, model=model)}
            ckpts.append(ck)
            print(f'  ckpt {step:>6d}  train {ck["train_mse"]:.5f}  test {ck["test_mse"]:.5f}', flush=True)

    # improvement over the last 300 steps, at the probe cadence
    w = max(300 // a.probe_every, 1)
    tail = [r['eval/test_mse'] for r in hist[-w:]]
    prev = [r['eval/test_mse'] for r in hist[-2 * w:-w]] or tail
    summary = {'wall_s': time.time() - t0, 'params': nparam, 'n_blocks': a.enc_layers + a.dec_layers,
               'n_tokens': n_tok, 'patch_dim': patch_dim, 'side': side, 'n_pixels': n_pix,
               'compression': n_pix / a.latent,
               'train_mse': evaluate(fwd, xtr[:10000], model=model), 'test_mse': evaluate(fwd, xte, model=model),
               'mean_baseline_train': mean_tr, 'mean_baseline_test': mean_te,
               's_per_step': st.median([r['train/s_per_step'] for r in hist]),
               'improve_pct_300': 100.0 * (st.mean(prev) - st.mean(tail)) / max(st.mean(prev), 1e-12),
               'improve_window_steps': 300, 'steps_done': hist[-1]['step'], 'pix_std': PIX_STD,
               'checkpoints': ckpts}
    json.dump({'cfg': dict(vars(a), exp_name=name), 'hist': hist, 'summary': summary},
              open(os.path.join(out_dir, 'run.json'), 'w'), indent=1)
    # the weights, so reconstructions can be rendered later without retraining. autoencoder.py has always
    # done this; this file did not, which is why the runs before this commit have no checkpoint. These are
    # small (the largest arm is 1.8M params) but .gitignore still keeps *.pt out of the repo.
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    print(f'{name} done: {summary["wall_s"]:.1f}s, train {summary["train_mse"]:.5f}, '
          f'test {summary["test_mse"]:.5f} (mean {mean_te:.5f}, ratio '
          f'{summary["test_mse"]/mean_te:.3f}), improving {summary["improve_pct_300"]:.2f}%/300')


if __name__ == '__main__':
    main()
