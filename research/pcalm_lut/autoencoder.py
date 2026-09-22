"""LUT autoencoder on Fashion-MNIST, plain backprop, MSE reconstruction.

A different question from the PC/DTP work on this branch: no predictive coding, no target propagation,
no paired backward stack. Just: can a LightMHL stack carry information through a narrow bottleneck --
can it exploit the combinatorial capacity of a low-dimensional vector?

    Linear{784 -> 64}
    [ LightMHL{n_heads=1, tables_per_head=64, nap=8} ] x 2L     residual blocks
    Linear{64 -> 784}

L is the encoding depth: the first L blocks are the encoder, the last L the decoder. Note 28 x 28 = 784,
not 768.

UNITS. The branch's loader standardises pixels as (x/255 - 0.1307)/0.3081, so training MSE is in
standardised units. Every MSE is reported BOTH ways -- standardised, and converted to [0,1] pixel units
by multiplying by 0.3081^2 -- because an MSE without its normalisation is not a number. (Those constants
are MNIST's, applied to Fashion-MNIST; that is the branch's existing convention, kept here so this run is
comparable with the rest, and flagged because the MSE scale depends on it.)
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

PIX_STD = 0.3081            # the loader's divisor; MSE_pixel = MSE_standardised * PIX_STD^2
LUT_KW = dict(n_anchor_pairs=8, read_top_n=2, read_tau=0.5, read_tau_learnable=True,
              confidence_form='margin', cell_mode='constant', forward_mode='scored',
              multi_head_input=False, initial_weights_noise=1e-3, head_dropout_rate=0.0)


class Autoencoder(nn.Module):
    """Linear in, 2L residual blocks of the chosen kind, linear out."""

    def __init__(self, in_dim=784, width=64, depth_L=2, kind='lut', n_tables=64, device='cuda',
                 seed=0, hidden=0):
        super().__init__()
        self.kind, self.width, self.depth_L = kind, width, depth_L
        self.n_blocks = 2 * depth_L
        torch.manual_seed(seed)
        self.enc = nn.Linear(in_dim, width)
        self.dec = nn.Linear(width, in_dim)
        # the branch's residual scaling, with the block count in place of the layer count
        self.ai = 1.0 / math.sqrt(max(self.n_blocks, 1) * width)
        if kind == 'lut':
            self.blocks = nn.ModuleList([
                LightMultiHeadLUT(input_dim=width, n_tables=n_tables, output_dim=width,
                                  random_seed=seed + 100 * (i + 1), device=torch.device(device), **LUT_KW)
                for i in range(self.n_blocks)])
        elif kind == 'mlp':
            g = torch.Generator(device='cpu').manual_seed(seed)
            self.hidden = hidden
            if hidden:
                # width -> hidden -> width, so the control can be given a LUT block's parameter budget
                self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, hidden, bias=False),
                                                           nn.Tanh(),
                                                           nn.Linear(hidden, width, bias=False))
                                             for _ in range(self.n_blocks)])
            else:
                self.blocks = nn.ParameterList([nn.Parameter(torch.randn(width, width, generator=g))
                                                for _ in range(self.n_blocks)])
        elif kind == 'linear':
            self.blocks = nn.ModuleList([])          # the PCA-equivalent control: bottleneck only
        else:
            raise ValueError(kind)
        self.to(device)

    def encode_depth(self, h, upto):
        for i in range(upto):
            h = self.block(i, h)
        return h

    def block(self, i, h):
        if self.kind == 'lut':
            return h + self.ai * self.blocks[i](h)
        if self.kind == 'mlp':
            if getattr(self, 'hidden', 0):
                return h + self.ai * self.blocks[i](h)
            return h + self.ai * F.linear(torch.tanh(h), self.blocks[i])
        return h

    def forward(self, x):
        h = self.enc(x)
        for i in range(self.n_blocks):
            h = self.block(i, h)
        return self.dec(h)


# ------------------------------------------------------------------------------- diagnostics --------
@torch.no_grad()
def lut_stats(model, x):
    """Per-block smallest-margin median, learned tau, and the addresses (for a flip rate)."""
    if model.kind != 'lut':
        return {}, None
    h = model.enc(x)
    mm, taus, addr = [], [], []
    for i in range(model.n_blocks):
        lut = model.blocks[i]
        d = h[:, lut.anchor_a] - h[:, lut.anchor_b]
        mm.append(float(d.abs().min(-1).values.median()))
        taus.append(float(lut.read_tau))
        addr.append(((d > 0).to(torch.int64) * lut.powers.view(1, 1, -1)).sum(-1))
        h = model.block(i, h)
    return {'m_min_mean': sum(mm) / len(mm), 'tau_mean': sum(taus) / len(taus),
            'm_min': mm, 'tau': taus}, addr


@torch.no_grad()
def branch_ratios(model, x):
    """||a_i * block_i(h)|| / ||h|| per block: how far the stack has departed from the linear
    autoencoder it starts as. Near zero means the residual blocks are still doing nothing and the model
    is the linear bottleneck plus a perturbation, whatever its parameter count says. Works for every
    block kind; the linear control has no blocks and reports nothing."""
    if model.n_blocks == 0 or model.kind == 'linear':
        return []
    h = model.enc(x)
    out = []
    for i in range(model.n_blocks):
        nh = model.block(i, h)
        out.append(float((nh - h).norm() / max(float(h.norm()), 1e-12)))
        h = nh
    return out


@torch.no_grad()
def flip_rate(prev, cur):
    """Fraction of (sample, table) address slots that changed since the previous probe."""
    if prev is None or cur is None:
        return float('nan')
    return float(sum(float((a != b).float().mean()) for a, b in zip(prev, cur)) / len(cur))


def evaluate(model, x, bs=4096):
    """Mean-squared error per PIXEL, averaged over pixels and samples, in standardised units."""
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            b = x[i:i + bs]
            tot += float((model(b) - b).pow(2).sum())
            n += b.numel()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='lut', choices=['lut', 'mlp', 'linear'])
    ap.add_argument('--depth-L', type=int, default=2)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--tables', type=int, default=64,
                    help='tables_per_head; n_heads is 1, so this is n_tables')
    ap.add_argument('--hidden', type=int, default=0,
                    help='mlp only: widening block width -> hidden -> width, for a parameter-matched '
                         'control. 0 keeps the plain width x width block.')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--probe-every', type=int, default=25)
    ap.add_argument('--out-dir', default='runs_ae')
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xtr, _ = load('fashion', train=True, device=dev)
    xte, _ = load('fashion', train=False, device=dev)
    model = Autoencoder(xtr.shape[1], a.width, a.depth_L, a.kind, a.tables, dev, a.seed,
                        hidden=a.hidden)
    opt = (torch.optim.Adam(model.parameters(), lr=a.lr) if a.optimizer == 'adam'
           else torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.0))
    loader = TensorLoader(xtr, xtr, batch_size=a.batch, seed=a.seed)      # targets are the inputs
    name = a.name or f'{a.kind}-L{a.depth_L}-{a.optimizer}{a.lr}'
    nblk = model.n_blocks
    nparam = sum(p.numel() for p in model.parameters())
    print(f'{name} | {a.kind} 2L={nblk} blocks, width {a.width}, {a.tables} tables | '
          f'{a.optimizer} lr {a.lr} | {nparam/1e6:.2f}M params', flush=True)

    # the baselines that make the MSE interpretable
    with torch.no_grad():
        mu = xtr.mean(0, keepdim=True)
        mean_mse_tr = float((xtr - mu).pow(2).mean())
        mean_mse_te = float((xte - mu).pow(2).mean())

    hist, it, prev_addr, t0 = [], iter(loader), None, time.time()
    for step in range(1, a.steps + 1):
        try:
            bx, _ = next(it)
        except StopIteration:
            it = iter(loader)
            bx, _ = next(it)
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        loss = (model(bx) - bx).pow(2).mean()          # MSE, mean over pixels AND batch
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dt = time.time() - ts
        if step % a.probe_every == 0 or step == 1:
            s, addr = lut_stats(model, xtr[:512])
            br = branch_ratios(model, xtr[:512])
            row = {'step': step, 'train/mse_batch': float(loss.detach()), 'train/s_per_step': dt,
                   'eval/train_mse': evaluate(model, xtr[:10000]), 'eval/test_mse': evaluate(model, xte),
                   'flips/mean': flip_rate(prev_addr, addr),
                   'branch/ratio_mean': (sum(br) / len(br)) if br else 0.0}
            for bi, bv in enumerate(br):
                row[f'branch/ratio_b{bi}'] = bv
            row.update({f'lut/{k}': v for k, v in s.items() if not isinstance(v, list)})
            prev_addr = addr
            hist.append(row)
            print(f'  step {step:>4d}  train {row["eval/train_mse"]:.5f}  test {row["eval/test_mse"]:.5f}'
                  f'  m_min {s.get("m_min_mean", float("nan")):.4f}  tau {s.get("tau_mean", float("nan")):.4f}'
                  f'  flips {row["flips/mean"]:.4f}  branch {row["branch/ratio_mean"]:.4f}'
                  f'  {dt*1e3:.1f} ms/step', flush=True)

    s, _ = lut_stats(model, xtr[:512])
    tail = [r['eval/test_mse'] for r in hist[-3:]]
    prev = [r['eval/test_mse'] for r in hist[-6:-3]] or tail
    summary = {'wall_s': time.time() - t0, 'params': nparam, 'n_blocks': nblk,
               'train_mse': evaluate(model, xtr[:10000]), 'test_mse': evaluate(model, xte),
               'mean_baseline_train': mean_mse_tr, 'mean_baseline_test': mean_mse_te,
               's_per_step': st.median([r['train/s_per_step'] for r in hist]),
               'still_improving_pct': 100.0 * (st.mean(prev) - st.mean(tail)) / max(st.mean(prev), 1e-12),
               'improve_window_steps': 3 * a.probe_every,
               'm_min_first': hist[0].get('lut/m_min_mean', float('nan')),
               'm_min_last': s.get('m_min_mean', float('nan')),
               'tau_last': s.get('tau_mean', float('nan')),
               'flips_last': hist[-1]['flips/mean'], 'pix_std': PIX_STD,
               'branch_first': hist[0]['branch/ratio_mean'], 'branch_last': hist[-1]['branch/ratio_mean']}
    out_dir = os.path.join(HERE, a.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump({'cfg': dict(vars(a), exp_name=name), 'hist': hist, 'summary': summary},
              open(os.path.join(out_dir, 'run.json'), 'w'), indent=1)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    print(f'{name} done: {summary["wall_s"]:.1f}s, train {summary["train_mse"]:.5f}, '
          f'test {summary["test_mse"]:.5f} (mean baseline {mean_mse_te:.5f}), '
          f'still improving {summary["still_improving_pct"]:.2f}%/75 steps')


if __name__ == '__main__':
    main()
