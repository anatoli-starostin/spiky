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
from spiky.lutorch.compression_mhl import CompressionMHL  # noqa: E402
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402
# The SAME augmentation object the ViT line uses -- imported, not reimplemented, so the two lines
# cannot drift apart. vit_autoencoder.py is left untouched.
from vit_autoencoder import augment  # noqa: E402

PIX_STD = 0.3081            # the loader's divisor; MSE_pixel = MSE_standardised * PIX_STD^2
LUT_KW = dict(n_anchor_pairs=8, read_top_n=2, read_tau=0.5, read_tau_learnable=True,
              confidence_form='margin', cell_mode='constant', forward_mode='scored',
              multi_head_input=False, initial_weights_noise=1e-3, head_dropout_rate=0.0)


class Autoencoder(nn.Module):
    """Linear in, 2L residual blocks of the chosen kind, linear out."""

    def __init__(self, in_dim=784, width=64, depth_L=2, kind='lut', n_tables=64, device='cuda',
                 seed=0, hidden=0, residual=True, gain_norm=False, block_norm='none',
                 lut_impl='light', norm_position='post', n_blocks=0, inner_out=-1, inner_in=-1, final_norm=False):
        super().__init__()
        self.kind, self.width, self.depth_L = kind, width, depth_L
        self.residual = residual
        # block_norm supersedes gain_norm; gain_norm=True is kept as the old spelling of 'gain'
        self.block_norm = 'gain' if gain_norm else block_norm
        self.gain_norm = self.block_norm == 'gain'
        self.norm_position = norm_position
        # n_blocks=0 keeps the original 2*depth_L convention (L counted as an ENCODING depth); an
        # explicit count is for specs that write "[block] x L" and mean L blocks total.
        self.n_blocks = n_blocks if n_blocks else 2 * depth_L
        torch.manual_seed(seed)
        self.enc = nn.Linear(in_dim, width)
        self.dec = nn.Linear(width, in_dim)
        # a_i = 1/sqrt(2L*width) is a RESIDUAL-BRANCH init: deliberately small because it scales a
        # correction added to a stream that already carries the signal. In non-residual mode the block IS
        # the signal path, so that factor would attenuate by ~16x per block and the run would measure
        # nothing but decay. Non-residual therefore drops it entirely and calibrates the init instead
        # (see calibrate_init).
        self.ai = 1.0 / math.sqrt(max(self.n_blocks, 1) * width) if residual else 1.0
        if kind == 'lut':
            # lut_impl='compression' wraps the same light LUT in CompressionMHL.
            # inner_out = -1 is the "no decompress" sentinel: eff_out becomes output_dim, the LUT emits
            # the full width itself and decompress is an Identity. inner_out = width keeps a real
            # Linear(width -> width) decompress after a LUT that emits `width`.
            # EITHER WAY THE BLOCK EMITS `width`. The sentinel does not change the block's output
            # dimension, so blocks stack homogeneously, the final decoder stays width -> in_dim, the
            # compression ratio is untouched, and a residual is well defined in both.
            self.blocks = nn.ModuleList([
                (LightMultiHeadLUT(input_dim=width, n_tables=n_tables, output_dim=width,
                                   random_seed=seed + 100 * (i + 1), device=torch.device(device),
                                   **LUT_KW)
                 if lut_impl == 'light' else
                 CompressionMHL(input_dim=width, output_dim=width, inner_in_dim=inner_in,
                                inner_out_dim=inner_out,
                                nap=LUT_KW['n_anchor_pairs'], tph=n_tables, n_heads=1, lut_impl='light',
                                random_seed=seed + 100 * (i + 1), device=torch.device(device),
                                **{k: v for k, v in LUT_KW.items()
                                   if k not in ('n_anchor_pairs', 'multi_head_input', 'forward_mode')},
                                light_forward_mode=LUT_KW['forward_mode']))
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
        # LayerNorm is applied to the BLOCK OUTPUT rather than pre-block: the failure being fixed is
        # an unbounded output (the margin score scales with ||h||), so normalising what comes OUT is
        # what removes it. A pre-block norm would fix the input scale but leave the output free to grow.
        self.lns = (nn.ModuleList([nn.LayerNorm(width) for _ in range(self.n_blocks)])
                    if self.block_norm == 'layernorm' and self.n_blocks else None)
        # Assigned None when off, so NO submodule is registered and no parameter name changes. That is
        # deliberate: inserting a module into an existing container is what renamed the FFN's second
        # Linear and broke every pre-existing checkpoint. A test below asserts the state_dict keys are
        # identical with the flag off.
        self.final_ln = nn.LayerNorm(width) if final_norm else None
        self.to(device)

    def encode_depth(self, h, upto):
        for i in range(upto):
            h = self.block(i, h)
        return h

    def raw_block(self, i, h):
        """The block's own output, before any skip or scaling."""
        if self.kind == 'lut':
            return self.blocks[i](h)
        if self.kind == 'mlp':
            if getattr(self, 'hidden', 0):
                return self.blocks[i](h)
            return F.linear(torch.tanh(h), self.blocks[i])
        return torch.zeros_like(h)

    def block(self, i, h):
        if self.kind == 'linear' or self.n_blocks == 0:
            return h
        # PRE-norm normalises the block's INPUT and leaves its output free. For a margin-score LUT,
        # whose per-table score is proportional to ||h||, that is the weaker of the two placements:
        # it bounds what the block reads, not what it emits, so the LAST block's output still reaches
        # the decoder unnormalised. Post-norm (the default) is what the divergence fix used.
        if self.block_norm == 'layernorm' and self.norm_position == 'pre':
            u = self.ai * self.raw_block(i, self.lns[i](h))
            return h + u if self.residual else u
        u = self.ai * self.raw_block(i, h)
        if self.block_norm == 'layernorm':
            return self.lns[i](h + u) if self.residual else self.lns[i](u)
        if self.gain_norm:
            # Hold each block's GAIN at 1 by rescaling its output to its input's RMS, per sample. A plain
            # sequential stack has nothing anchoring its scale: the calibrated init sets gain 1 at step 0,
            # training moves it, and the drift compounds over 2L blocks until the run explodes (measured:
            # gain reaches 6.6 and MSE 4e4 by step 300 at lr 1e-3). This is the smallest intervention
            # that removes that failure -- zero parameters, and unlike a LayerNorm it matches the input's
            # own scale rather than a fixed one, so per-sample brightness is not destroyed.
            scale = h.pow(2).mean(-1, keepdim=True).sqrt() / u.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-9)
            u = u * scale
        return h + u if self.residual else u

    @torch.no_grad()
    def out_params(self, i):
        """The parameters whose scale sets a block's output magnitude, for calibration."""
        if self.kind == 'lut':
            return [inner_lut(self.blocks[i]).tables]
        if self.kind == 'mlp':
            if getattr(self, 'hidden', 0):
                return [self.blocks[i][2].weight]
            return [self.blocks[i]]
        return []

    @torch.no_grad()
    def calibrate_init(self, x, verbose=True):
        """Layer-sequential unit-variance init for the NON-RESIDUAL stack.

        A LightMHL block at its default init emits almost nothing (table entries start at ~1e-3), so
        without a skip the signal would die at the first block; a dense block with N(0,1) weights would
        explode by sqrt(fan_in). Rather than hand-pick an analytic constant per block kind, each block is
        forwarded on one real batch and its output-side parameters are rescaled so the block preserves
        the RMS of its input. That is the same rule for every arm, which is what makes the comparison
        between them fair, and it is reported rather than assumed."""
        h = self.enc(x)
        rms = lambda t: float(t.pow(2).mean().sqrt())            # noqa: E731
        trace = [('enc', rms(h))]
        for i in range(self.n_blocks):
            target = rms(h)
            u = self.raw_block(i, h)
            got = rms(u)
            scale = target / max(got, 1e-12)
            for p_ in self.out_params(i):
                p_.mul_(scale)
            h = self.block(i, h)
            trace.append((f'block{i}', rms(h)))
            if verbose:
                print(f'    calibrate block {i}: out RMS {got:.4e} -> target {target:.4e} '
                      f'(x{scale:.3f}), stream RMS after {rms(h):.4f}', flush=True)
        return trace

    def forward(self, x):
        h = self.enc(x)
        for i in range(self.n_blocks):
            h = self.block(i, h)
        # One more norm between the last block and the output Linear. With pre-block norms the last
        # block's output is the ONE tensor in the stack that nothing normalises before it leaves, which
        # is where the margin score's unbounded output would escape; this closes that gap.
        if self.final_ln is not None:
            h = self.final_ln(h)
        return self.dec(h)


# ------------------------------------------------------------------------------- diagnostics --------
def inner_lut(block):
    """The LightMultiHeadLUT itself, whether the block is one directly or a CompressionMHL wrapping one.

    CompressionMHL keeps it as `lut_light`, so every probe that reaches into anchors, tau or tables has
    to go through here rather than assume the block IS the LUT.
    """
    return getattr(block, 'lut_light', block)


@torch.no_grad()
def lut_stats(model, x):
    """Per-block smallest-margin median, learned tau, and the addresses (for a flip rate)."""
    if model.kind != 'lut':
        return {}, None
    h = model.enc(x)
    mm, taus, addr, ssum, nin, nout = [], [], [], [], [], []
    for i in range(model.n_blocks):
        blk = model.blocks[i]
        lut = inner_lut(blk)
        # what the LUT actually reads: CompressionMHL's compress runs before the anchors are taken, and
        # the pre-norm (when on) before that, so the margins below must be measured on the SAME tensor
        # the layer addresses with, not on the block input.
        z = h
        if model.block_norm == 'layernorm' and model.norm_position == 'pre':
            z = model.lns[i](z)
        if hasattr(blk, 'compress'):
            z = blk.compress(z)
        d = z[:, lut.anchor_a] - z[:, lut.anchor_b]
        mm.append(float(d.abs().min(-1).values.median()))
        taus.append(float(lut.read_tau))
        # summed confidence score across tables -- the quantity that grows with ||h|| under
        # confidence_form='margin', i.e. the direct read on the amplification we diagnosed. Taken from
        # the layer's own confidence_score() so it cannot drift from the forward's definition.
        ssum.append(float(lut.confidence_score(d).sum(-1).mean()))
        addr.append(((d > 0).to(torch.int64) * lut.powers.view(1, 1, -1)).sum(-1))
        nin.append(float(h.norm(dim=-1).mean()))
        h = model.block(i, h)
        nout.append(float(h.norm(dim=-1).mean()))
    return {'m_min_mean': sum(mm) / len(mm), 'tau_mean': sum(taus) / len(taus),
            'score_sum_mean': sum(ssum) / len(ssum), 'm_min': mm, 'tau': taus,
            'score_sum': ssum, 'norm_in': nin, 'norm_out': nout}, addr


@torch.no_grad()
def branch_ratios(model, x):
    """Per block, the quantity that means something for that stack's topology.

    RESIDUAL mode: ||a_i block(h)|| / ||h||, the size of the update against the stream it is added to --
    how far the model has departed from the linear autoencoder that its identity path gives it for free.

    NON-RESIDUAL mode: ||block(h)|| / ||h||, the block's GAIN. There is no identity path to depart from,
    so departure is not defined; what matters instead is whether the stack preserves the signal. A gain
    far from 1 compounding over 2L blocks is how a plain sequential stack explodes or collapses, and
    that is what this then measures.
    """
    if model.n_blocks == 0 or model.kind == 'linear':
        return []
    h = model.enc(x)
    out = []
    for i in range(model.n_blocks):
        nh = model.block(i, h)
        num = (nh - h) if model.residual else nh
        out.append(float(num.norm() / max(float(h.norm()), 1e-12)))
        h = nh
    return out


@torch.no_grad()
def flip_rate(prev, cur):
    """Fraction of (sample, table) address slots that changed since the previous probe."""
    if prev is None or cur is None:
        return float('nan')
    return float(sum(float((a != b).float().mean()) for a, b in zip(prev, cur)) / len(cur))


def evaluate(model, x, bs=4096):
    """Mean-squared error per PIXEL, averaged over pixels and samples, in standardised units.

    ALWAYS in eval mode, and always on the tensor as given -- augmentation is applied in the training
    loop and never here, so the held-out number is un-augmented by construction. At the settings this
    file uses nothing in the stack is mode-dependent (no dropout, and LayerNorm is identical in both),
    so this changes no existing result; it is here so that adding a mode-dependent module later cannot
    silently corrupt the curve, which is exactly what happened on the ViT line.
    """
    was_training = model.training
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            b = x[i:i + bs]
            tot += float((model(b) - b).pow(2).sum())
            n += b.numel()
    if was_training:
        model.train()
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
    ap.add_argument('--no-residual', action='store_true',
                    help='plain sequential stack h = block(h). Drops the residual a_i scaling and '
                         'calibrates each block to preserve its input RMS instead.')
    ap.add_argument('--block-norm', default='none', choices=['none', 'gain', 'layernorm'],
                    help="none: nothing anchors the block scale. gain: parameter-free per-sample RMS "
                         "rescale of the block output to its input's RMS. layernorm: nn.LayerNorm with "
                         "learnable affine on the block output.")
    ap.add_argument('--gain-norm', action='store_true',
                    help='hold each block gain at 1 by matching its output RMS to its input RMS, per '
                         'sample. Parameter-free; intended for the non-residual stack, which otherwise '
                         'has nothing anchoring its scale.')
    ap.add_argument('--lut-impl', default='light', choices=['light', 'compression'],
                    help="'compression' builds the block as CompressionMHL wrapping the same light "
                         'LUT, with both inner dims at the -1 no-projection sentinel')
    ap.add_argument('--norm-position', default='post', choices=['post', 'pre'],
                    help='where --block-norm layernorm sits. post (default) normalises the block '
                         "OUTPUT, which is what the margin score's unbounded output needed; pre "
                         'normalises its input and leaves the last block output free')
    ap.add_argument('--n-blocks', type=int, default=0,
                    help='explicit block count; 0 keeps the 2*depth_L convention')
    ap.add_argument('--augment', action='store_true',
                    help='TRAIN-ONLY random horizontal flip (p=0.5) and random translation of up to '
                         '--aug-pad px, using vit_autoencoder.augment so the two lines share one '
                         'implementation. The augmented image is BOTH input and target. Off by default.')
    ap.add_argument('--aug-pad', type=int, default=2, help='translation range in px, +/- this many')
    ap.add_argument('--final-norm', action='store_true',
                    help='one more LayerNorm between the last block and the output Linear. Default OFF '
                         'and, when off, no module is registered, so parameter names are unchanged.')
    ap.add_argument('--inner-in', type=int, default=-1,
                    help="CompressionMHL's inner_in_dim. -1 is the no-compress sentinel (the LUT reads "
                         'the block input directly); a positive value builds a learned '
                         'Linear(width -> inner_in) before the LUT, with no nonlinearity between them.')
    ap.add_argument('--inner-out', type=int, default=-1,
                    help="CompressionMHL's inner_out_dim. -1 is the no-decompress sentinel (the LUT "
                         'emits the full width and decompress is Identity); a positive value adds a '
                         'real Linear decompress. The BLOCK OUTPUT WIDTH is the same either way.')
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
                        hidden=a.hidden, residual=not a.no_residual, gain_norm=a.gain_norm,
                        block_norm=a.block_norm, lut_impl=a.lut_impl, norm_position=a.norm_position,
                        n_blocks=a.n_blocks, inner_out=a.inner_out, inner_in=a.inner_in,
                        final_norm=a.final_norm)
    init_trace = None
    if a.no_residual and model.n_blocks and model.block_norm != 'layernorm':
        print('  non-residual: dropping a_i (now 1.0) and calibrating the init', flush=True)
        init_trace = model.calibrate_init(xtr[:512])
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
        # the augmented image is the TARGET as well as the input: an autoencoder reconstructs what it
        # was fed. Train stream only -- evaluate() never sees this.
        if a.augment:
            bx = augment(bx, 28, a.aug_pad)
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
            # per-block series too, flattened one key per block, so the ledger can show where in the
            # stack a norm or a score is growing rather than only its average over blocks
            for k, v in s.items():
                if isinstance(v, list):
                    row.update({f'lut/{k}_b{bi}': bv for bi, bv in enumerate(v)})
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
               'residual': not a.no_residual, 'a_i': model.ai, 'gain_norm': a.gain_norm,
               'block_norm': model.block_norm,
               'init_rms_trace': init_trace,
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
