"""Probe A — top-n soft read-out at EVAL ONLY, on an already-trained checkpoint.

WHY. §II.6(2) of LIGHTMHL_SURVEY.md names the layer's DISCONTINUITY as the leading
suspect for the +0.019 residual at 48K: crossing a hyperplane z[a]=z[b] swaps the whole
gathered row for an unrelated one, and the confidence score does not damp the jump (at
NAP=8 with the other margins at 0.5, `margin` still scores 0.326 when one margin is
exactly zero). A dense FFN is Lipschitz. This probe asks what that discontinuity costs,
WITHOUT retraining: read n cells instead of 1 and blend them.

WHAT THE BLEND IS. It is not an invention -- it is the paper's own Eq 13 with N > 1.
LookupFFN's score is the single dominant term of a softmax over all 2^K codes,
    w(b) proportional to exp(<z, b>) = exp(sum_j b_j d_j),   b in {-1,+1}^K
whose argmax is b* = sign(d) with value sum_j |d_j|. Flipping bit j gives
sum_j|d_j| - 2|d_j|, so the 1-flip neighbour along j carries relative weight

    u_j = exp(-2 |d_j|)                              (u_0 = 1 for the argmax cell)

Cheapest neighbours are therefore the SMALLEST-margin bits, which is exactly the
"nearest cells by margin" reading. We take the argmax plus the (n-1) smallest-margin
flips, normalise their weights to sum to 1, and gather that convex combination.

THE CONTROL, and why it is exact rather than approximate. The per-table confidence score
s_t multiplies the whole blend and the weights are normalised, so at n=1 the weight is
u_0/u_0 = 1 and the layer computes bit-for-bit what it computed before. n=1 MUST
reproduce the run's recorded bpb. If it does not, the probe is wrong and no n>1 number
from it means anything.

Eval only: no gradient, no training-code change, no retraining. The training path in
light_multi_head_lut.py is untouched -- this monkey-patches the module's forward for the
duration of a scoring pass.

    python probe_soft_readout.py --run exp_g_0190_... --n 1 2 3 4
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, '..', 'tools')
sys.path.insert(0, os.path.abspath(TOOLS))
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir                       # noqa: E402
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes  # noqa: E402

from model_build import build_model                            # noqa: E402
from fixed_eval import evaluate_bpb_fixed, eval_config          # noqa: E402

from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT   # noqa: E402
from spiky.lutorch.fast_multi_head_lut import _confidence_score    # noqa: E402


def _soft_forward(self, x, n_read: int, tau: float = 1.0, rand_nb: bool = False):
    """LightMultiHeadLUT.forward with a top-`n_read` blended read-out.

    Shapes follow the module's own `_forward_multi_head` exactly; the ONLY change is that
    the single gathered row becomes a normalised convex combination of `n_read` rows.
    Deliberately always takes the torch path (never `_fused_eval`), because the native
    kernel returns only the packed index and throws the margins away -- and the margins
    are what choose the neighbours.
    """
    if self.multi_head_input:
        B, H, T, K = x.shape[0], self.n_heads, self.tables_per_head, self.n_anchor_pairs
        idx_a = self.anchor_a.reshape(1, H, T * K).expand(B, H, T * K)
        idx_b = self.anchor_b.reshape(1, H, T * K).expand(B, H, T * K)
        d = (torch.gather(x, 2, idx_a) - torch.gather(x, 2, idx_b)).view(B, H, T, K)
        bits = (d.detach() > 0).to(torch.int64)                       # [B,H,T,K]
        powers = self.powers.view(1, 1, 1, -1)
        index = (bits * powers).sum(dim=-1)                           # [B,H,T]
        offset = self.table_offset.view(1, H, T)
        flat = self.tables.reshape(H * T * self.table_size, self.output_dim)
        n_bags, bag = B * H, T
    else:
        B, K = x.shape[0], self.n_anchor_pairs
        d = (x[:, self.anchor_a] - x[:, self.anchor_b])               # [B,T,K]
        bits = (d.detach() > 0).to(torch.int64)
        powers = self.powers.view(1, 1, -1)
        index = (bits * powers).sum(dim=-1)                           # [B,T]
        offset = self.table_offset.view(1, -1)
        flat = self.tables.reshape(self.n_tables * self.table_size, self.output_dim)
        n_bags, bag = B, self.n_tables

    score = _confidence_score(d, self.confidence_form, self.confidence_gain)
    m = d.detach().abs()                                              # [..., K]

    # --- the argmax cell, weight u_0 = 1 -------------------------------------------
    idx_list = [index]
    w_list = [torch.ones_like(score)]

    if n_read > 1:
        # the (n_read-1) cheapest bit flips = the smallest margins
        mv, mj = torch.topk(m, k=n_read - 1, dim=-1, largest=False)   # [..., n-1]
        if rand_nb:
            # FALSIFICATION CONTROL: keep the same weights but flip a RANDOM bit instead
            # of the smallest-margin one. If this helps as much, the gain is generic
            # output smoothing, not the cell boundary the hypothesis is about.
            mj = torch.randint_like(mj, 0, m.shape[-1])
        for i in range(n_read - 1):
            j = mj[..., i]                                            # which bit to flip
            pw = self.powers[j]                                       # 2^(K-1-j)
            bit = torch.gather(bits, -1, j.unsqueeze(-1)).squeeze(-1)
            # bit==1 -> clear it (subtract); bit==0 -> set it (add)
            idx_list.append(index + pw * (1 - 2 * bit))
            w_list.append(torch.exp(-2.0 * mv[..., i] / tau))

    wsum = torch.stack(w_list, 0).sum(0)                              # normaliser
    out = None
    for idx_i, w_i in zip(idx_list, w_list):
        flat_idx = (idx_i + offset).reshape(-1)
        psw = (score * (w_i / wsum)).reshape(-1).to(flat.dtype)
        offsets = torch.arange(n_bags, device=flat.device, dtype=torch.long) * bag
        part = F.embedding_bag(flat_idx, flat, offsets=offsets, mode='sum',
                               per_sample_weights=psw)
        out = part if out is None else out + part
    if self.multi_head_input:
        return out.view(B, H, self.output_dim)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--n', type=int, nargs='+', default=[1, 2, 3, 4])
    ap.add_argument('--tau', type=float, nargs='+', default=[1.0],
                    help='temperature on the neighbour weight exp(-2m/tau); '
                         'tau=1 is the plain softmax-over-codes weight')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None, help='write results json here')
    ap.add_argument('--random-neighbour', action='store_true',
                    help='falsification control: blend a RANDOM cell, not the nearest')
    a = ap.parse_args()

    run_dir = a.run if os.path.isabs(a.run) else os.path.join(HERE, a.run)
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    ck = a.checkpoint or os.path.join(run_dir, 'checkpoint.pt')
    assert os.path.exists(ck), f'checkpoint not found: {ck}'
    ec = eval_config(cfg)

    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    token_bytes = get_token_bytes(device=a.device)
    model = build_model(cfg, tok.get_vocab_size(), device=a.device)
    sd = torch.load(ck, map_location=a.device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    print(f'loaded {ck}: missing={len(missing)} unexpected={len(unexpected)} '
          f'params={sum(p.numel() for p in model.parameters()):,}')

    res = {'run': cfg['exp_name'], 'checkpoint': ck,
           'load_missing_keys': len(missing), 'load_unexpected_keys': len(unexpected),
           'eval': ec, 'scores': {}}

    # --- control 0: the UNPATCHED layer, i.e. the module exactly as it trains/evals ---
    base = evaluate_bpb_fixed(model, tok, token_bytes, cfg['seq_len'], a.device, **ec)
    res['baseline_unpatched'] = base
    print(f'\n[control 0] unpatched forward           bpb = {base:.8f}')

    orig = LightMultiHeadLUT.forward
    try:
        for n in a.n:
            for tau in (a.tau if n > 1 else [1.0]):
                LightMultiHeadLUT.forward = (
                    lambda self, x, _n=n, _t=tau: _soft_forward(
                        self, x, _n, _t, a.random_neighbour))
                b = evaluate_bpb_fixed(model, tok, token_bytes, cfg['seq_len'],
                                       a.device, **ec)
                res['scores'][f'{n}' if n == 1 else f'{n}@tau{tau:g}'] = b
                tag = '  <-- CONTROL, must match [control 0]' if n == 1 else ''
                lbl = f'n={n}' if n == 1 else f'n={n} tau={tau:g}'
                print(f'[{lbl:>14}] blended read-out   bpb = {b:.8f}   '
                      f'delta vs unpatched {b - base:+.8f}{tag}')
    finally:
        LightMultiHeadLUT.forward = orig

    if '1' in res['scores']:
        drift = abs(res['scores']['1'] - base)
        res['control_n1_abs_drift'] = drift
        res['control_n1_ok'] = bool(drift < 1e-6)
        print(f"\ncontrol: |n=1 - unpatched| = {drift:.3e}  -> "
              f"{'PASS' if drift < 1e-6 else '*** FAIL ***'}")
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=2)
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()


# --- control appendix -----------------------------------------------------------------
# The unpatched module under no_grad takes `_fused_eval` (a native CUDA kernel that
# computes the address AND the score in one pass, never materialising the margins), while
# the patched forward always takes the torch path -- it needs the margins to pick the
# neighbours. So "unpatched vs n=1" mixes two changes: kernel-vs-torch arithmetic and the
# blend. Disabling the kernel isolates them: torch-unpatched vs n=1 must agree to float
# equality, since at n=1 the blend is the identity.
