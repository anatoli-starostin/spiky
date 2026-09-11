"""Boundary jumps of TRAINED Gen-3 LightMHL layers on real activations. Eval only; nothing is trained.

Checkpoints (local, 16K steps, margin score, nap 8, inner 48->48, H=4, tph=128 -- the same
math as the paper config, different head/table split):
  exp_g_0193_B16k_light_margin_tph128_noznorm_seed1   read_top_n = 1
  exp_g_0194_B16k_light_margin_blend_n2_tau_auto_seed1 read_top_n = 2

For every layer, real val tokens go through the model and the LUT input z [N, H, 48] and head
output y [N, H, 48] are captured. For sampled (token, head, table, bit j) the jump the output
would take if u_j crossed zero with the table's other margins as observed is, per table:
  n=1: dy_h = s_t|_{u_j=0} * (W_t[c^(j)] - W_t[c])        (s_t evaluated with u_j set to 0)
  n=2: at a bit flip the blend is continuous (the two cells swap weights 1/2 <-> 1/2); the jump is
       at an ARGMIN SWITCH of the two smallest margins j1, j2 (tie at m = |u_j1|):
       dy_h = s_t * w1 * (W_t[c^(j2)] - W_t[c^(j1)]),  w1 = sigmoid(-2m/tau)
It is reported relative to ||y_h|| (this head's output at that token) and, pushed through the
layer's decompress, relative to the whole FFN output ||decompress(y)|| at that token.
Also reported: how close real tokens sit to a boundary (|u_j| quantiles).

    python continuity_probe_trained.py [<run_dir> ...]     (default: the two runs above)

The boundary score s_t|_{u_j=0} uses each layer's OWN confidence_form / confidence_gain /
sharp_margin_gamma (identical to the original 'margin' call for margin runs), and read_top_n is
read from each run's config.
"""
import json
import math
import os
import statistics as st
import sys

os.environ.setdefault('LUT_DISABLE_COMPILE', '1')
REPO = '/home/astarostin/projects/spiky'
FR = os.path.join(REPO, 'experiments', 'ffn_replacement')
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.join(FR, 'distill'))
import torch                                                          # noqa: E402
import distill_ffn as D                                               # noqa: E402  (loader helpers)
from model_build import build_model                                   # noqa: E402
from spiky.lutorch.fast_multi_head_lut import _confidence_score       # noqa: E402

RUNS = sys.argv[1:] or ['exp_g_0193_B16k_light_margin_tph128_noznorm_seed1',
                        'exp_g_0194_B16k_light_margin_blend_n2_tau_auto_seed1']
ROWS, SAMPLES_PER_LAYER = 8, 4000
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
g = torch.Generator().manual_seed(0)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def summ(xs):
    return (f'median {st.median(xs):.3g}  p10 {q(xs, .1):.3g}  p90 {q(xs, .9):.3g}  '
            f'p99 {q(xs, .99):.3g}')


tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
vocab = tok.get_vocab_size()

for run in RUNS:
    rd = os.path.join(FR, 'runs_corrected', run)
    if not os.path.isdir(rd):
        rd = os.path.join(FR, 'runs', run)
    cfg = json.load(open(os.path.join(rd, 'config.json')))
    n = int(cfg.get('lut_read_top_n', 1))
    model = build_model(cfg, vocab, device=dev)
    sd = torch.load(os.path.join(rd, 'checkpoint.pt'), map_location=dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    loader = D.tokenizing_distributed_data_loader_bos_bestfit(tok, 16, cfg['seq_len'], split='val', device=dev)
    idx = D.take_rows(loader, ROWS, skip_rows=12)
    got = {}
    hooks = [blk.ffn.lut_light.register_forward_hook(
        lambda mod, inp, out, _i=i: got.__setitem__(_i, (inp[0].detach(), out.detach())))
        for i, blk in enumerate(model.blocks)]
    with torch.no_grad():
        x = model.tok_emb(idx)
        for blk in model.blocks:
            x = blk(x, model.rope.cos, model.rope.sin)
    for h_ in hooks:
        h_.remove()

    print('=' * 100)
    _l0 = model.blocks[0].ffn.lut_light
    print(f'{run}  (form={_l0.confidence_form} gain={_l0.confidence_gain} gamma={_l0.sharp_margin_gamma}; '
          f'read_top_n={n}; missing={len(missing)} unexpected={len(unexpected)} keys; '
          f'{idx.numel():,} val tokens)')
    all_rel_h, all_rel_ffn, all_u = [], [], []
    for li, blk in enumerate(model.blocks):
        lut, dec = blk.ffn.lut_light, blk.ffn.decompress
        H, T, NAP, Dd = lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs, lut.output_dim
        K = 1 << NAP
        z, y = got[li]
        z, y = z.double(), y.double()
        Nt = z.shape[0]
        W = lut.tables.detach().double().view(H, T, K, Dd)
        a = lut.anchor_a.view(1, H, T * NAP).expand(Nt, H, T * NAP)
        b = lut.anchor_b.view(1, H, T * NAP).expand(Nt, H, T * NAP)
        d = (torch.gather(z, 2, a) - torch.gather(z, 2, b)).view(Nt, H, T, NAP)
        powers = 2 ** torch.arange(NAP - 1, -1, -1, device=d.device)
        c = ((d > 0).long() * powers).sum(-1)                                # [N, H, T]
        ffn_out = dec(y.reshape(Nt, H * Dd).to(dec.weight.dtype)).double()  # [N, E]
        Wdec = dec.weight.detach().double()                                  # [E, H*Dd]
        tau = lut.log_tau.exp().item() if n == 2 else None
        rel_h, rel_ffn = [], []
        ti = torch.randint(Nt, (SAMPLES_PER_LAYER,), generator=g)
        hi = torch.randint(H, (SAMPLES_PER_LAYER,), generator=g)
        tt = torch.randint(T, (SAMPLES_PER_LAYER,), generator=g)
        for s_i in range(SAMPLES_PER_LAYER):
            i, h, t = int(ti[s_i]), int(hi[s_i]), int(tt[s_i])
            dm = d[i, h, t].clone()
            if n == 1:
                j = int(torch.randint(NAP, (1,), generator=g))
                dm[j] = 0.0
                s0 = _confidence_score(dm.view(1, 1, NAP), lut.confidence_form, lut.confidence_gain,
                                       lut.sharp_margin_gamma).item()
                cc = int(c[i, h, t])
                delta = s0 * (W[h, t, cc ^ (1 << (NAP - 1 - j))] - W[h, t, cc])
            else:
                order = dm.abs().argsort()
                j1, j2 = int(order[0]), int(order[1])
                mtie = dm[j1].abs().item()
                dm[j2] = math.copysign(mtie, dm[j2].item())
                s0 = _confidence_score(dm.view(1, 1, NAP), lut.confidence_form, lut.confidence_gain,
                                       lut.sharp_margin_gamma).item()
                w1 = 1.0 / (1.0 + math.exp(2.0 * mtie / tau))
                cc = int(c[i, h, t])
                delta = s0 * w1 * (W[h, t, cc ^ (1 << (NAP - 1 - j2))] - W[h, t, cc ^ (1 << (NAP - 1 - j1))])
            rel_h.append((delta.norm() / y[i, h].norm()).item())
            dffn = Wdec[:, h * Dd:(h + 1) * Dd] @ delta
            rel_ffn.append((dffn.norm() / ffn_out[i].norm()).item())
        u_abs = d.abs().flatten()
        sub = u_abs[torch.randint(u_abs.numel(), (200000,), generator=g).to(u_abs.device)].tolist()
        all_rel_h += rel_h
        all_rel_ffn += rel_ffn
        all_u += sub
        extra = f'  tau={tau:.4f}' if tau else ''
        print(f'  L{li}{extra}: jump/||y_h|| {summ(rel_h)}')
        print(f'      jump/||FFN out|| {summ(rel_ffn)}   |u_j| {summ(sub)}   '
              f'frac |u_j|<1e-2: {sum(v < 1e-2 for v in sub) / len(sub):.4f}')
    print(f'  ALL LAYERS: jump/||y_h|| {summ(all_rel_h)}')
    print(f'              jump/||FFN out|| {summ(all_rel_ffn)}')
    print(f'              |u_j| {summ(all_u)}   frac |u_j|<1e-2: {sum(v < 1e-2 for v in all_u) / len(all_u):.4f}')
    del model, sd, got
    torch.cuda.empty_cache()
