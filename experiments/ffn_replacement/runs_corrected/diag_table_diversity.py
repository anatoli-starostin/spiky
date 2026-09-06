"""Is exp_n_0184's LUT redundant rather than under-capacity? Table DIVERSITY, not occupancy.

The hypothesis under test: 75.5M table parameters buy far less function than they look like
they should, because the tables span few directions, are dominated by their mean row, repeat
the same subspace as each other, and sum to an output that a handful of them would reproduce.
Dead cells are explicitly NOT the story -- under Light every addressed row gets gradient.

Everything runs on CPU: the GPU belongs to exp_n_0185.

  (1) effective rank of each 256x48 table (participation ratio and entropy-based)
  (2) mean-row dominance: ||mean_row||^2 vs E||row - mean_row||^2 -- is it a bias term?
  (3) cross-table subspace overlap, stratified same-head / same-layer / cross-layer
  (4) output-level redundancy on REAL validation tokens: how much of a head's summed output
      survives if you keep only the top-k of its 256 table contributions

    python diag_table_diversity.py [run_dir]
"""
import json
import os
import sys

import torch

FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
RC = os.path.join(FR, 'runs_corrected')
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))

RUN = sys.argv[1] if len(sys.argv) > 1 else 'exp_n_0184_B16k_light_bnorm_seed1'
D = os.path.join(RC, RUN)
DEV = 'cpu'
torch.manual_seed(0)


def q(t, ps=(0.0, 0.25, 0.5, 0.75, 1.0)):
    t = t.flatten().float()
    if t.numel() > 400_000:
        t = t[torch.randperm(t.numel())[:400_000]]
    return torch.quantile(t, torch.tensor(ps)).tolist()


def show(name, t, n=None):
    lo, p25, med, p75, hi = q(t)
    cnt = n if n is not None else t.numel()
    print(f'   {name:<34} n={cnt:<7,} min {lo:8.4g}  p25 {p25:8.4g}  med {med:8.4g}  '
          f'p75 {p75:8.4g}  max {hi:8.4g}')


def eff_rank(W):
    """(participation ratio, entropy effective rank) for a batch of matrices [N, r, c]."""
    s = torch.linalg.svdvals(W.float())                      # [N, min(r,c)]
    s2 = s.pow(2)
    pr = s2.sum(-1).pow(2) / s2.pow(2).sum(-1).clamp_min(1e-30)
    p = s2 / s2.sum(-1, keepdim=True).clamp_min(1e-30)
    ent = -(p * p.clamp_min(1e-30).log()).sum(-1)
    return pr, ent.exp(), s


def main():
    cfg = json.load(open(os.path.join(D, 'config.json')))
    sd = torch.load(os.path.join(D, 'checkpoint.pt'), map_location=DEV)
    keys = [k for k in sd if k.endswith('lut_light.tables')]
    print(f'run: {RUN}')
    print(f'VERIFIED FROM CONFIG: depth={cfg["depth"]} H={cfg["lut_n_heads"]} '
          f'tph={cfg["lut_tables_per_head"]} nap={cfg["lut_n_anchor_pairs"]} '
          f'(cells={2**cfg["lut_n_anchor_pairs"]}) d_in={cfg["lut_inner_in_dim"]} '
          f'd_out={cfg["lut_inner_out_dim"]}')
    print(f'VERIFIED FROM CHECKPOINT: {len(keys)} table tensors, '
          f'shape {tuple(sd[keys[0]].shape)}, dtype {sd[keys[0]].dtype}, '
          f'impl={cfg["lut_impl"]}, gate={cfg["lut_confidence_form"]}, '
          f'bf16={cfg["lut_use_bf16"]}')
    n_tab_total = sum(sd[k].shape[0] for k in keys)
    print(f'   => {n_tab_total:,} tables of {sd[keys[0]].shape[1]}x{sd[keys[0]].shape[2]}, '
          f'{sum(sd[k].numel() for k in keys):,} table parameters')

    H, TPH, DOUT = cfg['lut_n_heads'], cfg['lut_tables_per_head'], cfg['lut_inner_out_dim']

    # ---------------- (1) effective rank ----------------
    print('\n' + '=' * 100)
    print('(1) EFFECTIVE RANK of each table\'s 256x48 row matrix   [max possible = 48]')
    print('=' * 100)
    all_pr, all_ent, per_layer = [], [], []
    for li, k in enumerate(keys):
        W = sd[k].float()                                     # [n_tables, cells, d_out]
        pr, ent, _ = eff_rank(W)
        all_pr.append(pr), all_ent.append(ent)
        per_layer.append((li, pr, ent))
        print(f'   layer {li}: participation ratio  med {pr.median():6.3f}  '
              f'[{pr.min():.3f}, {pr.max():.3f}]      entropy eff-rank  med {ent.median():6.3f}'
              f'  [{ent.min():.3f}, {ent.max():.3f}]   ({pr.numel()} tables)')
    PR, ENT = torch.cat(all_pr), torch.cat(all_ent)
    print()
    show('participation ratio (all tables)', PR)
    show('entropy eff-rank (all tables)', ENT)
    print(f'\n   fraction of tables with participation ratio < 4 : '
          f'{(PR < 4).float().mean():.1%}   < 8: {(PR < 8).float().mean():.1%}   '
          f'< 16: {(PR < 16).float().mean():.1%}   (of 48 possible)')

    # ---------------- (2) mean-row dominance ----------------
    print('\n' + '=' * 100)
    print('(2) MEAN-ROW DOMINANCE — is a table just an expensive bias?')
    print('=' * 100)
    ratios = []
    for li, k in enumerate(keys):
        W = sd[k].float()
        mu = W.mean(dim=1)                                    # [n_tables, d_out]
        mu_sq = mu.pow(2).sum(-1)                             # ||mean_row||^2
        var = (W - mu.unsqueeze(1)).pow(2).sum(-1).mean(-1)   # E||row - mean||^2
        r = mu_sq / var.clamp_min(1e-30)
        ratios.append(r)
        print(f'   layer {li}: ||mean||^2 / E||row-mean||^2   med {r.median():9.4g}   '
              f'[{r.min():.4g}, {r.max():.4g}]')
    R = torch.cat(ratios)
    print()
    show('mean/variance ratio (all tables)', R)
    print(f'\n   >1 means the constant part carries more energy than the addressable part.')
    print(f'   fraction of tables with ratio > 1: {(R > 1).float().mean():.1%}   '
          f'> 10: {(R > 10).float().mean():.1%}')

    # ---------------- (3) cross-table subspace overlap ----------------
    print('\n' + '=' * 100)
    print('(3) CROSS-TABLE SUBSPACE OVERLAP  (mean squared cosine of principal angles, top-k)')
    print('=' * 100)
    bases = []
    for k in keys:
        W = sd[k].float()
        Wc = W - W.mean(dim=1, keepdim=True)                  # centre: subspace of variation
        U, S, Vh = torch.linalg.svd(Wc, full_matrices=False)
        bases.append(Vh)                                      # [n_tables, min, d_out]
    B = torch.stack(bases)                                    # [L, n_tables, m, d_out]
    L = B.shape[0]

    def overlap(k_top, pairs):
        vals = []
        for (l1, t1), (l2, t2) in pairs:
            A = B[l1, t1, :k_top]                             # [k, d_out]
            C = B[l2, t2, :k_top]
            s = torch.linalg.svdvals(A @ C.T)
            vals.append((s.pow(2).sum() / k_top).item())       # 1.0 = identical subspace
        return torch.tensor(vals)

    n_pairs = 1500
    strata = {}
    g = torch.Generator().manual_seed(0)
    same_head, diff_head, diff_layer = [], [], []
    for _ in range(n_pairs):
        l = torch.randint(0, L, (1,), generator=g).item()
        h = torch.randint(0, H, (1,), generator=g).item()
        a, b = torch.randint(0, TPH, (2,), generator=g).tolist()
        same_head.append(((l, h * TPH + a), (l, h * TPH + b)))
        h2 = (h + 1 + torch.randint(0, H - 1, (1,), generator=g).item()) % H
        diff_head.append(((l, h * TPH + a), (l, h2 * TPH + b)))
        l2 = (l + 1 + torch.randint(0, L - 1, (1,), generator=g).item()) % L
        diff_layer.append(((l, h * TPH + a), (l2, h2 * TPH + b)))
    strata['same head, same layer'] = same_head
    strata['diff head, same layer'] = diff_head
    strata['diff layer'] = diff_layer
    for k_top in (4, 8):
        print(f'   top-{k_top} subspaces   (1.0 = identical, '
              f'{k_top/DOUT:.3f} = random {k_top}-dim subspaces in R^{DOUT})')
        for name, pr in strata.items():
            v = overlap(k_top, pr)
            print(f'      {name:<26} med {v.median():.4f}   IQR '
                  f'[{v.quantile(0.25):.4f}, {v.quantile(0.75):.4f}]   n={len(pr)}')

    # ---------------- (4) output-level redundancy on real tokens ----------------
    print('\n' + '=' * 100)
    print('(4) OUTPUT-LEVEL REDUNDANCY on REAL validation tokens')
    print('=' * 100)
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    from model_build import build_model
    from spiky.lutorch.fast_multi_head_lut import _confidence_score

    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    ld = tokenizing_distributed_data_loader_bos_bestfit(tok, 2, 512, split='val', device=DEV)
    x_ids, _ = next(iter(ld))
    m = build_model(cfg, tok.get_vocab_size(), device=DEV)
    m.load_state_dict(sd, strict=False)
    m.eval()

    print(f'   {x_ids.numel():,} real val tokens\n')
    print(f'   {"layer":<7}{"align ratio":>13}{"eff rank":>11}{"top-1":>9}{"top-4":>9}'
          f'{"top-16":>9}{"top-64":>9}{"tables for 90%":>16}')
    for li, blk in enumerate(m.blocks):
        ffn = blk.ffn
        rec = {}

        def hook(mod, inp, _out, rec=rec, ffn=ffn):
            z = ffn.compress(inp[0]).view(inp[0].shape[0], ffn.n_heads, ffn.inner_in_dim)
            lut = ffn.lut_light
            B_, Hh, T = z.shape[0], lut.n_heads, lut.tables_per_head
            NAP = lut.n_anchor_pairs
            ia = lut.anchor_a.reshape(1, Hh, T * NAP).expand(B_, Hh, T * NAP)
            ib = lut.anchor_b.reshape(1, Hh, T * NAP).expand(B_, Hh, T * NAP)
            d = (torch.gather(z, 2, ia) - torch.gather(z, 2, ib)).view(B_, Hh, T, NAP)
            idx = ((d > 0).to(torch.int64) * lut.powers.view(1, 1, 1, -1)).sum(-1)
            flat = lut.tables.reshape(Hh * T * lut.table_size, lut.output_dim)
            rows = flat[(idx + lut.table_offset.view(1, Hh, T)).reshape(-1)] \
                .view(B_, Hh, T, lut.output_dim)
            score = _confidence_score(d, lut.confidence_form, lut.confidence_gain)
            rec['C'] = (rows * score.unsqueeze(-1))           # [B, H, T, d_out] contributions

        h = ffn.register_forward_hook(hook)
        with torch.no_grad():
            m(x_ids)
        h.remove()

        C = rec['C']                                          # [N, H, T, d_out]
        N, Hh, T, _ = C.shape
        C = C.reshape(N * Hh, T, C.shape[-1])                 # one (token, head) per row-block
        tot = C.sum(1)                                        # the head's actual output
        num = tot.pow(2).sum(-1)                              # ||sum C_t||^2
        den = C.pow(2).sum(-1).sum(-1)                        # sum_t ||C_t||^2
        align = (num / den.clamp_min(1e-30))                  # 1 = orthogonal, T = identical
        # effective rank of the contribution set per (token, head)
        s = torch.linalg.svdvals(C)
        s2 = s.pow(2)
        er = s2.sum(-1).pow(2) / s2.pow(2).sum(-1).clamp_min(1e-30)
        # keep only the top-k contributions by norm; how much of the output survives?
        mag = C.norm(dim=-1)                                  # [NH, T]
        order = mag.argsort(dim=1, descending=True)
        Cs = torch.gather(C, 1, order.unsqueeze(-1).expand_as(C))
        cum = Cs.cumsum(dim=1)
        frac = (cum.norm(dim=-1) / tot.norm(dim=-1, keepdim=True).clamp_min(1e-30))
        need90 = (frac >= 0.9).float().argmax(dim=1) + 1
        print(f'   {li:<7}{align.median():>13.3f}{er.median():>11.2f}'
              f'{frac[:, 0].median():>9.3f}{frac[:, 3].median():>9.3f}'
              f'{frac[:, 15].median():>9.3f}{frac[:, 63].median():>9.3f}'
              f'{need90.float().median():>16.0f}')
        del rec, C

    print(f'\n   align ratio: 1 = the {TPH} contributions are mutually orthogonal, '
          f'{TPH} = all identical.')
    print(f'   "top-k" is the fraction of the head output norm recovered by the k '
          f'largest contributions.')
    print(f'   "tables for 90%" is how many of {TPH} are needed to reach 90% of the output.')


if __name__ == '__main__':
    main()
