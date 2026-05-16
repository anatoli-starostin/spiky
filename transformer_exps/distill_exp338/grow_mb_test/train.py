"""3-phase MultiBit grow-redundancy test on layer-0 distill.

Phase 1 — train MultiBit(K=4, in=10, tph=1024, on=32) for 50k steps on
layer_0.pt teacher pairs. Should reach ~0.999 sign_acc.

Phase 2 — grow tph 1024 -> 1280 (+25%, 256 new tables) via
`grow_multibit_lut(new_init_std=1e-5)`. New tables preserve the old
1024-slot state and start with tiny latent (~1e-5). Sign_acc should be
~identical post-grow (new votes ≈ 0).

Phase 3 — continue training 25k steps. Key question: do the new tables'
latents GROW (meaning the model finds use for extra capacity)? Or stay
near 1e-5 (confirming they're redundant)?

Logs per-table latent L2 norm distribution: mean/max for old (0..1024) and
new (1024..1280) buckets. Also sign_acc throughout.

NB: there's no Adam-managed params in this single-LUT stack, so we only
drive the MultiBitPermutationLUTOptimizer. No upstream gradient contamination.
"""
import sys, os, json, time, math, csv, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from spiky.lutorch.multi_bit_permutation_lut import MultiBitPermutationLUT
from spiky.lutorch.multi_bit_permutation_lut_optimizer import MultiBitPermutationLUTOptimizer
from spiky.lutorch.ranking_tools import _canonical_borda_m

from grow_multibit_lut import grow_multibit_lut


DEVICE = 'cuda:0'
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(EXP_DIR, '..', 'data'))
N_OUTPUTS = 32


RANK_T = 0.01  # was 0.1 — sharper loss, closer to hard sign


def pair_soft_sign(v, tri_i, tri_j, T=RANK_T):
    diff = v[:, tri_i] - v[:, tri_j]
    return diff / (T + diff.abs())


def forward_lut(lut, x, borda_m):
    """Single MultiBit LUT -> dominance [B, 1, P] -> Borda project -> [B, E]."""
    out = lut(x)  # [B, 1, P]
    return torch.einsum('bhp,kp->bhk', out, borda_m).squeeze(1)


def evaluate(lut, X, Y_pair_sign, tri_i, tri_j, borda_m, eval_bs=8192, n_eval=32768):
    lut.eval()
    mse_acc = 0.0; sgn_acc = 0.0; count = 0
    with torch.no_grad():
        for s in range(0, n_eval, eval_bs):
            xi = X[s:s+eval_bs]; yi = Y_pair_sign[s:s+eval_bs]
            vp = forward_lut(lut, xi, borda_m)
            ss = pair_soft_sign(vp, tri_i, tri_j)
            mse_acc += F.mse_loss(ss, yi, reduction='sum').item()
            sgn_acc += (ss.sign() == yi).float().sum().item()
            count += yi.numel()
    lut.train()
    return mse_acc / count, sgn_acc / count


def per_table_latent_stats(lut, old_tph=None):
    """Return dict with per-table L2 norm statistics.
    If old_tph given, splits into 'old' (< old_tph) and 'new' buckets.
    """
    # latent_bf16: [n_heads*tph, table_dim, output_nap]
    lat = lut.latent_bf16.float()
    H, tph = lut.n_heads, lut.tph
    lat_per_head = lat.view(H, tph, lut.table_dim, lut.output_nap)
    # per-table L2 norm: sqrt(sum over table_dim * output_nap)
    norms = lat_per_head.reshape(H, tph, -1).norm(dim=-1)  # [H, tph]
    stats = {
        'all_mean': float(norms.mean().item()),
        'all_max':  float(norms.max().item()),
        'all_min':  float(norms.min().item()),
    }
    if old_tph is not None and 0 < old_tph < tph:
        old_n = norms[:, :old_tph]
        new_n = norms[:, old_tph:]
        stats.update({
            'old_mean': float(old_n.mean().item()),
            'old_max':  float(old_n.max().item()),
            'old_min':  float(old_n.min().item()),
            'new_mean': float(new_n.mean().item()),
            'new_max':  float(new_n.max().item()),
            'new_min':  float(new_n.min().item()),
        })
    return stats


def lr_scale_factory(warmup, total):
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    return fn


def train_phase(lut, X, Y_pair_sign, tri_i, tri_j, borda_m,
                steps, batch_size, lr, log_every, csv_w, csv_f,
                phase_label, old_tph=None, stats_csv_w=None, stats_csv_f=None):
    warmup = max(1, int(0.1 * steps))
    lr_scale = lr_scale_factory(warmup, steps)
    mb_opt = MultiBitPermutationLUTOptimizer(
        [lut], lr=lr, beta1=0.9, beta2=0.999, lr_schedule_fn=lr_scale,
    )
    N = X.shape[0]
    for step in range(steps):
        idx = torch.randint(0, N, (batch_size,), device=DEVICE)
        x = X[idx].detach().requires_grad_(True)
        v_pred = forward_lut(lut, x, borda_m)
        soft_s = pair_soft_sign(v_pred, tri_i, tri_j)
        target = Y_pair_sign[idx]
        loss = F.mse_loss(soft_s, target)

        mb_opt.zero_grad()
        loss.backward()
        mb_opt.step()

        if (step + 1) % log_every == 0 or step == 0:
            mse_eval, sgn_eval = evaluate(lut, X, Y_pair_sign, tri_i, tri_j, borda_m)
            print(f'  [{phase_label}] step {step+1:>6}/{steps} | '
                  f'train mse {loss.item():.4f} | eval mse {mse_eval:.4f} | '
                  f'sign_acc {sgn_eval:.4f}')
            csv_w.writerow([phase_label, step+1, f'{loss.item():.6f}',
                            f'{mse_eval:.6f}', f'{sgn_eval:.6f}'])
            csv_f.flush()
            if stats_csv_w is not None:
                s = per_table_latent_stats(lut, old_tph=old_tph)
                row = [phase_label, step+1, f'{s["all_mean"]:.6e}', f'{s["all_max"]:.6e}',
                       f'{s["all_min"]:.6e}',
                       f'{s.get("old_mean", -1):.6e}', f'{s.get("old_max", -1):.6e}',
                       f'{s.get("new_mean", -1):.6e}', f'{s.get("new_max", -1):.6e}']
                stats_csv_w.writerow(row)
                stats_csv_f.flush()
    mb_opt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--phase1-steps', type=int, default=25000)
    ap.add_argument('--phase3-steps', type=int, default=25000)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--log-every', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    # Config matching mb_a3_in10_tph01024_on32_033554k
    ap.add_argument('--input-nap', type=int, default=10)
    ap.add_argument('--output-nap', type=int, default=32)
    ap.add_argument('--tph-phase1', type=int, default=1024)
    ap.add_argument('--tph-phase3', type=int, default=1280)  # +25%
    ap.add_argument('--bit-width', type=int, default=4)
    ap.add_argument('--init-std', type=float, default=1e-3)
    ap.add_argument('--new-init-std', type=float, default=1e-5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Data
    data = torch.load(os.path.join(DATA_DIR, f'layer_{args.layer}.pt'), weights_only=False)
    X = data['inputs'].to(DEVICE)             # [N, n_inputs]
    Y_dom = data['outputs'].to(DEVICE)        # [N, P]
    N = X.shape[0]
    n_inputs = X.shape[1]
    borda_m = _canonical_borda_m(N_OUTPUTS).to(DEVICE)
    Y_borda = torch.einsum('bp,kp->bk', Y_dom, borda_m)
    tri_i, tri_j = torch.triu_indices(N_OUTPUTS, N_OUTPUTS, offset=1).unbind(0)
    tri_i, tri_j = tri_i.to(DEVICE), tri_j.to(DEVICE)
    Y_pair_sign = (Y_borda[:, tri_i] - Y_borda[:, tri_j]).sign()
    print(f'Data: X {tuple(X.shape)}, Y_pair_sign {tuple(Y_pair_sign.shape)}')

    # CSVs
    metrics_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
    metrics_w = csv.writer(metrics_f)
    metrics_w.writerow(['phase', 'step', 'train_mse', 'eval_mse', 'sign_acc'])
    stats_f = open(os.path.join(EXP_DIR, 'latent_stats.csv'), 'w', newline='')
    stats_w = csv.writer(stats_f)
    stats_w.writerow(['phase', 'step', 'all_mean', 'all_max', 'all_min',
                      'old_mean', 'old_max', 'new_mean', 'new_max'])

    # Phase 1: train original model
    print(f'\n=== PHASE 1: training MultiBit K={args.bit_width} in={args.input_nap} '
          f'tph={args.tph_phase1} on={args.output_nap} for {args.phase1_steps} steps ===')
    lut = MultiBitPermutationLUT(
        n_inputs=n_inputs, n_outputs=N_OUTPUTS, n_heads=1,
        input_nap=args.input_nap, output_nap=args.output_nap,
        tph=args.tph_phase1, bit_width=args.bit_width,
        random_seed=args.seed + 1000,
        initial_weights_noise=args.init_std,
        device=DEVICE,
    ).to(DEVICE)
    print(f'  bit params: {args.tph_phase1 * (1 << args.input_nap) * args.output_nap:,} '
          f'(× K={args.bit_width} bits)')
    t0 = time.time()
    train_phase(lut, X, Y_pair_sign, tri_i, tri_j, borda_m,
                steps=args.phase1_steps, batch_size=args.batch_size,
                lr=args.lr, log_every=args.log_every,
                csv_w=metrics_w, csv_f=metrics_f, phase_label='phase1',
                old_tph=None, stats_csv_w=stats_w, stats_csv_f=stats_f)
    mse1, sgn1 = evaluate(lut, X, Y_pair_sign, tri_i, tri_j, borda_m)
    print(f'  END PHASE 1: mse={mse1:.4f} sign_acc={sgn1:.4f} '
          f'({(time.time()-t0)/60:.1f} min)')

    # Phase 2: grow
    print(f'\n=== PHASE 2: grow {args.tph_phase1} -> {args.tph_phase3} '
          f'(+{args.tph_phase3 - args.tph_phase1}, init_std={args.new_init_std}) ===')
    lut_grown = grow_multibit_lut(lut, new_tph=args.tph_phase3,
                                  seed=args.seed + 2000, new_init_std=args.new_init_std)
    del lut
    torch.cuda.empty_cache()
    # Sanity: forward on grown LUT should be close to the original (new votes ≈ 0).
    mse2, sgn2 = evaluate(lut_grown, X, Y_pair_sign, tri_i, tri_j, borda_m)
    print(f'  POST-GROW: mse={mse2:.4f} sign_acc={sgn2:.4f}')
    metrics_w.writerow(['grow', 0, 'N/A', f'{mse2:.6f}', f'{sgn2:.6f}'])
    metrics_f.flush()
    s = per_table_latent_stats(lut_grown, old_tph=args.tph_phase1)
    print(f'  latent stats: old_mean={s["old_mean"]:.4e} old_max={s["old_max"]:.4e}  '
          f'new_mean={s["new_mean"]:.4e} new_max={s["new_max"]:.4e}')
    stats_w.writerow(['grow', 0, f'{s["all_mean"]:.6e}', f'{s["all_max"]:.6e}',
                      f'{s["all_min"]:.6e}',
                      f'{s["old_mean"]:.6e}', f'{s["old_max"]:.6e}',
                      f'{s["new_mean"]:.6e}', f'{s["new_max"]:.6e}'])
    stats_f.flush()

    # Phase 3: continue training with new tables
    print(f'\n=== PHASE 3: continue training for {args.phase3_steps} steps '
          f'(tracking old vs new table latent growth) ===')
    t3 = time.time()
    train_phase(lut_grown, X, Y_pair_sign, tri_i, tri_j, borda_m,
                steps=args.phase3_steps, batch_size=args.batch_size,
                lr=args.lr, log_every=args.log_every,
                csv_w=metrics_w, csv_f=metrics_f, phase_label='phase3',
                old_tph=args.tph_phase1, stats_csv_w=stats_w, stats_csv_f=stats_f)
    mse3, sgn3 = evaluate(lut_grown, X, Y_pair_sign, tri_i, tri_j, borda_m)
    s_final = per_table_latent_stats(lut_grown, old_tph=args.tph_phase1)
    print(f'  END PHASE 3: mse={mse3:.4f} sign_acc={sgn3:.4f} '
          f'({(time.time()-t3)/60:.1f} min)')
    print(f'  latent stats final: old_mean={s_final["old_mean"]:.4e} '
          f'new_mean={s_final["new_mean"]:.4e}  '
          f'(new vs init-std={args.new_init_std}: '
          f'ratio={s_final["new_mean"]/args.new_init_std:.2e})')

    metrics_f.close()
    stats_f.close()

    summary = {
        'experiment': 'multibit_grow_redundancy_test',
        'layer': args.layer,
        'tph_phase1': args.tph_phase1,
        'tph_phase3': args.tph_phase3,
        'input_nap': args.input_nap,
        'output_nap': args.output_nap,
        'bit_width': args.bit_width,
        'init_std': args.init_std,
        'new_init_std': args.new_init_std,
        'phase1_steps': args.phase1_steps,
        'phase3_steps': args.phase3_steps,
        'phase1_end': {'mse': mse1, 'sign_acc': sgn1},
        'post_grow': {'mse': mse2, 'sign_acc': sgn2},
        'phase3_end': {'mse': mse3, 'sign_acc': sgn3},
        'latent_stats_phase3_end': s_final,
    }
    with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n=== SUMMARY ===')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
