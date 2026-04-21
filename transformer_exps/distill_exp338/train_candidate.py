"""Train a candidate LUT stack to distill one layer's out_proj mapping.

Usage:
    python train_candidate.py <config.json> [--layer 3] [--steps 20000]

config.json describes the candidate stack. The first LUT always has
partition_sets by head (4 x 16 within 64). Stack layout:

    x [64] -> LUT_0(partition) -> dom0 -> D2V(d_0) -> LUT_1 -> dom1 -> ... -> LUT_{k-1} -> dom_out [P=496]

Each LUT is specified as {"input_nap", "tph", "output_nap"}. For LUT_i (i>0),
n_inputs is set to d_{i-1} from the previous D2V output dim. The final LUT
emits n_outputs=E=32 so its dominance has P=C(E,2)=496 matching the target.

Optimizer: BitPermutationLUTOptimizer on all LUT latents + plain Adam on D2V
LayerNorms (elementwise_affine). MSE loss on dominance output; sign accuracy
reported as the distillation metric.
"""
import os, sys, json, argparse, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.bit_permutation_lut_optimizer import BitPermutationLUTOptimizer
from spiky.lutorch.permutational_lut import PermutationalLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import DominanceToVector, _canonical_borda_m

DATA_DIR = '/home/starost/spiky/transformer_exps/distill_exp338/data'
DEVICE = 'cuda:0'
N_INPUTS = 64       # H*d_v (from exp338)
N_OUTPUTS = 32      # E
N_PAIRS = N_OUTPUTS * (N_OUTPUTS - 1) // 2   # 496
H_PARTITION = 4     # 4 heads
D_V_PARTITION = 16
PARTITION_SETS = [list(range(h*D_V_PARTITION, (h+1)*D_V_PARTITION)) for h in range(H_PARTITION)]


def bit_params_count(luts):
    # Counts "logical signs" per LUT: same formula for Bit and Perm variants.
    # For PermutationalLut this is the float weight count; the comparison with
    # BitPermLUT is at equal logical-signs (each float is 32-bit-equivalent).
    total = 0
    for lut in luts:
        nt = lut.tph * (lut.n_heads if hasattr(lut, 'n_heads') else 1)
        total += nt * (1 << lut.input_nap) * lut.output_nap
    return total


def build_stack(candidate_cfg, base_seed):
    """candidate_cfg = {'module_type': 'bit'|'perm' (default 'bit'),
                        'luts': [{...}], 'd_intermediate': [...]}"""
    luts = candidate_cfg['luts']
    module_type = candidate_cfg.get('module_type', 'bit')
    d_inter = candidate_cfg.get('d_intermediate', [])
    assert len(d_inter) == max(0, len(luts) - 1), 'd_intermediate mismatch'

    modules = nn.ModuleList()
    n_in = N_INPUTS
    for i, L in enumerate(luts):
        is_last = (i == len(luts) - 1)
        n_out = N_OUTPUTS if is_last else d_inter[i]
        part = PARTITION_SETS if i == 0 else None
        if module_type == 'bit':
            lut = BitPermutationLUT(
                n_inputs=n_in, n_outputs=n_out, n_heads=1,
                input_nap=L['input_nap'], output_nap=L['output_nap'], tph=L['tph'],
                random_seed=base_seed + i,
                initial_weights_noise=1e-3,
                latent_dtype='bf16', soft_backward=True,
                device=DEVICE,
                partition_sets=part,
            )
        elif module_type == 'perm':
            lut = PermutationalLut(
                n_inputs=n_in, n_outputs=n_out, n_heads=1,
                input_nap=L['input_nap'], output_nap=L['output_nap'], tph=L['tph'],
                pair_mode='scrambled', soft_mode='rational',
                aggregation='matmul',
                random_seed=base_seed + i,
                initial_weights_noise=1e-3,
                device=DEVICE,
                partition_sets=part,
            )
        else:
            raise ValueError(f"unknown module_type={module_type}")
        modules.append(lut)
        if not is_last:
            # BitPermLUT emits pair-dominance -> need D2V; PermLut emits a
            # vector directly -> no D2V needed between stages.
            if module_type == 'bit':
                d2v = DominanceToVector(n_out).to(DEVICE)
                modules.append(d2v)
            n_in = n_out
    return modules


def stack_forward(modules, x):
    """Forward through the stack and return E-dim output.

    For BitPermLUT stacks: LUT emits pair-dominance, final output Borda-projected.
    For PermLut stacks:    LUT already emits an E-dim vector — no Borda projection.
    Intermediate LUTs in multi-LUT stacks have a D2V after them (BitPermLUT only).
    """
    out = x
    last_module = None
    for mod in modules:
        if isinstance(mod, BitPermutationLUT):
            out = mod(out)             # [B, 1, P_i]
            last_module = mod
        elif isinstance(mod, PermutationalLut):
            out = mod(out)             # [B, 1, d_i]  — already a vector
            last_module = mod
        else:  # D2V
            out = mod(out).squeeze(1)  # [B, d_i]
    if isinstance(last_module, BitPermutationLUT):
        return torch.einsum('bhp,kp->bhk', out, last_module.dom_borda_m).squeeze(1)
    # PermLut: drop the size-1 heads dim and return [B, n_outputs].
    return out.squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='path to candidate config .json')
    ap.add_argument('--layer', type=int, default=3)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--log-every', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-2,
                    help='LR for both Adam and bit-opt. Higher than end-to-end '
                         '(~1e-3) because there is no upstream noise.')
    ap.add_argument('--schedule', choices=['const', 'warmup_cosine'],
                    default='warmup_cosine')
    ap.add_argument('--load-teacher-pairs', action='store_true',
                    help='For single-LUT ceiling check: import the teacher '
                         'out_proj anchor_pairs + idx_a/idx_b so the student '
                         'only needs to learn the sign bits.')
    ap.add_argument('--out-dir', default=None,
                    help='directory to save metrics.csv and summary.json '
                         '(defaults to candidate config dir)')
    args = ap.parse_args()

    with open(args.config) as f:
        cand = json.load(f)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.config))
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    data = torch.load(os.path.join(DATA_DIR, f'layer_{args.layer}.pt'), weights_only=False)
    X = data['inputs'].to(DEVICE)                 # [N, 64]
    Y_dom = data['outputs'].to(DEVICE)            # [N, P=496]
    N = X.shape[0]

    modules = build_stack(cand, base_seed=args.seed + 1000)
    bit_luts = [m for m in modules if isinstance(m, BitPermutationLUT)]
    perm_luts = [m for m in modules if isinstance(m, PermutationalLut)]
    luts = bit_luts + perm_luts
    n_bit_params = bit_params_count(luts)

    # Borda-project teacher dominance -> E-dim, then take the pairwise sign
    # of its 32 components. Only the RANKING of the Borda vector matters
    # downstream (next layer's TinyAnchorPairsLookup compares x_a vs x_b);
    # matching the full 496-dim teacher dominance was overconstrained.
    borda_m = _canonical_borda_m(N_OUTPUTS).to(DEVICE)           # [E, P]
    Y_borda = torch.einsum('bp,kp->bk', Y_dom, borda_m)         # [N, E]
    tri_i, tri_j = torch.triu_indices(N_OUTPUTS, N_OUTPUTS, offset=1).unbind(0)
    tri_i = tri_i.to(DEVICE); tri_j = tri_j.to(DEVICE)
    Y_pair_sign = (Y_borda[:, tri_i] - Y_borda[:, tri_j]).sign()  # [N, P=496] ±1
    print(f'layer {args.layer}: X {tuple(X.shape)}  Y_dom {tuple(Y_dom.shape)}  '
          f'-> Y_borda {tuple(Y_borda.shape)}  -> Y_pair_sign {tuple(Y_pair_sign.shape)}')
    print(f'  Y_pair_sign balance: +1 frac={( Y_pair_sign > 0).float().mean().item():.4f}')

    if args.load_teacher_pairs:
        pairs_path = os.path.join(DATA_DIR, 'teacher_pairs.pt')
        tp = torch.load(pairs_path, weights_only=False)[args.layer]
        assert len(bit_luts) == 1 and not perm_luts, \
            '--load-teacher-pairs requires single BitPermLUT stack'
        bit_luts[0].load_pairs(
            anchor_pairs_a=tp['anchor_pairs_a'].to(DEVICE),
            anchor_pairs_b=tp['anchor_pairs_b'].to(DEVICE),
            idx_a=tp['idx_a'].to(DEVICE),
            idx_b=tp['idx_b'].to(DEVICE),
        )
        print(f'  loaded teacher pairs from layer {args.layer}')

    # Parameter groups.
    adam_params = [p for m in modules for p in m.parameters() if p.requires_grad]
    print(f'bit LUTs: {len(bit_luts)}, perm LUTs: {len(perm_luts)}  |  '
          f'bit-params (logical signs): {n_bit_params:,}  |  '
          f'Adam params: {sum(p.numel() for p in adam_params):,}')
    for i, L in enumerate(luts):
        tag = '(partition)' if i == 0 else ''
        print(f'  LUT[{i}] {tag}: n_in={L.n_inputs} n_out={L.n_outputs} '
              f'in_nap={L.input_nap} tph={L.tph} out_nap={L.output_nap}')

    # LR schedule.
    total_steps = args.steps
    warmup = max(1, int(0.1 * total_steps))
    if args.schedule == 'const':
        lr_scale = lambda step: 1.0
    else:
        def lr_scale(step):
            if step < warmup:
                return step / warmup
            p = (step - warmup) / max(1, total_steps - warmup)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    adam_opt = (
        torch.optim.Adam(adam_params, lr=args.lr)
        if adam_params else None
    )
    adam_sched = (
        torch.optim.lr_scheduler.LambdaLR(adam_opt, lr_scale)
        if adam_opt is not None else None
    )
    bit_opt = (
        BitPermutationLUTOptimizer(
            bit_luts, lr=args.lr, beta1=0.9, beta2=0.999, lr_schedule_fn=lr_scale,
        ) if bit_luts else None
    )

    csv_path = os.path.join(out_dir, 'metrics.csv')
    csv_f = open(csv_path, 'w')
    csv_f.write('step,mse,sign_acc\n')

    t0 = time.time()
    bs = args.batch_size
    for modules_ in [modules]:
        for mod in modules_:
            mod.train()

    best = {'step': -1, 'mse': float('inf'), 'sign_acc': 0.0}

    # Temperature for soft pair-sign of student's Borda output.
    RANK_T = 0.1

    def pair_soft_sign(v):
        # v: [B, E]
        diff = v[:, tri_i] - v[:, tri_j]          # [B, P]
        return diff / (RANK_T + diff.abs())

    for step in range(args.steps):
        idx = torch.randint(0, N, (bs,), device=DEVICE)
        # BitPermutationLUT's custom autograd.Function only fires its backward
        # (and thus feeds the bit-opt hook) if an input carries requires_grad.
        x = X[idx].detach().requires_grad_(True)
        v_pred = stack_forward(modules, x)                 # [B, E]
        soft_s = pair_soft_sign(v_pred)                    # [B, P] in (-1,1)
        target = Y_pair_sign[idx]                          # [B, P] ±1
        loss = F.mse_loss(soft_s, target)

        if adam_opt is not None: adam_opt.zero_grad()
        if bit_opt is not None: bit_opt.zero_grad()
        loss.backward()
        if adam_opt is not None:
            adam_opt.step()
            adam_sched.step()
        if bit_opt is not None: bit_opt.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            with torch.no_grad():
                eval_bs = 8192
                mse_acc = 0.0; sgn_acc = 0.0; n_eval = 0
                for s in range(0, 32768, eval_bs):
                    xi = X[s:s+eval_bs]; yi = Y_pair_sign[s:s+eval_bs]
                    vp = stack_forward(modules, xi)
                    ss = pair_soft_sign(vp)
                    mse_acc += F.mse_loss(ss, yi, reduction='sum').item()
                    # Hard rank accuracy: student pair sign vs teacher pair sign.
                    sgn_acc += (ss.sign() == yi).float().sum().item()
                    n_eval += yi.numel()
                mse_eval = mse_acc / n_eval
                sgn_eval = sgn_acc / n_eval
            print(f'  step {step+1:>6}/{args.steps} | train mse {loss.item():.4f} | '
                  f'eval mse {mse_eval:.4f} | eval sign_acc {sgn_eval:.4f}')
            csv_f.write(f'{step+1},{mse_eval:.6f},{sgn_eval:.6f}\n')
            csv_f.flush()
            if sgn_eval > best['sign_acc']:
                best = {'step': step+1, 'mse': mse_eval, 'sign_acc': sgn_eval}
    csv_f.close()
    if bit_opt is not None: bit_opt.close()

    elapsed = time.time() - t0
    summary = {
        'layer': args.layer,
        'steps': args.steps,
        'bit_params': n_bit_params,
        'n_luts': len(luts),
        'final_mse': mse_eval,
        'final_sign_acc': sgn_eval,
        'best_sign_acc': best['sign_acc'],
        'best_step': best['step'],
        'best_mse': best['mse'],
        'time_min': round(elapsed/60, 2),
        'candidate': cand,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n=== DONE ===')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
