"""
Detailed profile of PermutationalLut at exp265 config (in=6, out=32, tph=2048).
Splits forward and backward into:
  - anchor_pairs_lookup
  - l_projection
  - permutational_part (soft_vote + scatter)
"""
import sys, os, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spiky.lutorch.permutational_lut import PermutationalLut
from spiky.lutorch.multi_head_lut import _compute_anchor_data
from spiky.lutorch.l_projection import _lprojection_backward
from spiky.lutorch.anchor_pairs_lookup import _anchor_pairs_backward

DEVICE = 'cuda:0'
BATCH = 64 * 32   # B*T tokens, matching exp265
N_INPUTS = 32
N_OUTPUTS = 32
INPUT_NAP = 6
OUTPUT_NAP = 32
TPH = 2048
N_HEADS = 1
SEED = 42
WARMUP = 5
ITERS = 30


def make_perm_lut(soft_mode='rational'):
    return PermutationalLut(
        n_inputs=N_INPUTS, n_outputs=N_OUTPUTS,
        input_nap=INPUT_NAP, output_nap=OUTPUT_NAP,
        n_heads=N_HEADS, tph=TPH,
        pair_mode='scrambled',
        soft_mode=soft_mode,
        temperature=0.1,
        random_seed=SEED, device=DEVICE,
        recompute_in_backward=True,
        initial_weights_noise=0.001,
    )


def time_fn(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def profile_breakdown(soft_mode):
    print(f'\n========== soft_mode={soft_mode} ==========')

    perm = make_perm_lut(soft_mode).to(DEVICE)
    perm.train()

    inner = perm.inner
    weights = inner.projection.weights
    a_a = inner.lookup.anchor_pairs_a
    a_b = inner.lookup.anchor_pairs_b
    powers = inner.lookup.powers
    cmp_eps = inner.lookup.cmp_eps
    n_tables = N_HEADS * TPH

    x = torch.randn(BATCH, N_INPUTS, device=DEVICE)
    batch_offset = (
        torch.arange(BATCH, device=DEVICE, dtype=torch.long)
        .repeat_interleave(n_tables) * N_INPUTS
    ).contiguous()

    # ===== Forward components =====

    # F1: anchor_pairs_lookup
    def fwd_anchor():
        return _compute_anchor_data(x, a_a, a_b, powers, cmp_eps, 1)
    t_fwd_anchor = time_fn(fwd_anchor)

    # F2: l_projection (table gather + reshape into [B, H, tph, P])
    lookup_indices, _, _, _, _ = _compute_anchor_data(x, a_a, a_b, powers, cmp_eps, 1)
    table_indices = torch.arange(n_tables, device=DEVICE, dtype=torch.long).unsqueeze(0).expand(BATCH, -1)

    def fwd_lproj():
        out = weights[table_indices, lookup_indices]
        out = out.view(BATCH, N_HEADS, TPH, OUTPUT_NAP)
        return out
    t_fwd_lproj = time_fn(fwd_lproj)

    # F3: permutational_part — soft_vote + scatter into [B, H, N]
    raw = weights[table_indices, lookup_indices].view(BATCH, N_HEADS, TPH, OUTPUT_NAP)
    idx_a = perm.idx_a.view(1, N_HEADS, TPH * OUTPUT_NAP).expand(BATCH, -1, -1)
    idx_b = perm.idx_b.view(1, N_HEADS, TPH * OUTPUT_NAP).expand(BATCH, -1, -1)

    def fwd_perm_part():
        d = perm._signed_vote(raw)
        src = d.reshape(BATCH, N_HEADS, TPH * OUTPUT_NAP)
        out = torch.zeros(BATCH, N_HEADS, N_OUTPUTS, device=DEVICE, dtype=raw.dtype)
        out.scatter_add_(2, idx_a, src)
        out.scatter_add_(2, idx_b, -src)
        return out
    t_fwd_perm = time_fn(fwd_perm_part)

    # Full forward (sanity)
    def fwd_full():
        with torch.no_grad():
            return perm(x)
    t_fwd_full = time_fn(fwd_full)

    # ===== Backward components =====

    # B1: anchor_pairs_recompute (the inner LUT does this in backward)
    def bwd_anchor_recompute():
        with torch.no_grad():
            return _compute_anchor_data(x, a_a, a_b, powers, cmp_eps, 1)
    t_bwd_anchor_recompute = time_fn(bwd_anchor_recompute)

    # B2: l_projection backward — scatter_add of grad_output into weight grads
    grad_per_table = torch.randn(BATCH, n_tables, OUTPUT_NAP, device=DEVICE)
    lookup_alt_indices = lookup_indices.unsqueeze(-1)
    main_weight = alt_weight = None

    def bwd_lproj():
        return _lprojection_backward(
            grad_per_table, weights, lookup_indices, lookup_alt_indices,
            main_weight, alt_weight, False, 1,
        )
    t_bwd_lproj = time_fn(bwd_lproj)

    # B3: anchor_pairs_backward — input grad
    _, _, lookup_alt_deltas, anchor1_ids, anchor2_ids = \
        _compute_anchor_data(x, a_a, a_b, powers, cmp_eps, 1)
    w_grad, li_grad, lai_grad = _lprojection_backward(
        grad_per_table, weights, lookup_indices, lookup_alt_indices,
        main_weight, alt_weight, False, 1,
    )

    def bwd_anchor():
        return _anchor_pairs_backward(
            x, anchor1_ids, anchor2_ids, lookup_alt_deltas,
            True, batch_offset, 0.5,
            li_grad, lai_grad, None,
        )
    t_bwd_anchor = time_fn(bwd_anchor)

    # B4: permutational backward — gradient through soft_vote + scatter
    # Build a fresh raw with grad
    raw_g = weights[table_indices, lookup_indices].view(BATCH, N_HEADS, TPH, OUTPUT_NAP).detach().requires_grad_(True)

    def bwd_perm_part():
        d = perm._signed_vote(raw_g)
        src = d.reshape(BATCH, N_HEADS, TPH * OUTPUT_NAP)
        out = torch.zeros(BATCH, N_HEADS, N_OUTPUTS, device=DEVICE, dtype=raw_g.dtype)
        out.scatter_add_(2, idx_a, src)
        out.scatter_add_(2, idx_b, -src)
        loss = out.sum()
        if raw_g.grad is not None:
            raw_g.grad = None
        loss.backward()
    t_bwd_perm = time_fn(bwd_perm_part)

    # Full forward + backward (sanity)
    def fwd_bwd_full():
        xx = x.detach().requires_grad_(True)
        out = perm(xx)
        target = torch.randn_like(out)
        loss = ((out - target) ** 2).mean()
        loss.backward()
    t_full_fwd_bwd = time_fn(fwd_bwd_full)
    t_full_bwd = t_full_fwd_bwd - t_fwd_full

    print(f'  Forward components:')
    print(f'    anchor_pairs_lookup : {t_fwd_anchor:>7.2f} ms')
    print(f'    l_projection        : {t_fwd_lproj:>7.2f} ms')
    print(f'    permutational_part  : {t_fwd_perm:>7.2f} ms')
    print(f'    sum of components   : {t_fwd_anchor + t_fwd_lproj + t_fwd_perm:>7.2f} ms')
    print(f'    full forward (eval) : {t_fwd_full:>7.2f} ms')
    print()
    print(f'  Backward components:')
    print(f'    anchor_recompute    : {t_bwd_anchor_recompute:>7.2f} ms')
    print(f'    l_projection_bwd    : {t_bwd_lproj:>7.2f} ms')
    print(f'    anchor_pairs_bwd    : {t_bwd_anchor:>7.2f} ms')
    print(f'    permutational_part  : {t_bwd_perm:>7.2f} ms')
    print(f'    sum of components   : {t_bwd_anchor_recompute + t_bwd_lproj + t_bwd_anchor + t_bwd_perm:>7.2f} ms')
    print(f'    full backward       : {t_full_bwd:>7.2f} ms')
    print()
    print(f'  Total step (full fwd+bwd): {t_full_fwd_bwd:>7.2f} ms')

    del perm
    torch.cuda.empty_cache()


print(f'PermutationalLut profile')
print(f'  config: in_nap={INPUT_NAP}, out_nap={OUTPUT_NAP}, tph={TPH}, '
      f'n_inputs={N_INPUTS}, n_outputs={N_OUTPUTS}, n_heads={N_HEADS}')
print(f'  batch: {BATCH} tokens, warmup={WARMUP}, iters={ITERS}')

for sm in ['rational', 'sigmoid', 'ste']:
    profile_breakdown(sm)
