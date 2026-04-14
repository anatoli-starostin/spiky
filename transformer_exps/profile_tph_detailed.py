"""
Detailed profiling of LUT operations: anchor lookup vs lprojection, forward vs backward.
Profiles a single MultiHeadLut with varying tph (OutProj-like: input=32, n_heads=1, n_outputs=32).
"""
import sys, os, time, functools
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spiky.lutorch.multi_head_lut import MultiHeadLut, MultiHeadLutFunction, _compute_anchor_data
from spiky.lutorch.l_projection import _lprojection_backward
from spiky.lutorch.anchor_pairs_lookup import _anchor_pairs_backward
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy

DEVICE = 'cuda:0'
E = 32
NAP = 6
BATCH = 128 * 32  # B*T tokens
SEED = 42
WARMUP = 5
ITERS = 50


def make_lut(tph):
    return MultiHeadLut(
        input_dim=E, n_heads=1, n_outputs=E,
        n_anchor_pairs=NAP, tables_per_head=tph,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
        initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=SEED, device=DEVICE, recompute_in_backward=True,
    )


def time_fn(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000  # ms


def profile_lut(tph):
    lut = make_lut(tph).to(DEVICE)
    lut.train()
    x = torch.randn(BATCH, E, device=DEVICE, requires_grad=True)

    weights = lut.projection.weights
    a_a = lut.lookup.anchor_pairs_a
    a_b = lut.lookup.anchor_pairs_b
    powers = lut.lookup.powers
    cmp_eps = lut.lookup.cmp_eps
    n_tables = weights.shape[0]

    # Precompute batch_offset
    batch_size = BATCH
    input_dim = E
    n_lookup_tables = n_tables
    batch_offset = (
        torch.arange(batch_size, device=DEVICE, dtype=torch.long)
        .repeat_interleave(n_lookup_tables) * input_dim
    ).contiguous()

    # --- Profile anchor lookup (forward) ---
    def anchor_fwd():
        return _compute_anchor_data(x.detach(), a_a, a_b, powers, cmp_eps, 1)
    t_anchor_fwd = time_fn(anchor_fwd)

    # --- Profile lprojection (forward, non-smooth) ---
    lookup_indices, _, _, _, _ = _compute_anchor_data(x.detach(), a_a, a_b, powers, cmp_eps, 1)
    table_indices = torch.arange(n_tables, device=DEVICE, dtype=torch.long).unsqueeze(0).expand(BATCH, -1)

    def lproj_fwd():
        return weights[table_indices, lookup_indices]
    t_lproj_fwd = time_fn(lproj_fwd)

    # --- Profile reshape + sum (tables_per_head reduction) ---
    raw_output = weights[table_indices, lookup_indices]  # [B, n_tables, n_outputs]

    def reshape_sum():
        return raw_output.view(BATCH, 1, tph, E).sum(dim=2)
    t_sum = time_fn(reshape_sum)

    # --- Profile full forward ---
    def full_fwd():
        out = lut(x.detach())
        return out
    t_full_fwd = time_fn(full_fwd)

    # --- Profile full forward + backward ---
    def full_fwd_bwd():
        xx = x.detach().requires_grad_(True)
        out = lut(xx)
        loss = out.sum()
        loss.backward()
        return

    t_full_fwd_bwd = time_fn(full_fwd_bwd)

    t_bwd = t_full_fwd_bwd - t_full_fwd

    # --- Profile backward components ---
    # Backward anchor recompute
    def anchor_recompute():
        with torch.no_grad():
            return _compute_anchor_data(x.detach(), a_a, a_b, powers, cmp_eps, 1)
    t_anchor_recompute = time_fn(anchor_recompute)

    # Backward lprojection
    grad_output = torch.randn(BATCH, n_tables, E, device=DEVICE)
    lookup_alt_indices = lookup_indices.unsqueeze(-1)
    main_weight = alt_weight = None

    def lproj_bwd():
        return _lprojection_backward(
            grad_output, weights, lookup_indices, lookup_alt_indices,
            main_weight, alt_weight, False, 1,
        )
    t_lproj_bwd = time_fn(lproj_bwd)

    # Backward anchor pairs
    _, _, lookup_alt_deltas, anchor1_ids, anchor2_ids = \
        _compute_anchor_data(x.detach(), a_a, a_b, powers, cmp_eps, 1)
    w_grad, li_grad, lai_grad = _lprojection_backward(
        grad_output, weights, lookup_indices, lookup_alt_indices,
        main_weight, alt_weight, False, 1,
    )

    def anchor_bwd():
        return _anchor_pairs_backward(
            x.detach(), anchor1_ids, anchor2_ids, lookup_alt_deltas,
            True, batch_offset, 0.5,
            li_grad, lai_grad, None,
        )
    t_anchor_bwd = time_fn(anchor_bwd)

    params = sum(p.numel() for p in lut.parameters())

    del lut
    torch.cuda.empty_cache()

    return {
        'tph': tph,
        'params': params,
        'full_fwd': t_full_fwd,
        'anchor_fwd': t_anchor_fwd,
        'lproj_fwd': t_lproj_fwd,
        'sum': t_sum,
        'full_bwd': t_bwd,
        'anchor_recompute': t_anchor_recompute,
        'lproj_bwd': t_lproj_bwd,
        'anchor_bwd': t_anchor_bwd,
        'total': t_full_fwd_bwd,
    }


print(f'Profiling MultiHeadLut: input_dim={E}, n_heads=1, n_outputs={E}, nap={NAP}')
print(f'Batch size: {BATCH} tokens, warmup={WARMUP}, iters={ITERS}')
print()

header = f'{"tph":>6} | {"params":>10} | {"fwd":>7} | {"anc_f":>7} | {"lproj_f":>7} | {"sum":>7} | {"bwd":>7} | {"anc_re":>7} | {"lproj_b":>7} | {"anc_b":>7} | {"total":>7}'
print(header)
print('-' * len(header))

for tph in [64, 128, 256, 512, 1024, 2048, 4096]:
    try:
        r = profile_lut(tph)
        print(f'{r["tph"]:>6} | {r["params"]:>10,} | '
              f'{r["full_fwd"]:>6.1f}ms | {r["anchor_fwd"]:>6.1f}ms | {r["lproj_fwd"]:>6.1f}ms | {r["sum"]:>6.1f}ms | '
              f'{r["full_bwd"]:>6.1f}ms | {r["anchor_recompute"]:>6.1f}ms | {r["lproj_bwd"]:>6.1f}ms | {r["anchor_bwd"]:>6.1f}ms | '
              f'{r["total"]:>6.1f}ms')
    except RuntimeError as e:
        print(f'{tph:>6} | OOM: {e}')
        break
