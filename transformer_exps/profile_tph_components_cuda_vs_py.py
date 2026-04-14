"""
Detailed component profile: CUDA vs PyTorch for each operation.
anchor_fwd, lproj_fwd, lproj_bwd, anchor_bwd — with and without custom kernels.
"""
import sys, os, time, importlib
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DEVICE = 'cuda:0'
E = 32
NAP = 6
BATCH = 128 * 32
SEED = 42
WARMUP = 5
ITERS = 50


def setup(use_cuda):
    if not use_cuda:
        os.environ['SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS'] = '1'
    else:
        os.environ.pop('SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS', None)

    import spiky.lutorch.l_projection as lp_mod
    importlib.reload(lp_mod)
    import spiky.lutorch.multi_head_lut as mhl_mod
    importlib.reload(mhl_mod)
    import spiky.lutorch.anchor_pairs_lookup as apl_mod
    importlib.reload(apl_mod)

    from spiky.lutorch.multi_head_lut import MultiHeadLut, _compute_anchor_data
    from spiky.lutorch.l_projection import _lprojection_backward
    from spiky.lutorch.anchor_pairs_lookup import _anchor_pairs_backward
    from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy

    return MultiHeadLut, _compute_anchor_data, _lprojection_backward, _anchor_pairs_backward, UncertaintyMode, AnchorSamplingPolicy


def time_fn(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def profile(tph, use_cuda):
    MultiHeadLut, _compute_anchor_data, _lprojection_backward, _anchor_pairs_backward, UncertaintyMode, AnchorSamplingPolicy = setup(use_cuda)

    lut = MultiHeadLut(
        input_dim=E, n_heads=1, n_outputs=E,
        n_anchor_pairs=NAP, tables_per_head=tph,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
        initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=SEED, device=DEVICE, recompute_in_backward=True,
    ).to(DEVICE)
    lut.train()

    x = torch.randn(BATCH, E, device=DEVICE)
    weights = lut.projection.weights
    a_a = lut.lookup.anchor_pairs_a
    a_b = lut.lookup.anchor_pairs_b
    powers = lut.lookup.powers
    cmp_eps = lut.lookup.cmp_eps
    n_tables = weights.shape[0]

    batch_offset = (
        torch.arange(BATCH, device=DEVICE, dtype=torch.long)
        .repeat_interleave(n_tables) * E
    ).contiguous()

    # anchor fwd
    def anchor_fwd():
        return _compute_anchor_data(x, a_a, a_b, powers, cmp_eps, 1)
    t_anc_f = time_fn(anchor_fwd)

    # lproj fwd (gather)
    lookup_indices, lookup_alt_indices, lookup_alt_deltas, anchor1_ids, anchor2_ids = \
        _compute_anchor_data(x, a_a, a_b, powers, cmp_eps, 1)
    table_indices = torch.arange(n_tables, device=DEVICE, dtype=torch.long).unsqueeze(0).expand(BATCH, -1)

    def lproj_fwd():
        return weights[table_indices, lookup_indices]
    t_lproj_f = time_fn(lproj_fwd)

    # lproj bwd
    grad_output = torch.randn(BATCH, n_tables, E, device=DEVICE)
    lookup_alt_indices_bwd = lookup_indices.unsqueeze(-1)

    def lproj_bwd():
        return _lprojection_backward(
            grad_output, weights, lookup_indices, lookup_alt_indices_bwd,
            None, None, False, 1,
        )
    t_lproj_b = time_fn(lproj_bwd)

    # anchor bwd
    w_grad, li_grad, lai_grad = _lprojection_backward(
        grad_output, weights, lookup_indices, lookup_alt_indices_bwd,
        None, None, False, 1,
    )

    def anchor_bwd():
        return _anchor_pairs_backward(
            x, anchor1_ids, anchor2_ids, lookup_alt_deltas,
            True, batch_offset, 0.5,
            li_grad, lai_grad, None,
        )
    t_anc_b = time_fn(anchor_bwd)

    del lut
    torch.cuda.empty_cache()
    return t_anc_f, t_lproj_f, t_lproj_b, t_anc_b


print(f'Profiling components: input_dim={E}, n_heads=1, n_outputs={E}, nap={NAP}, batch={BATCH}')
print(f'Warmup={WARMUP}, iters={ITERS}')
print()

header = (f'{"tph":>6} | '
          f'{"anc_f_cu":>8} {"anc_f_py":>8} {"ratio":>6} | '
          f'{"lprj_f_cu":>9} {"lprj_f_py":>9} {"ratio":>6} | '
          f'{"lprj_b_cu":>9} {"lprj_b_py":>9} {"ratio":>6} | '
          f'{"anc_b_cu":>8} {"anc_b_py":>8} {"ratio":>6}')
print(header)
print('-' * len(header))

for tph in [64, 128, 256, 512, 1024, 2048, 4096]:
    try:
        anc_f_c, lproj_f_c, lproj_b_c, anc_b_c = profile(tph, use_cuda=True)
        anc_f_p, lproj_f_p, lproj_b_p, anc_b_p = profile(tph, use_cuda=False)
        print(f'{tph:>6} | '
              f'{anc_f_c:>7.1f}ms {anc_f_p:>7.1f}ms {anc_f_p/anc_f_c:>5.2f}x | '
              f'{lproj_f_c:>8.1f}ms {lproj_f_p:>8.1f}ms {lproj_f_p/lproj_f_c:>5.2f}x | '
              f'{lproj_b_c:>8.1f}ms {lproj_b_p:>8.1f}ms {lproj_b_p/lproj_b_c:>5.2f}x | '
              f'{anc_b_c:>7.1f}ms {anc_b_p:>7.1f}ms {anc_b_p/anc_b_c:>5.2f}x')
    except RuntimeError as e:
        print(f'{tph:>6} | OOM: {e}')
        break
