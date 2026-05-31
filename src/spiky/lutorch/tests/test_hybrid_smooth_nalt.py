"""Math correctness tests for hybrid_smooth backward_mode with n_alt ∈ [1, NAP].

Covers:
  1. Forward: matches an explicit top-(n_alt+1) softmax reference on tiny inputs.
  2. Sequential argmin == topk (on values, ordering not guaranteed but set equal).
  3. autograd.gradcheck (fp64) on a small instance for dx, dW, dlog_T_soft, dlog_T_sel.
  4. Eager == @torch.compile (within bf16 noise).
  5. Sweep n_alt = 1, 2, 3, NAP — each correct.
"""
import math
import pytest
import torch
import torch.nn.functional as F

from spiky.lutorch.tiny_multi_head_lut import (
    _hybrid_smooth_kalt_fwd_autograd,
    _TinyMHLutHybridSmoothMemEff,
    _TinyMHLutHybridSmoothKalt,
    _soft_bit_matrix_msb,
)


# -------- Reference: explicit top-(n_alt+1) softmax forward, no shortcuts. --------
def reference_forward(x, weights, T_soft, T_sel, anchor_a, anchor_b, powers,
                      n_heads, tph, table_dim, n_alt):
    """Naive but readable reference. Computes the exact top-(n_alt+1) softmax
    over Hamming-1 neighbors at the n_alt smallest |abs_p| positions."""
    B, _ = x.shape
    n_tables = anchor_a.shape[0]
    n_outputs = weights.shape[2]
    NAP = anchor_a.shape[1]
    device = x.device

    # d, abs_p, main_index — same as production.
    d = x[:, anchor_a] - x[:, anchor_b]                              # [B, n_tables, NAP]
    bits = (d > 0).to(torch.int64)
    main_index = (bits * powers.view(1, 1, -1)).sum(dim=-1)          # [B, n_tables]
    abs_d = d.abs()
    abs_p = abs_d / (T_soft + abs_d)                                  # [B, n_tables, NAP]

    # Top-n_alt smallest abs_p positions (set, not ordered).
    if n_alt == NAP:
        topk_pos = torch.arange(NAP, device=device).view(1, 1, -1).expand(B, n_tables, -1)
    else:
        _, topk_pos = torch.topk(abs_p, k=n_alt, dim=-1, largest=False)

    delta_ts = 2.0 * abs_p.gather(-1, topk_pos)                       # [B, n_tables, n_alt]
    flip_powers = powers.to(main_index.dtype)[topk_pos]
    alt_indices = main_index.unsqueeze(-1) ^ flip_powers              # [B, n_tables, n_alt]

    # Softmax over [main(=0), -delta_ts/T_sel].
    logits_alts = -delta_ts / T_sel
    logits = torch.cat([torch.zeros_like(logits_alts[..., :1]), logits_alts], dim=-1)
    probs = torch.softmax(logits, dim=-1)                              # [B, n_tables, n_alt+1]

    # Gather rows & accumulate.
    table_offset = torch.arange(n_tables, device=device,
                                dtype=main_index.dtype) * table_dim
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    main_flat_idx = (main_index + table_offset.view(1, -1)).reshape(-1)
    main_rows = F.embedding(main_flat_idx, weights_flat).view(B, n_tables, n_outputs)
    out_pt = main_rows * probs[..., 0:1]
    for k in range(n_alt):
        ai = (alt_indices[..., k] + table_offset.view(1, -1)).reshape(-1)
        rk = F.embedding(ai, weights_flat).view(B, n_tables, n_outputs)
        out_pt = out_pt + rk * probs[..., k + 1:k + 2]
    return out_pt.view(B, n_heads, tph, n_outputs).sum(dim=2)


# -------- Tiny-shape config helper (small enough for gradcheck). --------
def tiny_inputs(NAP=4, tph=4, n_heads=1, n_outputs=3, input_dim=8, B=2,
                dtype=torch.float64, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    n_tables = n_heads * tph
    table_dim = 2 ** NAP
    x = torch.randn(B, input_dim, device=device, dtype=dtype, generator=g) * 0.3
    weights = torch.randn(n_tables, table_dim, n_outputs,
                          device=device, dtype=dtype, generator=g) * 0.5
    log_T_soft = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
    log_T_sel = torch.tensor(math.log(0.5), device=device, dtype=dtype, requires_grad=True)
    # MSB-first powers.
    powers = (1 << torch.arange(NAP - 1, -1, -1, device=device, dtype=torch.int64))
    # Random anchor pairs (distinct a, b for each table×NAP slot).
    a = torch.randint(0, input_dim, (n_tables, NAP), device=device,
                      generator=g, dtype=torch.int64)
    b = (a + 1 + torch.randint(0, input_dim - 1, (n_tables, NAP),
                                device=device, generator=g, dtype=torch.int64)
         ) % input_dim
    return dict(
        x=x.requires_grad_(True),
        weights=weights.requires_grad_(True),
        log_T_soft=log_T_soft, log_T_sel=log_T_sel,
        anchor_a=a, anchor_b=b, powers=powers,
        n_heads=n_heads, tph=tph, table_dim=table_dim,
    )


# ============================================================
# Tests
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_alt", [1, 2, 3, 4])
def test_forward_matches_reference(n_alt):
    """Forward exactly matches a from-scratch top-(n_alt+1) softmax reference."""
    NAP = 4
    p = tiny_inputs(NAP=NAP, dtype=torch.float64, device=DEVICE, seed=11)
    out_kalt = _hybrid_smooth_kalt_fwd_autograd(
        p["x"], p["weights"], p["log_T_soft"], p["log_T_sel"],
        p["anchor_a"], p["anchor_b"], p["powers"],
        p["n_heads"], p["tph"], p["table_dim"], n_alt,
    )
    out_ref = reference_forward(
        p["x"], p["weights"],
        p["log_T_soft"].exp(), p["log_T_sel"].exp(),
        p["anchor_a"], p["anchor_b"], p["powers"],
        p["n_heads"], p["tph"], p["table_dim"], n_alt,
    )
    assert torch.allclose(out_kalt, out_ref, atol=1e-10, rtol=1e-10), \
        f"max-diff = {(out_kalt - out_ref).abs().max().item():.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_alt", [1, 2, 3])
def test_seq_argmin_matches_topk_as_set(n_alt):
    """Sequential argmin and torch.topk pick the same set of positions (order
    differs by sort stability; we compare as sorted sets)."""
    NAP = 5
    B, n_tables = 3, 7
    abs_p = torch.rand(B, n_tables, NAP, device=DEVICE)

    # topk path
    _, topk_pos = torch.topk(abs_p, k=n_alt, dim=-1, largest=False)

    # sequential argmin path
    abs_mask = abs_p.clone()
    INF = torch.finfo(abs_p.dtype).max
    pos_list = []
    for _k in range(n_alt):
        idx_k = abs_mask.argmin(dim=-1, keepdim=True)
        pos_list.append(idx_k)
        abs_mask = abs_mask.scatter(-1, idx_k, INF)
    seq_pos = torch.cat(pos_list, dim=-1)

    # Sets must match (any sort order).
    assert torch.equal(topk_pos.sort(-1).values, seq_pos.sort(-1).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_alt", [1, 2, 3, 4])
def test_gradcheck(n_alt):
    """Autograd through the (compiled-but-eager) function passes gradcheck (fp64)."""
    NAP = 4
    p = tiny_inputs(NAP=NAP, B=2, tph=3, n_heads=1, n_outputs=2, input_dim=6,
                    dtype=torch.float64, device=DEVICE, seed=7)

    def f(x, weights, log_T_soft, log_T_sel):
        return _hybrid_smooth_kalt_fwd_autograd(
            x, weights, log_T_soft, log_T_sel,
            p["anchor_a"], p["anchor_b"], p["powers"],
            p["n_heads"], p["tph"], p["table_dim"], n_alt,
        )

    inputs = (p["x"], p["weights"], p["log_T_soft"], p["log_T_sel"])
    # gradcheck calls backward() multiple times; torch.compile's donated-buffer
    # optimisation forbids that. Disable it locally for this test.
    with torch._functorch.config.patch(donated_buffer=False):
        # nondet_tol=1e-8 covers argmin's index-tie nondeterminism (very rare in fp64).
        assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4,
                                        nondet_tol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_alt", [1, 2, 3, 4])
def test_eager_vs_compile(n_alt):
    """torch.compile path produces same output and grads as eager (within fp32 noise)."""
    NAP = 4
    p = tiny_inputs(NAP=NAP, B=4, tph=6, n_heads=2, n_outputs=5, input_dim=10,
                    dtype=torch.float32, device=DEVICE, seed=21)

    compiled = torch.compile(_hybrid_smooth_kalt_fwd_autograd)

    # Eager forward + backward.
    out_e = _hybrid_smooth_kalt_fwd_autograd(
        p["x"], p["weights"], p["log_T_soft"], p["log_T_sel"],
        p["anchor_a"], p["anchor_b"], p["powers"],
        p["n_heads"], p["tph"], p["table_dim"], n_alt,
    )
    g = torch.randn_like(out_e)
    out_e.backward(g)
    dx_e = p["x"].grad.detach().clone()
    dW_e = p["weights"].grad.detach().clone()
    dTs_e = p["log_T_soft"].grad.detach().clone()
    dTl_e = p["log_T_sel"].grad.detach().clone()

    # Reset grads & rebuild leaves.
    p["x"].grad = None; p["weights"].grad = None
    p["log_T_soft"].grad = None; p["log_T_sel"].grad = None

    out_c = compiled(
        p["x"], p["weights"], p["log_T_soft"], p["log_T_sel"],
        p["anchor_a"], p["anchor_b"], p["powers"],
        p["n_heads"], p["tph"], p["table_dim"], n_alt,
    )
    out_c.backward(g)

    assert torch.allclose(out_c, out_e, atol=1e-5, rtol=1e-5)
    assert torch.allclose(p["x"].grad, dx_e, atol=1e-5, rtol=1e-5)
    assert torch.allclose(p["weights"].grad, dW_e, atol=1e-4, rtol=1e-4)
    assert torch.allclose(p["log_T_soft"].grad, dTs_e, atol=1e-4, rtol=1e-4)
    assert torch.allclose(p["log_T_sel"].grad, dTl_e, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_alt", [1, 2, 3, 4])
def test_memeff_matches_kalt(n_alt):
    """The memory-efficient autograd.Function path produces the same forward
    output and the same input/weight/temperature gradients as the plain autograd
    kalt path (within fp64 numerical noise)."""
    NAP = 4
    p_kalt = tiny_inputs(NAP=NAP, B=4, tph=5, n_heads=2, n_outputs=3, input_dim=9,
                          dtype=torch.float64, device=DEVICE, seed=51)
    p_me = tiny_inputs(NAP=NAP, B=4, tph=5, n_heads=2, n_outputs=3, input_dim=9,
                        dtype=torch.float64, device=DEVICE, seed=51)

    out_k = _hybrid_smooth_kalt_fwd_autograd(
        p_kalt["x"], p_kalt["weights"], p_kalt["log_T_soft"], p_kalt["log_T_sel"],
        p_kalt["anchor_a"], p_kalt["anchor_b"], p_kalt["powers"],
        p_kalt["n_heads"], p_kalt["tph"], p_kalt["table_dim"], n_alt,
    )
    g = torch.randn_like(out_k)
    out_k.backward(g)

    from spiky.lutorch.tiny_multi_head_lut import _soft_bit_matrix_msb
    bit_mat = _soft_bit_matrix_msb(NAP, DEVICE, dtype=torch.float64)
    out_m = _TinyMHLutHybridSmoothMemEff.apply(
        p_me["x"], p_me["weights"], p_me["log_T_soft"], p_me["log_T_sel"],
        p_me["anchor_a"], p_me["anchor_b"], p_me["powers"], bit_mat,
        p_me["n_heads"], p_me["tph"], p_me["table_dim"], n_alt, False, False,
    )
    out_m.backward(g)

    # Tolerances allow for torch.compile's fp32 internal accumulators in the
    # kalt path (the memeff path is exact fp64 throughout). gradcheck on the
    # memeff path separately confirms the math itself is accurate to fp64.
    assert torch.allclose(out_m, out_k, atol=1e-6, rtol=1e-6), \
        f"out diff {(out_k - out_m).abs().max().item():.3e}"
    assert torch.allclose(p_me["x"].grad, p_kalt["x"].grad, atol=1e-5, rtol=1e-5), \
        f"dx diff {(p_kalt['x'].grad - p_me['x'].grad).abs().max().item():.3e}"
    assert torch.allclose(p_me["weights"].grad, p_kalt["weights"].grad,
                          atol=1e-5, rtol=1e-5), \
        f"dW diff {(p_kalt['weights'].grad - p_me['weights'].grad).abs().max().item():.3e}"
    assert torch.allclose(p_me["log_T_soft"].grad, p_kalt["log_T_soft"].grad,
                          atol=1e-5, rtol=1e-5)
    assert torch.allclose(p_me["log_T_sel"].grad, p_kalt["log_T_sel"].grad,
                          atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_alt", [1, 2, 3, 4])
def test_memeff_gradcheck(n_alt):
    """gradcheck on the memory-efficient autograd.Function (fp64)."""
    NAP = 4
    p = tiny_inputs(NAP=NAP, B=2, tph=3, n_heads=1, n_outputs=2, input_dim=6,
                    dtype=torch.float64, device=DEVICE, seed=77)

    from spiky.lutorch.tiny_multi_head_lut import _soft_bit_matrix_msb
    bit_mat = _soft_bit_matrix_msb(NAP, DEVICE, dtype=torch.float64)
    def f(x, weights, log_T_soft, log_T_sel):
        return _TinyMHLutHybridSmoothMemEff.apply(
            x, weights, log_T_soft, log_T_sel,
            p["anchor_a"], p["anchor_b"], p["powers"], bit_mat,
            p["n_heads"], p["tph"], p["table_dim"], n_alt, False, False,
        )

    inputs = (p["x"], p["weights"], p["log_T_soft"], p["log_T_sel"])
    assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-4,
                                    nondet_tol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_alt", [1, 2, 3, 4])
def test_memeff_dense_input_grad_matches_on_fwd_and_dW(n_alt):
    """dense_input_grad=True only changes the input-gradient computation.
    Forward output and weight gradient must be bit-identical to the self-
    consistent path. (gradcheck on the dense path is intentionally not used —
    the K-row soft surrogate is non-self-consistent w.r.t. the actual forward,
    so numerical diff disagrees by design.)"""
    NAP = 4
    p_s = tiny_inputs(NAP=NAP, B=4, tph=5, n_heads=2, n_outputs=3, input_dim=9,
                      dtype=torch.float64, device=DEVICE, seed=101)
    p_d = tiny_inputs(NAP=NAP, B=4, tph=5, n_heads=2, n_outputs=3, input_dim=9,
                      dtype=torch.float64, device=DEVICE, seed=101)
    from spiky.lutorch.tiny_multi_head_lut import _soft_bit_matrix_msb
    bit_mat = _soft_bit_matrix_msb(NAP, DEVICE, dtype=torch.float64)

    out_s = _TinyMHLutHybridSmoothMemEff.apply(
        p_s["x"], p_s["weights"], p_s["log_T_soft"], p_s["log_T_sel"],
        p_s["anchor_a"], p_s["anchor_b"], p_s["powers"], bit_mat,
        p_s["n_heads"], p_s["tph"], p_s["table_dim"], n_alt, False, False,
    )
    out_d = _TinyMHLutHybridSmoothMemEff.apply(
        p_d["x"], p_d["weights"], p_d["log_T_soft"], p_d["log_T_sel"],
        p_d["anchor_a"], p_d["anchor_b"], p_d["powers"], bit_mat,
        p_d["n_heads"], p_d["tph"], p_d["table_dim"], n_alt, False, True,
    )
    assert torch.allclose(out_s, out_d, atol=1e-12, rtol=1e-12), \
        f"forward differs between dense={False,True}: max-diff {(out_s-out_d).abs().max().item():.3e}"

    g = torch.randn_like(out_s)
    out_s.backward(g)
    out_d.backward(g)

    assert torch.allclose(p_s["weights"].grad, p_d["weights"].grad,
                          atol=1e-12, rtol=1e-12), \
        f"dW differs between dense flags: max-diff {(p_s['weights'].grad-p_d['weights'].grad).abs().max().item():.3e}"
    # dx and dT's SHOULD differ — that's the whole point. Just sanity-check
    # that both are finite and non-zero.
    assert torch.isfinite(p_d["x"].grad).all() and p_d["x"].grad.abs().max() > 0
    assert torch.isfinite(p_d["log_T_sel"].grad).all()
    assert torch.isfinite(p_d["log_T_soft"].grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_alt", [1, 2, 3, 4])
def test_kalt_manual_forward_matches_reference(n_alt):
    """Forward output of `_TinyMHLutHybridSmoothKalt` matches the explicit
    top-(n_alt+1) softmax reference. Same forward semantics as the autograd
    `_hybrid_smooth_kalt_fwd_autograd` and the memeff path; only backward
    differs (kalt uses K-row soft input grad surrogate via _soft_lut_bwd_body)."""
    NAP = 4
    p = tiny_inputs(NAP=NAP, B=3, tph=5, n_heads=1, n_outputs=4, input_dim=8,
                    dtype=torch.float32, device=DEVICE, seed=88)
    bit_mat = _soft_bit_matrix_msb(NAP, DEVICE, dtype=torch.float32)
    out_kalt = _TinyMHLutHybridSmoothKalt.apply(
        p["x"], p["weights"], p["log_T_soft"], p["log_T_sel"],
        p["anchor_a"], p["anchor_b"], bit_mat, p["powers"],
        p["n_heads"], p["tph"], p["table_dim"], n_alt, False,
    )
    out_ref = reference_forward(
        p["x"], p["weights"],
        p["log_T_soft"].exp(), p["log_T_sel"].exp(),
        p["anchor_a"], p["anchor_b"], p["powers"],
        p["n_heads"], p["tph"], p["table_dim"], n_alt,
    )
    assert torch.allclose(out_kalt, out_ref, atol=1e-5, rtol=1e-5), \
        f"out diff {(out_kalt - out_ref).abs().max().item():.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kalt_n_alt_1_matches_legacy_hybrid_smooth():
    """At n_alt=1, the generalised kalt manual path produces the same forward
    output as the legacy `_TinyMHLutHybridSmooth` (sigmoid form). They use
    different parameterisations of the same 2-row softmax, so outputs match
    bit-for-bit when probs are computed identically."""
    from spiky.lutorch.tiny_multi_head_lut import _TinyMHLutHybridSmooth
    NAP = 4
    p = tiny_inputs(NAP=NAP, B=4, tph=5, n_heads=1, n_outputs=3, input_dim=8,
                    dtype=torch.float32, device=DEVICE, seed=123)
    bit_mat = _soft_bit_matrix_msb(NAP, DEVICE, dtype=torch.float32)
    out_legacy = _TinyMHLutHybridSmooth.apply(
        p["x"], p["weights"], p["log_T_soft"], p["log_T_sel"],
        p["anchor_a"], p["anchor_b"], bit_mat, p["powers"],
        p["n_heads"], p["tph"], p["table_dim"], False,
    )
    out_kalt = _TinyMHLutHybridSmoothKalt.apply(
        p["x"], p["weights"], p["log_T_soft"], p["log_T_sel"],
        p["anchor_a"], p["anchor_b"], bit_mat, p["powers"],
        p["n_heads"], p["tph"], p["table_dim"], 1, False,
    )
    assert torch.allclose(out_legacy, out_kalt, atol=1e-6, rtol=1e-6), \
        f"out diff {(out_legacy - out_kalt).abs().max().item():.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_n_alt_nap_equals_full_ball():
    """When n_alt=NAP, no topk slicing happens — output must equal the explicit
    full-Hamming-1-ball reference (no index reordering risk)."""
    NAP = 3
    p = tiny_inputs(NAP=NAP, B=3, tph=4, n_heads=1, n_outputs=3, input_dim=8,
                    dtype=torch.float64, device=DEVICE, seed=33)
    out = _hybrid_smooth_kalt_fwd_autograd(
        p["x"], p["weights"], p["log_T_soft"], p["log_T_sel"],
        p["anchor_a"], p["anchor_b"], p["powers"],
        p["n_heads"], p["tph"], p["table_dim"], NAP,
    )
    out_ref = reference_forward(
        p["x"], p["weights"],
        p["log_T_soft"].exp(), p["log_T_sel"].exp(),
        p["anchor_a"], p["anchor_b"], p["powers"],
        p["n_heads"], p["tph"], p["table_dim"], NAP,
    )
    assert torch.allclose(out, out_ref, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
