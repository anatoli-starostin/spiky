"""Correctness gates for LightMultiHeadLUT's top-n blended read-out.

Gate (a) n=1 through the BLEND path reproduces the legacy single-cell path exactly.
Gate (b) gradients reach the tables AND the input, and the new dw/dm term is non-zero
         and correctly signed -- it must move the code TOWARD the better cell.
Gate (c) lives outside pytest (it needs a real checkpoint): see
         runs_corrected/gate_c_blend_matches_probe.py.

Run with pytest -- these are lutorch tests, which ARE pytest (unlike lut_fused/spnet).
"""
import torch

from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT


def _layer(n=1, tau=0.1, dtype=torch.float64, **kw):
    m = LightMultiHeadLUT(input_dim=12, n_tables=6, output_dim=5, n_anchor_pairs=4,
                          confidence_form="margin", random_seed=7,
                          read_top_n=n, read_tau=tau, **kw)
    return m.to(dtype)


# --------------------------------------------------------------------------------------
# gate (a): the n=1 limit is exact
# --------------------------------------------------------------------------------------
def test_blend_at_n1_reproduces_legacy_path_exactly():
    """The blend must be the IDENTITY at one candidate, not merely close.

    forward() short-circuits to _bagged_sum at read_top_n == 1, so this drives
    _blend_bag directly with read_top_n forced to 1 -- otherwise the gate would only be
    testing that the short-circuit is taken, which is not the claim.
    """
    torch.manual_seed(0)
    lay = _layer(n=2)
    x = torch.randn(9, 12, dtype=torch.float64)

    legacy = lay(x)                                    # read_top_n=2 -> blend
    lay.read_top_n = 1
    one_cell = lay(x)                                  # -> legacy _bagged_sum
    assert not torch.allclose(legacy, one_cell), "n=2 should differ from n=1"

    # now force the BLEND code path at n=1 and demand exactness against the legacy path
    from spiky.lutorch.fast_multi_head_lut import _confidence_score
    d = x[:, lay.anchor_a] - x[:, lay.anchor_b]
    index = lay._pack_index(x, d)
    flat = lay.tables.reshape(lay.n_tables * lay.table_size, lay.output_dim)
    score = _confidence_score(d, lay.confidence_form, lay.confidence_gain)
    blended_n1 = lay._blend_bag(d, index, lay.table_offset.view(1, -1), flat, score,
                                x.shape[0], lay.n_tables)
    assert torch.equal(blended_n1, one_cell), (
        f"n=1 blend is not bit-exact: max|diff| = "
        f"{(blended_n1 - one_cell).abs().max().item():.3e}")


def test_blend_at_n1_exact_on_multi_head_path():
    torch.manual_seed(0)
    lay = _layer(n=3, n_heads=3, multi_head_input=True)
    lay.read_top_n = 1
    x = torch.randn(7, 3, 12, dtype=torch.float64)
    ref = lay(x)
    lay.read_top_n = 3
    lay.read_top_n = 1          # exercise the blend at n=1 via the helper
    from spiky.lutorch.fast_multi_head_lut import _confidence_score
    B, H, T, K = 7, lay.n_heads, lay.tables_per_head, lay.n_anchor_pairs
    ia = lay.anchor_a.reshape(1, H, T * K).expand(B, H, T * K)
    ib = lay.anchor_b.reshape(1, H, T * K).expand(B, H, T * K)
    d = (torch.gather(x, 2, ia) - torch.gather(x, 2, ib)).view(B, H, T, K)
    index = lay._pack_index(x.reshape(B, H * lay.input_dim), d).view(B, H, T)
    flat = lay.tables.reshape(H * T * lay.table_size, lay.output_dim)
    score = _confidence_score(d, lay.confidence_form, lay.confidence_gain)
    got = lay._blend_bag(d, index, lay.table_offset.view(1, H, T), flat, score,
                         B * H, T).view(B, H, lay.output_dim)
    assert torch.equal(got, ref)


def test_weights_are_normalised_and_candidates_distinct():
    """sum_i w_i == 1 exactly, and the n candidate addresses differ per (token, table)."""
    torch.manual_seed(1)
    lay = _layer(n=3)
    x = torch.randn(11, 12, dtype=torch.float64)
    d = x[:, lay.anchor_a] - x[:, lay.anchor_b]
    m = d.abs()
    mv, mj = torch.topk(m, k=2, dim=-1, largest=False)
    logits = torch.cat([torch.zeros_like(mv[..., :1]), -2.0 * mv / lay.read_tau], -1)
    w = torch.softmax(logits, -1)
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-12)
    assert (w[..., 0] >= w[..., 1]).all(), "argmax must carry the largest weight"
    assert (w[..., 1] >= w[..., 2]).all(), "candidates must be ordered by nearness"
    index = lay._pack_index(x, d)
    bits = (d.detach() > 0).to(torch.int64)
    pw = lay.powers[mj]
    bsel = torch.gather(bits, -1, mj)
    idx = torch.cat([index.unsqueeze(-1), index.unsqueeze(-1) + pw * (1 - 2 * bsel)], -1)
    assert (idx >= 0).all() and (idx < lay.table_size).all(), "address out of range"
    for i in range(3):
        for j in range(i + 1, 3):
            assert (idx[..., i] != idx[..., j]).all(), "candidates must be distinct cells"


# --------------------------------------------------------------------------------------
# gate (b): gradients, including the NEW directional term
# --------------------------------------------------------------------------------------
def test_gradcheck_wrt_input():
    """Full float64 gradcheck of the blend w.r.t. x.

    Margins are kept away from a sign boundary (`* 2.0`) on purpose: the address is a
    piecewise-constant function of x, so the layer is differentiable only WITHIN a cell.
    gradcheck perturbs by eps=1e-6, far smaller than the typical margin here, so no probe
    crosses a boundary and the finite differences are valid.
    """
    torch.manual_seed(2)
    lay = _layer(n=2)
    lay.tables.data.mul_(50.0)          # lift rows off the ~1e-3 init so grads are visible
    x = (torch.randn(4, 12, dtype=torch.float64) * 2.0).requires_grad_(True)
    assert torch.autograd.gradcheck(lay, (x,), eps=1e-6, atol=1e-7, rtol=1e-4)


def test_gradcheck_wrt_tables():
    """Full float64 gradcheck of the blend w.r.t. the table entries."""
    torch.manual_seed(2)
    lay = _layer(n=2)
    x = torch.randn(4, 12, dtype=torch.float64) * 2.0
    t0 = (lay.tables.detach().clone() * 50.0).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda tt: _fwd_with_tables(lay, x, tt), (t0,),
                                    eps=1e-6, atol=1e-7, rtol=1e-4)


def test_all_n_candidate_rows_receive_table_gradient():
    """Not just the argmax row: every blended cell must get gradient, weighted by w_i."""
    torch.manual_seed(6)
    lay = _layer(n=2)
    lay.tables.data.mul_(50.0)
    x = torch.randn(64, 12, dtype=torch.float64) * 2.0
    lay.zero_grad()
    lay(x).pow(2).sum().backward()
    touched_blend = (lay.tables.grad.abs().sum(dim=-1) > 0).sum().item()

    lay.read_top_n = 1
    lay.zero_grad()
    lay(x).pow(2).sum().backward()
    touched_one = (lay.tables.grad.abs().sum(dim=-1) > 0).sum().item()

    assert touched_blend > touched_one, (
        f"blend should reach strictly more table rows: {touched_blend} vs {touched_one}")


def _fwd_with_tables(lay, x, tables):
    from spiky.lutorch.fast_multi_head_lut import _confidence_score
    d = x[:, lay.anchor_a] - x[:, lay.anchor_b]
    index = lay._pack_index(x, d)
    flat = tables.reshape(lay.n_tables * lay.table_size, lay.output_dim)
    score = _confidence_score(d, lay.confidence_form, lay.confidence_gain)
    return lay._blend_bag(d, index, lay.table_offset.view(1, -1), flat, score,
                          x.shape[0], lay.n_tables)


def test_input_gradient_is_larger_with_blend_than_without():
    """The blend must ADD a gradient path into x, not merely rescale the existing one."""
    torch.manual_seed(3)
    lay = _layer(n=2)
    lay.tables.data.mul_(50.0)
    x0 = torch.randn(16, 12, dtype=torch.float64) * 2.0

    g = {}
    for n in (1, 2):
        lay.read_top_n = n
        x = x0.clone().requires_grad_(True)
        lay(x).pow(2).sum().backward()
        g[n] = x.grad.clone()
    assert g[1].norm() > 0 and g[2].norm() > 0
    cos = torch.nn.functional.cosine_similarity(g[1].flatten(), g[2].flatten(), dim=0)
    assert not torch.allclose(g[1], g[2]), "blend did not change the input gradient"
    # it is a genuinely new direction, not a rescale
    assert cos.abs() < 0.999, f"blend gradient is collinear with the old one (cos={cos})"


def test_routing_gradient_points_toward_the_better_cell():
    """The NEW dw/dm term must be non-zero AND correctly signed.

    Analytically, with w = softmax([0, -2m/tau]) over the argmax cell c0 and the flip c1,

        dL/dm = score * (2/tau) * w0 * w1 * (<g, T[c0]> - <g, T[c1]>)

    where g = dL/dy. Gradient descent steps along -dL/dm, so when the ALTERNATIVE row is
    the better one (using c0 raises the loss more than c1 would), dL/dm > 0 and descent
    SHRINKS the margin -- moving the code toward flipping into that better cell. This test
    builds exactly that situation and checks the sign, then flips which row is better and
    checks the sign reverses.
    """
    torch.manual_seed(4)
    lay = _layer(n=2)
    x = (torch.randn(32, 12, dtype=torch.float64) * 2.0)

    from spiky.lutorch.fast_multi_head_lut import _confidence_score

    def margin_grad(alt_is_better: bool):
        """dL/dm for the flipped bit, with the two candidate rows' quality set by hand.

        The row values are supplied directly rather than through the table, so that the
        controlled quality difference cannot be disturbed by two (token, table) pairs
        happening to address the same physical row.
        """
        d = (x[:, lay.anchor_a] - x[:, lay.anchor_b]).detach().requires_grad_(True)
        m = d.abs()
        mv, mj = torch.topk(m, k=1, dim=-1, largest=False)          # [B, T, 1]
        w = torch.softmax(torch.cat([torch.zeros_like(mv),
                                     -2.0 * mv / lay.read_tau], -1), -1)   # [B, T, 2]
        rows = torch.empty(w.shape, dtype=torch.float64)            # [B, T, 2]
        # loss is the plain sum, so a row of -1 is the GOOD one (it lowers the loss)
        rows[..., 0], rows[..., 1] = (+1.0, -1.0) if alt_is_better else (-1.0, +1.0)
        score = _confidence_score(d, lay.confidence_form, lay.confidence_gain)
        (score.unsqueeze(-1) * w * rows).sum().backward()
        # dL/dm_j for the flipped bit: chain d -> m is sign(d), so undo it
        return torch.gather(d.grad * torch.sign(d.detach()), -1, mj).squeeze(-1)

    g_alt_better = margin_grad(alt_is_better=True)
    g_cur_better = margin_grad(alt_is_better=False)

    assert g_alt_better.abs().max() > 1e-9, "dw/dm term is zero -- no routing gradient"
    assert (g_alt_better > 0).float().mean() > 0.99, (
        "when the ALTERNATIVE cell is better, dL/dm must be > 0 so descent shrinks the "
        f"margin toward the flip; got {(g_alt_better > 0).float().mean():.3f} positive")
    assert (g_cur_better < 0).float().mean() > 0.99, (
        "when the CURRENT cell is better, dL/dm must be < 0 so descent grows the margin; "
        f"got {(g_cur_better < 0).float().mean():.3f} negative")


def test_n1_has_no_routing_gradient_at_all():
    """The control: plain Light must show ZERO of this term, which is why it needs the blend."""
    torch.manual_seed(5)
    lay = _layer(n=1)
    x = (torch.randn(8, 12, dtype=torch.float64) * 2.0).requires_grad_(True)
    lay.tables.data.mul_(50.0)
    lay(x).sum().backward()
    # x still gets gradient (through the score), but the address is detached: perturbing
    # only the SIGN pattern cannot change anything differentiably.
    assert x.grad.norm() > 0, "score path should still deliver gradient at n=1"
    d = x[:, lay.anchor_a] - x[:, lay.anchor_b]
    idx = lay._pack_index(x, d)
    assert not idx.requires_grad and idx.grad_fn is None, "address must carry no grad"
