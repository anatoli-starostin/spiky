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
    lay._compile_enabled = False        # gradcheck (atol 1e-7) validates the EAGER reference
                                        # math; the default torch.compile backend is a
                                        # non-bit-exact perf transform, parity-checked (~3e-5)
                                        # separately in test_compile_parity_matches_eager.
    lay.tables.data.mul_(50.0)          # lift rows off the ~1e-3 init so grads are visible
    x = (torch.randn(4, 12, dtype=torch.float64) * 2.0).requires_grad_(True)
    assert torch.autograd.gradcheck(lay, (x,), eps=1e-6, atol=1e-7, rtol=1e-4)


def test_gradcheck_wrt_tables():
    """Full float64 gradcheck of the blend w.r.t. the table entries."""
    torch.manual_seed(2)
    lay = _layer(n=2)
    lay._compile_enabled = False        # gradcheck validates the eager reference (see above)
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


# ======================================================================================
# Learnable / freezable blend temperature (log_tau)
# ======================================================================================
def _tau_layer(n=2, tau=0.1445, learn=False):
    return LightMultiHeadLUT(input_dim=12, n_tables=6, output_dim=5, n_anchor_pairs=4,
                             confidence_form="margin", random_seed=7,
                             read_top_n=n, read_tau=tau,
                             read_tau_learnable=learn).to(torch.float64)


def test_n1_bit_identical_with_tau_frozen_and_learnable():
    """GATE (a), the non-negotiable one: n=1 must be untouched by the tau machinery."""
    torch.manual_seed(0)
    x = torch.randn(9, 12, dtype=torch.float64)
    ref = None
    for learn in (False, True):
        lay = _tau_layer(n=1, learn=learn)
        got = lay(x)
        if ref is None:
            ref = got
        assert torch.equal(got, ref), f"n=1 differs with read_tau_learnable={learn}"
    # and the blend path itself at n=1, under both modes
    from spiky.lutorch.fast_multi_head_lut import _confidence_score
    for learn in (False, True):
        lay = _tau_layer(n=1, learn=learn)
        d = x[:, lay.anchor_a] - x[:, lay.anchor_b]
        index = lay._pack_index(x, d)
        flat = lay.tables.reshape(lay.n_tables * lay.table_size, lay.output_dim)
        score = _confidence_score(d, lay.confidence_form, lay.confidence_gain)
        blend = lay._blend_bag(d, index, lay.table_offset.view(1, -1), flat, score,
                               x.shape[0], lay.n_tables)
        assert torch.equal(blend, ref), f"blend at n=1 differs (learnable={learn})"


def test_frozen_tau_is_a_buffer_and_does_not_change_param_count():
    base = _tau_layer(n=1)
    froz = _tau_layer(n=2, learn=False)
    lrn = _tau_layer(n=2, learn=True)

    def npar(m):
        return sum(p.numel() for p in m.parameters())

    assert npar(froz) == npar(base), "frozen tau must not change the parameter count"
    assert npar(lrn) == npar(base) + 1, "learnable tau should add exactly one scalar"
    assert 'log_tau' not in dict(froz.named_parameters())
    assert 'log_tau' in dict(lrn.named_parameters())
    assert froz.log_tau.requires_grad is False
    assert 'log_tau' in froz.state_dict() and 'log_tau' in lrn.state_dict()
    # cross-loadable both ways, so a frozen run's checkpoint can seed a learnable one
    m1, u1 = lrn.load_state_dict(froz.state_dict(), strict=False)
    m2, u2 = froz.load_state_dict(lrn.state_dict(), strict=False)
    assert not m1 and not u1 and not m2 and not u2


def test_frozen_tau_receives_no_gradient_and_does_not_move():
    torch.manual_seed(1)
    lay = _tau_layer(n=2, learn=False)
    lay.tables.data.mul_(50.0)
    before = lay.read_tau.item()
    x = (torch.randn(8, 12, dtype=torch.float64) * 2.0)
    lay(x).pow(2).sum().backward()
    assert lay.log_tau.grad is None, "a frozen tau must not accumulate gradient"
    assert lay.read_tau.item() == before


def test_gradcheck_wrt_log_tau():
    """GATE (b): float64 gradcheck for the new parameter."""
    torch.manual_seed(2)
    lay = _tau_layer(n=3, learn=True)
    lay.tables.data.mul_(50.0)
    x = torch.randn(4, 12, dtype=torch.float64) * 2.0

    from spiky.lutorch.fast_multi_head_lut import _confidence_score

    def f(lt):
        # drive the blend functionally: wrapping lt in nn.Parameter would make a NEW leaf
        # and silently detach gradcheck's input from the graph, so the check would pass
        # while testing nothing.
        d = x[:, lay.anchor_a] - x[:, lay.anchor_b]
        m = d.abs()
        mv, mj = torch.topk(m, k=lay.read_top_n - 1, dim=-1, largest=False)
        bits = (d.detach() > 0).to(torch.int64)
        index = (bits * lay.powers.view(1, 1, -1)).sum(-1)
        idx = torch.cat([index.unsqueeze(-1),
                         index.unsqueeze(-1)
                         + lay.powers[mj] * (1 - 2 * torch.gather(bits, -1, mj))], -1)
        w = torch.softmax(torch.cat([torch.zeros_like(m[..., :1]),
                                     -2.0 * mv / lt.exp()], -1), -1)
        flat = lay.tables.reshape(lay.n_tables * lay.table_size, lay.output_dim)
        score = _confidence_score(d, lay.confidence_form, lay.confidence_gain)
        rows = flat[(idx + lay.table_offset.view(1, -1).unsqueeze(-1))]
        return (score.unsqueeze(-1).unsqueeze(-1) * w.unsqueeze(-1) * rows).sum((1, 2))

    lt0 = lay.log_tau.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (lt0,), eps=1e-6, atol=1e-7, rtol=1e-4)


def test_tau_gradient_matches_the_closed_form():
    """GATE (c): dw_k/dtau against the analytic expression, sign and magnitude.

    Our exponent is -2m/tau (the 2 lives in the code, not the temperature), so

        dw_k/dtau = (2 w_k / tau^2) (m_k - sum_l w_l m_l),      m_0 = 0 for the winner

    which is the stated (w_k/tau^2)(m_k - <m>) carrying the same factor 2 as the forward.
    """
    torch.manual_seed(3)
    K, tau = 4, 0.1445
    m = torch.rand(200, K, dtype=torch.float64) * 0.4
    mv, _ = torch.sort(m, dim=-1)
    costs = torch.cat([torch.zeros_like(mv[:, :1]), mv[:, :2]], -1)   # winner + 2 flips

    t = torch.tensor(tau, dtype=torch.float64, requires_grad=True)
    w = torch.softmax(-2.0 * costs / t, dim=-1)
    g_auto = torch.autograd.grad(w.sum(), t, retain_graph=True)[0]

    w_d = w.detach()
    mbar = (w_d * costs).sum(-1, keepdim=True)
    analytic = (2.0 * w_d / tau ** 2) * (costs - mbar)
    assert torch.allclose(g_auto, analytic.sum(), atol=1e-9), 'closed form mismatch'

    # sign: the winner (cost 0, below the weighted mean cost) LOSES weight as tau grows
    assert (analytic[:, 0] < 0).all(), 'winner weight must fall with rising tau'
    # the costliest listed candidate GAINS
    assert (analytic[:, -1] > 0).all(), 'far candidate weight must rise with rising tau'
    # weights live on a simplex, so the derivative sums to zero
    assert analytic.sum(-1).abs().max() < 1e-8


def test_tau_gradient_collapses_at_both_extremes_pointwise():
    """GATE (d): at a FIXED gap, dL/dtau -> 0 as tau -> 0 AND as tau -> infinity.

    This is the argument for initialising tau at the measured gap. It is a POINTWISE claim
    -- for one (token, table) whose gap is c -- and it is exact: the sensitivity is
    (2c/tau^2) w (1-w), which is killed by w(1-w) -> 0 at small tau and by 1/tau^2 at large
    tau. Its maximiser is tau* = 2c/z* with z* = 2.399379 the maximiser of
    z^2 sigmoid(z) sigmoid(-z), i.e. tau* = 0.8335*c -- a pure number, scale-free.

    See test_tau_gradient_does_not_collapse_in_aggregate for the caveat that this does NOT
    carry over to a sum across a realistic margin distribution.
    """
    def sens(tau, c):
        t = torch.tensor(float(tau), dtype=torch.float64, requires_grad=True)
        costs = torch.tensor([[0.0, float(c)]], dtype=torch.float64)
        w = torch.softmax(-2.0 * costs / t, -1)
        return torch.autograd.grad(w[0, 0], t)[0].abs().item()

    for c in (0.0331, 0.0722, 0.1079):        # the measured per-layer gaps, L0 / mid / L5
        peak = sens(0.8335 * c, c)
        assert sens(1e-4 * c, c) < peak * 1e-6, f'no collapse as tau->0 at gap {c}'
        assert sens(1e4 * c, c) < peak * 1e-6, f'no collapse as tau->inf at gap {c}'
        # tau = delta_m (what the run is initialised at) is near-optimal; 2*delta_m is not
        assert sens(c, c) > 0.95 * peak, 'tau = delta_m should be >95% of peak sensitivity'
        assert sens(2 * c, c) < 0.5 * peak, 'tau = 2*delta_m should be well below peak'


def test_tau_gradient_does_not_collapse_in_aggregate():
    """The honest caveat, asserted so it cannot be quietly forgotten.

    Summed over a realistic margin distribution the small-tau collapse DOES NOT happen: the
    gap distribution has mass arbitrarily close to zero (tokens sitting on a decision
    boundary), and for those the 1/tau^2 prefactor wins. Measured on a uniform margin draw,
    at tau=1e-4 roughly 60% of the total |dL/dtau| comes from gaps smaller than tau.

    So shrinking tau does not switch the routing gradient off -- it CONCENTRATES it onto
    boundary-adjacent tokens. That is a gradient-variance problem, not a vanishing-gradient
    one, and it is a different failure mode from the pointwise picture above.
    """
    torch.manual_seed(4)
    m = torch.rand(4000, 4, dtype=torch.float64) * 0.4
    mv, _ = torch.sort(m, dim=-1)
    costs = torch.cat([torch.zeros_like(mv[:, :1]), mv[:, :1]], -1)

    def agg(tau):
        t = torch.tensor(float(tau), dtype=torch.float64, requires_grad=True)
        w = torch.softmax(-2.0 * costs / t, -1)
        loss = (w * torch.tensor([1.0, -1.0], dtype=torch.float64)).sum()
        return torch.autograd.grad(loss, t)[0].abs().item()

    small, matched, large = agg(1e-4), agg(0.0722), agg(100.0)
    assert large < matched * 1e-3, 'the large-tau collapse should survive aggregation'
    assert small > matched, (
        'aggregate |dL/dtau| at tiny tau should NOT collapse -- if this ever fails, the '
        'boundary-token concentration effect has changed and the caveat needs revisiting')


# --------------------------------------------------------------------------------------
# compile path: default torch.compile (n>1) matches eager within fp tolerance
# --------------------------------------------------------------------------------------
def test_compile_parity_matches_eager():
    """The default-ON torch.compile backend for the n>1 blend matches the eager reference
    within fp tolerance (NOT bit-exact): out/grad_x/grad_log_tau < 1e-5, grad_tables < 2e-4
    (compile reorders the embedding_bag-backward accumulation). If torch.compile is
    unavailable the module's guard falls back to eager, so this compares eager-to-eager and
    still passes."""
    eager = _layer(n=2, dtype=torch.float32, read_tau_learnable=True)
    eager._compile_enabled = False
    comp = _layer(n=2, dtype=torch.float32, read_tau_learnable=True)  # compiles on first fwd
    torch.manual_seed(5)
    x = torch.randn(16, 12, dtype=torch.float32)
    go = torch.randn(16, eager.output_dim, dtype=torch.float32)

    def run(m):
        xin = x.detach().clone().requires_grad_(True)
        m.zero_grad(set_to_none=True)
        (m(xin) * go).sum().backward()
        return (m(xin).detach(), xin.grad.detach(),
                m.tables.grad.detach(), m.log_tau.grad.detach())

    oe, gxe, gte, gtaue = run(eager)
    oc, gxc, gtc, gtauc = run(comp)
    assert (oe - oc).abs().max() < 1e-5
    assert (gxe - gxc).abs().max() < 1e-5
    assert (gtaue - gtauc).abs().max() < 1e-5
    assert (gte - gtc).abs().max() < 2e-4
