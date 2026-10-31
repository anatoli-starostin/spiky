"""CPU unit tests for DominanceLUT (CUDA test skipped when unavailable). Run: pytest test_dominance_lut.py -q"""
import itertools
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dominance_lut_ref as dl
from dominance_lut_ref import DominanceLUT, LUTSpec, borda_reference, dominance_matrix, anchors_equal


def spec(h_in=1, h_out=1, tph=4, nap=3, d_in=8, d_out=8):
    return LUTSpec(h_in=h_in, h_out=h_out, tph=tph, nap=nap, d_in=d_in, d_out=d_out)


def rows_of(m, x):
    """independent read: delta rows [B,G,tph,nop] of the addressed cells (python loops)."""
    _, c, _, _ = m.addresses(x)
    B, G, T = c.shape
    out = torch.empty(B, G, T, m.nop)
    for b, g, t in itertools.product(range(B), range(G), range(T)):
        out[b, g, t] = m.weights[g, t, c[b, g, t]]
    return out


# (1) shapes / dtypes / device
@pytest.mark.parametrize("hh", [(1, 1), (3, 3), (1, 3), (3, 1)])
def test_shapes_dtypes(hh):
    m = DominanceLUT(spec(*hh), output_anchors="random", nop=5)
    x = torch.randn(7, hh[0], 8)
    y = m(x)
    assert y.shape == (7, hh[1], 8) and y.dtype == torch.float32 and y.device == x.device
    assert m.weights.shape == (max(hh), 4, 8, 5)
    assert m.anchor_a.shape == (max(hh), 4, 3) and m.out_a.shape == (max(hh), 4, 5)
    with pytest.raises(ValueError):
        m(torch.randn(7, hh[0], 9))
    assert m.double()(x.double()).dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_matches_cpu():
    m = DominanceLUT(spec(2, 2), output_anchors="random", nop=4)
    x = torch.randn(16, 2, 8)
    y = m(x)
    yc = m.cuda()(x.cuda())
    assert yc.is_cuda and torch.allclose(yc.cpu(), y, atol=1e-5)
    yc.sum().backward()
    assert m.weights.grad.is_cuda


# (2) scatter route == explicit D route == brute force
@pytest.mark.parametrize("vote", ["delta", "sign"])
@pytest.mark.parametrize("normalize", ["none", "count"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_borda_vs_explicit_D(vote, normalize, seed):
    g = torch.Generator().manual_seed(seed)
    G, tph, nap, nop, d = 2, 5, 3, 6, 7
    pairs = torch.randint(0, d, (G, tph, nop, 2), generator=g)
    pairs[..., 1] = (pairs[..., 0] + 1 + torch.randint(0, d - 1, (G, tph, nop), generator=g)) % d   # a != b
    pairs[:, 0, 1] = pairs[:, 0, 0]                                   # repeated pair inside a table
    pairs[:, 1, 2] = pairs[:, 1, 2].flip(-1)                          # reversed copy elsewhere
    m = DominanceLUT(spec(G, G, tph, nap, d, d), nop=nop, output_anchors="given", output_pairs=pairs,
                     vote=vote, normalize=normalize, seed=seed)
    x = torch.randn(9, G, d, generator=g)
    y = m(x)
    votes = rows_of(m, x)
    votes = votes.sign() if vote == "sign" else votes
    ref = borda_reference(votes, m.out_a, m.out_b, d, normalize)
    assert torch.allclose(y, ref, atol=1e-6)
    # dense debug path agrees too
    m.dense_D = True
    assert torch.allclose(m(x), ref, atol=1e-6)
    # and a fully independent per-output formula
    y2 = torch.zeros_like(ref)
    for b, gg, t, k in itertools.product(range(9), range(G), range(tph), range(nop)):
        a_, b_ = int(m.out_a[gg, t, k]), int(m.out_b[gg, t, k])
        v = votes[b, gg, t, k]
        w = m.inv_count[gg] if normalize == "count" else torch.ones(d)
        y2[b, gg, a_] += v * w[a_]
        y2[b, gg, b_] -= v * w[b_]
    assert torch.allclose(y, y2, atol=1e-6)


# (3) antisymmetry and zero-sum
def test_antisymmetric_zero_sum():
    m = DominanceLUT(spec(2, 2, 6, 3, 9, 9), nop=4, output_anchors="random")
    x = torch.randn(5, 2, 9)
    votes = rows_of(m, x)
    D = dominance_matrix(votes, m.out_a, m.out_b, 9)
    assert torch.allclose(D, -D.transpose(-1, -2))
    assert torch.allclose(D.diagonal(dim1=-2, dim2=-1), torch.zeros(5, 2, 9))
    y = m(x)
    assert y.sum(-1).abs().max() < 1e-5            # outputs of each head sum to zero
    assert DominanceLUT(spec(3, 1), output_anchors="random")(torch.randn(4, 3, 8)).sum(-1).abs().max() < 1e-5  # fan-in
    # consequence: a bias breaks it (and is the only way to add a common offset)
    mb = DominanceLUT(spec(), output_anchors="random", add_bias=True)
    with torch.no_grad():
        mb.bias.fill_(1.0)
    assert (mb(torch.randn(4, 1, 8)).sum(-1) - 8).abs().max() < 1e-5
    # rank: per table the delta->output map is an incidence matrix of rank <= min(nop, d_out-1); total <= d_out-1
    M = torch.zeros(9, 4 * 4)
    for t in range(4):
        for k in range(4):
            M[int(m.out_a[0, t, k]), t * 4 + k] += 1
            M[int(m.out_b[0, t, k]), t * 4 + k] -= 1
    assert torch.linalg.matrix_rank(M) <= 8 and M.sum(0).abs().max() == 0


# (4) swapping an output pair negates that delta's contribution
def test_swap_pair_negates():
    m = DominanceLUT(spec(), nop=3, output_anchors="random", seed=3)
    x = torch.randn(6, 1, 8)
    y = m(x)
    pairs = torch.stack([m.out_a, m.out_b], -1).clone()
    t, k = 2, 1
    pairs[0, t, k] = pairs[0, t, k].flip(0)
    m2 = DominanceLUT.like(m, output_pairs=pairs, weight_seed=0)
    m2.weights.data.copy_(m.weights.data)
    y2 = m2(x)
    _, c, _, _ = m.addresses(x)
    delta = m.weights[0, t, c[:, 0, t], k]                      # [B]
    a, b = int(m.out_a[0, t, k]), int(m.out_b[0, t, k])
    contrib = torch.zeros(6, 8)
    contrib[:, a] += delta
    contrib[:, b] -= delta
    assert torch.allclose(y2 - y, -2 * contrib[:, None], atol=1e-6)


# (5) addresses == the standard ManifestoHardLUT's
def test_addresses_match_manifesto():
    mh = pytest.importorskip("spiky.lutorch_ex").ManifestoHardLUT
    for hh in [(1, 1), (2, 2), (1, 2), (2, 1)]:
        sp = spec(*hh, tph=6, nap=4, d_in=12, d_out=12)
        std = mh(sp, seed=5)
        m = DominanceLUT(sp, nop=4, output_anchors="random", seed=5)
        assert torch.equal(std.anchor_a, m.anchor_a) and torch.equal(std.anchor_b, m.anchor_b)
        x = torch.randn(10, hh[0], 12)
        _, c, c_alt, u_abs = m.addresses(x)
        _, _, c_s, _, ua_s, calt_s = std._addresses(x)
        assert torch.equal(c, c_s.long()) and torch.equal(c_alt, calt_s.long()) and torch.allclose(u_abs, ua_s)


# (6) determinism / seeding / 'same'
def test_seeding_and_same():
    sp = spec(2, 2, 5, 3, 8, 8)
    a = DominanceLUT(sp, output_anchors="random", seed=1)
    b = DominanceLUT(sp, output_anchors="random", seed=1)
    c = DominanceLUT(sp, output_anchors="random", seed=2)
    assert anchors_equal(a, b) and torch.equal(a.weights, b.weights)
    assert not anchors_equal(a, c) and not torch.equal(a.weights, c.weights)
    s = DominanceLUT(sp, output_anchors="same", seed=1)
    assert torch.equal(s.out_a, s.anchor_a) and torch.equal(s.out_b, s.anchor_b)
    assert bool((a.out_a != a.out_b).all())
    assert bool((a.out_a < a.out_b).all())                         # random output pairs are canonical a<b
    for g, t in itertools.product(range(2), range(5)):            # distinct unordered pairs per table
        pr = {tuple(sorted((int(p), int(q)))) for p, q in zip(a.out_a[g, t], a.out_b[g, t])}
        assert len(pr) == a.nop
    with pytest.raises(ValueError):
        DominanceLUT(spec(d_in=8, d_out=9), output_anchors="same")
    with pytest.raises(ValueError):
        DominanceLUT(spec(), output_anchors="same", nop=4)
    DominanceLUT(spec(), output_anchors="random", nop=4)           # nop != nap fine otherwise
    with pytest.raises(ValueError):
        DominanceLUT(spec(), output_anchors="given")
    with pytest.raises(ValueError):
        DominanceLUT(spec(), output_anchors="given", output_pairs=torch.zeros(1, 4, 3, 2, dtype=torch.long))  # a == b
    st = DominanceLUT.like(a, weight_seed=99)
    assert anchors_equal(a, st) and not torch.equal(a.weights, st.weights)


# (7) gradient flow: only selected rows, matches a manual one-hot computation
@pytest.mark.parametrize("vote", ["delta", "sign"])
def test_table_gradients(vote):
    m = DominanceLUT(spec(1, 1, 4, 3, 8, 8), nop=4, output_anchors="random", vote=vote)
    x = torch.randn(11, 1, 8)
    w = torch.randn(11, 1, 8)
    (m(x) * w).sum().backward()
    _, c, _, _ = m.addresses(x)
    manual = torch.zeros_like(m.weights)
    for b, t, k in itertools.product(range(11), range(4), range(4)):
        a_, b_ = int(m.out_a[0, t, k]), int(m.out_b[0, t, k])
        manual[0, t, int(c[b, 0, t]), k] += w[b, 0, a_] - w[b, 0, b_]     # sign vote: straight-through = identity
    assert torch.allclose(m.weights.grad, manual, atol=1e-5)
    sel = torch.zeros(4, 8, dtype=torch.bool)
    for b, t in itertools.product(range(11), range(4)):
        sel[t, int(c[b, 0, t])] = True
    assert (m.weights.grad[0][~sel] == 0).all() and (m.weights.grad[0][sel].abs().sum(-1) > 0).any()


def test_input_gradient_like_standard():
    mh = pytest.importorskip("spiky.lutorch_ex").ManifestoHardLUT
    m = DominanceLUT(spec(), output_anchors="random")
    x = torch.randn(5, 1, 8, requires_grad=True)
    m(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    std = mh(spec(), seed=0)                                   # the standard layer also passes an input gradient
    x2 = torch.randn(5, 1, 8, requires_grad=True)
    std(x2).pow(2).sum().backward()
    assert x2.grad is not None and x2.grad.abs().sum() > 0
    xn = torch.randn(5, 1, 8)
    m(xn).sum().backward()                                     # no input grad requested -> no surrogate, no crash


# (8) one-layer learnability
def _fit(teacher, student, steps=300, lr=0.05, n=2048, seed=0, d=None):
    g = torch.Generator().manual_seed(seed)
    d = d or teacher.spec.d_in
    x = torch.randn(n, 1, d, generator=g)
    with torch.no_grad():
        y = teacher(x)
    base = y.pow(2).mean().item()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    first = None
    for _ in range(steps):
        loss = F.mse_loss(student(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    return F.mse_loss(student(x), y).item() / base, first / base


def test_learn_same_anchors():
    sp = spec(1, 1, 4, 3, 10, 10)
    teacher = DominanceLUT(sp, output_anchors="same", seed=0, weight_seed=100)
    student = DominanceLUT.like(teacher, weight_seed=200)
    rel, _ = _fit(teacher, student)
    assert rel < 0.1, rel


def test_learn_sign_vote_loss_decreases():
    sp = spec(1, 1, 4, 3, 10, 10)
    teacher = DominanceLUT(sp, output_anchors="same", seed=0, weight_seed=100)
    student = DominanceLUT.like(teacher, vote="sign", weight_seed=200)
    rel, first = _fit(teacher, student, steps=200, lr=0.02)
    assert rel < first, (rel, first)


# (9) output scale at init
def test_output_scale_default():
    m = DominanceLUT(spec(1, 1, 32, 8, 48, 48), output_anchors="random")
    assert abs(m.weight_init_std - (48 / (2 * 32 * 8)) ** 0.5) < 1e-12
    y = m(torch.randn(512, 1, 48))
    assert 0.9 < y.std().item() < 1.1, y.std().item()
    ys = DominanceLUT(spec(1, 1, 32, 8, 48, 48), output_anchors="same")(torch.randn(512, 1, 48))
    assert 0.9 < ys.std().item() < 1.1, ys.std().item()
    # the spec's suggestion 1/sqrt(tph*nop) alone would give sqrt(2/48) ~ 0.20
    m2 = DominanceLUT(spec(1, 1, 32, 8, 48, 48), output_anchors="random", weight_init_std=(32 * 8) ** -0.5)
    assert 0.15 < m2(torch.randn(512, 1, 48)).std().item() < 0.26


# (10) random output anchors: same-anchor student learns; different-anchor student is only reported
def test_random_output_anchors_student():
    sp = spec(1, 1, 4, 3, 10, 10)
    teacher = DominanceLUT(sp, nop=3, output_anchors="random", seed=0, weight_seed=100)
    same = DominanceLUT.like(teacher, weight_seed=200)
    rel, _ = _fit(teacher, same)
    assert rel < 0.1, rel
    diff = DominanceLUT(sp, nop=3, output_anchors="random", seed=0, output_seed=777, weight_seed=200)
    assert anchors_equal(teacher, diff, output=False) and not anchors_equal(teacher, diff)
    rel_d, _ = _fit(teacher, diff)
    print(f"\n[info] relative MSE: same output anchors {rel:.4f}; different output anchors {rel_d:.4f}")


# (11) sample_output_pairs: canonical, distinct per table, seeded-deterministic, unordered sets independent of orientation
def test_sample_output_pairs_canonical():
    from dominance_lut_ref import sample_output_pairs
    p = sample_output_pairs(12, 3, 6, 5, seed=7)
    assert p.shape == (3, 6, 5, 2) and bool((p[..., 0] < p[..., 1]).all())
    for g, t in itertools.product(range(3), range(6)):
        assert len({(int(a), int(b)) for a, b in p[g, t]}) == 5
    assert torch.equal(p, sample_output_pairs(12, 3, 6, 5, seed=7))
    assert not torch.equal(p, sample_output_pairs(12, 3, 6, 5, seed=8))
    with pytest.raises(ValueError):
        sample_output_pairs(3, 1, 1, 4, seed=0)                    # nop > C(3,2)
