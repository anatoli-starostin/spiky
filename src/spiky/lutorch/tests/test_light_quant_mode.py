"""LightMultiHeadLUT quant_mode ("p2_int8"; int4 deliberately deferred): the power-of-two two-cell read of
doc/research/lut_ablation/quantisation_simple.tex, implemented in spiky/lutorch/pow2_read.py.

Gates:
  scalars   q, c_q, round, window / skip / drop, log2 s -- including the COUNTED, BOUNDED differences from the ablation
            prototype's forms (bucketize thresholds, 8-entry c_q table, log2 of the formed score, torch.round half-even)
  packing   int8 round trip, range refusal; the quantiser stays generic in bit width (int4 deferred)
  train=int the straight-through training forward and the int32 shift-add read give the same value (exact in float64 on CPU)
  prototype the training forward (value and gradients) matches a frozen copy of the prototype's read when no boundary case
            occurs, and every boundary case is where the two formula forms are allowed to differ
  default   quant_mode=None leaves every existing path byte-identical and the state_dict keys unchanged
  export    CompressionMultiHeadLUT.export_quantised() matches the module, never mutates it, refuses unsupported layouts,
            and round-trips through its file format

Run with pytest (lutorch tests are pytest, unlike lut_fused / spnet).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from spiky.lutorch import pow2_read as P
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
from spiky.lutorch.quantised_light_ffn import QuantisedLightFFN

LN2 = math.log(2.0)
CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
PROTO_CTAB = [math.log2(1.0 + 2.0 ** -qq) for qq in range(8)]   # the prototype's table (python doubles)


def _ffn(mode="p2_int8", dev="cpu", dtype=torch.float64, seed=3, noise=0.3, tau=0.5, H=4, tph=16, nap=6, din=16, **kw):
    torch.manual_seed(seed)
    ffn = CompressionMultiHeadLUT(input_dim=32, output_dim=32, inner_in_dim=din, inner_out_dim=din, nap=nap, tph=tph,
                                  n_heads=H, lut_impl="light", confidence_form="learned_margin",
                                  learned_margin_freeze_g=True, read_top_n=2, read_tau=tau, read_tau_learnable=True,
                                  random_seed=seed, device=torch.device(dev), quant_mode=mode,
                                  initial_weights_noise=noise, **kw).to(dtype)
    ffn.lut_light._compile_enabled = False
    return ffn


# ------------------------------------------------------------------ config ---------------------------------------------
def test_presets_and_override_validation():
    c8 = P.resolve_quant_config("p2_int8")
    assert (c8["bits"], c8["offset"], c8["Q"], c8["L"], c8["kmax"], c8["lo"], c8["hi"], c8["J"], c8["C"]) == \
        (8, 0, 3, 8, 4, -3, 4, 64, 8)
    c8q = P.resolve_quant_config("p2_int8", {"Q": 1, "L": 6, "kmax": 2})
    assert (c8q["bits"], c8q["offset"], c8q["Q"], c8q["lo"], c8q["hi"]) == (8, 0, 1, -3, 2)
    assert P.resolve_quant_config(None) is None
    for bad in ({"J": 8}, {"C": 4}, {"bits": 4}, {"Q": 4}, {"L": 9}, {"kmax": 5}, {"kmax": 2, "L": 8}, {"Q": 1.0}):
        with pytest.raises(ValueError):
            P.resolve_quant_config("p2_int8", bad)
    for deferred in ("p2_int4", "p2_int2"):                            # int4 deliberately deferred: not a preset
        with pytest.raises(ValueError):
            P.resolve_quant_config(deferred)
    with pytest.raises(ValueError):
        P.resolve_quant_config(None, {"Q": 2})


@pytest.mark.parametrize("bad", [dict(confidence_form="margin"), dict(learned_margin_freeze_g=False), dict(read_top_n=1),
                                 dict(forward_mode="hard"), dict(multi_head_input=False), dict(anchor_mode="single")])
def test_light_refuses_unsupported_configurations(bad):
    kw = dict(input_dim=16, n_tables=16, output_dim=16, n_anchor_pairs=4, confidence_form="learned_margin",
              learned_margin_freeze_g=True, read_top_n=2, n_heads=4, multi_head_input=True, quant_mode="p2_int8")
    kw.update(bad)
    if kw.get("anchor_mode") == "single":
        kw["pool_size"] = 16
    with pytest.raises((ValueError, NotImplementedError)):
        LightMultiHeadLUT(**kw)


def test_compression_refuses_quant_mode_off_the_light_path():
    with pytest.raises(ValueError):
        CompressionMultiHeadLUT(input_dim=32, output_dim=32, inner_in_dim=16, inner_out_dim=16, nap=4, tph=8, n_heads=4,
                                lut_impl="fast", quant_mode="p2_int8")


# ------------------------------------------------------------------ scalars ---------------------------------------------
def test_round_half_up_ties():
    x = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 0.4, -0.6])
    assert P.round_half_up(x).tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 0.0, -1.0]
    # floor(x + 1/2) evaluated in float32: the largest float below 1/2 rounds UP (x + 0.5 == 1.0 after rounding). This is
    # the note's formula taken literally, the same in training and in the integer read, so it is consistent, not a bug.
    assert P.round_half_up(torch.tensor([0.49999997])).item() == 1.0
    # torch.round (the prototype) is half-to-even: the two differ exactly on the ties
    assert (torch.round(x[:6]) != P.round_half_up(x[:6])).sum() == 3


@pytest.mark.parametrize("dev", DEVICES)
def test_c_q_formula_is_bit_identical_to_the_prototype_table(dev):
    q = torch.arange(0, 70, dtype=torch.float32, device=dev)
    table = torch.tensor(PROTO_CTAB, dtype=torch.float32, device=dev)
    ref = torch.where(q < 8, table[q.clamp(max=7).long()], torch.zeros_like(q))
    assert torch.equal(P.c_q(q), ref)
    assert torch.all(P.c_q(q)[8:] == 0)


def _proto_q(mv, tau):
    theta = ((torch.arange(1, P.J + 1, device=mv.device, dtype=torch.float32) - 0.5) * tau * LN2 / 2.0).contiguous()
    return torch.bucketize(mv.contiguous(), theta, right=True).to(torch.float32), theta


@pytest.mark.parametrize("dev", DEVICES)
def test_q_direct_vs_prototype_thresholds_differ_only_within_one_ulp(dev):
    """Random smallest margins: the direct form and the 64-threshold count agree except at float ties, which must be
    within one ulp of a threshold. Adversarial margins placed exactly ON every threshold: count and bound the flips."""
    g = torch.Generator(device="cpu").manual_seed(0)
    tau = torch.tensor(0.4137, dtype=torch.float32, device=dev)
    mv = (torch.rand(2_000_000, generator=g) * 16.0).to(dev)
    qd = P.q_exponent(mv, tau)
    qp, theta = _proto_q(mv, tau)
    diff = qd != qp
    assert int(diff.sum()) <= 20, int(diff.sum())                       # ties are measure-zero for random margins
    if diff.any():
        near = (mv[diff].unsqueeze(-1) - theta).abs().min(dim=-1).values
        assert torch.all(near <= torch.finfo(torch.float32).eps * mv[diff].abs() * 2)
    # adversarial: exactly on the thresholds and one ulp either side
    on = torch.cat([theta, torch.nextafter(theta, theta + 1), torch.nextafter(theta, theta - 1)])
    qd, (qp, _) = P.q_exponent(on, tau), _proto_q(on, tau)
    flips = qd != qp
    assert int(flips.sum()) <= theta.numel()                            # at most one flip per threshold
    assert torch.all((qd[flips] - qp[flips]).abs() == 1)                 # and only by one step


def test_log2_score_log_domain_vs_log_of_formed_score_bounded():
    """The note's log-domain log2 s vs the prototype's log2(s): k' = round(log2 s - c_q) may differ only where the value is
    within float error of a rounding boundary; count those and bound them."""
    g = torch.Generator().manual_seed(1)
    m = torch.rand(400_000, 8, generator=g) * 2.0
    gz, beta, gamma = torch.tensor(0.0), torch.tensor(1.9), torch.tensor(1.2)
    direct = P.log2_score(m, gz, beta, gamma)
    s = m.sum(-1) * torch.exp(gz + gamma * F.logsigmoid(beta * m).sum(-1))
    formed = torch.log2(s.clamp_min(1e-30))
    assert float((direct - formed).abs().max()) < 1e-4
    kd, kp = P.round_half_up(direct), torch.round(formed)
    flips = kd != kp
    assert int(flips.sum()) <= 40, int(flips.sum())                     # 1e-4 of 400k
    frac = formed[flips] - torch.floor(formed[flips])
    assert torch.all((frac - 0.5).abs() < 1e-4)                         # every flip sits on a rounding boundary


def test_window_skip_and_drop():
    cfg = P.resolve_quant_config("p2_int8")
    m = torch.full((5, 8), 1.0)
    mv = torch.tensor([[0.0], [0.01], [0.2], [0.6], [3.0]])
    tau = torch.tensor(0.5)
    g, beta, gamma = torch.tensor(0.0), torch.tensor(2.0), torch.tensor(1.0)
    q, k, skip, drop = P.blend_exponents(m, mv, tau, g, beta, gamma, cfg)
    assert q.tolist() == torch.clamp(torch.floor(mv.squeeze(-1) * (2 / (0.5 * LN2)) + 0.5), 0, 64).tolist()
    assert torch.equal(drop, q > 3)
    assert torch.all((k >= -3) & (k <= 4))
    tiny = torch.full((3, 8), 1e-4)                                     # log2 s far below -3 -> skipped, k clamped to lo
    q, k, skip, drop = P.blend_exponents(tiny, tiny[:, :1], tau, g, beta, gamma, cfg)
    assert torch.all(skip) and torch.all(k == -3)
    zero = torch.zeros(2, 8)                                            # all-zero margins: log2 s = -inf -> skipped
    q, k, skip, drop = P.blend_exponents(zero, zero[:, :1], tau, g, beta, gamma, cfg)
    assert torch.all(skip)


def test_dropped_and_skipped_cells_keep_a_gradient():
    score = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
    mv = torch.tensor([[0.1], [2.0], [0.1]], requires_grad=True)
    tau = torch.tensor(0.5, requires_grad=True)
    q = torch.tensor([0.0, 6.0, 0.0])
    k = torch.tensor([1.0, 1.0, -3.0])
    skip = torch.tensor([False, False, True])
    drop = q > 3
    w = P.ste_blend_weights(score, mv, tau, q, k, skip, drop)
    assert w.detach().tolist() == [[2.0, 2.0], [2.0, 0.0], [0.0, 0.0]]  # values: 2^k', 2^(k'-q), zeros
    w[1, 1].backward(retain_graph=True)                                # dropped second cell
    assert score.grad[1] != 0 and mv.grad[1, 0] != 0 and tau.grad != 0
    score.grad = None
    w[2].sum().backward()                                              # skipped table
    assert score.grad[2] != 0


# ------------------------------------------------------------------ packing ---------------------------------------------
def test_pack_int8_round_trip_and_range():
    vals = torch.arange(-128, 128, dtype=torch.float32)
    W = vals[torch.randint(0, 256, (37, 5, 48), generator=torch.Generator().manual_seed(8))]
    W[0, 0, :48] = vals[:48]
    W[0, 1, :48] = vals[-48:]
    packed = P.pack_tables(W, 8)
    assert packed.dtype == torch.int8 and packed.shape == W.shape
    assert torch.equal(packed.to(torch.int32), W.to(torch.int32))       # whole bytes, read directly
    for bad in (128.0, -129.0):
        with pytest.raises(ValueError):
            P.pack_tables(torch.full((1, 1, 48), bad), 8)
    with pytest.raises(NotImplementedError):                            # no int4 packer yet (deferred)
        P.pack_tables(torch.zeros(1, 1, 48), 4)


def test_quantise_tables_saturation_and_rounding():
    W = torch.tensor([[[0.5, -0.5, 1.5, 300.0, -300.0, 0.0]]])
    e = torch.zeros(1, 6)
    q = P.quantise_tables(W, e, 8)
    assert q.tolist() == [[[1.0, 0.0, 2.0, 127.0, -128.0, 0.0]]]        # half up, then saturate


@pytest.mark.parametrize("bits,offset", [(4, -1), (6, 0), (8, 0)])
def test_quantiser_is_generic_in_bit_width(bits, offset):
    """The quantiser is not hard-wired to int8 (int4 is a deferred preset): exponents, range and saturation follow b and o."""
    H, T, K, D = 2, 3, 4, 5
    W = torch.randn(H * T, K, D, generator=torch.Generator().manual_seed(bits), dtype=torch.float64)
    e = P.head_chan_exponents(W, H, bits, offset)
    A = W.abs().reshape(H, -1, D).amax(dim=1)
    assert torch.equal(e, torch.ceil(torch.log2(A)) - (bits - 1) + offset)
    q = P.quantise_tables(W, e, bits)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    assert q.min() >= lo and q.max() <= hi and torch.equal(q, torch.round(q))
    sc = torch.pow(2.0, e).reshape(H, 1, 1, D).expand(H, T, 1, D).reshape(H * T, 1, D)
    r = W / sc
    inside = (r >= lo - 0.5) & (r < hi + 0.5)
    assert torch.all(((q - r).abs())[inside] <= 0.5)
    if offset < 0:                                                        # a finer step saturates the largest entries
        assert bool((q == lo).any() | (q == hi).any())


# ------------------------------------------------------------------ train value == integer read ------------------------
@pytest.mark.parametrize("mode", ["p2_int8"])
@pytest.mark.parametrize("dev", DEVICES)
def test_training_value_equals_integer_shift_add_read(mode, dev):
    for seed, tau in ((3, 0.5), (4, 0.05), (5, 2.0)):
        ffn = _ffn(mode, dev, seed=seed, tau=tau)
        lut = ffn.lut_light
        x = torch.randn(64, 32, dtype=torch.float64, device=dev, generator=torch.Generator(device=dev).manual_seed(seed))
        z = ffn.compress(x).view(64, 4, 16)
        with torch.no_grad():
            y_train, y_int = lut(z), lut.forward_int(z)
            _, mv, _ = P.blend_candidates(z[:, :, :1].expand(-1, -1, 8).unsqueeze(2), torch.zeros(64, 4, 1, dtype=torch.long,
                                                                                                    device=dev), lut.powers)
        if dev == "cpu":
            assert torch.equal(y_train, y_int)                          # exact: all values are dyadic, sums below 2^53
        else:
            torch.testing.assert_close(y_int, y_train, rtol=1e-12, atol=1e-14)   # CUDA embedding_bag float64 rounding


def test_the_read_exercises_skip_and_drop():
    ffn = _ffn("p2_int8", "cpu", tau=0.05)
    lut = ffn.lut_light
    x = torch.randn(256, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    z = ffn.compress(x).view(256, 4, 16)
    H, T, NAP = 4, 16, 6
    ia = lut.anchor_a.reshape(1, H, T * NAP).expand(256, H, T * NAP)
    ib = lut.anchor_b.reshape(1, H, T * NAP).expand(256, H, T * NAP)
    d = (torch.gather(z, 2, ia) - torch.gather(z, 2, ib)).view(256, H, T, NAP)
    index = ((d > 0).long() * lut.powers.view(1, 1, 1, -1)).sum(-1)
    m, mv, _ = P.blend_candidates(d, index, lut.powers)
    tau, g, beta, gamma = lut._quant_scalars()
    q, k, skip, drop = P.blend_exponents(m, mv, tau, g, beta, gamma, lut._quant)
    assert skip.any() and drop.any() and (~skip & ~drop).any()


# ------------------------------------------------------------------ against the frozen prototype -----------------------
def _prototype_read(lut, z, lo, hi, Q, wbits, woffset):
    """Frozen copy of lut_ablation/.../pow2_blend_read.py::_p2b_blend (no mantissa, no export, no counters) on the
    multi-head path. Returns the read and its per-table integers."""
    B, H, T, NAP = z.shape[0], lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs
    ia = lut.anchor_a.reshape(1, H, T * NAP).expand(B, H, T * NAP)
    ib = lut.anchor_b.reshape(1, H, T * NAP).expand(B, H, T * NAP)
    d = (torch.gather(z, 2, ia) - torch.gather(z, 2, ib)).view(B, H, T, NAP)
    index = ((d.detach() > 0).to(torch.int64) * lut.powers.view(1, 1, 1, -1)).sum(dim=-1)
    score = lut.confidence_score(d)
    W = lut.tables
    D = W.shape[-1]
    with torch.no_grad():
        A = W.detach().abs().reshape(H, T, -1, D).amax(dim=(1, 2))
        s = torch.ceil(torch.log2(A.float().clamp_min(1e-30))) - (wbits - 1) + woffset
        sc = torch.pow(2.0, s).reshape(H, 1, 1, D).expand(H, T, 1, D).reshape(H * T, 1, D)
        Wint = torch.clamp(torch.round(W.detach() / sc), -2 ** (wbits - 1), 2 ** (wbits - 1) - 1)
        Wq = Wint * sc
    flat = (Wq + (W - W.detach())).reshape(-1, D)
    m = d.abs()
    mv, mj = m.min(dim=-1, keepdim=True)
    bits = (d.detach() > 0).to(torch.int64)
    pw = lut.powers[mj]
    bsel = torch.gather(bits, -1, mj)
    idx = torch.cat([index.unsqueeze(-1), index.unsqueeze(-1) + pw * (1 - 2 * bsel)], dim=-1)
    tau = lut.read_tau.float()
    sf = score.float().unsqueeze(-1)
    x = 2.0 * mv.float() / tau
    e1, e2 = sf * torch.sigmoid(x), sf * torch.sigmoid(-x)
    with torch.no_grad():
        theta = ((torch.arange(1, 65, device=mv.device, dtype=torch.float32) - 0.5) * tau.detach() * LN2 / 2.0).contiguous()
        q = torch.bucketize(mv.detach().float().contiguous(), theta, right=True).to(torch.float32)
        ctab = torch.tensor(PROTO_CTAB, dtype=torch.float32, device=mv.device)
        cq = torch.where(q < 8, ctab[q.clamp(max=7).long()], torch.zeros_like(q))
        l2 = torch.log2(sf.detach().clamp_min(1e-30))
        kr = torch.round(l2 - cq)
        skip, above = kr < lo, kr > hi
        kc = kr.clamp(lo, hi)
        drop = q > Q
        b1, b2 = torch.pow(2.0, kc), torch.pow(2.0, kc - q)
        v1 = torch.where(skip, torch.zeros_like(b1), b1)
        v2 = torch.where(skip | drop, torch.zeros_like(b2), b2)
    s1 = e1 * (b1 / e1.detach().clamp_min(1e-30))
    s2 = e2 * (b2 / e2.detach().clamp_min(1e-30))
    psw = torch.cat([v1 + (s1 - s1.detach()), v2 + (s2 - s2.detach())], dim=-1)
    flat_idx = (idx + lut.table_offset.view(1, H, T, 1)).reshape(-1)
    offsets = torch.arange(B * H, device=flat.device, dtype=torch.long) * (T * 2)
    out = F.embedding_bag(flat_idx, flat, offsets=offsets, mode="sum", per_sample_weights=psw.reshape(-1).to(flat.dtype))
    return out.view(B, H, D), dict(q=q.squeeze(-1), kc=kc.squeeze(-1), skip=skip.squeeze(-1), Wint=Wint,
                                     l2=l2.squeeze(-1), cq=cq.squeeze(-1))


@pytest.mark.parametrize("mode", ["p2_int8"])
@pytest.mark.parametrize("dev", DEVICES)
def test_training_forward_matches_frozen_prototype(mode, dev):
    """float32, as trained. With this seed no boundary case occurs (asserted), so value and gradients must agree."""
    cfg = P.resolve_quant_config(mode)
    ffn_a = _ffn(mode, dev, dtype=torch.float32, seed=11)
    ffn_b = _ffn(mode, dev, dtype=torch.float32, seed=11)
    la, lb = ffn_a.lut_light, ffn_b.lut_light
    x = torch.randn(96, 32, device=dev, generator=torch.Generator(device=dev).manual_seed(2))
    za = ffn_a.compress(x).view(96, 4, 16)
    zb = ffn_b.compress(x).view(96, 4, 16)
    ya = la(za)
    yb, ref = _prototype_read(lb, zb, cfg["lo"], cfg["hi"], cfg["Q"], cfg["bits"], cfg["offset"])
    # integers first: count every difference between the two formula forms
    B, H, T, NAP = 96, 4, 16, 6
    with torch.no_grad():
        ia = la.anchor_a.reshape(1, H, T * NAP).expand(B, H, T * NAP)
        ib = la.anchor_b.reshape(1, H, T * NAP).expand(B, H, T * NAP)
        d = (torch.gather(za, 2, ia) - torch.gather(za, 2, ib)).view(B, H, T, NAP)
        index = ((d > 0).long() * la.powers.view(1, 1, 1, -1)).sum(-1)
        m, mv, _ = P.blend_candidates(d, index, la.powers)
        tau, g, beta, gamma = la._quant_scalars()
        q, k, skip, drop = P.blend_exponents(m, mv, tau, g, beta, gamma, cfg)
        e = P.head_chan_exponents(la.tables, H, cfg["bits"], cfg["offset"])
        Wint = P.quantise_tables(la.tables, e, cfg["bits"])
    assert int((q != ref["q"]).sum()) == 0
    assert int((k != ref["kc"]).sum()) == 0
    assert int((skip != ref["skip"]).sum()) == 0
    assert int((Wint != ref["Wint"]).sum()) == 0
    torch.testing.assert_close(ya, yb, rtol=1e-6, atol=1e-7)
    (ya * torch.linspace(-1, 1, ya.numel(), device=dev).view_as(ya)).sum().backward()
    (yb * torch.linspace(-1, 1, yb.numel(), device=dev).view_as(yb)).sum().backward()
    for (na, pa), (nb, pb) in zip(ffn_a.named_parameters(), ffn_b.named_parameters()):
        assert na == nb
        if pa.grad is None:
            assert pb.grad is None, na
            continue
        # tolerance relative to the gradient's scale: CUDA's gather backward is itself non-deterministic at ~1e-7
        torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-5, atol=1e-5 * float(pb.grad.abs().max()), msg=na)


def test_weight_rounding_differs_from_prototype_only_on_exact_ties():
    W = torch.randn(64, 32, 16, generator=torch.Generator().manual_seed(4)) * 0.3
    W[0, 0, :4] = torch.tensor([0.25, -0.25, 0.75, -0.75])             # force exact .5 ties at scale 2^-1 below
    e = torch.full((4, 16), -1.0)
    ours = P.quantise_tables(W, e, 8)
    proto = torch.clamp(torch.round(W / 0.5), -128, 127)
    diff = ours != proto
    assert int(diff.sum()) >= 2                                         # the forced ties do show up
    r = (W / 0.5)[diff]
    assert torch.all(r - torch.floor(r) == 0.5)                         # and every difference is an exact tie


# ------------------------------------------------------------------ default-off guard ---------------------------------
@pytest.mark.parametrize("n", [1, 2])
def test_quant_mode_none_is_byte_identical(n):
    kw = dict(input_dim=16, n_tables=16, output_dim=8, n_anchor_pairs=5, confidence_form="learned_margin",
              learned_margin_freeze_g=True, read_top_n=n, n_heads=4, multi_head_input=True, random_seed=5,
              read_tau=0.3, read_tau_learnable=True, initial_weights_noise=0.1)
    a = LightMultiHeadLUT(**kw)
    b = LightMultiHeadLUT(**kw, quant_mode=None, quant_overrides=None)
    q = LightMultiHeadLUT(**{**kw, "read_top_n": 2}, quant_mode="p2_int8")
    for m in (a, b, q):
        m._compile_enabled = False
    assert list(a.state_dict()) == list(b.state_dict()) == list(q.state_dict())       # checkpoints interchange
    for (k1, v1), (k2, v2) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(v1, v2), k1
    x = torch.randn(33, 4, 16, generator=torch.Generator().manual_seed(1))
    xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ya, yb = a(xa), b(xb)
    assert torch.equal(ya, yb)
    ya.pow(2).sum().backward()
    yb.pow(2).sum().backward()
    assert torch.equal(xa.grad, xb.grad)
    for (na, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is None:                                             # e.g. log_tau at n=1 (no blend)
            assert pb.grad is None, na
            continue
        assert torch.equal(pa.grad, pb.grad), na
    # and the default read is still the unquantised blend, bit for bit
    if n == 2:
        with torch.no_grad():
            H, T, NAP = 4, 4, 5
            ia = a.anchor_a.reshape(1, H, T * NAP).expand(33, H, T * NAP)
            ib = a.anchor_b.reshape(1, H, T * NAP).expand(33, H, T * NAP)
            d = (torch.gather(x, 2, ia) - torch.gather(x, 2, ib)).view(33, H, T, NAP)
            index = ((d > 0).long() * a.powers.view(1, 1, 1, -1)).sum(-1)
            flat = a.tables.reshape(H * T * a.table_size, 8)
            ref = a._blend_bag(d, index, a.table_offset.view(1, H, T), flat, a.confidence_score(d), 33 * H, T)
        assert torch.equal(ya.detach(), ref.view(33, H, 8))


def test_state_dict_loads_across_quant_mode():
    kw = dict(input_dim=16, n_tables=16, output_dim=8, n_anchor_pairs=5, confidence_form="learned_margin",
              learned_margin_freeze_g=True, read_top_n=2, n_heads=4, multi_head_input=True, random_seed=5,
              read_tau_learnable=True)
    plain = LightMultiHeadLUT(**kw)
    quant = LightMultiHeadLUT(**{**kw, "random_seed": 6}, quant_mode="p2_int8")
    quant.load_state_dict(plain.state_dict(), strict=True)
    assert torch.equal(quant.tables, plain.tables)


# ------------------------------------------------------------------ export ---------------------------------------------
@pytest.mark.parametrize("mode", ["p2_int8"])
def test_export_matches_module_and_never_mutates_it(mode, tmp_path):
    ffn = _ffn(mode, "cpu")
    before = {k: v.clone() for k, v in ffn.state_dict().items()}
    art = ffn.export_quantised()
    assert isinstance(art, QuantisedLightFFN)
    for k, v in ffn.state_dict().items():
        assert torch.equal(before[k], v), k
    x = torch.randn(50, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        ref = ffn(x)
    out = art(x)
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-13)
    assert art.table_bytes() == 4 * 16 * 64 * 16                        # one byte per int8 entry
    path = tmp_path / "layer.pt"
    art.to_file(str(path))
    back = QuantisedLightFFN.from_file(str(path))
    assert torch.equal(back(x), out)
    assert all(not p.requires_grad for p in art.buffers() if p.is_floating_point())


@pytest.mark.parametrize("bad", [dict(z_norm=True), dict(inner_residual=True)])
def test_export_refuses_when_something_sits_between_sum_and_decompress(bad):
    ffn = _ffn("p2_int8", "cpu", **bad)
    with pytest.raises(ValueError):
        ffn.export_quantised()


def test_export_refuses_without_quant_mode():
    ffn = _ffn(None, "cpu")
    with pytest.raises(ValueError):
        ffn.export_quantised()


@pytest.mark.parametrize("dev", DEVICES)
def test_int8_accumulation_chunked_equals_single_and_explicit_shift_add(dev):
    """The integer read is exact whatever the chunking, and equals an explicit per-cell (row << shift) sum."""
    g = torch.Generator().manual_seed(12)
    N, H, T, K, D = 37, 4, 16, 64, 16
    packed = torch.randint(-128, 128, (H * T * K, D), generator=g).to(torch.int8).to(dev)
    fi = torch.randint(0, H * T * K, (N, H, T, 2), generator=g).to(dev)
    q = torch.randint(0, 7, (N, H, T), generator=g).float().to(dev)
    k = torch.randint(-3, 5, (N, H, T), generator=g).float().to(dev)
    skip = (torch.rand(N, H, T, generator=g) < 0.2).to(dev)
    drop = q > 3
    group = P.shift_groups(q, k, skip, drop)
    P.check_shift_groups(group)
    one = P.int8_blend_read(packed, D, fi, group, chunk_bags=None)
    chunked = P.int8_blend_read(packed, D, fi, group, chunk_bags=7)
    rows = packed[fi].to(torch.int64)                                  # [N, H, T, 2, D]
    w = torch.where(group < P.N_SHIFTS, torch.pow(2, group.clamp(max=62)), torch.zeros_like(group))
    explicit = (rows * w.unsqueeze(-1)).sum(dim=(2, 3))
    assert torch.equal(one, chunked) and torch.equal(one.to(torch.int64), explicit)
    with pytest.raises(ValueError):
        P.check_shift_groups(torch.full((1, 1, 1, 2), -1, dtype=torch.long, device=dev))


@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_compiled_artefact_matches_eager_artefact():
    ffn = _ffn("p2_int8", "cuda", dtype=torch.float32, seed=31)
    art = ffn.export_quantised()
    x = torch.randn(300, 32, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3))
    art._compile_enabled = False                                        # this gate is the torch read, not the CUDA kernel
    eager = art._forward_torch(x)
    art._compile_enabled = True
    compiled = art._forward_torch(x)
    assert art._compiled is not None
    torch.testing.assert_close(compiled, eager, rtol=1e-6, atol=1e-6)
    with torch.no_grad():
        torch.testing.assert_close(eager, ffn(x), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_compiled_quant_forward_matches_eager():
    ffn = _ffn("p2_int8", "cuda", dtype=torch.float32, seed=21)
    lut = ffn.lut_light
    z = ffn.compress(torch.randn(128, 32, device="cuda")).view(128, 4, 16).detach()
    eager = lut(z)
    lut._compile_enabled = True
    lut._compiled_fwd = None
    compiled = lut(z)
    torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-6)
