"""Continuity probe for LightMultiHeadLUT (Gen 3) at cell boundaries. Eval only; nothing is trained.

Paper config (table v3): H=4, tph=128, d_in=d_out=48, nap=8, confidence_form='margin', read_top_n=1 (and 2).
(Table v2's random-init numbers were measured with H=8, tph=64.)

For a chosen (head h, table t, anchor pair j) we build an input with u_j = z[a_j] - z[b_j] = 0
EXACTLY, then step u_j to -eps and +eps (eps = 1e-9, float64). The bit j of table t flips, so the
cell c_t changes; every other margin in the model moves by at most eps. We record the margin
score on both sides and at 0, the cell index, and the jump in the head output y_h.

Also: a grid sweep u_j in [-1e-2, 1e-2]; the same probe at read_top_n=2; the n=2 "argmin switch"
(two margins crossing, which changes the SECOND cell); the min_j|u_j| counterfactual; the backward
dependence on the neighbouring cell; and the native fused CUDA eval kernel (fp32).

    LUT_DISABLE_COMPILE=1 python continuity_probe.py
"""
import os
os.environ.setdefault('LUT_DISABLE_COMPILE', '1')
import math                                                         # noqa: E402
import statistics as st                                             # noqa: E402

import torch                                                        # noqa: E402
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT    # noqa: E402
from spiky.lutorch.fast_multi_head_lut import _confidence_score     # noqa: E402

H, TPH, D, NAP = 4, 128, 48, 8   # paper geometry from table v3 (exp_g_0193); v2's numbers used H=8, TPH=64
FORM = os.environ.get('PROBE_FORM', 'margin')   # confidence form under test; 'margin' reproduces table v3
# sharp_margin's exponent; only read when PROBE_FORM=sharp_margin (None -> SHARP_MARGIN_GAMMA)
GAMMA = float(os.environ['PROBE_GAMMA']) if 'PROBE_GAMMA' in os.environ else None
K = 1 << NAP
EPS = 1e-9
N_DRAWS = 300
g = torch.Generator().manual_seed(0)
print(f'PROBE_FORM={FORM}  PROBE_GAMMA={GAMMA}')


def build(n, device='cpu', dtype=torch.float64, seed=1000):
    m = LightMultiHeadLUT(input_dim=D, n_tables=H * TPH, output_dim=D, n_anchor_pairs=NAP,
                          confidence_form=FORM, random_seed=seed, n_heads=H,
                          sharp_margin_gamma=GAMMA if FORM == 'sharp_margin' else None,
                          multi_head_input=True, read_top_n=n, read_tau=0.5, device=device)
    m._compile_enabled = False
    return m.to(dtype)


def forward_torch(m, z):
    """The autograd (training) path: grad mode on, so the fused eval kernel is skipped."""
    with torch.enable_grad():
        return m(z).detach()


def margins(m, z):
    a = m.anchor_a.reshape(1, H, TPH * NAP).expand(z.shape[0], H, TPH * NAP)
    b = m.anchor_b.reshape(1, H, TPH * NAP).expand(z.shape[0], H, TPH * NAP)
    return (torch.gather(z, 2, a) - torch.gather(z, 2, b)).view(z.shape[0], H, TPH, NAP)


def cells(d):
    powers = 2 ** torch.arange(NAP - 1, -1, -1)
    return ((d > 0).long() * powers).sum(-1)                       # MSB-first, as the module


def rows(m):
    return m.tables.detach().reshape(H * TPH, K, D)


def manual_y_n1(m, z, score_fn):
    """y_h = sum_t score_t * W_t[c_t], built from the module's own anchors/tables."""
    d = margins(m, z)
    c = cells(d)
    s = score_fn(d)
    W = rows(m).view(H, TPH, K, D)
    picked = W[torch.arange(H)[:, None], torch.arange(TPH)[None, :], c[0]]   # [H, T, D]
    return (s[0].unsqueeze(-1) * picked).sum(1).unsqueeze(0)


def margin_score(d):
    """The score of the form under test (FORM; 'margin' unless PROBE_FORM is set)."""
    return _confidence_score(d, FORM, 1.0, GAMMA if FORM == 'sharp_margin' else None)


def min_score(d):
    return d.abs().min(dim=-1).values


def set_margin(m, z, h, t, j, u):
    """Return a copy of z with u_j of (h, t) set to exactly u (moves coordinate a_j only)."""
    z = z.clone()
    a, b = int(m.anchor_a[h, t, j]), int(m.anchor_b[h, t, j])
    z[0, h, a] = z[0, h, b] + u
    return z


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def summary(name, xs):
    return (f'{name}: median {st.median(xs):.3g}  p10 {q(xs, .1):.3g}  p90 {q(xs, .9):.3g}  '
            f'min {min(xs):.3g}  max {max(xs):.3g}')


def draw(m):
    z = torch.randn(1, H, D, generator=g, dtype=torch.float64)
    h = int(torch.randint(H, (1,), generator=g))
    t = int(torch.randint(TPH, (1,), generator=g))
    j = int(torch.randint(NAP, (1,), generator=g))
    return set_margin(m, z, h, t, j, 0.0), h, t, j


# ------------------------------------------------------------------------------------------------
print('=' * 100)
print('0. The module output equals the formula  y_h = sum_t s_t W_t[c_t],  s = (sum_j |u_j|) prod_j sigmoid(2|u_j|)')
m1 = build(1)
z = torch.randn(4, H, D, generator=g, dtype=torch.float64)
y_mod = forward_torch(m1, z)
y_man = torch.cat([manual_y_n1(m1, z[i:i + 1], margin_score) for i in range(4)])
print(f'   max |module - formula| = {(y_mod - y_man).abs().max().item():.3e}   (scale |y| ~ {y_mod.abs().mean().item():.3e})')

# ------------------------------------------------------------------------------------------------
for n in (1, 2):
    m = build(n)
    print('=' * 100)
    print(f'1. BOUNDARY PROBE, read_top_n={n}: {N_DRAWS} draws of (random input, head, table, bit), eps={EPS:g}')
    rel_head, rel_table, absj, s0_over_med, flips_ok, ctrl, other_flips = [], [], [], [], 0, [], 0
    for _ in range(N_DRAWS):
        z0, h, t, j = draw(m)
        zl, zr = set_margin(m, z0, h, t, j, -EPS), set_margin(m, z0, h, t, j, +EPS)
        dl, d0, dr = margins(m, zl), margins(m, z0), margins(m, zr)
        cl, c0, cr = cells(dl), cells(d0), cells(dr)
        flips_ok += int(cl[0, h, t] != cr[0, h, t])
        other_flips += int(((cl != cr).sum() - (cl[0, h, t] != cr[0, h, t]).long()).item())
        s0 = margin_score(d0)
        s0_over_med.append((s0[0, h, t] / s0[0].median()).item())
        yl, yr = forward_torch(m, zl), forward_torch(m, zr)
        y_c = forward_torch(m, set_margin(m, z0, h, t, j, 3 * EPS))       # same side, no flip
        jump = (yr - yl)[0, h].norm().item()
        absj.append(jump)
        rel_head.append(jump / yr[0, h].norm().item())
        W = rows(m).view(H, TPH, K, D)
        rel_table.append(((W[h, t, cr[0, h, t]] - W[h, t, cl[0, h, t]]).norm()
                          / W[h, t, cl[0, h, t]].norm()).item())
        ctrl.append(((y_c - yr)[0, h].norm() / yr[0, h].norm()).item())
    print(f'   cell index flips across u_j=0 in {flips_ok}/{N_DRAWS} draws; other (table,bit) flips caused by the step: {other_flips}')
    print('   ' + summary('score at u_j=0 / median score of all tables at that input', s0_over_med))
    print('   ' + summary('|| y_h(+eps) - y_h(-eps) ||  (absolute)', absj))
    print('   ' + summary(f'|| jump || / || y_h ||  (head output, {TPH} tables summed)', rel_head))
    print('   ' + summary('control: same-side step 2*eps, relative', ctrl))
    if n == 1:
        print('   ' + summary('one table:  || W[c+] - W[c-] || / || W[c-] ||', rel_table))

    # one worked example with a grid sweep
    z0, h, t, j = draw(m)
    print(f'   example sweep  (head {h}, table {t}, bit {j}):')
    print(f'   {"u_j":>11} {"cell":>6} {"score s_t":>13} {"||y_h||":>12} {"||y_h - y_h(u=-1e-2)||":>24}')
    ref = forward_torch(m, set_margin(m, z0, h, t, j, -1e-2))[0, h]
    for u in (-1e-2, -1e-3, -1e-6, -EPS, 0.0, EPS, 1e-6, 1e-3, 1e-2):
        zu = set_margin(m, z0, h, t, j, u)
        du = margins(m, zu)
        yu = forward_torch(m, zu)[0, h]
        print(f'   {u:>11.1e} {int(cells(du)[0, h, t]):>6d} {margin_score(du)[0, h, t].item():>13.6e} '
              f'{yu.norm().item():>12.6e} {(yu - ref).norm().item():>24.6e}')

# ------------------------------------------------------------------------------------------------
print('=' * 100)
print('2. read_top_n=2: ARGMIN SWITCH (two margins of one table cross; the main cell stays, the second cell changes)')
m2 = build(2)
rel2 = []
for _ in range(100):
    z = torch.randn(1, H, D, generator=g, dtype=torch.float64)
    h = int(torch.randint(H, (1,), generator=g))
    t = int(torch.randint(TPH, (1,), generator=g))
    j1, j2 = [int(v) for v in torch.randperm(NAP, generator=g)[:2]]
    others = [abs(margins(m2, z)[0, h, t, k].item()) for k in range(NAP) if k not in (j1, j2)]
    r = 0.25 * min(others)                                   # j1, j2 are the two smallest margins
    z = set_margin(m2, z, h, t, j1, r)
    yl = forward_torch(m2, set_margin(m2, z, h, t, j2, r + EPS))   # argmin = j1
    yr = forward_torch(m2, set_margin(m2, z, h, t, j2, r - EPS))   # argmin = j2
    rel2.append(((yr - yl)[0, h].norm() / yr[0, h].norm()).item())
print('   ' + summary('|| jump || / || y_h ||', rel2))

# ------------------------------------------------------------------------------------------------
print('=' * 100)
print('3. COUNTERFACTUAL score = min_j |u_j|  (same anchors, tables, cells; read_top_n=1)')
cf_abs, cf_rel, cf_s0 = [], [], []
for _ in range(N_DRAWS):
    z0, h, t, j = draw(m1)
    cf_s0.append(min_score(margins(m1, z0))[0, h, t].item())
    yl = manual_y_n1(m1, set_margin(m1, z0, h, t, j, -EPS), min_score)
    yr = manual_y_n1(m1, set_margin(m1, z0, h, t, j, +EPS), min_score)
    cf_abs.append((yr - yl)[0, h].norm().item())
    cf_rel.append(((yr - yl)[0, h].norm() / yr[0, h].norm()).item())
print(f'   min-margin score AT u_j=0: max over draws = {max(cf_s0):.1e}')
print('   ' + summary('|| jump || absolute', cf_abs))
print('   ' + summary('|| jump || / || y_h ||', cf_rel))

# ------------------------------------------------------------------------------------------------
print('=' * 100)
print('4. BACKWARD: does anything depend on the neighbouring cell W_t[c^(j*)]?')
for n in (1, 2):
    m = build(n)
    z0 = torch.randn(1, H, D, generator=g, dtype=torch.float64)
    h, t = 3, 17
    d = margins(m, z0)
    jstar = int(d[0, h, t].abs().argmin())
    c = int(cells(d)[0, h, t])
    alt = c ^ (1 << (NAP - 1 - jstar))
    gout = torch.randn(1, H, D, generator=g, dtype=torch.float64)

    def grads(mod):
        mod.zero_grad(set_to_none=True)
        zz = z0.clone().requires_grad_(True)
        (mod(zz) * gout).sum().backward()
        return mod.tables.grad.detach().view(H, TPH, K, D).clone(), zz.grad.detach().clone()

    tg, zg = grads(m)
    with torch.no_grad():
        m.tables.view(H, TPH, K, D)[h, t, alt] += torch.randn(D, generator=g, dtype=torch.float64)
    tg2, zg2 = grads(m)
    yl = forward_torch(m, z0)
    print(f'   read_top_n={n}: |dL/dW_t[c]| = {tg[h, t, c].abs().max().item():.3e}   '
          f'|dL/dW_t[alt]| = {tg[h, t, alt].abs().max().item():.3e}   '
          f'input grad change after perturbing W_t[alt]: {(zg2 - zg).abs().max().item():.3e}')

# ------------------------------------------------------------------------------------------------
print('=' * 100)
print('5. NATIVE FUSED CUDA EVAL KERNEL (fp32, no grad, read_top_n=1) vs the torch path')
if torch.cuda.is_available():
    mc = build(1, device='cuda', dtype=torch.float32)
    e32 = 1e-4
    rel_native, rel_torchc, agree = [], [], []
    for _ in range(100):
        z0, h, t, j = draw(m1)
        z0 = z0.float().cuda()
        zl, zr = set_margin(mc, z0, h, t, j, -e32), set_margin(mc, z0, h, t, j, +e32)
        with torch.no_grad():
            yl_n, yr_n = mc(zl), mc(zr)                          # native fused eval path
        yl_t, yr_t = forward_torch(mc, zl), forward_torch(mc, zr)
        rel_native.append(((yr_n - yl_n)[0, h].norm() / yr_n[0, h].norm()).item())
        rel_torchc.append(((yr_t - yl_t)[0, h].norm() / yr_t[0, h].norm()).item())
        agree.append(max((yl_n - yl_t).abs().max().item(), (yr_n - yr_t).abs().max().item())
                     / yr_t.abs().max().item())
    print('   ' + summary('native kernel: || jump || / || y_h ||', rel_native))
    print('   ' + summary('torch path   : || jump || / || y_h ||', rel_torchc))
    print('   ' + summary('max |native - torch| / max|y|', agree))
else:
    print('   CUDA not available -- skipped')
