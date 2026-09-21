# PC (A, symmetric) and PC-ALM (B, constraint) over paired forward/backward LightMHL — derivation v2

Supersedes DERIVATION.md where they differ (the couplings are now fixed by the brief). Derivation only.
**[CODE]** = read from the repo, **[DERIVED]**, **[ASSUMPTION]**, **[FLAG]** = something the brief leaves
under-determined that I think must be settled before coding.

## 0. Grounded facts

* **No PC/ALM-for-LUT code exists** beyond what I wrote for this study (`research/pcalm_lut/pcalm.py`), and no
  paired / inverse / backward LUT construct exists anywhere in the repo. Confirmed, not re-searched. [CODE]
* **Residual convention** [CODE] (`lut_stack.py`, the paper's App. B topology):
  `h_1 = a_1 W_1 x`, `h_i = h_{i-1} + a_i LUT_i(h_{i-1})`, `yhat = a_L W_L h_{L-1}`.
* **Pre-multiplier** [CODE]: `a_i = 1/sqrt(L*N)`. Your 0.0884 is `1/sqrt(4*32)` — the depth-4 width-32 case.
  At L = 64, N = 32 it is `1/sqrt(2048) = 0.0221`. Depth-dependent, not a constant.
* **compress / decompress do NOT wrap the LUT in the 32->32 configuration** [CODE]: the layer is a bare
  LightMHL 32 -> 32 plus the residual add. They exist only in the LM stack (`CompressionMHL`, 384 -> H*48 -> 384).
* **LightMHL config under study** [CODE]: `read_top_n=2, forward_mode='scored', confidence_form='margin',
  cell_mode='constant', 32->32, n_tables=16, table_size=256 (nap=8), tau=0.5 learnable (log_tau),
  multi_head_input=False, output_heads=1`; parameters per layer: `tables [16,256,32]` and the scalar `log_tau`.

## 1. Layer semantics and Jacobian [CODE for the forward, DERIVED for the Jacobian]

Per table t: `d_j = z[a_j] - z[b_j]`, `m_j = |d_j|`, `c_t = pack(sign(d.detach()))`,
`s_t = (sum_j m_j) prod_j sigmoid(2 m_j)`, `j* = argmin_j m_j`, `c'_t = c_t ^ bit(j*)`,
`w_0 = sigmoid(2 m_{j*}/tau)`, `w_1 = 1 - w_0`, and

```
LUT(z) = sum_t s_t ( w_0 V[t,c_t] + w_1 V[t,c'_t] ) =: sum_t s_t u_t,      Delta_t := V[t,c_t] - V[t,c'_t]
```

```
dm_j/dz   = sign(d_j) (e_{a_j} - e_{b_j})
ds_t/dm_j = prod_k sigmoid(2 m_k) * [ 1 + 2 (sum_k m_k) (1 - sigmoid(2 m_j)) ]
sigma_t   := ds_t/dz = sum_j (ds_t/dm_j) sign(d_j)(e_{a_j} - e_{b_j})            (supported on 2*NAP coords)
omega_t   := dw_0/dz = (2/tau) w_0 w_1 sign(d_{j*}) (e_{a_{j*}} - e_{b_{j*}})    (ONE anchor pair)
J_LUT(z)  = sum_t [ u_t sigma_t^T + s_t Delta_t omega_t^T ]                      (2T rank-1 terms)
J_LUT^T v = sum_t [ sigma_t (u_t . v) + s_t omega_t (Delta_t . v) ]
```

For the residual layer `f_i(h) = h + a_i LUT^f_i(h)`: `J_{f_i} = I + a_i J_{LUT^f_i}`, and likewise
`g_i(h) = h + b_i LUT^g_i(h)`, `J_{g_i} = I + b_i J_{LUT^g_i}`. **[ASSUMPTION]** g is residual too and has its
own tables/anchors/tau; if g is meant non-residual, drop the `I` in its Jacobian everywhere below.

**Vanishing terms**: `dc/dz = 0`, `dV/dz = 0` — the address is detached, so the gathered content enters only
through the constants `u_t`, `Delta_t`. No term survives from the piecewise-constant gather.

## 2. Experiment A — symmetric PC [DERIVED]

Energy as fixed by the brief, with `h_0 = x` clamped:

```
E({h}) = sum_{i=1}^{L-1} [ || h_i - f_i(h_{i-1}) ||^2 + || h_{i-1} - g_i(h_i) ||^2 ]
r^f_i := h_i - f_i(h_{i-1}),   r^b_i := h_{i-1} - g_i(h_i)
```

**[FLAG] As written this energy has no data term and is minimised by any consistent trajectory** (in the
residual form, `h_i = h_{i-1}` with zero tables is a global minimum at E = 0). It only becomes a learning
problem if the top is pinned as well — either by clamping `h_{L-1}` (or a readout of it) to the target, or by
adding `1/2||a_L W_L h_{L-1} - y||^2` to E. Both are standard PC; they give different gradients. I write the
gradients for the *unpinned* energy and add the supervised term separately as `+ [i = L-1] a_L W_L^T (yhat - y)`
so either choice can be read off. **This is the single most important thing to settle before coding.**

`h_i` appears in exactly four terms (for `1 <= i <= L-1`; the first and last index drop the out-of-range ones):

| term | appearance of h_i | contribution to dE/dh_i |
|---|---|---|
| `||h_i - f_i(h_{i-1})||^2` | as the LHS | `+2 r^f_i` |
| `||h_{i+1} - f_{i+1}(h_i)||^2` | inside f | `-2 J_{f_{i+1}}(h_i)^T r^f_{i+1}` |
| `||h_{i-1} - g_i(h_i)||^2` | inside g | `-2 J_{g_i}(h_i)^T r^b_i` |
| `||h_i - g_{i+1}(h_{i+1})||^2` | as the LHS | `+2 r^b_{i+1}` |

```
dE/dh_i = 2 [ r^f_i + r^b_{i+1} - J_{f_{i+1}}(h_i)^T r^f_{i+1} - J_{g_i}(h_i)^T r^b_i ]   (+ supervised term at i = L-1)
```

with the transposes expanded by section 1:

```
J_{f_{i+1}}^T r^f_{i+1} = r^f_{i+1} + a_{i+1} sum_t [ sigma^f_t (u^f_t . r^f_{i+1}) + s^f_t omega^f_t (Delta^f_t . r^f_{i+1}) ]
J_{g_i}^T     r^b_i     = r^b_i     + b_i     sum_t [ sigma^g_t (u^g_t . r^b_i)     + s^g_t omega^g_t (Delta^g_t . r^b_i) ]
```

so, writing it out fully,

```
dE/dh_i = 2 [ r^f_i - r^f_{i+1} + r^b_{i+1} - r^b_i ]
        - 2 a_{i+1} sum_t [ sigma^f_t (u^f_t . r^f_{i+1}) + s^f_t omega^f_t (Delta^f_t . r^f_{i+1}) ]
        - 2 b_i     sum_t [ sigma^g_t (u^g_t . r^b_i)     + s^g_t omega^g_t (Delta^g_t . r^b_i)     ]
```

The bracketed part is the plain bidirectional error transport that the residual identity gives for free; the two
sums are the only places the LUT geometry enters, and they inject credit **only along the 2T anchor-difference
directions** of each layer.

**Learning gradients** (both LUTs get gradient directly from E, which is the point of the symmetric coupling).
For the forward LUT at level i, only the two addressed cells of each table receive anything:

```
dE/dV^f_i[t,c] = -2 a_i s^f_t ( w^f_0 1[c = c^f_t] + w^f_1 1[c = c'^f_t] ) r^f_i        (summed over the batch)
dE/dlog tau^f_i = -2 a_i sum_t s^f_t ( Delta^f_t . r^f_i ) * dw_0/dlog tau,
                  dw_0/dlog tau = -(2 m_{j*}/tau) w_0 w_1
```

and symmetrically for g, with `r^b_i` in place of `r^f_i`, `b_i` in place of `a_i`:

```
dE/dV^g_i[t,c] = -2 b_i s^g_t ( w^g_0 1[c = c^g_t] + w^g_1 1[c = c'^g_t] ) r^b_i
dE/dlog tau^g_i = -2 b_i sum_t s^g_t ( Delta^g_t . r^b_i ) * dw_0/dlog tau
```

**Inner loop.** `h_i <- h_i - eta_h dE/dh_i`. At fixed weights the residual stack is piecewise affine,
`r = A h + const`, `E = ||A h + const||^2`, Hessian `2 A^T A`, so descent needs `eta_h < 1 / sigma_max(A)^2`
(the factor 2 in E halves the usual `2/L` bound). A now contains **both** families of blocks, so
`sigma_max(A)` is larger than in the forward-only case and the admissible step is correspondingly smaller.
A is only piecewise constant in h: every address flip is a discontinuity of A, so monotone descent holds inside
a cell and can jump at a flip.

## 3. Experiment B — PC + ALM, constraint coupling [DERIVED]

Constraints are the forward maps, in residual form:

```
c_i(h) :  h_i - f_i(h_{i-1}) = h_i - h_{i-1} - a_i LUT^f_i(h_{i-1}) = 0,   i = 1..L-1,   h_0 = x clamped
```

Augmented Lagrangian (g does NOT appear):

```
L_rho({h},{lam}) = 1/2 || a_L W_L h_{L-1} - y ||^2 + sum_i [ lam_i . r^f_i + (rho/2) || r^f_i ||^2 ]
```

**Primal inner update** (one gradient step per inner iteration, Algorithm 1 style):

```
dL/dh_i = (lam_i + rho r^f_i) - J_{f_{i+1}}(h_i)^T (lam_{i+1} + rho r^f_{i+1}) + [i = L-1] a_L W_L^T (yhat - y)
        = (lam_i + rho r^f_i) - (lam_{i+1} + rho r^f_{i+1})
          - a_{i+1} sum_t [ sigma^f_t (u^f_t . e_{i+1}) + s^f_t omega^f_t (Delta^f_t . e_{i+1}) ],
          e_{i+1} := lam_{i+1} + rho r^f_{i+1}
h_i <- h_i - eta_h dL/dh_i
```

Completing the square shows the only difference from PC: each prediction target is shifted by `-lam_i/rho`,
i.e. `rho r` is replaced everywhere by the composite credit `lam + rho r`.

**Dual ascent**: `lam_i <- lam_i + alpha r^f_i`, once per inner iteration, `lam` initialised to 0 each sample.
`alpha = rho` with an exactly-solved inner problem is the classical method of multipliers; the finite-inference
variant uses `alpha` as a free rate.

**Concrete rho schedule** (the paper keeps rho = 1; if we want one, this is the standard Bertsekas rule with
the stability coupling made explicit):

```
rho_0 = 1, alpha = 1, eta_h = c / (sigma_max(A)^2 (2 rho + alpha)) with c ~ 2 (the bound is < 4)
after each outer (weight) step k:  if ||r^{(k)}|| > gamma ||r^{(k-1)}||  then  rho <- min(beta rho, rho_max)
gamma = 0.5, beta = 2, rho_max = 32,  and RE-DERIVE eta_h from the bound whenever rho changes
```

The re-derivation is not optional: raising rho without shrinking eta_h walks straight out of the stability
region `eta_h sigma^2 (2 rho + alpha) < 4`.

**The backward LUT in B.** g is outside L. Two jobs: (i) realise the downward pass, i.e. warm-start the inner
argmin by `h_{i-1} <- g_i(h_i)` from a clamped target downward (instead of, or blended with, the forward-pass
init), and (ii) be trained by its own reconstruction objective:

```
R = mu * sum_i || h_{i-1} - g_i(f_i(h_{i-1})) ||^2
```

**[FLAG] On which `h` should R be evaluated?** Three readings, materially different:
* **feedforward h** (`h_{i-1}` from a plain forward pass): g learns to invert f on the *data manifold as the
  model currently sees it*. Stable, decoupled from the relaxation, and it can be trained in the same pass with
  no extra inner loop. My default recommendation.
* **relaxed h** (the `h*` at the end of the inner loop): g learns to invert f on the states the relaxation
  actually visits — which is what a warm-start needs — but it makes g's target depend on g's own warm-start
  (a moving target, feedback loop) unless the warm-start is stop-gradiented.
* **both** (R on feedforward h, warm-start from g): the compromise; g is trained on a stable target but used on
  the relaxed states, with a distribution shift between train and use that must be monitored.
Note R is a *second objective*, not part of L: its gradient must not flow into f (otherwise f is being trained
to be invertible, which is a different model). So `R` should use `f_i(h_{i-1})` with f **detached**, unless we
deliberately want the invertibility pressure on f — state which. Gradients, with `q_i := h_{i-1} - g_i(f_i(h_{i-1}))`:

```
dR/dV^g_i[t,c] = -2 mu b_i s^g_t ( w^g_0 1[c = c^g_t] + w^g_1 1[c = c'^g_t] ) q_i     (addresses from z = f_i(h_{i-1}))
dR/dlog tau^g_i = -2 mu b_i sum_t s^g_t ( Delta^g_t . q_i ) * dw_0/dlog tau
```

## 4. Double-backward / HVP audit (compiled path always on at read_top_n = 2)

| place | needs | verdict |
|---|---|---|
| A: `dE/dh` inner step | first-order grad of a scalar | **fine compiled** |
| B: primal `dL/dh` inner step | first-order | **fine compiled** |
| B: dual step `lam += alpha r` | no derivatives | fine |
| A/B: weight gradients (V, log_tau) at fixed h | first-order | **fine compiled** |
| R (g's reconstruction) | first-order through g, f detached | **fine compiled** |
| `sigma_max(A)` for the step-size rule | `A v` (JVP) + `A^T u` | `A^T u` = one backward; **`A v` by finite differences** (what the existing code does) |
| exact inner argmin (Newton / Gauss-Newton / CG on the normal equations) | HVP through the LUT | **blocked** -> `LUT_DISABLE_COMPILE=1`, an eager copy of the layer, or FD-HVP `(grad(h+eps v) - grad(h))/eps` |
| curvature-adaptive eta_h (per-mode Jury bound from the live spectrum) | HVP / JVP | **blocked** as above; FD is the cheap way out |
| implicit differentiation of `h*` w.r.t. theta (if we ever want exact bilevel gradients) | second-order | **blocked**; not needed by Algorithm 1 |
| line search / energy monitoring | function evaluations | fine |

FD caveat: the address is piecewise constant, so an FD probe with `eps` large enough to cross a boundary
measures a discontinuity, not a derivative. Keep `eps` well below the typical smallest margin (measured median
`m_{j*} ~ 0.15` at init) — `eps ~ 1e-3` is two orders below that.

## 5. A vs B on paper

**What B buys.** Credit at every layer: `lam` integrates the residual and at the fixed point equals the BP
adjoint, so the weight update is the BP gradient. Measured at gate 1 on plain MLPs (N=32, L=64, T=2L):
PC-ALM had 0/62 layers with zero gradient and +0.93 mean cosine to BP; PC had 19-20/62 layers receiving
**exactly zero** gradient and +0.88 on the rest.

**What B costs.** One `h`-sized dual state per layer; one dual step per inner iteration (negligible); the same
T inner iterations as PC. Measured: PC-ALM 1314 ms/step vs PC 1364 ms/step vs BP 6 ms/step at that cell. B also
costs the *structural* property that its fixed point is the forward pass (`r -> 0`), so **B does not relocate
addresses at convergence** — A does, because its optimum sits off the forward pass. In B the address search is
transient, in A it is the equilibrium.

**What could make B fail: conditioning, not rank.** `2T = 32 = width`, so `J_LUT` is generically full rank.
The two halves of the spectrum behave very differently:

* **score block** `u_t sigma_t^T`. As all margins grow, `prod_k sigmoid(2 m_k) -> 1` and
  `(1 - sigmoid(2 m_j)) -> 0`, so `ds_t/dm_j -> 1`: the score path does **not** die at large margins, it
  saturates to an O(1) direction `sum_j sign(d_j)(e_{a_j} - e_{b_j})`. It collapses in the opposite regime:
  when *all* NAP margins are small, `prod_k sigmoid(2 m_k) -> 2^{-NAP} = 1/256` and `sum_k m_k -> 0`, so both
  `s_t` and `ds_t/dz` vanish. (With only *one* small margin the product is ~0.5, not 2^-8 — the collapse needs
  the whole table to sit on a corner.)
* **routing block** `s_t Delta_t omega_t^T`, carrying `(2/tau) w_0 w_1`. This is maximal exactly at a boundary
  (`w_0 w_1 = 1/4` at `m_{j*} = 0`) and decays like `exp(-2 m_{j*}/tau)`. At tau = 0.5: `m_{j*} = 0` -> factor
  1.0; `m_{j*} = 0.5` -> 0.42; `m_{j*} = 1.5` -> 0.010. Relative to the score term the routing term scales like
  `s_t |Delta_t| (2/tau) w_0 w_1 / |u_t|`; with the measured init statistics (median `m_{j*} ~ 0.15`,
  `sum_j m_j ~ 3`) it is *comparable to or larger than* the score term near a boundary and two orders smaller
  once the smallest margin exceeds ~1.5.
* **consequence**: the conditioning of `J_LUT` is governed by how many tables currently sit near a boundary.
  Early in training many do and the relaxation genuinely moves addresses; as margins grow (which training
  encourages, since `s_t` rewards large margins) the routing block collapses and the relaxation degenerates into
  pure score-scaling. **That is the concrete failure mode for both A and B, and the reason to log the margin
  distribution over training, not just the flip counts.** A learnable `tau` can fight it (raising tau widens the
  window), which is an argument for keeping `log_tau` trainable and watching where it goes.
* Secondary conditioning note: each `omega_t` is a *single* anchor-difference direction, and tables share
  anchors, so the 2T vectors are far from orthogonal even when all blocks are active; the effective rank is
  below 2T in practice.

## 6. Knobs that must be fixed before coding (questions)

1. **What is clamped?** Input only, or input *and* target? For A this decides whether the energy is degenerate
   (section 2 [FLAG]) — if input-only, we must add the supervised readout term instead.
2. **Is there a readout at all in A**, or is the top state itself the prediction?
3. **g's form:** residual (`g = I + b_i LUT^g`) like f, or a bare LUT? Own `tau`, own anchors, own tables —
   or anchors tied to f?
4. **`b_i`**: same `1/sqrt(L N)` rule as `a_i`?
5. **R's evaluation point** (feedforward h / relaxed h / both) and **is f detached inside R?** (section 3 [FLAG])
6. **mu** (weight of R), and whether R is optimised jointly or in an alternating schedule.
7. **rho / alpha / eta_h**: accept `rho_0 = 1, alpha = 1, eta_h = 2/(sigma^2 (2 rho + alpha))` and the
   `gamma = 0.5, beta = 2, rho_max = 32` schedule above, or fix rho?
8. **Inner steps T**: `2L` as the paper, or more, given the conditioning argument?
9. **Shape and budget**: depth L, width N (32?), tables/layer (16?), nap (8?), batch, steps, seeds.
10. **Task**: Fashion-MNIST as at gate 1, or something where LUT capacity actually matters?
11. **Arms**: include BP as a control? (I would — it is the only arm with an unambiguous reference gradient.)
