# PC (A) and PC-ALM (B) over a stack of paired forward/backward LightMHL layers — derivation

Derivation only; nothing here is implemented. Tags: **[CODE]** read from the repo, **[DERIVED]** worked out
here, **[ASSUMPTION]** my choice, flagged because the repo does not settle it.

## 0. What the repo actually contains

**PC / PC-ALM implementation [CODE]** — `research/pcalm_lut/pcalm.py` (written for this study; the only PC code
in the repo). Energy, **summed over the batch** (inference in Algorithm 1 is per sample, so the primal gradient
and the dual step are in the same units; only the weight gradient is averaged):

```
L_rho(h, theta, lam) = sum_b [ 1/2 || yhat(h_{L-1}) - y ||^2 ]  +  sum_i lam_i . r_i  +  (rho/2) sum_i || r_i ||^2
r_i = h_i - f_i(h_{i-1}),   r_1 = h_1 - f_1(x)   (x clamped)
```

Loop structure [CODE]: init `h` from a forward pass and `lam = 0`; then `T-1` times { primal `h_i -= eta_h *
grad_{h_i} L_rho`; dual `lam_i += alpha * r_i` }; one final primal step; then one Adam step on
`grad_theta L_rho / |B|`. `pc` = the same code with `alpha = 0` (lam pinned to 0, so the energy is the PC
quadratic penalty `F_PC`). `bp` = ordinary autograd. Defaults: `T = 2L`, `alpha = rho = 1`,
`eta_h = 1 / sigma_max(A)^2` with `sigma_max` from power iteration (A·v by finite differences, Aᵀ·u by one
backward), stability check `eta_h sigma^2 (2 rho + alpha) < 4`.

**Paired forward/backward LightMHL: NOT IN THE REPO.** I searched the LUT modules, their tests, the experiment
trees and `claude/*.md` for an inverse / backward / feedback / tied second LUT: there is none. The only
"reconstruction" hits are internal to FastMHL/HyperplaneMHL (rebuilding the sign pattern of the chosen row
inside one layer's own backward) and `forward_int` / `export_quantised`, which are inference artefacts.
So the construction below is **[ASSUMPTION]**.

**Residual convention and pre-multipliers [CODE]** (`research/pcalm_lut/lut_stack.py`, following the paper's
App. B): `h_1 = a_1 W_1 x`, `h_i = h_{i-1} + a_i LUT_i(h_{i-1})`, `yhat = a_L W_L h_{L-1}`, with
`a_1 = 1/sqrt(D)`, `a_i = 1/sqrt(L N)`, `a_L = 1/(gamma0 N)`. **Correction to the number in the brief:**
`1/sqrt(L N) = 0.0884` is the depth-4, width-32 case (`1/sqrt(128)`); at the deep cell L=64, N=32 it is
`1/sqrt(2048) = 0.0221`. It is depth-dependent, not a constant.

**compress / decompress [CODE]:** *not* in this stack — the LUT is 32→32 and bare, wrapped only by the residual
add. They appear only in the LM configuration (`CompressionMHL`: 384 → H·48 → LUT → H·48 → 384).

**Forward semantics [CODE]**, per table t (T = 16 tables, K = 256 cells, NAP = 8, tau = 0.5 learnable):
`d_j = z[a_j] - z[b_j]`, `m_j = |d_j|`; `c_t = pack(sign(d.detach()))`; `s_t = (sum_j m_j) prod_j sigmoid(2 m_j)`;
`j* = argmin_j m_j`, `c'_t = c_t` with bit `j*` flipped; `w_0 = sigmoid(2 m_{j*}/tau)`, `w_1 = 1 - w_0`;
`LUT(z) = sum_t s_t ( w_0 V[t,c_t] + w_1 V[t,c'_t] )`.

## 1. Jacobian of one LightMHL layer [DERIVED]

Write `u_t = w_0 V[t,c_t] + w_1 V[t,c'_t]` (the blended row) and `Delta_t = V[t,c_t] - V[t,c'_t]`. Define the
two **input-side** vectors (both supported on at most 2·NAP = 16 coordinates of z):

```
sigma_t := d s_t / dz   = sum_j (d s_t / d m_j) sign(d_j) (e_{a_j} - e_{b_j}),
           d s_t/d m_j  = prod_k sigmoid(2 m_k) * [ 1 + 2 (sum_k m_k) (1 - sigmoid(2 m_j)) ]
omega_t  := d w_0 / dz  = (2/tau) w_0 w_1 * sign(d_{j*}) (e_{a_{j*}} - e_{b_{j*}})            (single pair!)
```

Then

```
J_LUT(z) = sum_t [ u_t sigma_t^T  +  s_t Delta_t omega_t^T ]          (2T rank-1 terms)
J_f_i(h) = I + a_i J_LUT(h)                                           (residual block)
J_LUT^T v = sum_t [ sigma_t (u_t . v) + s_t omega_t (Delta_t . v) ]   (what the inference step needs)
```

**What vanishes:** `d c_t/dz = 0` (address detached) and `d V/dz = 0`, so the gathered content contributes to J
only through the fixed vectors `u_t`, `Delta_t`. There is no term from the piecewise-constant part. Nothing
else is dropped: `m = |d|` is differentiable in the code.

## 2. The paired backward layer [ASSUMPTION]

Level i carries a forward map `f_i: h_{i-1} -> h_i` and a backward map `g_i: h_i -> h_{i-1}`, each a LightMHL
with its own tables and anchors, in the same residual form: `g_i(h) = h + b_i LUT^g_i(h)`. Three readings are
possible and the repo does not choose:

* **(a) second prediction term (my default):** `g` contributes its own quadratic residual to the energy, i.e. a
  two-sided PC energy. Both directions are then trained by the same relaxation.
* **(b) second constraint family:** `h_{i-1} = g_i(h_i)` enters as its own equality with multipliers `mu_i`.
* **(c) initializer only:** a backward sweep sets `h` from the target (target-propagation style) and `g` never
  enters the energy.

Everything below is written for **(a)** for Experiment A and for **(b)** for Experiment B, with the differences
noted; (c) changes only the initialisation and leaves all gradients as in the unpaired case.

## 3. Experiment A — basic PC with paired layers [DERIVED]

```
E({h}) = 1/2 || a_L W_L h_{L-1} - y ||^2
       + (rho_f/2) sum_{i=1}^{L-1} || h_i - f_i(h_{i-1}) ||^2            r^f_i
       + (rho_b/2) sum_{i=1}^{L-1} || h_{i-1} - g_i(h_i) ||^2            r^b_i          (h_0 = x clamped)
```

Inference update, `h_i <- h_i - eta_h grad_{h_i} E`, with `1 <= i <= L-1`:

```
grad_{h_i} E =  rho_f r^f_i                                   (own forward residual, as the LHS)
             -  rho_f J_{f_{i+1}}(h_i)^T r^f_{i+1}            (next layer's forward residual)   [i < L-1]
             -  rho_b J_{g_i}(h_i)^T r^b_i                    (own backward residual)
             +  rho_b r^b_{i+1}                               (next backward residual, as the LHS) [i < L-1]
             +  a_L W_L^T ( a_L W_L h_{L-1} - y )             (supervised term)                 [i = L-1 only]
```

Substituting section 1 (dropping the layer index on the inner sums):

```
J_{f_{i+1}}^T v = v + a_{i+1} sum_t [ sigma^f_t (u^f_t . v) + s^f_t omega^f_t (Delta^f_t . v) ]
J_{g_i}^T     v = v + b_i     sum_t [ sigma^g_t (u^g_t . v) + s^g_t omega^g_t (Delta^g_t . v) ]
```

so, e.g. for the forward path, the term `-rho_f J^T r^f_{i+1}` expands to
`-rho_f r^f_{i+1} - rho_f a_{i+1} sum_t [ sigma_t (u_t . r^f_{i+1}) + s_t omega_t (Delta_t . r^f_{i+1}) ]`:
the identity part is the plain PC error transport, and the LUT part injects credit **only** along the 2T
anchor-difference directions `sigma_t, omega_t`.

Learning phase (tables and tau; same form for `f` and `g`, with `r` the corresponding residual):

```
d E / d V_i[t, c] = - rho_f a_i  s_t ( w_0 1[c = c_t] + w_1 1[c = c'_t] )  r^f_i        (summed over the batch)
d E / d log tau_i = - rho_f a_i  sum_t s_t ( Delta_t . r^f_i ) * d w_0/d log tau,
                     d w_0 / d log tau = - (2 m_{j*} / tau) w_0 w_1
```

Only the two addressed cells per table receive gradient; `1[c = c_t]` is exactly the sparsity that makes a LUT
update local. (Under ALM, replace `rho_f r^f_i` by the composite credit `lam_i + rho r^f_i` — section 4.)

**Inner-loop convergence [DERIVED].** Stack the residuals `r(h) = A h + b` (linear in h at fixed weights up to
the LUT's piecewise-constant regions). The PC energy Hessian is `rho A^T A` plus the readout block, so gradient
descent converges iff `eta_h < 2 / (rho sigma_max(A)^2)`, i.e. the paper's bound at `alpha = 0`. With the
residual convention A has `I` on the diagonal, so `sigma_max(A) >= 1`; adding the backward family appends more
blocks and can only raise `sigma_max`, so **pairing forces a smaller step**. Because the address is piecewise
constant, A is only piecewise constant in h: each address flip is a discontinuity of A, and the iteration is a
gradient descent on a piecewise-quadratic energy. Monotone descent holds within a cell; at a flip the energy
can jump, which is exactly the "search" the address hypothesis wants and also the thing that can chatter.

## 4. Experiment B — PC + ALM [DERIVED]

Constraints (reading (b)): `r^f_i = h_i - f_i(h_{i-1}) = 0` with multipliers `lam_i`, and optionally
`r^b_i = h_{i-1} - g_i(h_i) = 0` with multipliers `mu_i`, for i = 1..L-1, `h_0 = x` clamped.

```
L_rho({h},{lam},{mu}) = 1/2 || a_L W_L h_{L-1} - y ||^2
   + sum_i [ lam_i . r^f_i + (rho/2)||r^f_i||^2 ]
   + sum_i [ mu_i  . r^b_i + (rho/2)||r^b_i||^2 ]
```

Primal (inner argmin, one gradient step per inner iteration as in Algorithm 1):

```
grad_{h_i} L_rho = (lam_i + rho r^f_i)  -  J_{f_{i+1}}^T (lam_{i+1} + rho r^f_{i+1})
                 -  J_{g_i}^T (mu_i + rho r^b_i)  +  (mu_{i+1} + rho r^b_{i+1})
                 +  [i = L-1] a_L W_L^T ( a_L W_L h_{L-1} - y )
```

i.e. **identical in form to PC with `rho r` replaced everywhere by the composite credit `lam + rho r`**
(completing the square: `lam.r + (rho/2)||r||^2 = (rho/2)||r - (-lam/rho)||^2 - ||lam||^2/(2 rho)`, so each
prediction target is shifted by `-lam_i/rho`).

Dual ascent: `lam_i <- lam_i + alpha r^f_i`, `mu_i <- mu_i + alpha r^b_i`, once per inner iteration
(`alpha = rho` with an exactly-solved inner problem recovers the classical method of multipliers).

Weight step: `theta <- theta - eta Adam( grad_theta L_rho )`, i.e. in the table gradient of section 3 replace
`rho_f r^f_i` by `(lam_i + rho r^f_i)` and `rho_b r^b_i` by `(mu_i + rho r^b_i)`.

**rho schedule.** The paper keeps `rho = 1` fixed and never anneals it [CODE: our default too]. Classical MM
would use `rho <- beta rho` when `||r^{k+1}|| > gamma ||r^k||` (Bertsekas, beta ~ 2-10, gamma ~ 0.25). Note the
coupling: the stability bound `eta_h sigma^2 (2 rho + alpha) < 4` means every rho increase must shrink eta_h,
so a rho schedule is really a joint (rho, eta_h) schedule.

**Where the backward LUT sits.** Under (b) it realises downward constraint propagation: `mu_i` carries credit
from level i-1 back to level i through `J_{g_i}^T`, in parallel with the upward `lam` path. Under (a) it is just
a second penalty. Either way it does **not** remove the need for `J_f^T`: PC/ALM already transport credit
downward through the forward layer's transpose. What `g` adds is a *learned* downward map whose own tables are
trained — i.e. it can represent a downward correction the forward transpose cannot.

## 5. Double-backward / HVP audit (compiled path is always on at read_top_n = 2)

| where | what it needs | verdict |
|---|---|---|
| primal step `grad_h L_rho` | first-order only | **fine compiled** |
| dual step | no derivatives | fine |
| weight step `grad_theta L_rho` at fixed h | first-order | **fine compiled** |
| `sigma_max(A)` power iteration | `A v` and `A^T u` | `A^T u` is one backward (fine); `A v` is a JVP — **use finite differences** (already done) |
| exact inner argmin by Newton / Gauss-Newton | HVP through the layer | **blocked**: needs `LUT_DISABLE_COMPILE=1`, an eager copy, or FD-HVP |
| curvature-adaptive `eta_h` (e.g. per-mode Jury bound from the current spectrum) | HVP / JVP | same as above |
| implicit differentiation of the converged `h*` w.r.t. theta | second-order | **blocked** (and not needed by Algorithm 1) |
| line search on the energy | function evaluations only | fine |

## 6. A vs B on paper

* **B buys** credit at every layer: the multiplier integrates the residual, so at the fixed point `lam` is the
  BP adjoint and the weight update is the BP gradient (measured here at gate 1: 0/62 dead layers for PC-ALM vs
  19-20/62 for PC, alignment +0.93 vs +0.88 on the live ones).
* **B costs** one extra state of the size of `h` per layer (plus `mu` if the backward constraints are included:
  2x again), one extra dual step per inner iteration (cheap), and the same T inner iterations as PC. Measured
  overhead at gate 1: PC-ALM 1314 ms/step vs PC 1364 ms/step vs BP 6 ms/step — the dual step is free, the inner
  loop is not.
* **The fixed point is the forward pass.** At convergence `r -> 0`, so `h` returns to its forward-pass value and
  the addresses return to the forward-pass addresses. B therefore does **not** perform persistent address search;
  A (plain PC) does, because its penalty optimum sits off the forward pass. This is the structural point from
  gate 0, and it is a property of the formulation, not of the implementation.
* **Rank vs conditioning.** `2T = 32 = width`, so `J_LUT` is generically full rank — degeneracy is not the
  problem, conditioning is:
  * the `omega_t` (routing) directions carry the factor `(2/tau) w_0 w_1` with `w_0 = sigmoid(2 m_{j*}/tau) >= 1/2`,
    so `w_0 w_1 <= 1/4` and decays like `exp(-2 m_{j*}/tau)` once the smallest margin exceeds tau. At tau = 0.5
    and a typical smallest margin of ~0.15 (measured at init, gate 0) this factor is ~0.23, but it collapses as
    margins grow during training;
  * the `sigma_t` (score) directions are O(1) and independent of tau;
  * each `omega_t` is a **single** anchor-difference direction `e_a - e_b`, and tables share anchors, so the 2T
    vectors are not close to orthogonal;
  * consequence: the spectrum of `J_LUT` splits into an O(1) score block and an exponentially suppressed routing
    block. The relaxation will move `h` mainly along score directions. **The failure mode for both A and B is
    that the boundary-crossing directions are the worst-conditioned ones**, so the inner loop needs either many
    steps or a per-block step size to make address search actually happen. That, not rank, is what I would watch.

## 7. Knobs to fix before any code (questions, not choices)

1. **Backward-layer role:** (a) second prediction term, (b) second constraint family with its own `mu`, or
   (c) initializer only?
2. **Tying:** are `g_i` and `f_i` independent (own tables and anchors), anchor-tied, or table-tied in some
   transposed sense? Independent doubles the parameters.
3. **Shape:** depth L, width N, tables/layer T, NAP (and therefore cells = 2^NAP), and whether the backward
   layer has the same shape.
4. **Penalties and rates:** `rho_f`, `rho_b` (equal?), `alpha`, `eta_h` rule (`1/sigma_max^2` as now?), inner
   steps `T` (`2L` as the paper, or more given the conditioning argument above), and any `rho` schedule.
5. **tau:** learnable per layer (as now) or frozen during the study, and at what value — it directly sets the
   routing-block conditioning.
6. **Task and clamping:** which dataset (Fashion-MNIST as in gate 1?), and is only the input clamped, or input
   and target (the "nudged" variant)?
7. **Budget:** batch size, steps, seeds; and whether BP is included as a third arm (I would).
8. **Readout:** keep the linear `a_L W_L` readout, or make the last layer a LUT too?
