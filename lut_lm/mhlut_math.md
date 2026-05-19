# MultiHeadLUT — Math

Two families of LUT modules sit on top of the same discrete idea: replace a learned linear projection with a learned row-lookup, indexed by signs of pairwise input comparisons. The two families differ in **how they smooth the discrete forward for training** — the §1 family uses an *uncertainty kernel* (the original manifesto formulation), the §2 family uses a *softmax over all rows of the table* (the newer construction).

---

## 0.  Basics

A Multi-Head LUT replaces a linear projection `y = W x` with a learned table-lookup. Input `x ∈ ℝ^{input_dim}`, output `y ∈ ℝ^{n_heads × n_outputs}`.

A module has `H` heads (`n_heads`) and `tph` tables per head. Total tables `T = H · tph`. Each table `t` owns a weight tensor

    W_t ∈ ℝ^{K × n_outputs} ,    K = 2^NAP

where `NAP` (number of anchor pairs) is the per-table input bit-width.

**Anchor pairs.** For each table, two integer lists are sampled once at init:

    a_t, b_t  ∈  [0, input_dim) ^ NAP

with **balanced full coverage** (each input dimension participates in equally many pairs across the module). Anchors are buffers; they don't move during training.

**Per-table row index** (the discrete part, used in every variant below):

    d_t[i]      = x[a_t[i]] − x[b_t[i]]                       i = 0 .. NAP−1
    bits_t[i]   = ( d_t[i] > 0 )                              ∈ {0, 1}
    chosen_t    = Σ_i bits_t[i] · 2^{NAP−1−i}                 ∈ [0, K)

**Module output**: per-head sum of per-table outputs

    y[h, :]  =  Σ_{t ∈ head h}  out_t

The `out_t ∈ ℝ^{n_outputs}` formula differs across §1 and §2. Defaults used everywhere unless stated: `NAP = 6, K = 64`.

---

## 1.  Canonical MultiHeadLUT — the manifesto formulation

The forward is a row gather; the soft variant blends the chosen row with the `K' = n_alternatives` "nearest" rows, weighted by an **uncertainty kernel**. The hard variant is the manifesto's deployed forward.

### 1.1  Forward — hard (the manifesto)

    out_t  =  W_t [ chosen_t , : ]

NAP comparisons, one row gather, no multiplications. Exactly the manifesto's deployed module.

### 1.2  Forward — soft (uncertainty blend)

The `K' = n_alternatives` "nearest" rows are obtained by 1-bit-flipping the `K'` anchor positions with the smallest `|d_t[i]|`. Call them `pos_1, … , pos_{K'}` (sorted by `|d_t[pos_i]|` ascending) and define

    alt_t_i           = chosen_t  XOR  ( 1 << ( NAP − 1 − pos_i ) )       (the row obtained by flipping bit pos_i)
    d_alt_t_i         = d_t[ pos_i ]                                       (the d value at the flipped position)

The **uncertainty kernel** (inverse-L1 form, manifesto default):

    u(|d|)  =  β / ( 1 + |d| )         with  β = 0.5  (uncertainty_bias)

Per-row blending weights:

    α_alt_t_i   =  u( |d_alt_t_i| )  /  K'                                  (one per alt)
    α_main_t    =  1  −  Σ_i  α_alt_t_i

Forward soft:

    out_t  =  α_main_t · W_t [ chosen_t , : ]   +   Σ_i  α_alt_t_i · W_t [ alt_t_i , : ]

**Tradeoff vs hard.** Forward now reads `1 + K'` rows per table and computes a `(K' + 1)`-term linear combination — `O(K' · n_outputs)` multiplications per table per token. The manifesto's "no multiplications at inference" property is lost in soft mode; it's a training-only forward unless you accept the bandwidth/compute cost at deploy.

**`n_alternatives = 1` is the manifesto's soft case**: blend the chosen row with one alt at the lowest-confidence bit, weighted by the uncertainty there. The `K' = 3` case (current default) is a more diffuse blend.

### 1.3  Backward

Forward soft is differentiable through `u` (since `α_main, α_alt_i` depend on `|d_alt_t_i|`) and the weights enter linearly — plain autograd.

Uncertainty derivative:

    du / d |d|     =  − β / ( 1 + |d| )²
    d |d| / d d    =  sign(d)
    so  du / d d   =  − β · sign(d) / ( 1 + |d| )²

Define

    grad_main_t      = ⟨ grad_out_t , W_t [ chosen_t , : ] ⟩
    grad_alt_t_i     = ⟨ grad_out_t , W_t [ alt_t_i  , : ] ⟩

Since `∂α_main / ∂α_alt_i = −1` and `α_alt_i = u(|d_alt_i|) / K'`, the chain gives

    dL / d d_alt_t_i  =  ( grad_alt_t_i − grad_main_t )  ·  ( − β · sign(d_alt_t_i) / ( 1 + |d_alt_t_i| )² )  /  K'

Scatter to anchors:

    dL / dx[ a_t[pos_i] ]  +=  dL / d d_alt_t_i
    dL / dx[ b_t[pos_i] ]  −=  dL / d d_alt_t_i

Weight gradient (linear in α's):

    dL / d W_t [ chosen_t  , : ]  +=  α_main_t   · grad_out_t
    dL / d W_t [ alt_t_i   , : ]  +=  α_alt_t_i  · grad_out_t                    (for each alt)

Note: the **backward x-gradient chain above does not depend on `smooth_mode`**. The forward toggle (§1.1 hard pick vs §1.2 soft blend) only changes which combination of rows is *output*; the backward still routes the same uncertainty-kernel gradient to `x` through the `K'` nearest alts (otherwise the module would have no `x`-gradient and be untrainable). What `smooth_mode` actually changes in the backward is the **weight gradient distribution**:

- `smooth_mode = True` (soft forward): `dL / dW[chosen]   = α_main · grad_out`,  `dL / dW[alt_i]  = α_alt_i · grad_out` (gradient is spread linearly across chosen + K′ alts).
- `smooth_mode = False` (hard forward): `dL / dW[chosen]   = grad_out`,  no alt-row weight gradient (hard `index_add`).

In implementation, the hard-mode case uses a standard STE wrapper: numerically the output equals `W[chosen]`, but autograd is routed through the same `α_main, α_alt_i` chain so that `dL/dx` still flows.

`n_alternatives = 1` matches the manifesto's backward exactly: one alt position contributes, kernel applied at the lowest-confidence bit. `n_alternatives ≥ 1` is required either way — without at least one alt, there's no x-gradient.

---

## 2.  SoftMultiHeadLUT — softmax over all rows

A different soft mechanism. Instead of blending chosen with `K'` nearby rows by uncertainty, **score every row of the table against `x` via a softmax**, then either pick the top row (hard, STE) or take the soft mixture (soft).

### 2.1  The new idea — soft signs × bit matrix → softmax

Replace each binary bit by a smooth proxy:

    p_t[i]  =  d_t[i] / ( T_soft + | d_t[i] | )           ∈ ( −1 , +1 )

`T_soft > 0` is a temperature: small T_soft makes `p` nearly ±1, large T_soft makes it nearly linear in `d`.

Define the **bit matrix** `B ∈ {−1, +1}^{NAP × K}` — column `k` is the ±1 expansion of integer `k`, MSB first:

    B[i, k]  =  +1   if bit (NAP−1−i) of k is 1,   else  −1

Row match scores (popcount-like at the hard limit):

    ts_t[k]  =  Σ_i  p_t[i] · B[i, k]      ∈ [ −NAP , +NAP ]

Intuition: `ts_t[k]` is maximised at the integer `k` whose bit pattern agrees with `sign(p_t)`. At the hard limit (`p ∈ {±1}^NAP`), `argmax_k ts_t[k] = chosen_t` of §0 — provably equal at fp32. So this construction is a smooth interpolation between "no information" (all `p = 0` → `ts = 0` → uniform softmax) and "the manifesto hard pick" (all `p = ±1` → ts maximised exactly at `chosen_t`).

Soft selector:

    sel_t[k]  =  softmax_k ( ts_t[:] / T_sel )       ∈ Δ^{K−1}

`T_sel > 0` is the row-selection temperature. Both temperatures are log-parametrised: `T_soft = exp(log_T_soft), T_sel = exp(log_T_sel)`, learnable.

### 2.2  Forward — hard (STE)

    chosen_t  =  argmax_k  ts_t[k]                  (= the sign-pack `chosen_t` of §0)
    out_t     =  W_t [ chosen_t , : ]

Numerically identical to §1.1 — the manifesto hard pick. In PyTorch the STE is written

    sel_hard           =  one_hot ( argmax_k ts_t )
    sel_through        =  sel_hard  −  sel_t.detach()  +  sel_t          (numerically = sel_hard)
    out_t              =  Σ_k  sel_through[k] · W_t [ k , : ]            (numerically = W_t[chosen_t])

so autograd flows through `sel_t` (§2.4) while the value matches the hard pick.

### 2.3  Forward — soft (true soft mixture)

    out_t  =  Σ_k  sel_t[k] · W_t [ k , : ]

A full convex combination over all `K = 2^NAP` rows. Output varies smoothly with `x` (no row jumps at sign flips). Cost: `K` row reads, softmax over `K`, `K`-term weighted sum — only practical with small NAPs.

### 2.4  Backward — full soft (over all K rows)

For the soft mixture (§2.3), plain autograd. For the hard-STE forward (§2.2), the STE wrapper makes autograd flow through the same chain. The math:

    d_sel_t[k]   =  ⟨ grad_out_t , W_t [ k , : ] ⟩
    d_z_t[k]     =  ( sel_t[k] · ( d_sel_t[k] − Σ_j sel_t[j] · d_sel_t[j] ) ) / T_sel              (softmax bwd)
    d_p_t[i]     =  Σ_k  d_z_t[k] · B[i, k]
    d_d_t[i]     =  d_p_t[i] · T_soft / ( T_soft + |d_t[i]| )²                                      (p = d/(T+|d|) deriv.)
    dL / dx[a_t[i]]  +=  d_d_t[i]
    dL / dx[b_t[i]]  −=  d_d_t[i]

Weight gradient depends on which forward (§2.2 vs §2.3) was used:

    §2.2 (STE hard):    dL / d W_t [ chosen_t , : ]  +=  grad_out_t                  (only chosen row)
    §2.3 (soft mix):    dL / d W_t [ k , : ]         +=  sel_t[k] · grad_out_t       (every row, sel-weighted)

Temperature gradients (out of the same chain):

    dL / d log_T_sel    =  − Σ_{t,k}  d_z_t[k] · ts_t[k]
    dL / d log_T_soft   =  − Σ_{t,i}  d_d_t[i] · d_t[i]

Every row of every table contributes. Saved-for-backward memory: `O(B · T · K)` for `sel` plus the `ts, p, d` intermediates.

### 2.5  Backward — top-K approximation

Compute `ts_t` as in §2.1 then **mask** to `−∞` at every row outside

    S_t  =  { chosen_t } ∪ { chosen_t XOR ( 1 << (NAP − 1 − pos_i) )  :  i = 1 .. K' }

where `pos_1 ,…, pos_{K'}` are the `K'` lowest-`|d|` anchor positions (same selection rule as §1.2). The softmax now renormalises over only `1 + K'` rows:

    sel_t[k]  =  softmax_k ( ts_masked_t[:] / T_sel )       — zero outside S_t

Everything else in §2.4 is unchanged.

Limits:

- `K' = 0`            →  `sel = δ_{chosen}` → only the chosen row contributes; backward is *almost* the §1 hard limit, except `x`-gradient still flows through the soft-sign Jacobian rather than vanishing.
- `K' = NAP`          →  Hamming-1 ball around `chosen_t` (`NAP + 1` rows total).
- `K' = K − 1`         →  recovers §2.4 exactly.

Useful because §2.4's `O(K)` memory becomes a bottleneck at large `NAP` or large batch — `top-K` keeps the same softmax-chain math at `O(K')` rows.

---

## 3.  How §1 and §2 relate

Same discrete forward (hard pick at `chosen_t`). Different smooth surrogates:

| | §1 — uncertainty blend (manifesto) | §2 — softmax over rows |
|---|---|---|
| Forward soft output | linear combination of chosen + `K'` nearest rows, weighted by `1/(1+\|d_alt\|)` | softmax-weighted combination over all `K` rows of `ts = soft_signs · B` |
| `n_alternatives = 1` / `K' = 1` | manifesto soft formulation: blend chosen with one alt | meaningless — §2 has no "alts" parameter; closest analog is §2.5 with `K' = 1` |
| Hard forward (`smooth_mode=False` / `hard=True`) | output = `W[chosen]`; `dL/dx` still flows via the uncertainty-kernel STE; weight grad is hard `index_add` | output = `W[chosen]` via the `sel_hard − sel_soft.detach() + sel_soft` STE; `dL/dx` still flows via the soft-sign chain; weight grad is hard `index_add` |
| `x`-gradient source (soft mode) | flip-induced "swap" `grad_alt_i − grad_main`, scaled by `−β·sign(d_alt)/(1+\|d_alt\|)²` | full softmax chain: every anchor position contributes `d_d_t[i] · T_soft/(T_soft+\|d\|)²` |
| Weight gradient (soft mode) | `α_main · grad_out` to chosen, `α_alt_i · grad_out` to each alt | `sel_t[k] · grad_out` to *every* row |
| Inference forward | §1.1 (hard pick), no multiplications | §2.2 (hard pick, same as §1.1), no multiplications |
| Inference compute (soft mode, if used) | `(1+K') · n_outputs` mults per table | `K · n_outputs` mults per table (impractical) |

Both collapse to the same hard inference (one row gather per table). They are alternative training-time smoothings of that hard forward.
