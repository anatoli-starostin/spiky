# Plan note: a `min_margin` confidence form for LightMHL (Gen 3)

Status: **spec only**. Nothing is implemented and nothing has been run. The PDF is not regenerated.
Decided context: nap = 8 for the whole paper set, 16K steps, batch 48, corrected eval, nebius-h100.

## 1. Math

Notation: margins `u_j = z[a_j] − z[b_j]`, `m_j = |u_j|`, `j = 1..n` (n = nap = 8),
`P = ∏_j σ(2 m_j) ∈ (0, 1]`, `g = confidence_gain`.

| form | score |
|---|---|
| `bounded` | `g · P` |
| `bounded_norm` | `g · P^(1/n)` |
| `margin` | `g · (Σ_j m_j) · P` |
| **`min_margin`** (new) | **`g · (min_j m_j) · P`** |

The name follows the existing set: `margin` is the sum-kernel, `min_margin` replaces the sum by the minimum.

**Range and scale.** `min_margin ∈ [0, g · min_j m_j)`, and exactly
`min_margin = (min_j m_j / Σ_j m_j) · margin ≤ margin / n`. On real margins it is far below that
bound (section 2): the median ratio `min_margin / margin` is **0.0175** (bound 1/n = 0.125).

**Derivative** (for `_confidence_score_and_dscore`). With `j* = argmin_j m_j`:

    ∂s/∂m_j = g · P · 1[j = j*]  +  2 · s · σ(−2 m_j)

The second term is the same `∂P/∂m_j = 2P σ(−2m_j)` term `margin` has. The first is where the forms
differ: `margin` adds `g·P` to **every** anchor, `min_margin` only to the argmin anchor. In the
existing helper's style (score without gain, multiply both by g at the end):

    mv, mi   = m.min(dim=-1)
    score    = mv * prob
    dscore_dm = 2.0 * score.unsqueeze(-1) * sig_neg  +  prob.unsqueeze(-1) * one_hot(mi, n)

**Ties.** `min` is not differentiable where two or more margins tie for the minimum; any convex
combination of the tied one-hots is a valid (Clarke) subgradient. **Use the one-hot at the index
`torch.min(dim=-1)` returns** (the first tied index). That is exactly what autograd produces for
`m.min(dim=-1).values`, which is what LightMHL's autograd path differentiates. So FastMHL's
analytic backward and LightMHL's autograd backward stay bit-identical. `torch.amin` splits the
gradient evenly among ties; that is also a valid subgradient, but it breaks that parity, so don't use it.
Exact ties need exactly equal float margins, e.g. a duplicated anchor pair inside one table, so
they are rare but not impossible.
Through the sign: `∂s/∂u_j = ∂s/∂m_j · sgn(u_j)`, and torch's `sgn(0) = 0`.

**At a cell boundary.** When one margin `m_k → 0`, it becomes the argmin, so

    s = g · m_k · σ(2 m_k) · C_k = (g/2) · C_k · m_k + O(m_k²),     C_k = ∏_{j≠k} σ(2 m_j) ∈ (0, 1]

The score **vanishes linearly** in the crossing margin, with a V shape in `u_k` (it is `|u_k|`).

**Continuity and differentiability of y.**
* **n = 1** (`y_h = Σ_t s_t W_t[c_t]`): the cell changes only where `s_t = 0`, and `s` is
  continuous, so **y is continuous everywhere** (C⁰, Lipschitz). It is **not differentiable** at a
  boundary in general. The one-sided derivatives in `u_k` are `−(g/2) C_k W_t[c]` at 0⁻ and
  `+(g/2) C_k W_t[c′]` at 0⁺, equal only if `W_t[c′] = −W_t[c]`. `s` also has kinks (but no jumps) where the argmin
  switches between two anchors. Summary: piecewise smooth, continuous, kinked.
* **n = 2** (blend): continuous at bit flips (as with `margin`, plus `s → 0`). **Still
  discontinuous at argmin switches**: when the two smallest margins tie, the second cell changes
  from `c^(j1)` to `c^(j2)` while `s = g · m_tie · P > 0` and `w1` is unchanged. The jump is
  `s·w1·(W[c^(j2)] − W[c^(j1)])`. `min_margin` shrinks it (smaller `s`) but does not remove it.

Checked numerically for n = 1 (`continuity_probe.py`, section 3): substituting `min_j |u_j|` gives
score exactly 0 at the boundary and a jump of 1.6e-9 of ‖y_h‖, against a median 5.4% for `margin`.

## 2. Scale and selectivity on real margins (`min_margin_scale.py`)

The cache `/tmp/margins_anchor.pt` used by `diag_confidence_forms.py` was **gone**: `/tmp` is
cleared at reboot. It was **regenerated** with `runs_corrected/dump_margins.py`: CPU, one forward
of 2 val rows through `sweep_s05_dout48_H4_tph256_c256_din32`, no training. `--trained` does the
same with that run's 4K checkpoint. The regenerated cache reproduces the published numbers
(bounded 0.0542 / 2.06 / 0.536, margin 0.2286 / 3.04 / 0.870, bounded_norm 0.6838 / 1.09 / 0.061),
so it is the same data. Also measured: the trained Gen-3 standard layer `exp_g_0193` (LightMHL
margin, H4 tph128 nap8, 16K) on real val tokens.

`gain→0.6838` is the convention arms C and D actually used: bounded_norm's mean at init, **not**
gate-off's 1.0. `gain→margin` matches `margin` at gain 1 on the same margins. `gain→1` matches gate-off.

| margins | form | mean | p75/p25 | within-token CV | gain→0.6838 | gain→margin | gain→1 |
|---|---|---|---|---|---|---|---|
| init (6.29M) | bounded | 0.0542 | 2.06 | 0.536 | 12.61 | 4.22 | 18.44 |
| | margin | 0.2286 | 3.04 | 0.870 | 2.99 | 1.00 | 4.37 |
| | bounded_norm | 0.6838 | 1.09 | 0.061 | 1.00 | 0.33 | 1.46 |
| | **min_margin** | **0.0059** | **7.32** | **1.782** | **116.3** | **38.9** | **170.1** |
| 4K trained (6.29M) | margin | 0.9441 | 3.63 | 0.872 | 0.72 | 1.00 | 1.06 |
| | **min_margin** | **0.0260** | **8.97** | **1.725** | **26.3** | **36.3** | **38.5** |
| exp_g_0193 trained | margin | 0.5473 | 5.12 | 0.983 | 1.25 | 1.00 | 1.83 |
| | **min_margin** | **0.0146** | **12.22** | **1.969** | **46.7** | **37.4** | **68.3** |

Fraction of (token, table) `min_margin` scores below 1e-3: 27% at init, 10% after 4K steps, 22% in
exp_g_0193. nap-dependence at init (pooled |u|): 0.358 (n=1), 0.042 (n=4), **0.0055 (n=8)**, 0.00016 (n=16).
It collapses faster than `bounded`.

**Reading.**
* **The scale mismatch is the dominant risk.** At gain 1, `min_margin` sits 39× below `margin`
  and 170× below gate-off at init. That is worse than `bounded`'s 18.4×, which failed by +0.225.
  **Every min_margin arm needs `confidence_gain` from step 0.**
* **Match to `margin`, with G ≈ 38.** The ratio to margin is stable through training (38.9 → 36.3 →
  37.4) because `min/Σ` is stable (median 0.0175). Matching to a fixed number drifts (116 → 26 → 47).
  3.1/3.2 run `margin` at gain 1, which self-normalises (0.229 at init → 0.94 by 4K), so G ≈ 38
  keeps 3.3 at 3.1's scale throughout.
* **Selectivity roughly doubles** (within-token CV 1.7–2.0 vs 0.87–0.98), and 10–27% of rows get
  near-zero weight. A gain fixes the mean, not this shape. Rows reached by near-boundary tokens
  get tiny table gradients (`s·g`), and the input gradient concentrates on the argmin anchor. The
  cost of continuity is gradient starvation near boundaries.
* **Confound:** 3.3 vs 3.1 changes continuity **and** selectivity together; one arm cannot
  attribute a difference to either.

## 3. Implementation touchpoints

| file | lines | change |
|---|---|---|
| `src/spiky/lutorch/fast_multi_head_lut.py` | 142–168 | comment block listing the forms |
| | 171–185 | `_confidence_score`: add the `min_margin` branch |
| | 188–219 | `_confidence_score_and_dscore`: add score + `dscore_dm` (argmin one-hot) and docstring |
| | 1210–1219 | class docstring |
| | 1337–1340 | validation tuple + message |
| | 133, 238, 964, 1020 | no change (they dispatch on the string) |
| `src/spiky/lutorch/light_multi_head_lut.py` | 58–63 | docstring |
| | 117–121 | validation tuple + message |
| | **397** | `_score_form_id` map: add `"min_margin": 3`. **Required**, or construction raises `KeyError` |
| | 700–703 | `_fused_eval` guard: return `None` for id 3 until the native kernel is rebuilt everywhere (falls back to the torch path) |
| | 549, 765 | no change |
| `src/spiky/lutorch/bh4_multi_head_lut.py` | 157–160 | validation tuple + message |
| `native/lutorch/lutorch.cu` | 203–207 | kernel comment: form 3 |
| | 222–238 | loop: also track `min_abs` (init +inf) |
| | 241–247 | `else if (score_form == 3) score = min_abs * exp(-acc);` |
| | 1368, 1399–1400 | binding docstring; validation `score_form > 2` → `> 3` |
| `src/spiky/lutorch/compression_mhl.py` | 121–129 | none functional (passes the string); optional comment |
| `experiments/ffn_replacement/tools/model_build.py` | 195–196 | none (`lut_confidence_form`, `lut_confidence_gain` pass through) |
| tests | `test_fast_mhl_forward_confidence.py:48`, `test_light_embedding_bag_fusion.py:26`, `test_light_multi_head_input.py:124` | add to `_FORMS` / parametrize. The numeric-derivative test must use margins with a separated minimum (finite differences fail at the kink) |
| | `runs_corrected/verify_scored_kernel.py:15` | add `3: 'min_margin'` |
| | new | boundary jump → 0 at n=1; analytic `dscore` equals autograd (tie convention) |

**Estimate:** Python + tests about 2–3 h. CUDA about 10 lines, plus rebuilding the native
extension on gpustar **and nebius-h100** (arch 9.0) and re-running `verify_scored_kernel.py`, about 1–2 h.
**Risks:**
* If nebius runs an un-rebuilt extension, no-grad CUDA eval at n=1 raises `value_error` on form 3.
  The `_fused_eval` guard above removes this dependency, at the cost of a slower eval.
* The fused eval kernel does **not** need the argmin: it is no-grad and emits only index + score.
* `lutorch` tests are pytest (unlike `lut_fused`/`spnet`).

## 4. Proposed table rows

* **3.3: soft forward n=1, `min_margin` confidence, single-alternative backward. Run it.**
  Gain G ≈ 38, matched to `margin`. It is the continuous analogue of 3.1 and differs only in the
  kernel. It is the cleanest test of Anatoly's continuity hypothesis, at the same cost as 3.1.
  Caveat: continuity and selectivity change together (see 2).
* **3.4: n=2 + `min_margin`. Defer.** n=2 is already continuous at bit flips, and `min_margin`
  does not remove n=2's argmin-switch discontinuity, so 3.4 would not test continuity. Run it only
  if 3.3 beats 3.1.
* 3.1/3.2 (`margin`, gain 1) need no gain: `margin` self-normalises during training.

## 5. FLOPs / MACs vs `margin`

Convention as in the PDF: comparisons count as FLOPs, so a min-reduction over n costs the same as
a sum-reduction. P = H·tph = 512, n = 8.

| row | fwd FLOPs | bwd FLOPs | fwd MACs | bwd MACs |
|---|---|---|---|---|
| 3.1 margin n=1 | 77,824 | 131,072 | 24,576 | 49,152 |
| **3.3 min_margin n=1** | **77,824** (Δ 0) | **127,488** (Δ −P(n−1) = −3,584) | 24,576 | 49,152 |
| 3.2 margin n=2 | 134,656 | 235,008 | 49,152 | 98,304 |
| 3.4 min_margin n=2 | 131,072 (Δ −3,584 if it reuses the blend's argmin) | 231,424 (Δ −3,584) | 49,152 | 98,304 |

The backward saves `n−1` adds per table because the `P` term lands on one anchor instead of `n`.
These deltas are inside the PDF's stated tolerance for scalar overheads, so writing "same as 3.1/3.2" is also defensible.
