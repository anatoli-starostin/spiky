# exp_g_0195 — quantisation study (STEP 3, overnight mission 005f4e58)

Eval-only PTQ on the **0195@16K** checkpoint (learnable-tau top-2 blend, corrected
eval bs48×100 skip-12). σ = 0.00335. Baseline fp32 = **1.160637**. Refs:
vanilla@16K seed2 1.161798, n=1 control 0193 1.172852. North star: a **no-multiply
int16** deployment of the LightMHL read-out.

## 3a — op catalog (eval forward, margin score, NAP=8, n=2 blend, output_dim=48)

Counts are **per (token, table)**; tables = [512, 256, 48], 128 tables/head × 4 heads,
6 layers. "MUL" = float multiply, the thing the north-star wants to eliminate.

| stage | adds/int | MUL | transcendental | notes |
|---|---|---|---|---|
| margins d_j=z[a]−z[b] | 8 sub | — | — | + 8 sign tests |
| pack address | 8 shift + 7 add (int) | — | — | integer address, no float MUL |
| score: 2m, Σlogsigmoid, exp | 14 add | — | **9** (8 logsigmoid + 1 exp) | `prob = exp(Σ logsigmoid(2m))` |
| score = (Σm)·prob | 7 add | **1** | — | |
| blend: min, logit, sigmoid | ~8 cmp/add | **1** (−2/tau folded) | **1** (sigmoid → w1) | w0 = 1−w1 |
| read-out psw = score·w | — | **2** | — | scalar per row |
| read-out Σ_i w_i·T[c_i][k] | 48 add | **96** | — | 2 rows × 48 dims — **the bulk** |
| accumulate over 128 tables | 48×127 add | — | — | |

**Per-table MUL ≈ 100, of which 96 are `score·w·table` in the read-out.** The address is
already integer/shift-only. The multiply cost is concentrated in one place (the weighted
table read); the transcendentals (10/table: 8 logsigmoid + exp + sigmoid) are elsewhere.

## 3b — PTQ ladder (eval-only fake-quant, tables symmetric, dequant for forward)

| rung | scheme | val_bpb | Δ vs fp32 | σ | vs 0193 control |
|---|---|---|---|---|---|
| 0 | fp32 baseline | 1.160637 | — | — | −3.65σ |
| 1 | **int16 tables, per-row** | 1.160636 | −0.000001 | −0.00σ | −3.65σ |
| 2 | **int8 tables, per-row** | 1.160634 | −0.000003 | −0.00σ | −3.65σ |
| 3 | int16 tables, per-tensor | 1.160640 | +0.000003 | +0.00σ | −3.65σ |
| 4 | int8 tables, per-tensor | 1.160685 | +0.000048 | +0.01σ | −3.64σ |
| 5 | **pow2 tables (shift-only read)** | 1.164086 | +0.003450 | **+1.03σ** | −2.62σ |
| 6 | pow2 tables + pow2 tau | 1.164813 | +0.004177 | +1.25σ | −2.40σ |

(per-row = one scale per 48-dim lookup vector; pow2 = each stored value → nearest signed
2^k, so `w·table` becomes a shift; pow2 tau = each layer's read_tau → nearest 2^k.)

## Reading

- **int8 tables are free** (−0.00σ), per-row *or* per-tensor. The stored table values are
  small and the score gate dominates; 8 bits capture them with no measurable loss. So the
  37.7M table params compress 4× with zero cost, int16 with zero cost.
- **Shift-only tables (rung 5) cost only +1.03σ** and the model *stays −2.62σ under the n=1
  control* and only ~+0.69σ over dense parity. The 96-per-table read-out multiplies —
  ~96% of all MULs — can be turned into **shifts** for ≈1σ. That is the single biggest
  step toward no-multiply and it nearly lands for free.
- **Rounding tau to a power of two adds only +0.22σ on top** (rung 6). The routing weight's
  own multiply (−2/tau) is cheap to remove too. This is consistent with 0195's learned tau
  sitting 1.17–1.34× off the nearest 2^k (see ANALYSIS_0195): not tight, but the +0.22σ
  says the blend tolerates the rounding.

## 3c — pow2 score & w, and the FULL shift-only read-out (MEASURED)

The MULs left after rung 6 are the **2 psw = score·w** scalars/table and the **1 score =
(Σm)·prob**. Rung 5's cheapness made "pow2 the rest too" look free; **it is not.** These
rungs round the *data-dependent* score and blend weights to powers of two as well (eager
path, patched `_blend_bag`; a no-rounding sanity eval reproduces 1.160637 exactly, Δ 0, so
the numbers are faithful).

| rung | scheme | val_bpb | Δ vs fp32 | σ | vs 0193 control |
|---|---|---|---|---|---|
| 7 | pow2 **w** only | 1.166091 | +0.005454 | +1.63σ | −2.02σ |
| 8 | pow2 **score** only | 1.164493 | +0.003856 | +1.15σ | −2.49σ |
| 9 | pow2 score **+** w | 1.171558 | +0.010921 | +3.26σ | −0.39σ |
| 10 | **FULL shift-only** (tables+tau+score+w) | 1.178680 | +0.018043 | **+5.39σ** | **+1.74σ** |

**The correction:** rounding score and w to pow2 is NOT cheap — together +3.26σ, and the
*full* multiply-free read-out is **+5.39σ, which lands ABOVE the n=1 control 0193**. A truly
multiply-free read-out erases the blend's entire advantage and then some. The earlier draft
called the remaining multiplies "cheap to remove" — that was wrong; only the *tables* shift
away cheaply. score·w carries the routing information the blend exists for, and hard pow2
rounding of it destroys that.

## Verdict for a no-multiply / int deployment

- **The right target is int16 (or int8) tables + real integer arithmetic**, not pure shifts.
  int8 tables are free; int16 trivially so. Keep score & w in int (multiply in int16/int32
  accumulate, round once) rather than forcing them to powers of two.
- **Shift-only *tables* (rung 5, +1.03σ) is a viable aggressive knob** on top of int score/w
  — it removes 96 of ~100 MUL/table and the model stays −2.62σ under the n=1 control. This is
  the sweet spot if a multiply-lean (not multiply-free) engine is the goal.
- **A fully multiply-free read-out is off the table** at this training length: +5.39σ, worse
  than n=1. The blend is not worth keeping if score·w must be shift-only.
- The **10 transcendentals/table** (8 logsigmoid + exp + sigmoid) remain the other integer-
  engine cost, addressed by LUT / piecewise-linear approximation — a separate untested axis.

## 3d — same ladder on 0203@48K (better-trained checkpoint)

Re-ran the table rungs on the 48K checkpoint (baseline 1.132300). Mixed vs the 16K read:

| rung | 0195@16K Δ | 0203@48K Δ |
|---|---|---|
| int16 per-row | −0.00σ | −0.00σ |
| int8 per-row | −0.00σ | −0.00σ |
| int8 per-tensor | +0.01σ | +0.02σ |
| **pow2 tables** | **+1.03σ** | **+1.85σ** |
| pow2 tables + tau | +1.25σ | +2.09σ |

- **int8/int16 tables stay free** on the better-trained model — the deployment verdict (int
  tables are free) is robust to training length. ✓
- **But shift-only pow2 tables cost MORE at 48K (+1.85σ vs +1.03σ)** — the opposite of the
  "usually more robust" expectation. The longer-trained tables carry finer structure that
  nearest-2^k rounding cannot represent; more training makes them *less* pow2-friendly, not
  more. Reinforces: int tables yes, shift-only tables only if ~2σ is affordable.

## Caveats & next

- Eval-only fake-quant (quantise→dequantise in fp for the forward); it measures the
  *representation* cost, not a real integer kernel's accumulation/rounding. A real int16
  kernel accumulates in int32 and rounds once — expected to match or beat fake-quant.
- Table rungs re-checked on **0203@48K** (§3d): int free, shift-only *costlier* with training.
  The pow2 score·w rungs (§3c) were not re-run on 48K; the +3.26σ penalty there is unlikely
  to narrow given shift-only tables got worse, not better, with more training.
- The pow2 score/w rungs use *hard* nearest-2^k rounding. A learned/annealed pow2 (or a
  2-bit mantissa "pow2×{1,1.5}") would sit between int and shift-only and might recover most
  of the +3.26σ — an untested middle ground if a multiply-lean engine needs score/w cheaper.
- Transcendental axis (piecewise-linear logsigmoid/exp) remains unmeasured.
