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

## What remains for a true no-multiply int engine

The MULs left after rung 6 are the **2 psw = score·w** scalars/table and the **1 score =
(Σm)·prob**. Removing them needs score and w themselves in pow2 (data-dependent →
runtime round-to-exponent, cheap) — an untested rung, the obvious next probe. The harder
cost is the **10 transcendentals/table** (8 logsigmoid + exp + sigmoid): a pure-integer
engine replaces these with LUTs / piecewise-linear approximations. **The transcendental
gate, not the multiplies, is the real work of an all-integer deployment** — the multiplies
shift away for ≈1σ; the score's logsigmoid/exp is where the approximation error will live.

## Caveats & next

- Eval-only fake-quant (quantise→dequantise in fp for the forward); it measures the
  *representation* cost, not a real integer kernel's accumulation/rounding. A real int16
  kernel accumulates in int32 and rounds once — expected to match or beat fake-quant.
- Measured on **0195@16K**. Re-run this ladder on the **0203@48K** checkpoint when it lands
  — a better-trained table may quantise differently (usually *more* robustly).
- Next rung: pow2 **score & w** (all-shift read-out); then a piecewise-linear logsigmoid to
  bound the transcendental-approximation error.
