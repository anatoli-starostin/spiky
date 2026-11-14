# Gradient backpropagation

How a gradient reaches the table rows, the address inputs and the learned scalars of a lookup layer whose address is an integer. The address is on [Hashing mechanics](hashing.md) and the read of the rows on [Readout mechanics](readout.md); results are on the Architectures pages. This page only describes backward mechanisms.

## Contents

| Section | Covers |
| --- | --- |
| [1. The problem](#1-the-problem) | channels and the row score (1.1), where the table gradient goes (1.2), value and gradient decoupling (1.3), the routes at a glance (1.4) |
| [2. Neighbour surrogates](#2-neighbour-surrogates) | the surrogate in general form (2.1), choices of $U$ (2.2), $k$ alternatives (2.3), the blend as value and gradient (2.4) |
| [3. Softmax surrogates over rows](#3-softmax-surrogates-over-rows) | all $K$ rows (3.1), restricted forms (3.2), soft address as the forward (3.3) |
| [4. Gradient through the confidence score](#4-gradient-through-the-confidence-score) | confidence read (4.1), hard forward with a score surrogate (4.2) |
| [5. Straight-through estimators for other reads](#5-straight-through-estimators-for-other-reads) | power-of-two int8 read (5.1), bit and vote tables (5.2) |
| [6. Early era: the `lut_fused` code](#6-early-era-the-lut_fused-code) | minimum-delta input gradient and token-pair competition (6.1), gradient policies (6.2) |
| [Open ideas](#open-ideas) |  |
| [Names in the sources](#names-in-the-sources) |  |

## Idea / model

### 1. The problem

#### 1.1 Three channels and the row score

Take one table with $K=2^{n_\text{ap}}$ rows $W[c]\in\mathbb{R}^{d_\text{out}}$ and probe margins $u_j$ as in [hashing 1.1](hashing.md#11-the-basic-construction-manifesto-math), where the absolute margins are $m_j=\lvert u_j\rvert$ and the deciding margin $u^\star=\min_j m_j$ belongs to the deciding probe $j^\star$ and the neighbour address $c'$ has that bit flipped. Write $g=\partial L/\partial y$ for the gradient of the loss at the table's output $y$. The **row score** of row $c$ is

$$r_c=\langle g,W[c]\rangle$$

Reading: this is the change of the loss per unit of weight that the output puts on row $c$ and every input gradient below is a combination of row scores.

A gradient has three **channels**:

- **W**: the table rows.
- **X**: the input $z$ of the table. A margin gradient is scattered to the anchors because $\partial u_j/\partial z[a_j]=1$ and $\partial u_j/\partial z[b_j]=-1$ hold: anchor $a_j$ receives $\partial L/\partial u_j$ and anchor $b_j$ receives its negative, and contributions to the same coordinate add up. A single-anchor probe has only the coordinate $z[a_j]$ (the margin is the coordinate itself, $u_j=z[a_j]$): $z[a_j]$ receives $\partial L/\partial u_j$ and nothing is subtracted anywhere, so the entries of the table no longer sum to zero.
- **P**: learned scalars, such as the temperatures of the surrogates and the scalars $\beta$ and $\gamma$ and $\tau$ of the confidence read.

The address $c$ is an integer without derivative, so the hard read $y=\sum_t W_t[c_t]$ has zero derivative with respect to the input $z$ and the X channel needs either a surrogate, a smooth function whose derivative stands in for the missing one, or a forward that already contains smooth quantities ([section 4](#4-gradient-through-the-confidence-score) below).

Every variant is therefore described by two functions: (a) the function computed in the forward pass and (b) the function whose gradient the input and the scalars receive. In the ablation table of the source the weights receive the gradient of (a) and the input and scalars the gradient of (b), with the weights held constant; the straight-through reads of [section 5](#5-straight-through-estimators-for-other-reads) below depart from this for the weights. The stop-gradient operator is written $\mathrm{sg}$ throughout.

**Where it lives.**

- The definitions follow the [`lutorch_ex` report][report-tex-grad], section "How the gradient reaches the input", and the forward and backward columns of [`lut_ablation_table_v4.tex`][frf-abl4-tableA] (Table A).

#### 1.2 Where the table gradient goes

Four choices exist for the weight gradient $\partial L/\partial W[\ell]$ of one table and one input vector $z$, where $\ell$ is the row index and the weights $U$ ([section 2.2](#22-choices-of-u) below) and $U_i$ with mean $\bar U$ ([section 2.3](#23-k-alternatives) below) and $\pi$ ([section 3.1](#31-all-k-rows-pinned-to-the-chosen-row) below) are the neighbour and softmax weights of those sections:

| choice | $\partial L/\partial W[\ell]$ | used by (sections below) |
| --- | --- | --- |
| hard row | $g$ at the addressed row, zero elsewhere | 2.1, 3.1 with a hard forward, 4.1 scaled by the score |
| blended rows | $(1-U)g$ at the addressed row and $Ug$ at the neighbour | 2.4, 3.1 with a hybrid forward |
| $k$ blended rows | $(1-\bar U)g$ at the addressed row and $U_i g/k$ at neighbour $i$ | 2.3 with the smooth read |
| all rows | $\pi(\ell)g$ for every $\ell$ | 3.3 |

Reading: the first three are the exact gradient of the function that the forward computes. The fourth puts weight on rows that the forward did not read; it can also be paired with a hard forward (a hard read whose table receives the soft weight gradient). The asymmetry of the LUTGPT line is deliberate: the weight gradient stays aligned with the forward, whereas the input side takes a dense surrogate (3.1) in place of the hard read's input gradient, which is zero or sits on a few coordinates. How much this choice matters is not settled; the later experiments made the picture more complicated (see the routes in 1.4).

**Density.** A row receives gradient only from the inputs whose address selects it: with $N$ inputs per step a row gets about $N/K$ inputs if addresses are spread evenly, against all $N$ inputs for a dense weight. A larger batch raises the count per row but not the fraction $1/K$ of inputs.

**Where it lives.**

- The weight backward of each module in the sections below (the last column of the table names the sections).
- Code at `e85a5ce`, [`tiny_multi_head_lut.py`][cal-tiny]: the global toggle [`_GLOBAL_SOFT_WEIGHT_GRAD`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L65) (a hard forward with the soft weight gradient, the fourth row).
- LUTGPT report, [subsection "Why the asymmetry"][cal-lutgpt-asym].

<details>
<summary>Measured gradient statistics of the d24 tables: visit counts and noise (one batch)</summary>

At d24 (context 2048, vocabulary 32768, one batch of the LUT model at steps 500 and 1500, [analysis](https://github.com/anatoli-starostin/spiky/blob/b38b57d70f4c30f877888ac148f9d9e60495b00d/experiments/lut_nanochat/results/lut-grad-spectra-analysis/README.md)): the median table has a Gini coefficient of hit counts of 0.57, the top 10% of cells take 42% of the hits and under 0.4% of cells have no hit; the gradient row norm follows the hit count (rank correlation 0.87 to 0.93). The gradients of different tables are almost independent (a redundancy ratio of 1.6 to 5 where 64 would be aligned), and the signal-to-noise ratio per micro-batch of the table gradients is about 0.2 to 0.5. The soft table gradient on a hard forward gave no gain at all at the 8k scale: 1.4983 for the hard forward with the soft weight gradient ([`exp445`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp445_soft_wgrad_hard_fwd)) and with the hard one ([`exp428`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp428_qkv_tph_x2)), against 1.4821 for the soft forward ([LUT attention](../architectures/transformers/lut-attention.md#what-was-found-with-lut-sdpa-bpe-context-512-vocabulary-32768)).

</details>

#### 1.3 Value and gradient decoupling

All hard-forward surrogates below use one pattern: a hard value with the gradient of a soft path $f$ as follows:

$$y=y_\text{hard}+f-\mathrm{sg}(f)$$

Reading: the value is exactly $y_\text{hard}$ and the gradient is that of the path $f$ alone. The two common forms differ in what is detached:

$$y=y_\text{hard}+\big(p_\text{soft} \mathrm{sg}(W)\big)-\mathrm{sg}\big(p_\text{soft} \mathrm{sg}(W)\big)\qquad\text{against}\qquad y=y_\text{soft}+\mathrm{sg}\big(y_\text{hard}-y_\text{soft}\big)$$

Reading: in the first (**decoupled**) form the soft path $p_\text{soft}$ carries only the address gradient, because the table is detached inside it, and the table gradient comes from the hard path alone, so only the addressed row is trained; in the second the soft path carries the weight gradient too and spreads it over all rows it mixes. The neighbour surrogate (2.1) and the softmax surrogates (3.1, 3.3) use the first form or an equivalent of it. Straight-through estimators for the parameters of the address itself have the same shape and are described with their models in the hashing page: the straight-through value of the ternary weights in [hashing 5](hashing.md#5-ternary-approach), the surrogate for the first-spike time and the bucket boundaries of the LIF digits in [hashing 6.1](hashing.md#61-lif-first-spike-bucket-digits), and the straight-through softmax of the trainable anchor logits, with a hard argmax forward and an annealed temperature, in [hashing 7](hashing.md#7-trainable-anchors).

**Where it lives.** The pattern appears in the code in three spellings, each an example of the same expression:

- `y_c + (surr - surr.detach())` in [`ManifestoHardLUT`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/manifesto_hard.py) ([`manifesto_hard.py`][ex-manhard]); the second form, `sel_hard - sel_soft.detach() + sel_soft`, in [`soft_multi_head_lut.py`][cal-soft].
- The decoupled form for a distillation task with LIF digits: LIF [README][lif-readme-st], section "Straight-through hard addressing".

#### 1.4 The routes at a glance

| route | forward value | input gradient is the gradient of | table gradient | section | code |
| --- | --- | --- | --- | --- | --- |
| neighbour surrogate | hard row | two-row blend with weight $U$ | addressed row | 2.1 | `lut_fused` ([`gt_lut_transformer.py`][lf-gt-bwd]); [`lutorch`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch) ([`anchor_pairs_lookup.py`][lt-apl-bwd]); the `e85a5ce` code ([`tiny_multi_head_lut.py`][cal-tiny-alt]); `lutorch_ex` ([`manifesto_hard.py`][ex-manhard]) |
| $k$ alternatives | hard row or smooth blend | blend with $k$ neighbours | addressed row, or $k+1$ rows | 2.3 | `lutorch` ([`anchor_pairs_lookup.py`][lt-apl-bwd2], [`l_projection.py`][lt-lproj-cls]); [`TinyMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L3666-L4389) ([code at `e85a5ce`][cal-tiny-alt2]) |
| soft Manifesto read | two-row blend | the same blend | both rows | 2.4 | `lutorch` (`smooth_mode` in [`l_projection.py`][lt-lproj-cls2]); [`FastMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L754-L1162) (hybrid, [`fast_multi_head_lut.py`][lt-fast-hybrid]); `lutorch_ex` ([`ManifestoSoftLUT`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/manifesto_soft.py), [`manifesto_soft.py`][ex-mansoft]) |
| all-rows surrogate | hard row or two-row blend | softmax over all $K$ rows, signs pinned to the chosen row | addressed row, or both rows | 3.1 | [`TinyMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L3666-L4389) ([code at `e85a5ce`][cal-tiny-soft]); [`FastMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L754-L1162) ([`fast_multi_head_lut.py`][lt-fast-soft]) |
| restricted surrogate | as above | the same softmax over the kept rows | as above | 3.2 | [`TinyMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L3666-L4389) (`soft_topk`, [code at `e85a5ce`][cal-tiny-topk]); [`FastMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/fast_multi_head_lut.py#L749-L943) (`ball` in the [older file][cal-fast-ball], `backward_topk` in the [later file][frf-fast-topk]) |
| soft address | softmax over all $K$ rows | itself | all rows | 3.3 | [`SoftMultiHeadLUT`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/soft_multi_head_lut.py#L45-L281) ([`soft_multi_head_lut.py`][cal-soft]); [`MatmulMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L4475-L4594) (in [`tiny_multi_head_lut.py`][cal-tiny-matmul]) |
| confidence read | rows scaled by the score | the score and the blend weight themselves | rows read, scaled by the score | 4.1 | `lutorch_ex` ([`ConfidenceLUT`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/confidence.py#L55-L114), [`confidence.py`][ex-conf]); [`LightMultiHeadLUT`](https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/light_multi_head_lut.py) ([`light_multi_head_lut.py`][frf-light]) |
| hard read, score surrogate | hard row | score read on detached tables | addressed row | 4.2 | [`LightMultiHeadLUT`](https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/light_multi_head_lut.py) ([`light_multi_head_lut.py`][frf-light-hard]) |
| power-of-two int8 read | integer read | exact float blend | float master rows | 5.1 | `lutorch_ex` ([`quantised_confidence.py`][ex-quantconf], [`_pow2.py`][ex-pow2]) |

### 2. Neighbour surrogates

A hard read has no derivative with respect to the input. The neighbour surrogate takes the gradient of the blend of the addressed row and its neighbour row instead, with the blend weight $U$ a function of the deciding margin. Subsection 2.1 gives the form for any $U$, 2.2 the choices of $U$, 2.3 the case of $k$ neighbours and 2.4 the blend used as the forward value too.

#### 2.1 The neighbour surrogate in general form

The forward value is the hard read $y=\sum_t W_t[c_t]$ of [readout 1.1](readout.md#11-hard-lookup-and-accumulate). Its gradient is taken from the blend of [readout 2.1](readout.md#21-manifesto-blend-soft-read) instead:

$$\tilde y=\sum_t\Big[(1-U_t)W_t[c_t]+U_tW_t[c'_t]\Big],\qquad U_t=U(u^\star)$$

with the addresses $c_t$ and $c'_t$ and the deciding probe of every table held fixed. Reading: $U\in(0,\frac12]$ is the share of the neighbour row and a function of the deciding margin only (the choices of $U$ are in 2.2 below); the value of $\tilde y$ is never used. The output is the hard value plus the surrogate term $\sum_tU_t\big(W_t[c'_t]-W_t[c_t]\big)$ with its own value subtracted out, computed on a detached table, so the term changes nothing in the value and adds gradient only to $U$ and through it to the input (the code spelling is in the "Where it lives" block of [1.3](#13-value-and-gradient-decoupling)).

Differentiating gives, per table, with $U'$ the derivative of $U$ with respect to the absolute margin,

$$\frac{\partial L}{\partial u_{j^\star}}=U'(u^\star) \mathrm{sgn}(u_{j^\star}) \big(r_{c'}-r_c\big)$$

Reading: if the neighbour row would lower the loss more than the addressed row, that is, $r_{c'}<r_c$ holds, the loss wants the margin to shrink towards the tie and so towards selecting the neighbour (the derivative $U'$ is negative). The input gradient is nonzero only at the two anchors of the deciding probe, with opposite signs; an input gives at most two nonzero input entries per table and, for pair probes, the entries of one table sum to zero. A single-anchor probe passes the same value to $z[a_{j^\star}]$ only. Any $U$ with a derivative can be used; an uncertainty function with learned temperatures also passes gradient to them (2.2 below).

Worked line for the rational uncertainty function $U(m)=\frac{0.5}{1+m}$ of 2.2: the derivative is $U'(m)=-\frac{0.5}{(1+m)^2}$ and so

$$\frac{\partial L}{\partial u_{j^\star}}=\frac{0.5 \mathrm{sgn}(u_{j^\star})}{(1+\lvert u_{j^\star}\rvert)^2}\big(r_c-r_{c'}\big)$$

Reading: the same two-row difference as above, scaled by a factor that is largest at the tie and falls off as $1/(1+m)^2$.

In this read the input gradient exists only through the neighbour surrogate: its weight $U$ is a design choice and not a derivative of anything the forward computes. The confidence read of [section 4](#4-gradient-through-the-confidence-score) below is an alternative route with no surrogate.

**Status:** current surrogate of the reference and fused hard cartridges of `lutorch_ex`, for both uncertainty functions and both anchor modes.

**Where it lives.**

- `lut_fused`: the ground-truth python [`gt_lut_transformer.py`][lf-gt-bwd2] (`LUT.backward`, `up()`; file added in cf45631e, 2025-12-04) and the CUDA runtime ([`lut_runtime.cu`][lf-rt-bwd] with the kernel logic in [`lut_runtime_kernels_logic.proto`][lf-rtlogic-prop], `propagate_through_detectors_*`); see 6.1 below.
- `lutorch`: `AnchorPairsLookupFunction.backward` in [`anchor_pairs_lookup.py`][lt-apl-bwd3] (file added in a7cf2c12, 2026-02-22) with the CUDA kernels in [`lutorch.cu`][lt-cu-bwd]; [`LProjection`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/l_projection.py#L175-L349) in [`l_projection.py`][lt-lproj-cls3].
- Code at `e85a5ce`: `TinyMultiHeadLut` ([`tiny_multi_head_lut.py`][cal-tiny], [`_multi_alt_bwd_body`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L600-L713) with one alternative; file added in 64ead82d, 2026-04-29) and `FastMultiHeadLut` ([`fast_multi_head_lut.py`][cal-fast]; file added in 79757af3, 2026-06-07).
- `lutorch_ex` (main; first added in 6693d8bb, 2026-10-01): `ManifestoHardLUT` in [`manifesto_hard.py`][ex-manhard] with [`manifesto_base.py`][ex-manbase] (`_addresses`, `_uncertainty`) and [`uncertainty.py`][ex-uncert]; the fused twin [`fused_manifesto.cu`][ex-fmcu] with the kernels [`softsign_surrogate_grad.cu`][ex-softsigncu] and [`single_anchor_input_grad.cu`][ex-singlecu]; report section "How the gradient reaches the input" ([report][report-tex-grad2]).


#### 2.2 Choices of $U$

The weight $U$ of [2.1](#21-the-neighbour-surrogate-in-general-form) is a function of $m=u^\star$ and enters the gradient only through its derivative: the surrogate multiplies $r_{c'}-r_c$ by it.

**The rational function is the Spiking Manifesto's own uncertainty function.** $U$ equals $\frac12$ at a tie and decreases to zero as the margin grows. The hard read with this $U$ (`ManifestoHardLUT` of `lutorch_ex`) is, according to the `lutorch_ex` report, the rule of the Manifesto as it was published. In the Manifesto $U$ is used only for the training gradient and inference uses the plain sum of the selected rows. The soft read that puts $U$ in the forward (2.4 below) goes beyond the Manifesto.

The other uncertainty functions are variants of ours. The rational and softsign weights are the ones that define the blend in [readout 2.1](readout.md#21-manifesto-blend-soft-read), so the column $U(m)$ below repeats those definitions and the derivative column is what the gradient needs. The uncertainty functions found in the code are

| name | $U(m)$ | derivative of $U$ with respect to $m$ | code |
| --- | --- | --- | --- |
| rational (`INVERSE_L1`, default; the Manifesto's) | $\frac{0.5}{1+m}$ | $-\frac{0.5}{(1+m)^2}$ | `lut_fused` (`up()`; [`LUTLayer.py`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/LUTLayer.py) added in 6364a7e9, 2025-11-14); `lutorch` ([`UncertaintyMode`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/lut_helpers.py#L1023-L1025)); `lutorch_ex` (`rational_uncertainty` in [`uncertainty.py`][ex-uncert], the cartridges and the fused kernels) |
| `INVERSE_QUADRATIC` | $\frac{0.5}{1+m^2}$ | $-\frac{m}{(1+m^2)^2}$ | `lutorch` ([`UncertaintyMode`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/lut_helpers.py#L1023-L1025)); see the remark below the table for `lutorch_ex` |
| rational with a learned breakpoint (mode `ste`) | $\frac{0.5}{T+m}$ | $-\frac{0.5}{(T+m)^2}$ | [`TinyMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L3666-L4389) only (first added in ab499fac, 2026-05-11) |
| softsign | $\sigma\big(-2\rho/T_\text{sel}\big)$ with $\rho=\frac{m}{T_\text{soft}+m}$ | $-U(1-U)\frac{2}{T_\text{sel}}\frac{T_\text{soft}}{(T_\text{soft}+m)^2}$ | [`TinyMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L3666-L4389) (soft backward mode, first added in 38008342, 2026-05-11); [`SoftMultiHeadLUT`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/soft_multi_head_lut.py#L45-L281) (file added in a9151e8e, 2026-05-16); [`FastMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L754-L1162) (in the first version of its file, 79757af3, 2026-06-07); `lutorch_ex` (softsign cartridges and [`softsign_surrogate_grad.cu`][ex-softsigncu]) |

Remark on versions. `lutorch_ex` had a function `rational_uncertainty_quadratic` (the $0.5/(1+m^2)$ uncertainty function) in `uncertainty.py` from its first commit 6693d8bb until 127aab1a (both 2026-10-01), which removed it; no cartridge or fused kernel selected it, and it is not on main. On main the cartridges and fused kernels select between the rational and the softsign kernel only (`uncertainty_func`). The learned breakpoint $T$ is not in `FastMultiHeadLut` (checked at 79757af3, e85a5ce and c4a750b).

Reading: all four are largest at a tie and fall to zero as the margin grows, so a bit that is far from flipping passes no gradient. The rational function has the derivative of largest size exactly at the tie; the quadratic function has derivative zero at the tie and a maximum of size at $m=1/\sqrt3$ (derived here), so a near-tied bit gets less gradient than a slightly separated one. The older code writes the backward as $-U' \mathrm{sgn}$ times $r_c-r_{c'}$ and this is the same expression as in 2.1.

The softsign function has two learned temperatures per layer (stored as logarithms). Their gradients are, per table and multiplied by $r_{c'}-r_c$ and by the temperature itself,

$$\frac{\partial U}{\partial T_\text{soft}}=U(1-U)\frac{2}{T_\text{sel}}\frac{m}{(T_\text{soft}+m)^2},\qquad \frac{\partial U}{\partial T_\text{sel}}=U(1-U)\frac{2\rho}{(T_\text{sel})^2}$$

The learned breakpoint $T$ of the third function receives $r_{c'}-r_c$ times $-0.5/(T+m)^2$ in the same way; its constant $0.5$ only rescales the whole input gradient. At $T=1$ it is the rational function.

<details>
<summary>Rational against softsign weights, with soft and hard reads (48k, fixed evaluation)</summary>

Rational against softsign with the soft and the hard read ([fused sweep README](https://github.com/anatoli-starostin/spiky/blob/b1a5a7131732f93595f53bf572299139eb1a0856/experiments/lutorch_ex_fused/README.md#results-on-the-full-170-shard-corpus-48k-steps--these-are-the-comparable-numbers), 48k steps, fixed evaluation, single seeds): hard read 1.1324 (rational) against 1.1378 (softsign); soft read 1.1186 (rational) against 1.1095 (softsign). The learned temperatures of the LUT SDPA models (48 per model) were not compared against fixed ones in a run we found ([FFN replacement, section 5](../architectures/transformers/ffn-replacement.md#5-lutorch_ex-cartridges-on-the-full-corpus-october-2026)).

</details>

**Where it lives.**

- `lut_fused`: `up()` in [`gt_lut_transformer.py`][lf-gt-up]; the same factor in the kernel logic of the CUDA runtime ([`lut_runtime_kernels_logic.proto`][lf-rtlogic]).
- `lutorch`: `UncertaintyMode` in [`anchor_pairs_lookup.py`][lt-apl-bwd4] (the rational and the quadratic kernel) and the backward kernels of [`lutorch.cu`][lt-cu-bwd2].
- Code at `e85a5ce`: `_multi_alt_bwd_body` in [`tiny_multi_head_lut.py`][cal-tiny-alt3] (the learned breakpoint) and its hybrid and soft backward modes (softsign); [`soft_multi_head_lut.py`][cal-soft]; the derivation of the softsign weight is in [`hybrid_smooth_math.tex`][cal-hybridtex].
- `lutorch_ex`: [`uncertainty.py`][ex-uncert], `_uncertainty` in [`manifesto_base.py`][ex-manbase-unc] and the kernel [`softsign_surrogate_grad.cu`][ex-softsigncu].

#### 2.3 $k$ alternatives

The $k$ probes with the smallest margins give $k$ neighbours with addresses $c^{(i)}$ that each have one bit flipped and with signed margins $\delta_i$ at the flipped probes. With $U_i=U(\lvert\delta_i\rvert)$ from 2.2 the surrogate is the smooth read of [readout 2.1](readout.md#21-manifesto-blend-soft-read),

$$\tilde y=\Big(1-\frac1k\sum_{i=1}^{k}U_i\Big)W[c]+\sum_{i=1}^{k}\frac{U_i}{k}W[c^{(i)}]$$

and the input gradient at the two anchors of flipped probe $i$ is

$$\frac{\partial L}{\partial \delta_i}=\frac{1}{k}U'(\lvert\delta_i\rvert) \mathrm{sgn}(\delta_i) \big(r_{c^{(i)}}-r_c\big)$$

Reading: the case $k=1$ is 2.1. The address module returns zero-valued carrier tensors; the read module's backward fills their gradients with the row scores $r_c$ and $r_{c^{(i)}}$ and the address module's backward turns them into the input gradient above. The table gradient is hard in the hard mode and puts the weight $U_i/k$ on neighbour $i$ in the smooth mode (1.2). The winner-take-all address of [hashing 6.3](hashing.md#63-winner-take-all-address) returns the same carriers with gaps to the winner and uses the same uncertainty functions. `TinyMultiHeadLut` in mode `ste` picks the alternatives by repeated argmin and recomputes them in the backward, so nothing is saved in the forward.

**Status:** legacy; `lutorch_ex` has $k=1$ only.

**Where it lives.**

- `lutorch`: `AnchorPairsLookupFunction.backward` in [`anchor_pairs_lookup.py`][lt-apl-bwd5] (the argument `n_alternatives`); the smooth read and backward of `LProjection` in [`l_projection.py`][lt-lproj-cls4].
- Code at `e85a5ce`: `_multi_alt_bwd_body` in [`tiny_multi_head_lut.py`][cal-tiny-alt4].
- `lutorch_ex`: none ($k=1$ only).

<details>
<summary>Number of alternatives of the smooth read, 1 to 6: no trend beyond noise</summary>

The number of alternatives of the smooth read, 1 to 6 (tiny BPE model, 43M parameters, 8k steps, old evaluation, single seeds): 1.5865 with 2 alternatives falling steadily to 1.5741 with 6 (3 alternatives 1.5805) ([LUT attention](../architectures/transformers/lut-attention.md#what-was-found-with-lut-sdpa-bpe-context-512-vocabulary-32768), forward and backward modes). For the confidence layer, the number of rows read (1 or 2) is worth 0.0105 ([FFN replacement, section 5](../architectures/transformers/ffn-replacement.md#5-lutorch_ex-cartridges-on-the-full-corpus-october-2026)).

</details>

#### 2.4 The blend as value and gradient

When the blend of 2.1 is also the forward value (the soft read of [readout 2.1](readout.md#21-manifesto-blend-soft-read)), the weights receive the gradient of the blend at fixed weights,

$$\frac{\partial L}{\partial W[c]}=(1-U) g,\qquad \frac{\partial L}{\partial W[c']}=U g$$

and the input gradient is the one of 2.1, now the exact gradient of the computed function. No surrogate is involved here: ordinary autograd on this forward, with the addresses held fixed, produces exactly these gradients. Reading: both rows are trained by every input that blends them, in proportion to their share; as $U\to0$ the weight gradient becomes the hard row of 1.2. The index set stays stop-gradient and the value is continuous across a bit flip, since at a tie both rows have weight one half and swap roles.

`FastMultiHeadLut` with `forward_mode="hybrid_smooth"` has the same forward and the same two-row weight gradient, but its input gradient comes from the all-rows surrogate of 3.1 below, or from the two-cell softmax of 3.2 below when `backward_topk=1`, which equals the blend.

**Status:** current in `lutorch_ex`; the hybrid form is the soft forward of the `FastMultiHeadLut` line.

**Where it lives.**

- `lutorch`: the smooth mode (`smooth_mode`) of [`MultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/multi_head_lut.py#L161-L485) and `LProjection` ([`l_projection.py`][lt-lproj-cls5]) reads the same blend with one alternative.
- Code at `e85a5ce`: `forward_mode="hybrid_smooth"` with [`_hybrid_smooth_weight_grad`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L515-L553) in [`fast_multi_head_lut.py`][lt-fast-hybrid2] (the older version is [here][cal-fast-hybrid]) and the hybrid mode of [`tiny_multi_head_lut.py`][cal-tiny-hs]; [`hybrid_smooth_math.tex`][cal-hybridtex].
- `lutorch_ex`: `ManifestoSoftLUT` in [`manifesto_soft.py`][ex-mansoft] and [`fused_manifesto_soft.py`][ex-fmsoft], with the fused kernel [`fused_manifesto.cu`][ex-fmcu].

<details>
<summary>Soft-trained model read hard: how much it loses (June report, old evaluation)</summary>

June report, fast test scale (context 512, vocabulary 32768), old evaluation: a model trained with the two-row `hybrid_smooth` forward scores 1.1842 evaluated that way and 1.2498 on the hard read (+0.066) ([`exp754`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp754_exp752_hybrid_smooth_16k)); the narrow-backbone version 1.2044 and 1.2778 (+0.073) ([`exp755`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp755_exp754_E192_dv32_16k)); trained hard from the start the two models reach 1.2162 and 1.2301 ([`exp752`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp752_exp750_recipe_16k), [`exp756`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp756_exp755_hard_16k)), and 24k hard steps match the 16k soft steps within noise, 1.2048 against 1.2044 ([`exp760`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp760_exp756_24k)). A soft-trained narrow model fine-tuned to hard for 4k steps reached 1.2513 ([`exp715`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp715_exp710_soft_to_hard)), worse than 1.2301 from the start. All numbers are in the table of [LUT attention](../architectures/transformers/lut-attention.md#what-was-found-with-lut-sdpa-bpe-context-512-vocabulary-32768) (forward and backward modes).

</details>

### 3. Softmax surrogates over rows

#### 3.1 All $K$ rows, pinned to the chosen row

The backward replaces the one-row read by a softmax over all rows whose signs are pinned to the bits that the forward actually chose. Let $s_j=\pm1$ be the chosen bit at probe $j$ and $\chi_j(\ell)=\pm1$ bit $j$ of row $\ell$ (most significant bit first) and define

$$p_j=s_j\frac{m_j}{T_\text{soft}+m_j},\qquad S(\ell)=\sum_jp_j\chi_j(\ell),\qquad \pi(\ell)=\mathrm{softmax}_\ell\Big(\frac{S(\ell)}{T_\text{sel}}\Big)$$

with the learned temperatures $T_\text{soft}$ and $T_\text{sel}$ (stored as logarithms, one pair per layer). The backward is the exact gradient of the soft read $\sum_\ell\pi(\ell)W[\ell]$ with respect to the input and the two temperatures, at fixed table weights. With the softmax-weighted mean score $\bar r=\sum_\ell\pi(\ell)r_\ell$ and

$$e_\ell=\pi(\ell)\big(r_\ell-\bar r\big),\qquad \frac{\partial L}{\partial p_j}=\frac{1}{T_\text{sel}}\sum_\ell e_\ell \chi_j(\ell),\qquad \frac{\partial L}{\partial u_j}=\frac{\partial L}{\partial p_j}\frac{T_\text{soft}}{(T_\text{soft}+m_j)^2}$$

the margin gradient is scattered to the anchors as in 1.1 (the factor $s_j \mathrm{sgn}(u_j)$ that the code multiplies in is one, except at exact ties). The temperature gradients are

$$\frac{\partial L}{\partial\log T_\text{sel}}=-\sum_\ell e_\ell\frac{S(\ell)}{T_\text{sel}},\qquad \frac{\partial L}{\partial\log T_\text{soft}}=-\sum_ju_j\frac{\partial L}{\partial u_j}$$

summed over inputs and tables. The table gradient is aligned with the forward: the hard row in a hard forward, two rows in a hybrid forward (2.4), nothing on the other rows.

Reading: because $p_j$ carries the sign of the chosen bit, the chosen row has the largest score and flipping bit $j$ lowers it by exactly the amount $2\lvert p_j\rvert$ in the score. The gradient is dense in all probes, unlike 2.1, and it is nonzero for the temperatures even with a hard forward, where they do not change the value. The softmax gradient also reads as a covariance. The score is linear in the $p_j$ and so $\pi$ factorises into independent bits, bit $j$ being $+1$ with probability $q_j=\sigma(2p_j/T_\text{sel})$ and the derivative of the output with respect to $p_j$ equals the factor $2q_j(1-q_j)/T_\text{sel}$ times the difference between the mean rows with bit $j$ set and with bit $j$ unset, both means taken with the weights $\pi$ (stated in the gradient notes for a linear score). The factorisation removes the normaliser but not the cost: $r_\ell$ is needed for every row, so the work is of order $K d_\text{out}$ per table and input.

Two remarks. With the log-sum-exp read over tables ([readout 3.1](readout.md#31-log-sum-exp-over-tables)) the same surrogate is used for the input and the temperatures, weighted per table by the log-sum-exp softmax, while the rows and $\tau$ get the exact gradient. When the margins come from hyperplanes, the same backward passes the margin gradient on to the hyperplane weights and bias; the full soft forward, whose exact gradient this backward is, can be kept as a gradient-check target.

**Status:** current for the `FastMultiHeadLut` line (hard forward with this backward); not in `lutorch_ex`.

**Where it lives.**

- `FastMultiHeadLut`: `_soft_lut_bwd_body` and the autograd functions of `forward_mode="hard"` and `"hybrid_smooth"` in [`fast_multi_head_lut.py`][lt-fast-soft2] (the older version with `backward_mode="dense_K"` is [here][cal-fast-soft]); the file was added in 79757af3 (2026-06-07) and has this backward, with the learned temperatures, from its first version.
- `TinyMultiHeadLut`: `backward_mode="soft"` in [`tiny_multi_head_lut.py`][cal-tiny-soft2] (soft backward mode first added in 38008342, 2026-05-11).
- [`HyperplaneMultiHeadLUT`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/hyperplane_multi_head_lut.py#L622-L891) in [`hyperplane_multi_head_lut.py`][lt-hyp-fullsoft] (file added in b3f6b077, 2026-07-23), whose method `forward_full_soft` is the full soft forward.
- Derivations: LUTGPT report ([subsection "Soft surrogate backward"][cal-lutgpt-soft]); ablation table ([row 2.1][frf-abl4-r21]); notes in [`lut_gradient_notes.tex`][cal-gradnotes].

<details>
<summary>All-rows backward against the Hamming-ball backward: 0.018 in favour of all rows</summary>

At the hard forward, fast test scale, old evaluation, 16k steps: the backward restricted to the Hamming ball of the chosen row reached 1.2360 ([`exp729`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp729_FastMHL_hard_ball)) and the backward through all rows 1.2178 ([`exp731`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp731_FastMHL_hard_dense_K)), so the all-rows surrogate is worth 0.018; the LUT SDPA report states that a one-row backward trains but converges to a substantially worse loss, without a number ([LUT attention](../architectures/transformers/lut-attention.md#what-was-found-with-lut-sdpa-bpe-context-512-vocabulary-32768), forward and backward modes). At the fixed evaluation and 16k steps, restricting the surrogate of the `FastMultiHeadLut` to the best neighbour cost 0.023 with one row and 0.010 with two rows ([FFN replacement, section 4](../architectures/transformers/ffn-replacement.md#4-fixed-evaluation-confidence-forms-and-parity-that-does-not-last-september-2026)).

</details>

#### 3.2 Restricted surrogates

The softmax of 3.1 is restricted to $1+k$ rows: the chosen row and the $k$ rows reached by flipping the $k$ probes with the smallest margins; for $k=n_\text{ap}$ this is the whole Hamming-1 ball of $n_\text{ap}+1$ rows. Non-kept rows carry no mass, the softmax is renormalised over the kept rows and the rest of the chain is unchanged; the weight gradient is unchanged too. Three forms of this restriction occur:

- **Soft top-k:** the $k$ smallest margins give the kept rows and the scores keep their signs $p_j$ as in 3.1.
- **Ball:** all $n_\text{ap}$ flips are kept ($k=n_\text{ap}$) and the scores use the unsigned magnitudes $\lvert p_j\rvert$ in place of $p_j$.
- **Backward top-k:** the same restricted softmax as soft top-k, computed from the kept rows only, so the other rows are never touched; with $k=1$ it is the two-cell form below.

**Two cells.** For $k=1$ the kept rows are $c$ and $c'$ and they differ only at bit $j^\star$ so their scores differ by $2\lvert p_{j^\star}\rvert=2\rho$ and the weight of $c'$ is the softsign uncertainty function $U=\sigma(-2\rho/T_\text{sel})$ of 2.2. The softmax derivatives sum to zero over the kept rows, which gives $e_c+e_{c'}=0$ and the two rows agree on every bit except $j^\star$ so that $\partial L/\partial p_j=0$ holds for all $j\ne j^\star$ and

$$\frac{\partial L}{\partial u_{j^\star}}=\frac{2}{T_\text{sel}}U(1-U)\frac{T_\text{soft}}{(T_\text{soft}+u^\star)^2} \mathrm{sgn}(u_{j^\star}) \big(r_c-r_{c'}\big)=U'(u^\star) \mathrm{sgn}(u_{j^\star}) \big(r_{c'}-r_c\big)$$

Reading: the two-cell restriction is exactly the softsign neighbour surrogate of 2.1, with the same temperature gradients, because both are the derivative of the same two-row blend. This was derived here from the closed form that the two-cell backward kernel implements (one scatter at $j^\star$ and no bit-matrix product) and is stated in the ablation table (row 2.3 states that the weight on $c'$ is exactly the forward weight $w_t$ of row 2.2). With a hybrid forward and two cells (row 2.4) the backward equals the forward; the input gradient then reaches only the probe $j^\star$ and not all probes as in the full softmax of row 2.2.

**Cost.** The cost depends on how the kept scores are formed, not on the value. Masking the non-kept rows before the softmax still forms the product over all $K$ rows, so it saves memory but not work; forming $r_\ell$ for all rows and then gathering the ball costs the same as 3.1; gathering only the kept rows costs of order $1+k$ rows per table.

**Status:** cheaper than 3.1; kept as a tool and as a candidate for a native sparse kernel.

**Where it lives.**

| form | cost | code |
| --- | --- | --- |
| soft top-k | masks, still all $K$ rows | [`TinyMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L3666-L4389), `backward_mode="soft_topk"` in [`tiny_multi_head_lut.py`][cal-tiny-topk2] (first added in c2a89a44, 2026-05-16) |
| ball | forms $r_\ell$ for all rows, gathers the ball | `ball` mode, [`_ball_lut_bwd_body`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/fast_multi_head_lut.py#L200-L284) in the older [`fast_multi_head_lut.py`][cal-fast] (in the first version of the file, 79757af3, 2026-06-07) |
| ball, gathered | kept rows only | `ball_gather`, [`_ball_gather_lut_bwd_body`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/fast_multi_head_lut.py#L312-L398) in the older [`fast_multi_head_lut.py`][cal-fast] (79757af3, 2026-06-07) |
| backward top-k | kept rows only, $1+k$ per table | `backward_topk`: `_soft_lut_bwd_body_topk` and, for $k=1$, [`_hybrid_smooth_2cell_bwd`](https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/fast_multi_head_lut.py#L673-L757) in [`fast_multi_head_lut.py`][frf-fast-topk2] (first added in bd438793, 2026-09-06) |

Rows 2.3 and 2.4 of the [ablation table][frf-abl4-r2324] state the two-cell forms.

<details>
<summary>Restricted against full softmax backward, with hard and hybrid forward</summary>

Restricted (Hamming ball) against full softmax backward in the 16k LUT SDPA recipe, old evaluation: hard forward 1.2360 against 1.2178; `hybrid_smooth` forward with the ball backward 1.2115 evaluated soft and 1.2810 hard, about 0.07 soft-to-hard as with the full backward. At 8 address bits in the feed-forward on the fixed evaluation the full surrogate was better by 0.023 (hard, one row) and 0.010 (`hybrid_smooth`) ([LUT attention](../architectures/transformers/lut-attention.md#what-was-found-with-lut-sdpa-bpe-context-512-vocabulary-32768), [FFN replacement, section 4](../architectures/transformers/ffn-replacement.md#4-fixed-evaluation-confidence-forms-and-parity-that-does-not-last-september-2026)). A comparison at several numbers of address bits was not found.

</details>

#### 3.3 Soft address as the forward

The soft lookup of 3.1 without pinning the signs can serve as the forward itself:

$$p_j=\frac{u_j}{T_\text{soft}+\lvert u_j\rvert},\qquad y=\sum_\ell\mathrm{softmax}_\ell\Big(\frac{S(\ell)}{T_\text{sel}}\Big)W[\ell]$$

with $S$ as in 3.1. Reading: as both temperatures tend to zero this tends to the hard lookup; at finite temperatures every row has weight and receives the weight gradient $\pi(\ell)g$ (the fourth line of 1.2), and the input and temperatures get the exact autograd gradient of the computed function, with no surrogate.

Two variants exist: a Gumbel-noise option and a straight-through option, built as a hard selector minus the detached soft selector plus the soft selector (1.3): the value is the one-hot row, the table gradient is that of the one-hot (the hard row), and the input gradient is that of the soft selector, which is the gradient of 3.1 (the signs are not pinned, but the argmax of the soft scores is the chosen row).

The forward needs the product over all $K$ rows, of order $K d_\text{out}$ per table and input, with buffers of size $K$ per input and table; a matrix-multiplication form of it exists. It is a reference and an upper bound and not a deployable read, because deployment uses the hard read.

**Status:** reference; not deployable.

**Where it lives.**

- `SoftMultiHeadLUT` in [`soft_multi_head_lut.py`][cal-soft] (file added in a9151e8e, 2026-05-16): the Gumbel option and the straight-through option `hard=True`, written `sel_hard - sel_soft.detach() + sel_soft`.
- `MatmulMultiHeadLut` and `backward_mode` `"soft"` in [`tiny_multi_head_lut.py`][cal-tiny-matmul2] (`MatmulMHLut` first added in 4d2f44d4, 2026-05-23).
- `forward_full_soft` of [`hyperplane_multi_head_lut.py`][lt-hyp-fullsoft2].

<details>
<summary>Soft forward against hard forward at 8k steps: a 0.016 gain at five times the time</summary>

At the 8k scale (89M parameters, context 512, vocabulary 32768, old evaluation): the soft forward gave 1.4821 against 1.4983 for the hard forward, a gain of 0.016 that appears only after step 3,400, at five times the time; the soft table gradient on the hard forward gave 1.4983, no gain ([LUT attention](../architectures/transformers/lut-attention.md#what-was-found-with-lut-sdpa-bpe-context-512-vocabulary-32768), forward and backward modes).

</details>

### 4. Gradient through the confidence score

#### 4.1 Confidence read

The confidence read multiplies each table's read by the smooth score $s_t=\big(\sum_jm_j\big)\prod_j\sigma(\beta m_j)^{\gamma}$ and, at two rows, blends the neighbour with the weight $v_t=\sigma(-2u^\star/\tau)$ of readout 2.2. The address is detached: there is no straight-through estimator, no surrogate and no surrogate temperature. The forward value itself contains $s_t$ and $v_t$ and so the input gradient is the exact gradient of the computed function and reaches the input through them: through $s_t$ at all $n_\text{ap}$ margins, and at two rows through $v_t$ at the deciding margin. Writing $P=\prod_j\sigma(\beta m_j)$ gives $s=(\sum_jm_j)P^{\gamma}$ as an equivalent form of the score, and with one row per table $\partial L/\partial s=r_c$ and

$$\frac{\partial L}{\partial u_j}=r_c \frac{\partial s}{\partial m_j} \mathrm{sgn}(u_j),\qquad \frac{\partial s}{\partial m_j}=P^{\gamma}+\gamma\beta s \sigma(-\beta m_j)$$

Reading: the implementations let autograd differentiate the score and the closed form is derived here, with the one fact $\frac{d}{dm}\log\sigma(\beta m)=\beta\sigma(-\beta m)$ for the factors of the product; for $\beta=2$ and $\gamma=1$ it is the analytic derivative of the `margin` form in [`lut_mechanisms.tex`][frf-mech-forms], cross-checked there against autograd. With one row, a single scalar $r_c$ multiplies a profile over the probes that depends only on the margins, because $\partial s/\partial m_j$ is positive for positive $\beta$ and $\gamma$. The gradient therefore moves all margins of the table together: it widens them when $r_c$ is negative, since a larger score lowers the loss, and narrows them when $r_c$ is positive. It never compares the addressed row with another row. At two rows the table output is $s\big((1-v)W[c]+vW[c']\big)$ and

$$\frac{\partial L}{\partial s}=(1-v)r_c+vr_{c'},\qquad \frac{\partial L}{\partial v}=s \big(r_{c'}-r_c\big),\qquad \frac{\partial v}{\partial u^\star}=-\frac{2}{\tau}v(1-v)$$

so the margin gradient is $\mathrm{sgn}(u_j) \frac{\partial L}{\partial s}\frac{\partial s}{\partial m_j}$ at every probe plus $\mathrm{sgn}(u_{j^\star}) \frac{\partial L}{\partial v}\frac{\partial v}{\partial u^\star}$ at the deciding probe: only this last term compares the two rows. The table gradient goes to both rows with the weights $s(1-v)$ and $sv$ in the forward. The scalars $\beta$ and $\gamma$ and $\tau$ get the exact gradient. The blend generalises to $n$ rows with softmax weights over the $n$ rows ([readout 2.2](readout.md#22-confidence-read-confidencelut)) and the gradient passes through those weights in the same way.

**Two routes.** In the hard Manifesto read (2.1) the integer address has no derivative, so the input gradient exists only through the neighbour surrogate, a function the forward does not compute. In the confidence read the input gradient is the exact gradient of the computed function through the smooth score and blend weight, and no surrogate is needed; the price is that every row is multiplied by $s_t$ and so the read is not multiplication-free ([readout 5.1](readout.md#51-power-of-two-int8-read-p2_int8) gives the power-of-two form). The profile of $\partial s/\partial m_j$ depends on the score form of [readout 2.3](readout.md#23-confidence-score-forms): in the `bounded` and `bounded_norm` forms, whose score is a product of sigmoids without the margin sum and stays below one, the derivative contains the factor $\sigma(-2m_j)$ only and so vanishes as the margins grow, and a well-separated decision stops producing input gradient, whereas the `margin` form has the extra term $P^{\gamma}$ that tends to one (the mechanisms note calls this a gradient floor). No quantity of size $K$ per input exists in this backward, since one or two rows are read.

**Status:** default read of `lutorch_ex`.

**Where it lives.**

- `lutorch_ex`: `_score`, `_blend_v` and `_forward_impl` in [`confidence.py`][ex-conf] (file added in f4edeac2, 2026-10-03) with the fused read [`_fused_read.py`][ex-fusedread] and the fused backward [`fused_confidence.cu`][ex-fconfcu]; report section "How the gradient reaches the input" ([report][report-tex-grad3]).
- `LightMultiHeadLUT` in [`light_multi_head_lut.py`][frf-light] (file added in 93de410d, 2026-09-06): the blend over $n$ rows is the option `read_top_n=n`; section "LightMHL" of [`lut_mechanisms.tex`][frf-mech-light].
- The read itself is [readout 2.2](readout.md#22-confidence-read-confidencelut).

<details>
<summary>Confidence layer against the surrogate route at equal geometry (16k and 48k)</summary>

Fast test scale, fixed evaluation, validation bits per byte, single seeds. At 16k steps ([LightMHL survey](https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/runs_corrected/LIGHTMHL_SURVEY.md#3-all-lightmhl-runs-at-16k)): the confidence layer with the `margin` score at the geometry of the `FastMultiHeadLut` (48.5M parameters) reached 1.1840 against 1.1870 for the surrogate route ([`exp_n_0194`](https://github.com/anatoli-starostin/spiky/tree/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/runs_corrected/exp_n_0194_light_margin_znorm_nap7_seed1), [`exp_n_0127`](https://github.com/anatoli-starostin/spiky/tree/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/runs_corrected/exp_n_0127_grid_H4d48_nap7_tph128)). At 48k steps, two other models: the confidence layer with 105.1M parameters ([`exp_n_0200`](https://github.com/anatoli-starostin/spiky/tree/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/runs_corrected/exp_n_0200_light_margin_znorm_nap8_tph256_48k_seed1)) reached 1.1345 and the surrogate route with 180.6M parameters ([`exp_n_0171`](https://github.com/anatoli-starostin/spiky/tree/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/runs_corrected/exp_n_0171_0118at48k_compmhl_H4_nap9_tph256_in48_out48)) reached 1.1345 (1.134538 against 1.134511), while the dense model reached 1.1154, so both are 0.019 behind it. One row against two rows, at 16k steps: 1.1696 against 1.1600 ([FFN replacement, section 4](../architectures/transformers/ffn-replacement.md#4-fixed-evaluation-confidence-forms-and-parity-that-does-not-last-september-2026) and [section 5](../architectures/transformers/ffn-replacement.md#5-lutorch_ex-cartridges-on-the-full-corpus-october-2026)).

</details>

#### 4.2 Hard forward with a score surrogate

The forward is the hard read; the backward is the gradient of the score read of 4.1 on detached weights (1.3), $y=y_\text{hard}+f-\mathrm{sg}(f)$ with $f$ the scored read, and the score is divided by a detached mean over the tables of the head:

$$\hat s_t=\frac{s_t}{\mathrm{sg}\big(\bar s+\epsilon\big)}$$

where $\bar s$ is that mean score. Reading: the score is absent from the forward value, so nothing restrains it; with an unbounded score the injected gradient grows with the margins and has no restoring signal (the code comment records a divergence without the rescale), and the detached mean keeps it of order one while preserving the relative confidence of the tables. The value of $f-\mathrm{sg}(f)$ is zero and the table gradient is the plain hard row.

**Status:** a control, kept as evidence that the confidence score is meant to be in the forward; not a candidate.

**Where it lives.**

- `LightMultiHeadLUT`, forward mode `"hard"` ([`_hard_read`][frf-light]); arms 3.3 and 3.4 of the [ablation table][frf-abl4-r33].

### 5. Straight-through estimators for other reads

#### 5.1 Power-of-two int8 read

The deployed read of [readout 5.1](readout.md#51-power-of-two-int8-read-p2_int8) replaces the float read of the confidence read (4.1) by integer arithmetic. In the float read a table with two rows contributes $s(1-v)W[c]+svW[c']$, where $s$ is the score, $v$ the blend weight and $W$ the float rows. In the integer read the two row weights are rounded to $2^{k'}$ and $2^{k'-q}$ with integers $k'$ and $q$, and the rows are rounded to int8 values $\hat W$ with one shared power of two $2^{e}$ as scale. Rounding has zero derivative almost everywhere, so training needs a rule for what to differentiate. The rule is: the forward computes the integer read exactly, and the backward behaves as if the float read had been computed, with two straight-through replacements, one for the rows and one for the row weights.

*Rows.* The float master rows $W$ are kept and trained. The row used in the forward is

$$\hat W2^{e}+\big(W-\mathrm{sg}(W)\big)$$

where $\mathrm{sg}$ is the stop-gradient. The second term is zero in value, so the forward sees the quantised row $\hat W2^{e}$, and its derivative with respect to $W$ is the identity. Every master row receives the gradient as if no rounding happened, also the entries that the rounding clips at the end of the int8 range.

*Row weights.* The weights $s(1-v)$ and $sv$ are what carries the gradient to the score, the blend weight and through them to $\beta$, $\gamma$, $\tau$ and the input. The forward needs the weights $2^{k'}$ and $2^{k'-q}$ instead. Both are obtained by multiplying the float weight by a ratio that is treated as a constant:

$$\hat w_1=s(1-v) \mathrm{sg}\Big(\frac{2^{k'}}{s(1-v)}\Big),\qquad \hat w_2=s v \mathrm{sg}\Big(\frac{2^{k'-q}}{s v}\Big)$$

The value of $\hat w_1$ is exactly $2^{k'}$, since the float weight cancels against the ratio, and its derivative is the one of $s(1-v)$ scaled by that constant ratio. The same holds for $\hat w_2$. With one row per table the weight is $s \mathrm{sg}\big(2^{k'}/s\big)$ in the same way. The integers $k'$ and $q$ are recomputed from $s$, $v$ and the margins at every forward pass and carry no gradient.

*What differs from the float read.* Three things:

- The row scores $r_c$ that enter the gradients of the score and the blend (4.1) are computed on the quantised rows, since those are what the forward read.
- The gradient of a row is the forward weight times $g$, that is $2^{k'}g$ at the addressed row and $2^{k'-q}g$ at the neighbour, and not $s(1-v)g$ and $svg$, because the value is multiplied by the integer weight.
- Some tables are skipped in the forward (the exponent $k'$ is below the allowed window) or have the neighbour dropped (the exponent $q$ is too large). The ratios still use the clamped exponent for them, so the input and the scalars keep a gradient through such a cell and it can come back later. The row gradient of that cell is zero, because its forward weight is zero.

The trained forward equals the deployed integer read up to round-off.

**Status:** current for the quantised confidence read.

**Where it lives.**

- `lutorch_ex`: `ste_tables`, `ste_blend_weights` and `ste_cell_weights` in [`_pow2.py`][ex-pow2-ste] (file added in 5f0e5d8c, 2026-10-03); `quant_bwd` in [`pow2_int8_read.cu`][ex-p2cu-bwd]; module docstring of [`quantised_confidence.py`][ex-quantconf].
- The read and the integers $k'$ and $q$ are in [readout 5.1](readout.md#51-power-of-two-int8-read-p2_int8).

<details>
<summary>Power-of-two int8 read in training: +0.002 to +0.003 against the float read</summary>

Fake-quantised int8 power-of-two read in training against the float read ([fused sweep README](https://github.com/anatoli-starostin/spiky/blob/b1a5a7131732f93595f53bf572299139eb1a0856/experiments/lutorch_ex_fused/README.md#results-on-the-full-170-shard-corpus-48k-steps--these-are-the-comparable-numbers); 48k steps, full corpus, fixed evaluation): +0.0023 with one row (1.0993 against 1.0970) and +0.0033 with two rows (1.0899 against 1.0865) ([FFN replacement, section 5](../architectures/transformers/ffn-replacement.md#5-lutorch_ex-cartridges-on-the-full-corpus-october-2026)).

</details>

#### 5.2 Bit and vote tables

The tables of the vote line ([readout 6](readout.md#6-votes-and-dominance)) do not store real numbers. A row holds votes, and each vote is either a single bit $\pm1$ (bit tables) or a bounded value in $[-\frac12,\frac12]$ (vote tables). In the forward the vote is a hard function of a trained float number, the latent $\ell$: the bit is $b=\mathrm{sign}(\ell)$, and the vote of a vote table is $\pm\frac12$ in the `ste` mode. Such a function has derivative zero almost everywhere. The straight-through rule used here replaces it, in the backward only, by the derivative of a smooth function of the same shape, which is large when the latent is near zero and small when it is far from zero. The page gives the form of the bit gradient first and then of the votes.

*Bits.* The forward uses the bits $b$ directly. The backward scatters the output gradient $g$ into the latents of the entries that were read, multiplied by the factor

$$\frac{T}{(T+\lvert\ell\rvert)^2}=\frac{d}{d\ell}\frac{\ell}{T+\lvert\ell\rvert}$$

where $T$ is the gate temperature. The right-hand side shows where the factor comes from: it is the derivative of the rational soft sign $\ell/(T+\lvert\ell\rvert)$, a smooth stand-in for $\mathrm{sign}(\ell)$ (the same shape as the softsign of 2.2). So a latent far from zero is nearly frozen, since its bit is unlikely to flip, and a latent near zero moves freely. Only the entries read in the forward receive gradient, and the packed bits carry none.

*Votes.* The forward is $\pm\frac12$, and the stand-in is the rational vote $\frac{w}{2(T+\lvert w\rvert)}$ of the raw weight $w$, so the backward factor is its derivative $\frac{T}{2(T+\lvert w\rvert)^2}$. The quantisation levels of a vote (`vote_quant_levels`) use a straight-through gradient too.

*Input side.* The vote read is a hard read, so its input gradient comes from a neighbour surrogate (2.1), which needs the votes of the addressed row and of the neighbour row. By default these are the discrete $\pm1$ votes. An opt-in variant takes them from the continuous latent instead; its docstring warns that it can under-train early, when the latents are small.

**Status:** experimental; the vote line was set aside ([readout 6](readout.md#6-votes-and-dominance)).

**Where it lives.** Code at `e85a5ce`.

- Bit tables: [`_VoteGatherFn`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/bit_permutation_lut.py#L260-L329) in [`bit_permutation_lut.py`][cal-bit] (docstring: "STE + rational Jacobian gate"; the gate temperature is `ste_gate_temperature`) and [`multi_bit_permutation_lut.py`][cal-multibit].
- Votes: mode `ste` of [`PermutationalLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/permutational_lut.py#L294-L669) in [`permutational_lut.py`][cal-perm].
- Forms of the votes: [readout 6.2](readout.md#62-signed-votes-permutationallut-bitpermutationlut).

### 6. Early era: the `lut_fused` code

A word on *synapses*, which appear in this section. `lut_fused` inherits its vocabulary from the `spnet` implementation, which is much closer to biology and was built around sparse connections: a table cell is a *lookup neuron* and its row is a list of synapses to output neurons, each synapse with its own weight, learning rate and weight range. The later code (`lutorch`, `lutorch_ex`) is much closer to classic deep learning, with a table as a dense weight tensor and a row as a vector of the output width. When a synapse is mentioned below, read it as a table weight with extra per-connection bookkeeping.

#### 6.1 Minimum-delta input gradient and token-pair competition

`lut_fused` implements the neighbour surrogate with the rational uncertainty function. The forward stores per table the address $c$ and the index $j^\star$ of the probe with the smallest absolute margin and its signed margin $u_{j^\star}$ and nothing that depends on the number of rows $K$ in size. The backward computes the number $G=\langle g,W[c]-W[c']\rangle$ and this is $r_c-r_{c'}$ and then adds

$$G\cdot\frac{0.5 \mathrm{sgn}(u_{j^\star})}{(1+\lvert u_{j^\star}\rvert)^2}$$

to the gradient at anchor $a_{j^\star}$ and subtracts it at anchor $b_{j^\star}$ (ties count as sign minus). Reading: this is the rational surrogate of 2.1 and 2.2, written with the negated derivative $-U'$ of the uncertainty function.

In the concatenating token-pair (attention-like) layer an address is built by concatenating the bits of a query-side lookup, a key-side lookup and a positional lookup. The neighbour is then taken in one source only: the input gradient goes to the query side or to the key side, whichever has the smaller absolute minimum margin, and in addition to the positional encoding if its margin is the smallest of all three. The table gradient goes to the addressed row of the concatenated index. The layer also has a mixed-source variant (called the product variant in the code; the name is historical and no values are multiplied) that builds no concatenation: one address and one smallest margin are taken over all three sources together, so the neighbour gradient goes to whichever source holds the smallest margin. Its backward (experimental) flips that single global smallest-margin probe over the mixed-source anchors, and each anchor's gradient goes to the source it came from, so there is no query-versus-key competition and no separate positional rule. Its forward stores no margins and the backward recomputes them. The forward of both variants, with the address layouts and the causal pairs $i<j$, is on [readout 1.3](readout.md#13-the-forward-of-lut_fused).

**Status:** origin and reference; the package is deprecated.

**Where it lives.** `lut_fused`.

- Ground-truth python [`gt_lut_transformer.py`][lf-gt] (`LUT.backward` and the backward of the attention class; file added in cf45631e, 2025-12-04).
- [`LUTLayer.py`][lf-layer] (file added in 6364a7e9, 2025-11-14) with the CUDA kernels `propagate_through_detectors_*` (input gradient) and `gather_w_gradients_*` (weight gradient), run on two streams ([`LUT_RUNTIME_DATA_FLOW.md`][lf-flow]).
- The mixed-source (product) variant: `forward_step_product` in [`lut_runtime.cu`][lf-rt-product] with the logic in [`lut_runtime_kernels_logic.proto`][lf-rtlogic-product]; its backward is the experimental `backward_backprop_product` path.

#### 6.2 Gradient policies

This part of `lut_fused` is unconventional. At that time we did not follow the PyTorch guidelines for gradients and optimisers: the layer decides itself how the weight gradient is delivered, and it does not always go through the standard autograd path to an optimiser. We tried to benefit from the sparsity of the gradient (one batch touches few rows), but this matters much less when training with big batches, where most rows are touched. The policies below describe what the code does, not a pattern to copy.

The weight gradient of a `lut_fused` layer is delivered in one of three forms:

| policy | what it computes | notes |
| --- | --- | --- |
| Dense | the weight gradient as a dense tensor, for an autograd optimiser | the default |
| Sparse | only the touched entries, converted from the dense buffer on the device and returned as a sparse tensor | |
| Internal | the layer updates its weights in place during the backward, as plain SGD without momentum | needs a learning-rate hook `hook(weights) -> float`; cannot be combined with the `INTEGERS` mode of the forward, which accumulates the output in int32 fixed point ([readout 1.3](readout.md#13-the-forward-of-lut_fused)) |

An option divides the gradient by its largest absolute value before the update (Dense and Sparse only). Each synapse type has a learning rate between $0$ and $1$, a minimum and maximum weight, an initial weight and initial noise, so weights are clipped to a range per synapse type and some synapses can learn more slowly. The weight-gradient kernels already multiply the gradient they write by the learning rate of the synapse type (one by default), so the Dense and Sparse forms hold this scaled gradient and not the plain output gradient, and the Internal form applies the negated external rate on top. The `lutorch` and `lutorch_ex` modules return an ordinary dense gradient to the optimiser and have no such policy.

**Status:** deprecated; relevant if in-place SGD or per-synapse learning rates are wanted again.

**Where it lives.** `lut_fused`, [`LUTLayer.py`][lf-layer]; the programming guide, section "Gradient policies" ([guide][lf-pg-policies]).

| name | meaning |
| --- | --- |
| [`GradientType`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/LUTLayer.py#L18-L22) | the form of the delivered gradient: Dense, Sparse or Internal |
| [`GradientPolicy`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/LUTLayer.py#L26-L33) | the policy object: a [`GradientType`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/LUTLayer.py#L18-L22) and the flag `normalized` |
| `normalized` | the division by the largest absolute value |
| [`SynapseMeta`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/LUTLayer.py#L37-L46) | per synapse type: learning rate, minimum and maximum weight, initial weight and noise |
| [`_process_gradients`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/LUTLayer.py#L581-L623) | the step that applies the policy |

## Open ideas

- The ranking-aligned (Walsh-basis) table update moved to [Special LUT optimisers 1.1](../optimisation/special-optimisers.md#11-walsh-basis-ranking-aligned-table-update). *Untested.*

## Names in the sources

"Ablation Gen 1", "ablation Gen 2" and "ablation Gen 3" below are the labels that the [ablation table][frf-abl4-tableA2] itself gives to its math (the Manifesto rule, the softsign and all-rows math, the confidence score); they do not label code versions.

| idea | names |
| --- | --- |
| neighbour surrogate with a rational uncertainty function | "uncertainty function", `INVERSE_L1`, `up()` (`lut_fused`), "two-alternatives backward", "Manifesto hard", "ablation Gen 1", `n_alternatives=1` (`lutorch`), "hard forward, soft backward" (report) |
| softsign neighbour surrogate | softsign cartridge (`lutorch_ex`), "ablation Gen 2" restricted to two cells, `backward_topk=1`, `hybrid_smooth_n_alt=1` ([`FastMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L754-L1162)), restricted softmax over the two cells (ablation table), row 2.3 of the ablation table |
| softmax surrogate over all rows | soft surrogate, `dense_K` ([`FastMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L754-L1162)), "always soft" backward, `backward_mode="soft"` ([`TinyMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L3666-L4389)), "all-rows backward" |
| restricted softmax surrogate | `ball`, `ball_gather`, `backward_topk` ([`FastMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L754-L1162)), `soft_topk`, "Hamming-1 ball backward" |
| gradient only through the score | Light, "no STE", "scored-only" ([`ConfidenceLUT`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/confidence.py#L55-L114)), `forward_mode="hard"` with a score in the backward only ([`LightMHL`](https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/light_multi_head_lut.py)), "ablation Gen 3" |
| value and gradient decoupling | "decoupled straight-through", `y_c + (surr - surr.detach())`, `sel_hard - sel_soft.detach() + sel_soft`, `(hard - soft).detach() + soft` |
| STE gate of a bit latent | "STE + rational Jacobian gate", `ste_gate_temperature`, "rational soft sign Jacobian" |

[report-tex-grad3]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/lutorch_ex/lutorch_ex_report.tex#L642-L678
[report-tex-grad2]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/lutorch_ex/lutorch_ex_report.tex#L642-L678
[report-tex-grad]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/lutorch_ex/lutorch_ex_report.tex#L642-L678
[ex-manbase]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/manifesto_base.py
[ex-manbase-unc]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/manifesto_base.py#L229-L241
[ex-manhard]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/manifesto_hard.py
[ex-mansoft]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/manifesto_soft.py
[ex-fmsoft]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/fused_manifesto_soft.py
[ex-uncert]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/uncertainty.py
[ex-conf]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/confidence.py
[ex-fusedread]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/_fused_read.py
[ex-pow2]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/_pow2.py
[ex-pow2-ste]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/_pow2.py#L134-L192
[ex-quantconf]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/quantised_confidence.py
[ex-softsigncu]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/native/lutorch/softsign_surrogate_grad.cu
[ex-singlecu]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/native/lutorch/single_anchor_input_grad.cu
[ex-fmcu]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/csrc/fused_manifesto.cu
[ex-fconfcu]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/csrc/fused_confidence.cu
[ex-p2cu-bwd]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/csrc/pow2_int8_read.cu#L862
[lt-apl-bwd5]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/anchor_pairs_lookup.py#L529-L621
[lt-apl-bwd4]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/anchor_pairs_lookup.py#L529-L621
[lt-apl-bwd3]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/anchor_pairs_lookup.py#L529-L621
[lt-apl-bwd2]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/anchor_pairs_lookup.py#L529-L621
[lt-apl-bwd]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/anchor_pairs_lookup.py#L529-L621
[lt-cu-bwd2]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/native/lutorch/lutorch.cu#L445-L491
[lt-cu-bwd]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/native/lutorch/lutorch.cu#L445-L491
[lt-lproj-cls5]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/l_projection.py#L175-L349
[lt-lproj-cls4]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/l_projection.py#L175-L349
[lt-lproj-cls3]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/l_projection.py#L175-L349
[lt-lproj-cls2]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/l_projection.py#L175-L349
[lt-lproj-cls]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/l_projection.py#L175-L349
[lt-fast-soft2]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L225-L314
[lt-fast-hybrid2]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L511-L749
[lt-fast-soft]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L225-L314
[lt-fast-hybrid]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L511-L749
[lt-hyp-fullsoft2]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/hyperplane_multi_head_lut.py#L876-L891
[lt-hyp-fullsoft]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/hyperplane_multi_head_lut.py#L876-L891
[lf-layer]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/LUTLayer.py
[lf-rt-product]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/native/spiky/lut_fused/lut_runtime.cu#L789-L966
[lf-rt-bwd]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/native/spiky/lut_fused/lut_runtime.cu#L205-L369
[lf-rtlogic]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/native/spiky/lut_fused/lut_runtime_kernels_logic.proto
[lf-rtlogic-product]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/native/spiky/lut_fused/lut_runtime_kernels_logic.proto#L2017-L2395
[lf-rtlogic-prop]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/native/spiky/lut_fused/lut_runtime_kernels_logic.proto#L821-L867
[lf-gt]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/tests/gt_lut_transformer.py
[lf-gt-up]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/tests/gt_lut_transformer.py#L28-L30
[lf-gt-bwd2]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/tests/gt_lut_transformer.py#L156-L190
[lf-gt-bwd]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/tests/gt_lut_transformer.py#L156-L190
[lf-flow]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/spiky/cuda/LUT_RUNTIME_DATA_FLOW.md
[lf-pg-policies]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/spiky/programming_guide.md#gradient-policies
[cal-fast]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/fast_multi_head_lut.py
[cal-fast-soft]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/fast_multi_head_lut.py#L98-L192
[cal-fast-hybrid]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/fast_multi_head_lut.py#L497-L738
[cal-fast-ball]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/fast_multi_head_lut.py#L200-L398
[cal-tiny]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py
[cal-tiny-matmul2]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L4414-L4594
[cal-tiny-topk2]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L1348-L1445
[cal-tiny-soft2]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L1136-L1237
[cal-tiny-hs]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L1805-L1885
[cal-tiny-alt4]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L600-L713
[cal-tiny-alt3]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L600-L713
[cal-tiny-matmul]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L4414-L4594
[cal-tiny-topk]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L1348-L1445
[cal-tiny-soft]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L1136-L1237
[cal-tiny-alt2]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L600-L713
[cal-tiny-alt]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L600-L713
[cal-soft]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/soft_multi_head_lut.py
[cal-bit]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/bit_permutation_lut.py
[cal-multibit]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/multi_bit_permutation_lut.py
[cal-perm]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/permutational_lut.py
[cal-hybridtex]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/docs/hybrid_smooth_math.tex
[cal-gradnotes]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/claude-memory/secondary/papers/Permutations-is-all-you-need/paper/lut_gradient_notes.tex
[cal-lutgpt-soft]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/claude-memory/secondary/papers/Permutations-is-all-you-need/paper/lutgpt_research_report.tex#L190-L257
[cal-lutgpt-asym]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/claude-memory/secondary/papers/Permutations-is-all-you-need/paper/lutgpt_research_report.tex#L258-L264
[frf-fast-topk2]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/fast_multi_head_lut.py#L562-L757
[frf-fast-topk]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/fast_multi_head_lut.py#L562-L669
[frf-light]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/light_multi_head_lut.py
[frf-light-hard]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/light_multi_head_lut.py#L1093-L1123
[frf-mech-light]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/doc/lutorch/lut_mechanisms.tex#L128-L188
[frf-mech-forms]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/doc/lutorch/lut_mechanisms.tex#L189-L257
[frf-abl4-tableA2]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/doc/research/lut_ablation/lut_ablation_table_v4.tex#L52-L113
[frf-abl4-r33]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/doc/research/lut_ablation/lut_ablation_table_v4.tex#L106-L109
[frf-abl4-r2324]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/doc/research/lut_ablation/lut_ablation_table_v4.tex#L91-L97
[frf-abl4-r21]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/doc/research/lut_ablation/lut_ablation_table_v4.tex#L81-L86
[frf-abl4-tableA]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/doc/research/lut_ablation/lut_ablation_table_v4.tex#L52-L113
[lif-readme-st]: https://github.com/anatoli-starostin/spiky/blob/ccd0616c84e0efbe71c2177368f68b48512fc456/experiments/lif_detectors_mhl/README.md#straight-through-hard-addressing-mirrors-the-teacher
