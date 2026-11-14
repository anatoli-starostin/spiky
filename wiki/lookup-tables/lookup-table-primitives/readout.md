# Readout mechanics

How the output of a layer is formed from the rows that the tables read. The address of each read is on [Hashing mechanics](hashing.md); gradients through a read are on [Gradient backpropagation](backpropagation.md); results are on the Architectures pages. This page only describes readout mechanisms.

## Contents

| Section | Covers |
| --- | --- |
| [1. The read stage](#1-the-read-stage) | hard lookup-and-accumulate (1.1), one form for all reads (1.2), the forward of [`lut_fused`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lut_fused) (1.3) |
| [2. Weighted reads](#2-weighted-reads) | Manifesto blend (2.1), confidence read (2.2), confidence-score forms (2.3), input-conditioned cells (2.4) |
| [3. Other aggregations over tables](#3-other-aggregations-over-tables) | log-sum-exp (3.1), sparse-output rows (3.2) |
| [4. Output projection](#4-output-projection) | dense map back to model width |
| [5. Quantised reads](#5-quantised-reads) | power-of-two int8 read (5.1), precision of stored rows (5.2) |
| [6. Votes and dominance](#6-votes-and-dominance) | dominance votes with Borda sum (6.1), signed votes (6.2), rank-coded decode (6.3) |
| [Open ideas](#open-ideas) |  |

## Idea / model

### 1. The read stage

#### 1.1 Hard lookup-and-accumulate

*Source: [`lutorch_ex` report][report-tex-heads], section "Heads and the multi-head LUT"; code [`manifesto_hard.py`][ex-manhard], [`lut_base.py`][ex-lutbase], [`lut_spec.py`][ex-spec].*

From the hashing page: a table has $K=2^{n_\text{ap}}$ rows, each a vector $W_t[c]\in\mathbb{R}^{d_\text{out}}$ of the output width. The hash gives it the address $c_t$ and the margins $u_j$ of its $n_\text{ap}$ probes. Write $m_j=\lvert u_j\rvert$ for the absolute margins and $u^\star=\min_j m_j$ for the deciding margin; the neighbour address $c'_t$ is the address with the deciding bit flipped. A head sums the reads of its $\mathrm{tph}$ tables.

The basic read, **lookup-and-accumulate** (LAC), returns the addressed row of every table and adds them:

$$y=\sum_{t=1}^{\mathrm{tph}}W_t[c_t]\in\mathbb{R}^{d_\text{out}}$$

Reading: one row per table, no weights; $y$ is the output of one head. It needs only row fetches and additions, no multiplication, and its cost does not depend on the number of rows.

**Head patterns** ([`LUTSpec`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/lut_spec.py#L39-L102) with $h_\text{in}$ input heads and $h_\text{out}$ output heads, $\max(h_\text{in},h_\text{out})$ table groups): for $h_\text{in}=h_\text{out}$ each head reads its own input slice and the head outputs stay separate (concatenated by the wrapper, [section 4](#4-output-projection) below); for $h_\text{in}=1$ one input is shared by all groups; for $h_\text{out}=1$ the group outputs are summed into one head (fan-in). Any other combination is refused.

Only the addressed rows are read, so in a hard read only they receive weight gradient; the neighbour $c'_t$ enters a hard read only through the gradient. The reads of 1.2 and the following sections are weighted and use the margins and $c'_t$ in the value.

**Older code.** In [`lutorch`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch) the read is split in two: [`LProjection`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/l_projection.py#L175-L349) ([`l_projection.py`][lt-lproj]) returns one row per table, and [`MultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/multi_head_lut.py#L15-L214) ([`multi_head_lut.py`][lt-mhl]) sums over the tables of a head. In `lut_fused` ([`LUTLayer.py`][lf-layer]) a table cell is a *lookup neuron* with sparse (or full) synapses to the output neurons, and the output is the synaptic sum in float32 or int32; the synapse lists are grown by the synapse-growth engine. The forward of `lut_fused` is described in [1.3](#13-the-forward-of-lut_fused). The synapse vocabulary is inherited from the `spnet` implementation, which is closer to biology and built around sparse connections; the later code works with dense weight tensors (see [backpropagation 6](backpropagation.md#6-early-era-the-lut_fused-code)).

<details>
<summary>Hard read against the weighted reads on one geometry (48k, fixed evaluation)</summary>

Hard read against the weighted reads on one geometry, fast test scale (context 512, vocabulary 32768), full corpus, 48k steps, fixed evaluation, single seeds ([fused sweep README](https://github.com/anatoli-starostin/spiky/blob/b1a5a7131732f93595f53bf572299139eb1a0856/experiments/lutorch_ex_fused/README.md#results-on-the-full-170-shard-corpus-48k-steps--these-are-the-comparable-numbers)): the hard Manifesto read reached 1.1324 with the rational weight and 1.1378 with softsign, the soft Manifesto read 1.1186 and 1.1095, the confidence-weighted read of one row 1.0970 and of two rows 1.0865, against 1.0894 for the dense feed-forward. At 16k steps on 4 shards the one-row hard read of the `FastMultiHeadLut` reached 1.1810, its soft Manifesto read (`hybrid_smooth`, which blends two rows) 1.1694 and the confidence layer reading one row 1.1696 (all without the total-variation penalty; the dense model reached 1.1651) ([FFN replacement, section 4](../architectures/transformers/ffn-replacement.md#4-fixed-evaluation-confidence-forms-and-parity-that-does-not-last-september-2026) and [section 5](../architectures/transformers/ffn-replacement.md#5-lutorch_ex-cartridges-on-the-full-corpus-october-2026)).

</details>

#### 1.2 One form for all reads

*Source: [`lutorch_ex` README][ex-readme-cartridges], section "Cartridges"; [`lut_base.py`][ex-lutbase].*

In `lutorch_ex` a read module (a *cartridge* in the code, a *lookup variant* in the report) differs from the others only in the coefficients of two rows per table:

$$y=\sum_{t=1}^{\mathrm{tph}}\Big(a_t W_t[c_t]+b_t W_t[c'_t]\Big)$$

Reading: $a_t$ weighs the addressed row and $b_t$ the neighbour row; the sections below define the symbols in the table.

| read | $a_t$ | $b_t$ | section |
| --- | --- | --- | --- |
| hard | $1$ | $0$ | 1.1 |
| Manifesto blend | $1-U_t$ | $U_t$ | 2.1 below |
| confidence, one row | $s_t$ | $0$ | 2.2 below |
| confidence, two rows | $s_t(1-v_t)$ | $s_t v_t$ | 2.2 below |
| power-of-two int8 | $2^{k'}$ | $2^{k'-q}$ | 5.1 below |

The fused CUDA twins of these reads ([`fused_manifesto_hard.py`][ex-fhard], [`fused_manifesto_soft.py`][ex-fsoft], [`fused_confidence.py`][ex-fconf], [`fused_quantised_confidence.py`][ex-fquant]) compute the same function.

#### 1.3 The forward of `lut_fused`

*Source: [`lut_runtime.cu`][lf-rt] (`forward_step_concat`, `forward_step_product`) with the kernel logic in [`lut_runtime_kernels_logic.proto`][lf-rtlogic], at the pinned commit. The backward of the same kernels is [backpropagation 6.1](backpropagation.md#61-minimum-delta-input-gradient-and-token-pair-competition).*

The forward of `lut_fused` is a hard read of 1.1 with its own address rules; the address is built from the input by the comparisons of [hashing 1.1](hashing.md#11-the-basic-construction-manifesto-math) in the following way.

**Non-sequence layer.** For each table (a *detector*) and each probe $j=0,\dots,n_\text{ap}-1$ with anchors $a_j,b_j$ the bit is set when $z[a_j]$ exceeds $z[b_j]$ by a margin. With $\delta_j=z[a_j]-z[b_j]$ and the kernel parameter `cmp_eps` $=\varepsilon$ (`ROUNDED_COMPARISON_GREATER`), the bit is

$$\text{bit}_j=1\iff \delta_j>0\ \text{and}\ \delta_j\ge\varepsilon$$

Reading: for $\varepsilon=0$ this is the plain comparison $\delta_j>0$; for $\varepsilon>0$ a difference smaller than $\varepsilon$ in absolute value gives bit $0$. The address is packed **LSB-first**, $c=\sum_j\text{bit}_j 2^{j}$ (probe $0$ is the lowest bit; compare the MSB-first packing of `lutorch_ex` in the hashing page). The same loop keeps the *deciding probe*: the first probe with the minimum $\lvert\delta_j\rvert$ (strict comparison, so the earliest wins a tie) and its signed $\delta_j$ are stored for the backward. An anchor slot that is unset (the maximum index value in the pair list) is skipped: its bit stays $0$ and it never takes part in the margin. The read is the hard sum of 1.1 only: nothing but the addressed row is read in the forward and the stored margin is used by the backward alone. The row is added to the output through the synapses, either to all outputs (fully connected: direct indexing) or to the sparse list of outputs of the lookup neuron (sparse path, [3.2 below](#32-sparse-output-rows)). With the `INTEGERS` mode the output is accumulated in int32 fixed point: values are converted with the scale $\mathtt{0x000FFFFF}\times$`int_rescaler` and converted back to floats at the end.

**Concatenating token-pair layer.** For a sequence of length $T$ each pair of tokens $i<j$ gives one address per detector, $c=Q 2^{n_\text{ap}+p_d}+K 2^{p_d}+PE$, where $Q$ is the address of $n_\text{ap}$ bits of the lookup on the later token $j$, $K$ the same on the earlier token $i$, $PE$ the $p_d$ sign bits of a positional vector indexed by the distance $j-i-1$ and the detector (here $p_d$ is the positional dimension, and the symbols $Q$, $K$ are the query- and key-side addresses). The row is added to the output of the later token $j$, which sums over all $i<j$: timestep $0$ has no earlier token and so no output.

**Mixed-source variant (`product` in the code).** The kernel is called `forward_step_product`. The name "product" is historical and nothing is multiplied; this page calls the variant by what it does. No concatenation: each probe takes its anchors from three sources, the first input at token $i$, the second input at token $j$ and the positional embedding, and compares them as above. The sources are laid out in one of two ways, *sequential* $`[\text{inp}_1\mid\text{inp}_2\mid\text{pos}]`$ (the anchor index falls in one of three consecutive ranges) or *sliced* $`[i_{1,k},i_{2,k},p_k]`$ (the anchor index $a$ selects source $a\bmod 3$ and slice $\lfloor a/3\rfloor$). One address is formed from all probes and so one smallest margin is taken over all three sources together. The forward stores no margins; the backward recomputes the address and the smallest margin. The pairs are causal, $i<j$, as above.

**Why these two modes exist.** In the early versions we tried to implement the attention primitive following the Spiking Manifesto paper, and the concatenating token-pair layer and the product variant are the modes used for that. Unfortunately such approaches scale poorly on a GPU.

### 2. Weighted reads

#### 2.1 Manifesto blend (soft read)

*Source: [`manifesto_soft.py`][ex-mansoft], [`manifesto_base.py`][ex-manbase] (`_uncertainty`), [`uncertainty.py`][ex-uncert]. Older: `forward_mode="hybrid_smooth"` of [`FastMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L754-L1162) ([`fast_multi_head_lut.py`][lt-fast-hybrid]), derivation in [`hybrid_smooth_math.tex`][cal-hybridtex].*

The neighbour row joins the read with a weight called the **uncertainty** $U_t$ of the deciding margin:

$$y=\sum_{t=1}^{\mathrm{tph}}\Big((1-U_t)W_t[c_t]+U_t W_t[c'_t]\Big)$$

Reading: the weight $U_t$ lies between zero and one half, is largest for a small deciding margin, and is exactly one half at zero margin; at a bit flip the two rows swap roles with equal weights, so the value is continuous in the input. Two uncertainty functions exist. `rational` has no parameters:

$$U=\frac{0.5}{1+u^\star}$$

`softsign` squashes the margin first and adds two learned temperatures $T_\text{soft}$ and $T_\text{sel}$ (one pair per layer, stored as logarithms), with $\sigma$ the logistic sigmoid:

$$\rho=\frac{u^\star}{T_\text{soft}+u^\star},\qquad U=\sigma\Big(-\frac{2\rho}{T_\text{sel}}\Big)$$

The softsign weight is the exact softmax over the two rows $c$ and $c'$ with a score gap of $2\rho$ between them at the temperature $T_\text{sel}$ above, and `hybrid_smooth` of `FastMultiHeadLut` is the same formula.

The hard Manifesto module returns the hard value of 1.1 and uses the same $U$ only in its backward (see [Gradient backpropagation](backpropagation.md#21-the-neighbour-surrogate-in-general-form)).

**More than one neighbour.** Flipping each of the $k$ least confident bits gives $k$ neighbours, one per flipped bit. `LProjection` in smooth mode, with $\delta_i$ the margin of flipped probe $i$ and $U_i=0.5/(1+\lvert\delta_i\rvert)$ (`INVERSE_QUADRATIC` squares $\delta_i$ in the denominator), reads

$$y=\Big(1-\frac1k\sum_{i=1}^{k}U_i\Big)W[c]+\sum_{i=1}^{k}\frac{U_i}{k}W[c^{(i)}]$$

Reading: the addressed row keeps the remaining mass; $k=1$ is the rational blend. [`TinyMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L3666-L4389) ([`tiny_multi_head_lut.py`][cal-tiny-hs], `hybrid_smooth_n_alt`) instead gives the $k+1$ rows softmax weights over their score gaps, with $k$ up to $n_\text{ap}$ (the whole Hamming-1 ball); [`LightMultiHeadLUT`](https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/light_multi_head_lut.py) does the same inside the confidence read (2.2).

**Status:** superseded in `lutorch_ex` by the confidence read (2.2).

<details>
<summary>Soft against hard forward, rational against softsign, number of alternatives (ten-arm sweep)</summary>

Soft against hard forward, rational against softsign and the number of alternatives, from the ten-arm sweep ([fused sweep README](https://github.com/anatoli-starostin/spiky/blob/b1a5a7131732f93595f53bf572299139eb1a0856/experiments/lutorch_ex_fused/README.md#results-on-the-full-170-shard-corpus-48k-steps--these-are-the-comparable-numbers); fast test scale, 48k steps, full corpus, fixed evaluation, single seeds): under the hard Manifesto read the rational weight was better (1.1324 against 1.1378), under the soft read softsign was (1.1095 against 1.1186), and soft beat hard for both weights. The two-row confidence read beat the one-row read (1.0865 against 1.0970). The number of alternatives of the smooth read (1 to 6) hardly mattered in a tiny BPE model: 1.5865 with 2 alternatives falling steadily to 1.5741 with 6 ([LUT attention](../architectures/transformers/lut-attention.md#what-was-found-with-lut-sdpa-bpe-context-512-vocabulary-32768), forward and backward modes). At the 16k LUT SDPA scale the soft-to-hard cost is 0.065 to 0.073 in the June report ([LUT attention](../architectures/transformers/lut-attention.md#what-was-found-with-lut-sdpa-bpe-context-512-vocabulary-32768)).

</details>

#### 2.2 Confidence read ([`ConfidenceLUT`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/confidence.py#L55-L114))

*Source: [`confidence.py`][ex-conf], [`confidence_base.py`][ex-confbase]; [`lutorch_ex` report][report-tex-variants], section "Lookup variants"; LookupFFN reference [`lookup_ffn.py`][lffn-py].*

Instead of blending rows, a smooth **confidence score** $s_t$ of all margins scales the whole table. With learned positive scalars $\beta$ and $\gamma$ (one pair per layer, stored as logarithms):

$$s_t=\Big(\sum_{j}m_j\Big)\exp\Big(\gamma\sum_{j}\log\sigma(\beta m_j)\Big)$$

Reading: large only if every bit is far from flipping; each margin near zero multiplies the second factor by a factor of about $2^{-\gamma}$ per margin. At $\beta=2$ and $\gamma=1$ this is the score of LookupFFN ([Zeng et al.][lffn-pmlr]; see [hashing 6.2](hashing.md#62-bh4-coordinate-sign-addressing-lookupffn-style)). With one row per table (`read_top_n=1`):

$$y=\sum_{t=1}^{\mathrm{tph}}s_t W_t[c_t]$$

With two rows (`read_top_n=2`) the neighbour joins with the weight $v_t=\sigma(-2u^\star/\tau)$ where $\tau$ is a learned temperature:

$$y=\sum_{t=1}^{\mathrm{tph}}s_t\Big((1-v_t)W_t[c_t]+v_t W_t[c'_t]\Big)$$

Reading: the score gates what the table contributes, the blend inside decides which row it reads; a table that is unsure of its address contributes less. For single-anchor probes the margin is $m_j=\lvert z[a_j]\rvert$ ([hashing, section 2](hashing.md#2-single-anchor-hashing)). The address is an integer, so the input gets gradient only through the score $s_t$ and the weight $v_t$ alone.

`LightMultiHeadLUT` ([`light_multi_head_lut.py`][frf-light]) generalises the blend to $n$ rows (`read_top_n=n`): the addressed row and the $n-1$ neighbours at the smallest margins, with weights $\pi$ over the $n$ rows:

$$\pi=\mathrm{softmax}\Big(0,-\frac{2m_{j_1}}{\tau},\dots,-\frac{2m_{j_{n-1}}}{\tau}\Big)$$

so $n=2$ is the formula above and $n=1$ the one-row read.

**Status:** default read of `lutorch_ex`.

<details>
<summary>Confidence layer: one row against two rows, three-seed spread, and the dense feed-forward</summary>

One row against two rows, the score alone against the blend, and the dense feed-forward: the confidence layer reading two rows reached 1.0865 and reading one row 1.0970 (three seeds: mean 1.0961, spread 0.0013), against 1.0894 for the dense feed-forward ([fused sweep README](https://github.com/anatoli-starostin/spiky/blob/b1a5a7131732f93595f53bf572299139eb1a0856/experiments/lutorch_ex_fused/README.md#results-on-the-full-170-shard-corpus-48k-steps--these-are-the-comparable-numbers), fast test scale, 48k steps, 170 shards, fixed evaluation). On 4 shards at 16k steps the same two readings are 1.1600 and 1.1696 against 1.1651 dense ([lut_ablation README](https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/lut_ablation/README.md#the-21-runs)), and at d24 a two-row fork converged to the same loss at 18% more time per step ([FFN replacement, section 4](../architectures/transformers/ffn-replacement.md#4-fixed-evaluation-confidence-forms-and-parity-that-does-not-last-september-2026) to [section 6](../architectures/transformers/ffn-replacement.md#6-gpt-2-scale-d24-nanochat-october-to-november-2026)).

</details>

#### 2.3 Confidence-score forms

*Source: [`light_multi_head_lut.py`][frf-light], [`fast_multi_head_lut.py`][frf-fast] ([`_confidence_score`](https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/fast_multi_head_lut.py#L213-L252)); math in [`lut_mechanisms.tex`][frf-mech-forms]; `soft_winner` in [`tiny_multi_head_lut.py`][cal-tiny-winner].*

`LightMultiHeadLUT` has a family of scores $s_t$ in the read $y=\sum_t s_t W_t[c_t]$ of one head. With the shorthand $P=\prod_j\sigma(2m_j)$ where every sum and product runs over the probes of the table:

| form | score $s_t$ |
| --- | --- |
| `bounded` | $P$ |
| `bounded_norm` | $P^{1/n_\text{ap}}$ |
| `margin` | $\big(\sum_j m_j\big)P$ |
| `min_margin` | $\big(\min_j m_j\big)P$ |
| `tanh_margin` | $\big(\sum_j m_j\big)\prod_j\tanh(2m_j)$ |
| `sharp_margin` | $\big(\sum_j m_j\big)P^{\gamma}$ with a fixed $\gamma>1$ |
| `learned_margin` | $e^{g}\big(\sum_j m_j\big)\prod_j\sigma(\beta m_j)^{\gamma}$ with learned $g$ and $\beta$ and $\gamma$ |

Reading: `margin` is the LookupFFN score; `learned_margin` with the log-gain $g$ frozen at $0$ is the score of 2.2, which is how `lutorch_ex` has it. `min_margin` and `tanh_margin` vanish at every cell boundary, so the read is continuous across a bit flip; the other forms jump. A constant `confidence_gain` rescales any form (it could be absorbed in the output projection of [section 4](#4-output-projection)). Products are computed in the log domain. In `LightMultiHeadLUT`, `forward_mode="hard"` returns the plain row sum of 1.1 and uses the score only in the backward.

`TinyMultiHeadLut` has the older `soft_winner` read. Let $p_j=u_j/(T_\text{soft}+\lvert u_j\rvert)$ be the squashed margins and $b_{k,j}=\pm1$ the bit pattern of the row with number $k$ in the table. The code score of row $k$ is $\sum_j p_j b_{k,j}$ and the hard row is weighted by its softmax mass over all $K$ rows:

$$y=\sum_{t=1}^{\mathrm{tph}}\mathrm{softmax}_k\Big(\frac{\sum_j p_j b_{k,j}}{T_\text{sel}}\Big)\Big[c_t\Big]W_t[c_t]$$

**Status:** `margin` adopted. `bounded`, `min_margin`, `tanh_margin` and `soft_winner` are dead ends; `sharp_margin` and `learned_margin` are variants of `margin`.

<details>
<summary>Confidence-score forms: margin against bounded_norm and the collapse of the product form</summary>

The comparison of forms, fixed evaluation (batch 48 by 100 steps, 12 leading rows skipped), single seeds ([LightMHL survey](https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/runs_corrected/LIGHTMHL_SURVEY.md#5-what-each-knob-is-worth-16k-one-seed-each-sd--000335)): at 16k steps and equal geometry the `margin` form beat `bounded_norm` by 0.0346 bits per byte (1.1729 against 1.2075), about ten seed spreads; the product form `bounded` collapses at 8 anchor pairs (an 18-fold attenuation of the forward) and was stopped at step 3,200 with a gap of 0.225. At 4k steps the selectivity across tables alone, the one thing in which `bounded_norm` and `bounded` with a matched gain differ, was worth 0.006 bits per byte, inside the seed spread of 0.0096 ([LookupFFN line](https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/runs_corrected/LOOKUPFFN_LINE.md#arm-c-does-selectivity-buy-anything)).

</details>

#### 2.4 Input-conditioned cells

*Source: [`light_multi_head_lut.py`][frf-light], option `cell_mode`.*

A cell may return more than a constant row. With $x_h$ the head's input slice (the gated modes need equal input and output widths) and $s_t$ the score of 2.2:

- `gated_affine` stores two vectors $f_c$ and $g_c$ per cell; the head output is the sum of $\sum_t s_t f_{c_t}$ and $\big(\sum_t s_t g_{c_t}\big)\odot x_h$ with $\odot$ the elementwise product. The sums over tables factorise, so one gather over the wide row suffices.
- `gated_multiply` returns only the product term, which is $\big(\sum_t s_t g_{c_t}\big)\odot x_h$ for the head.
- `margin_readout` adds to the constant read the term $\sum_t s_t A_t m_t$ with a matrix $A_t$ of size $d_\text{out}\times n_\text{ap}$ per table and the table margins $m_t$ (signed or absolute, by an option).
- `codebook` stores one scalar $w_c$ per cell and decodes the vector of all products $s_t w_{c_t}$ with one shared matrix.

**Status:** experimental.

`margin_readout` ([`exp_g_0214`](https://github.com/anatoli-starostin/spiky/tree/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/runs_corrected/exp_g_0214_B48k_light_marginreadout_H8_tph64_nap8_seed1), 1.1305) and `codebook` ([`exp_g_0215`](https://github.com/anatoli-starostin/spiky/tree/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/runs_corrected/exp_g_0215_B48k_light_codebook_H8_tph48_nap8_seed1), stopped at step 17,000, 1.2166) were run against constant cells ([`exp_g_0207`](https://github.com/anatoli-starostin/spiky/tree/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/runs_corrected/exp_g_0207_B48k_light_margin_n1_H8_tph64_nap8_seed1), 1.1313); `gated_multiply` was not.

### 3. Other aggregations over tables

#### 3.1 Log-sum-exp over tables

*Code [`fast_multi_head_lut.py`][lt-fast], option `exp_outputs`.*

The sum over the tables of a head can be replaced by a soft maximum with a learned temperature $\tau=\mathrm{softplus}(\tau_\text{raw})$ floored at a small constant. Per output coordinate, with the hard address unchanged, the `sum` scale reads:

$$y=\mathrm{tph}\cdot\tau\Big(\log\sum_{t}e^{W_t[c_t]/\tau}-\log\mathrm{tph}\Big)$$

Reading: for large $\tau$ this is the plain sum, for small $\tau$ it is $\mathrm{tph}$ times the maximum over tables. The default scale `mean` drops the factor $\mathrm{tph}$ and the offset, giving the bare soft maximum $\tau\log\sum_t e^{W_t[c_t]/\tau}$ of the rows. Rows are initialised as logarithms (`logspace`), spread and offset chosen so that the output starts near zero. Opt-in, hard forward only, float32 only. It costs one exponential per table and output coordinate.

**Status:** experimental.

On the Walker2d control task the readout was used for the actor of the [lutwalker2d](../lookup-related-problems/luts-to-spikes.md) experiment. The exp19 README tabulates a plain-sum actor (return 5488.4 ± 179.9, three seeds) next to the log-sum-exp actor (5403.8 ± 34.4, three seeds, *claimed*, runs not carried on main); the soft maximum gave no gain, and the learned temperature rose from its initial value toward the sum in every one of those runs ([exp19 README](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/experiments/walker2d-lut/exp19_lut-lse-expmlpcrit-t32/README.md)).

#### 3.2 Sparse-output rows

*Code [`SparseScatter`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/ranking_tools.py#L975-L1031) in [`ranking_tools.py`][cal-rank], `sparse_scatter_n_outputs` in [`tiny_multi_head_lut.py`][cal-tiny]; `lut_fused` synapses ([`LUTLayer.py`][lf-layer]).*

A row stores only $n_\text{sp}<d_\text{out}$ values. Table $t$ owns a fixed random index set $I_t$ of $n_\text{sp}$ output coordinates, drawn without replacement per table, and the head output is

$$y[i]=\sum_{t}\sum_{k:I_t[k]=i}W_t[c_t][k]$$

Reading: rows are narrower, outputs overlap across tables; storage per table is $K n_\text{sp}$ instead of $K d_\text{out}$ values. `TinyMultiHeadLut` can balance the sets so that every output coordinate receives about the same number of values. The sparse synapse lists of `lut_fused` (1.1) are the same idea. The dominance read (6.1 below) is a special case where each stored value is added to one output and subtracted from another.

**Status:** set aside.

### 4. Output projection

*Source: [`lutorch_ex` report][report-tex-outproj], section "The output projection"; code [`projection.py`][ex-proj] ([`ProjectionMHL`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/projection.py#L69-L273), current), [`fp8.py`][ex-fp8], [`deploy.py`][ex-deploy]; [`compression_mhl.py`][lt-comp] (research predecessor). The input projection is on the hashing page.*

The head outputs, flattened, are mapped back to model width by one dense matrix, the **output projection** (`decompress`):

$$\text{out}=D [y_1;\dots;y_{h_\text{out}}]+b_D,\qquad D\in\mathbb{R}^{D_\text{out}\times h_\text{out}d_\text{out}}$$

Reading: $[y_1;\dots;y_{h_\text{out}}]$ is the concatenation of the head outputs and $D_\text{out}$ is the model width. This is the same as summing one matrix product per head. Both $D$ and $b_D$ start at zero, so a fresh layer outputs exactly zero and is a bypass inside a residual block. It is the only dense matrix after the tables; rows stay narrow because the stored size of a table is $K d_\text{out}$ values.

**Variants.**

- Either projection can be switched off when the widths already match; switching both off is refused. Without output projection (`inner_out_dim=-1` in [`CompressionMHL`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/compression_mhl.py#L59-L258)) rows have full width and the heads are summed.
- `CompressionMHL` has an *independent* mode (one projection per head, equal to one matrix on the concatenated head outputs) and a *joint* mode (one shared projection, heads summed first). `inner_residual` adds the table input to its output when the widths agree.
- The power-of-two scale of the quantised read (5.1 below) is multiplied exactly into the columns of $D$ at export.
- The matrix products can run in bfloat16, or in fp8 (e4m3 operands, e5m2 gradients, one dynamic scale per tensor, float32 master weights). Addressing always receives a float32 input, so reduced precision here cannot change an address.

<details>
<summary>No projections against compressed projections: 330M against 67M parameters</summary>

With and without projections, at fast test scale and the old evaluation (single seeds, 16k steps, 4 heads of width 48 unless said, 128 tables per head of 256 cells): the dense baseline [`exp_n_0135`](https://github.com/anatoli-starostin/spiky/tree/3ac72cca40c63b77d509d4ccff33bb85d43e515e/experiments/ffn_replacement/runs/exp_n_0135_untied_vanilla_baseline_16k) reached 1.2014; the LUT with no projections [`exp_n_0136`](https://github.com/anatoli-starostin/spiky/tree/3ac72cca40c63b77d509d4ccff33bb85d43e515e/experiments/ffn_replacement/runs/exp_n_0136_fastmhl_raw_H4_nap8_tph128) 1.2057 with 330.7M parameters; compressing only the output [`exp_n_0138`](https://github.com/anatoli-starostin/spiky/tree/3ac72cca40c63b77d509d4ccff33bb85d43e515e/experiments/ffn_replacement/runs/exp_n_0138_outcompress_only_H4_nap8_tph128) 1.2125 with 66.9M; compressing the input as well [`exp_n_0121`](https://github.com/anatoli-starostin/spiky/tree/3ac72cca40c63b77d509d4ccff33bb85d43e515e/experiments/ffn_replacement/runs/exp_n_0121_ffnsw_nap8_tph128_16k) 1.1915 with 67.4M. At the same 67.4M parameters the inner width was swept as heads times width: 8 heads of width 24 [`exp_n_0132`](https://github.com/anatoli-starostin/spiky/tree/3ac72cca40c63b77d509d4ccff33bb85d43e515e/experiments/ffn_replacement/runs/exp_n_0132_grid_H8d24_nap8_tph128) 1.1926, 2 heads of width 96 [`exp_n_0131`](https://github.com/anatoli-starostin/spiky/tree/3ac72cca40c63b77d509d4ccff33bb85d43e515e/experiments/ffn_replacement/runs/exp_n_0131_grid_H2d96_nap8_tph128) 1.1888 and 1 head of width 192 [`exp_n_0137`](https://github.com/anatoli-starostin/spiky/tree/3ac72cca40c63b77d509d4ccff33bb85d43e515e/experiments/ffn_replacement/runs/exp_n_0137_grid_H1d192_nap8_tph128) 1.1945 ([FFN replacement, section 1](../architectures/transformers/ffn-replacement.md#1-routed-feed-forward-with-compression-small-scale-august-2026-old-evaluation)). These results should be revisited on the fixed evaluation: 1.2057 for anchor pairs reading the full-width input directly, without any projection, is interesting and not that bad, even at 330.7M parameters. Especially interesting is that compressing only the output makes it much worse (1.2125 with 66.9M parameters), while compressing the input as well recovers it (1.1915).

</details>

### 5. Quantised reads

#### 5.1 Power-of-two int8 read (`p2_int8`)

*Source: [`_pow2.py`][ex-pow2], [`quantised_confidence.py`][ex-quantconf], [`deployed_quantised_confidence.py`][ex-deployedq], [`deploy.py`][ex-deploy]; derivation in the last sections of [`lut_ablation_table_v4.tex`][frf-abl4] and the report section "The int8 lookup and decode". Predecessors on the unmerged branch `ffn_replacement_fix`: [`pow2_read.py`][frf-pow2], [`quantised_light_ffn.py`][frf-qlight].*

The two-row confidence read (2.2) with weights restricted to powers of two, so that the read is shifts and integer additions. Rows are stored as int8 with one shared power-of-two exponent $e$ per head and output channel:

$$\hat W=\mathrm{clamp}\Big(\mathrm{round}\Big(\frac{W}{2^{e}}\Big),-128,127\Big),\qquad e=\lceil\log_2\max\lvert W\rvert\rceil-7$$

Reading: the maximum runs over all tables and rows of the head, per output channel, so the largest entry uses the full int8 range. The weights $s(1-v)$ and $sv$ of 2.2 are rounded: their ratio becomes $2^{-q}$ for an integer $q$ and the first weight becomes $2^{k'}$ for an integer $k'$ as follows:

$$q=\mathrm{round}\Big(\frac{2u^\star}{\tau\ln 2}\Big),\qquad k'=\mathrm{round}\big(\log_2 s-c_q\big),\qquad c_q=\log_2\big(1+2^{-q}\big)$$

Reading: the ratio of the two weights is about $2^{-q}$ and so the addressed row has the weight $s/(1+2^{-q})$ approximately, and $2^{k'}$ is that weight with its base-2 logarithm $\log_2 s-c_q$ rounded to an integer. For large $q$ the value $c_q$ is taken as zero. Each table then adds two shifted rows to an int32 accumulator kept in units of $2^{-6}$ with left shifts $k'+6$ for the addressed row and $k'+6-q$ for the neighbour:

$$2^{k'}\hat W[c]+2^{k'-q}\hat W[c']$$

and the head output is the accumulator scaled once by $2^{e}$ at the end. Limits: $k'$ is clamped to the window from $-3$ to $4$ and a table whose $k'$ falls below it is skipped (it still receives gradient); the neighbour is dropped when $q$ exceeds 3. With one row the read is $2^{k'}\hat W[c]$ with $k'=\mathrm{round}(\log_2 s)$ and no neighbour.

**Training and export.** The float rows together with $\beta$ and $\gamma$ and $\tau$ are trained through straight-through estimators of the roundings, with the gradient of the exact float read, so the trained forward equals the deployed integer read up to round-off. At export the factor $2^{e-6}$ is multiplied exactly into the output projection ([section 4](#4-output-projection) above), so the deployed cartridge ([`DeployedQuantisedConfidenceLUT`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/deployed_quantised_confidence.py#L21-L132)) returns the raw int32 accumulator. Export needs an output projection and one output head per table group (no fan-in).

The row accumulation needs no multiplier; the integers $q$ and $k'$ are computed per table from the margins in floating point.

<details>
<summary>Int8 power-of-two read against the float read: two rows tie with the dense feed-forward</summary>

Int8 power-of-two read against the float read, one and two rows ([fused sweep README](https://github.com/anatoli-starostin/spiky/blob/b1a5a7131732f93595f53bf572299139eb1a0856/experiments/lutorch_ex_fused/README.md#results-on-the-full-170-shard-corpus-48k-steps--these-are-the-comparable-numbers); fast test scale, 48k steps, full corpus, fixed evaluation, single seeds): +0.0023 with one row (1.0993 against 1.0970) and +0.0033 with two (1.0899 against 1.0865), so the two-row int8 read ties with the dense feed-forward (1.0894). [FFN replacement, section 5](../architectures/transformers/ffn-replacement.md#5-lutorch_ex-cartridges-on-the-full-corpus-october-2026).

</details>

#### 5.2 Precision of stored rows

*Source: precision tables in the [`lutorch_ex` README][ex-readme-spec]; the quantisation study in [`LIGHTMHL_SURVEY.md`][frf-survey]; `use_fp8` and `vote_quant_levels` in [`permutational_lut.py`][cal-perm].*

Storage and read precisions found in the code, from finest to coarsest:

- float32 master rows (default); bfloat16 or float16 rows with float32 accumulation (the `lutorch_ex` Manifesto and confidence classes, not the quantised ones);
- per-row or per-tensor int8 and int16 rows;
- rows rounded to the nearest power of two, so that the product with a weight is a shift;
- int8 rows with a shared exponent and power-of-two weights (5.1);
- fp8 e4m3 rows with a per-tensor dynamic scale and straight-through gradient, and quantised votes ([`PermutationalLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/permutational_lut.py#L294-L669), 6.2 below);
- 1-bit and K-bit votes (6.2 below); int32 accumulation (`lut_fused`).

Addressing always runs in float32, so a rounded margin cannot flip a bit. The quantisation study applies rungs of this ladder after training to a trained confidence checkpoint: int8 and int16 rows, shift-only rows, a power-of-two version of the blend temperature $\tau$ and finally power-of-two score and weights.

<details>
<summary>Post-training quantisation of one trained layer: which roundings cost nothing</summary>

Post-training quantisation of one trained layer ([study](https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/runs_corrected/exp_g_0195_B16k_light_margin_blend_n2_tau_learn0p5_seed1/QUANT_STUDY_0195.md), 16k steps, fixed evaluation, seed spread 0.00335, baseline 1.1606): int16 and int8 tables, per row or per tensor, cost nothing measurable; tables rounded to powers of two cost +0.0035 (+0.0062 at the 48k checkpoint); rounding the data-dependent score and blend weights to powers of two as well costs +0.0109 together, and the fully multiply-free read-out +0.0180, behind the one-row layer. Conclusion: integer rows with a real integer product of score and weight, not pure shifts ([FFN replacement, section 5](../architectures/transformers/ffn-replacement.md#5-lutorch_ex-cartridges-on-the-full-corpus-october-2026)).

</details>

### 6. Votes and dominance

The reads above add vectors in output space. The reads of this section store *votes on pairs of outputs* and form the output by Borda counting, for order-coded signals (outputs whose ranking carries the information).

#### 6.1 Dominance votes with Borda sum ([`DominanceLUT`](https://github.com/anatoli-starostin/spiky/blob/c7fdbf4a06c72e15da546a641e07e08dda187221/experiments/lutorch_ex_fused/dominance_stack/dominance_lut_ref.py#L112-L273))

*Source: the unmerged branch `dominance_stack_ablation`: [`dominance_lut_ref.py`][dom-ref], [`dominance_fast.py`][dom-fast], [`dominance_conf.py`][dom-conf].*

The address is the standard sign-bit address, but a row stores $n_\text{op}$ deltas instead of $d_\text{out}$ values. Table $t$ owns $n_\text{op}$ pairs of outputs, and pair number $k$ joins output $p_k$ to output $q_k$ of the same head. Delta $k$ of the addressed row is the vote that output $p_k$ should exceed output $q_k$ by this amount. The head output is the Borda score of the antisymmetric dominance matrix $D$ built from the votes, written $v_{tk}$ for table $t$ and entry $k$ of its addressed row:

$$y_j=\sum_{b}D[j,b]=\sum_{(t,k):p_k=j}v_{tk}-\sum_{(t,k):q_k=j}v_{tk}$$

Reading: each vote adds to one output and subtracts from another, so $y$ sums to zero over outputs and a common offset cannot be represented (a bias option exists). The matrix $D$ is never built: the sums are two scatter-adds, or a product with a fixed signed incidence matrix in the fast variant. Here Borda is the row sum of signed margins, not a count of wins. Output pairs: `same` as the input pairs (needs equal input and output widths and $n_\text{op}$ equal to $n_\text{ap}$ probes), `random` (distinct pairs per table) or `given`. `vote='sign'` uses plus or minus one with a straight-through gradient; `normalize='count'` divides each output by the number of votes that touch it. The confidence variant ([`DominanceConfFast`](https://github.com/anatoli-starostin/spiky/blob/c7fdbf4a06c72e15da546a641e07e08dda187221/experiments/lutorch_ex_fused/dominance_stack/dominance_conf.py#L67-L123)) multiplies each row by the one-row score of 2.2 before the Borda sum. Storage per table is $K n_\text{op}$ values.

**Status:** experimental.

<details>
<summary>Dominance vectors between blocks in the permutational series</summary>

In the byte-level permutational series dominance vectors were the interface between blocks ([Permutational transformers](../architectures/transformers/permutational.md#idea--model)). The capacity of the dominance read has not been measured separately.

</details>

#### 6.2 Signed votes (`PermutationalLut`, [`BitPermutationLUT`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/bit_permutation_lut.py#L777-L920))

*Source: [`permutational_lut.py`][cal-perm]; [`bit_permutation_lut.py`][cal-bit], [`multi_bit_permutation_lut.py`][cal-multibit], [`bit_permutation_lut_ex.py`][cal-bitex]; address front end [`tiny_anchor_pairs_lookup.py`][cal-tapl].*

Same pair-vote output as 6.1, with bounded or binary votes, for $N$ ranked outputs and $\binom{N}{2}$ output pairs.

**Saturating votes** (`PermutationalLut`). A table reads $n_\text{op}$ raw weights $w$ at its address and turns each into a vote between minus one half and one half, with a fixed temperature $\tau$ as follows:

$$v=\frac{w}{2(\tau+\lvert w\rvert)}\quad\text{(rational)},\qquad v=\sigma\Big(\frac{w}{\tau}\Big)-\frac12\quad\text{(sigmoid)},\qquad v=\pm\frac12\quad\text{(ste)}$$

`ste` is hard in the forward pass and uses the rational form in the backward pass. The votes are scattered to output pairs as in 6.1 and the result is divided by a scale; the default `borda` scale is $\sqrt{N-1}$ and the alternative `clt` scale is $\sqrt{\bar n}$ with $\bar n$ the mean number of votes per output. Pairs are `aligned` (output pairs equal input pairs, equal widths) or `scrambled` (independent pairs with full coverage). `vote_quant_levels` quantises each vote to equispaced levels with a straight-through gradient. The dominance vector over all pairs can be returned instead of the Borda scores.

**1-bit and K-bit votes** (`BitPermutationLUT`, [`MultiBitPermutationLUT`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/multi_bit_permutation_lut.py#L341-L443)). The address is the sign-bit address (int16, at most 15 probes). A row holds $n_\text{op}$ signed bits $b$ that are stored as the sign of a float latent and packed 32 per int32. Each output pair $p$ has one channel, which adds the votes routed to it as an integer; here $b_{t,c,k}$ is bit $k$ of row $c$ of table $t$ and $\text{scale}$ is a fixed constant:

$$d_p=\text{scale}\cdot\sum_{(t,k)\to p}b_{t,c_t,k}$$

Reading: $d$ is a dominance vector with one entry per pair; [section 6.3](#63-rank-coded-decode) below turns it into a Borda vector. Training goes through a straight-through gate $T/(T+\lvert\text{latent}\rvert)^2$ on the latent, where $T$ is the gate temperature. The multi-bit version stores integers $q$ of $K$ bits, $K$ being 2, 4 or 8, obtained as $q=\mathrm{round}(\text{latent}\cdot 2^{K-1})$ after clamping, and reads them as the midrise values $(2q+1)/2^{K}$ with the constant offset this adds per output kept as a bias.

[`BitPermutationLUTEx`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/bit_permutation_lut_ex.py#L217-L364) adds a second stage. The first stage is unchanged except for where the votes go: instead of landing on the output pairs they land on the slots of a larger *voting space*, with $T$ output tables of $2^m$ slots each (random routing, summed per slot). Then each output table is read like a winner-take-all address ([hashing 6.3](hashing.md#63-winner-take-all-address)): the slot with the largest total is the winner, and its index $0,\dots,2^m-1$ is read as $m$ bits, that is $m$ signs of plus or minus one. The table owns $m$ fixed output pairs, and these signs are added to the dominance vector at those pairs. So an output table casts a hard vote of $m$ signs, chosen by the largest slot, and the gradient goes through the runner-up slots as in hashing 6.3. [`BitPermutationLUTLin`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/bit_permutation_lut_ex.py#L371-L464) replaces the second stage by a learned float matrix from the voting space to the pairs, followed by a layer norm without parameters, so that training and evaluation use the same deterministic path.

**Status:** experimental; `PermutationalLut` is the float reference for the bit versions.

<details>
<summary>Bit width of the votes at byte level (1, 2, 4, 8 bits) and 1-bit blocks</summary>

Bit width of the votes in the byte-level series (context 128, 25k steps, old evaluation, single seeds): in the output table 1, 2, 4 and 8 bits gave 1.4004, 1.3905, 1.3721 and 1.4139, and inside the blocks one-bit query, key and value tables with a 4-bit output projection reached 1.4322 against 1.4918 for all-4-bit ([Permutational transformers](../architectures/transformers/permutational.md#what-was-found), what was found). The two second-stage variants and a corrected comparison with the dense baseline are not available: the fixed evaluation was never run for bit tables.

</details>

#### 6.3 Rank-coded decode

*Source: [`ranking_tools.py`][cal-rank] ([`DominanceToVector`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/ranking_tools.py#L484-L522), [`VectorToDominance`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/ranking_tools.py#L605-L768), [`DominanceCanonicalize`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/ranking_tools.py#L838-L862), [`BordaCanonicalize`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/ranking_tools.py#L865-L890), [`SparseDominanceCanonicalize`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/ranking_tools.py#L771-L835)), [`rank_quantize.py`][cal-rq].*

The decode step between layers in the vote line. With $N$ ranked items, `DominanceToVector` turns a dominance vector $d$ with one entry per pair into a Borda vector:

$$v=\mathrm{LayerNorm}(Bd)$$

Reading: $B$ is the incidence matrix with entry $+1$ at the first item of a pair and $-1$ at the second and $0$ elsewhere, all scaled by the factor $1/\sqrt{N-1}$ for $N$ items; the LayerNorm has an affine part. This is the decode half of 6.1 and 6.2.

The reverse, `VectorToDominance`, takes pairwise signs of a vector $x$ as follows: $d_p=+1$ if $x_a>x_b$ and $-1$ otherwise (ties give minus one), or the soft form $d_p=(x_a-x_b)/(\theta+\lvert x_a-x_b\rvert)$ with a temperature $\theta$ or a straight-through version. Composing the two (`DominanceCanonicalize`) maps a vote vector that is not transitive to a totally ordered one, which also removes magnitude information; `BordaCanonicalize` does the same starting from a Borda vector; `SparseDominanceCanonicalize` works on a random subset of pairs per head. [`RankQuantize`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/rank_quantize.py#L41-L67) replaces a vector by its normalised ranks with a straight-through gradient. These steps act on the signal between reads, not on rows.

**Status:** research line; not used in the `lutorch_ex` layers.

<details>
<summary>Canonicalising the output projection to signs: early BPE run against later recipe</summary>

Canonicalising the output of the output projection to signs (the Borda projection and back) gave 1.7088 in an early BPE run ([`exp071`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp071_canonicalize_out)); dropping the canonicalisation in a later recipe was better, 1.6042 against 1.6212 at 8k steps and 1.4803 against 1.4906 at 48k, and a canonicalisation after the output projection closed a gradient path for magnitudes at a cost of 0.013 in a byte-level run ([Permutational transformers](../architectures/transformers/permutational.md#magnitude-leakage), magnitude leakage; old evaluation, the with-canonicalisation numbers are quoted from run configs).

</details>

## Open ideas

- Cell mode `gated_multiply` (2.4): no run is known. *Untested.*
- A power-of-two read with shift-only rows but a real integer product of score and weight, as the quantisation study of 5.2 suggests. *Untested.*

[report-tex-outproj]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/lutorch_ex/lutorch_ex_report.tex#L557-L571
[report-tex-variants]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/lutorch_ex/lutorch_ex_report.tex#L230-L331
[report-tex-heads]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/lutorch_ex/lutorch_ex_report.tex#L332-L472
[ex-readme-spec]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/README.md#cartridges-specification
[ex-readme-cartridges]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/README.md#cartridges
[ex-spec]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/lut_spec.py
[ex-lutbase]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/lut_base.py
[ex-proj]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/projection.py
[ex-fp8]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/fp8.py
[ex-deploy]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/deploy.py
[ex-manbase]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/manifesto_base.py
[ex-manhard]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/manifesto_hard.py
[ex-mansoft]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/manifesto_soft.py
[ex-uncert]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/uncertainty.py
[ex-conf]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/confidence.py
[ex-confbase]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/confidence_base.py
[ex-pow2]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/_pow2.py
[ex-quantconf]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/quantised_confidence.py
[ex-deployedq]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/deployed_quantised_confidence.py
[ex-fhard]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/fused_manifesto_hard.py
[ex-fsoft]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/fused_manifesto_soft.py
[ex-fconf]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/fused_confidence.py
[ex-fquant]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/fused_quantised_confidence.py
[lt-lproj]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/l_projection.py
[lt-mhl]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/multi_head_lut.py
[lt-fast]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py
[lt-fast-hybrid]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/fast_multi_head_lut.py#L511-L749
[lt-comp]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/compression_mhl.py
[lf-layer]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/LUTLayer.py
[lf-rt]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/native/spiky/lut_fused/lut_runtime.cu
[lf-rtlogic]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/native/spiky/lut_fused/lut_runtime_kernels_logic.proto
[lffn-py]: https://github.com/anatoli-starostin/spiky/blob/93de410db25f3b139a0539c8a17b395e49043d5a/research/lookupffn/lookup_ffn.py
[lffn-pmlr]: https://proceedings.mlr.press/v202/zeng23a.html
[cal-hybridtex]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/docs/hybrid_smooth_math.tex
[cal-tiny]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py
[cal-tiny-winner]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L1025-L1052
[cal-tiny-hs]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_multi_head_lut.py#L4092-L4179
[cal-rank]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/ranking_tools.py
[cal-rq]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/rank_quantize.py
[cal-perm]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/permutational_lut.py
[cal-bit]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/bit_permutation_lut.py
[cal-bitex]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/bit_permutation_lut_ex.py
[cal-multibit]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/multi_bit_permutation_lut.py
[cal-tapl]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/tiny_anchor_pairs_lookup.py
[frf-light]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/light_multi_head_lut.py
[frf-fast]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/fast_multi_head_lut.py
[frf-mech-forms]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/doc/lutorch/lut_mechanisms.tex#L189-L257
[frf-abl4]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/doc/research/lut_ablation/lut_ablation_table_v4.tex
[frf-pow2]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/pow2_read.py
[frf-qlight]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/quantised_light_ffn.py
[frf-survey]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/experiments/ffn_replacement/runs_corrected/LIGHTMHL_SURVEY.md#6d-exp_g_0195--assigned-to-nebius-deliberately-not-run-on-gpustar
[dom-ref]: https://github.com/anatoli-starostin/spiky/blob/c7fdbf4a06c72e15da546a641e07e08dda187221/experiments/lutorch_ex_fused/dominance_stack/dominance_lut_ref.py
[dom-fast]: https://github.com/anatoli-starostin/spiky/blob/c7fdbf4a06c72e15da546a641e07e08dda187221/experiments/lutorch_ex_fused/dominance_stack/dominance_fast.py
[dom-conf]: https://github.com/anatoli-starostin/spiky/blob/c7fdbf4a06c72e15da546a641e07e08dda187221/experiments/lutorch_ex_fused/dominance_stack/dominance_conf.py
