# Hashing mechanics

How an input vector is turned into table addresses. The read of the selected rows is on [Readout mechanics](readout.md); gradients through an integer address are on [Gradient backpropagation](backpropagation.md); results are on the Architectures pages. This page only describes hashing mechanisms.

## Contents

| Section | Covers |
| --- | --- |
| [1. Pairwise comparison hashing](#1-pairwise-comparison-hashing) | basic Manifesto math (1.1), anchor sampling policies (1.2), mixed address widths (1.3) |
| [2. Single-anchor hashing](#2-single-anchor-hashing) | sign of one coordinate |
| [3. Hyperplanes](#3-hyperplanes) | what each probe tests (3.1), free affine hyperplanes (3.2) |
| [4. Input projections](#4-input-projections) | learned linear map in front of the tables |
| [5. Ternary approach](#5-ternary-approach) | hyperplanes with weights in {-1, 0, +1} |
| [6. Other models](#6-other-models) | LIF digits (6.1), BH4 (6.2), winner-take-all (6.3), never tested (6.4) |
| [7. Trainable anchors](#7-trainable-anchors) | learning which coordinates a bit compares |
| [Open ideas](#open-ideas) |  |

## Idea / model

### 1. Pairwise comparison hashing

#### 1.1 The basic construction (Manifesto math)

*Source: [`lutorch_ex` report][report-tex-onetable] ([PDF][report-pdf]), section "One table"; Spiking Manifesto ([arXiv:2512.11843][manifesto]).*

A lookup table computes an **address** from the input and returns the vector stored there. The address must be cheap, and similar inputs should usually get the same address (a locality-sensitive hash, LSH). In Izhikevich's *Spiking Manifesto* information is carried by firing order, and a sign-bit address encodes it: each bit records which of two neurons fired first.

**Related prior work.** The construction is close to random ferns ([Özuysal, Fua and Lepetit, *Fast Keypoint Recognition in Ten Lines of Code*, CVPR 2007][ferns-2007]; [Özuysal, Calonder, Lepetit and Fua, *Fast Keypoint Recognition Using Random Ferns*, IEEE TPAMI 32(3), 2010][ferns-2010]): a fern concatenates $S$ binary pixel-pair comparisons into an index of a table of class statistics, and several ferns with independently drawn random pairs are combined. Here the same pair tests index tables of trained vectors, trained by gradient.

**From input to address.** A table receives $z\in\mathbb{R}^{d_\text{in}}$ (one head's slice of the layer input) and examines it with $n_\text{ap}$ probes (`nap`). An **anchor** is the index of one coordinate of $z$. A probe on an **anchor pair** $(a_j,b_j)$ computes the **margin** $u_j$, a bit $\beta_j$, and the address $c$:

$$u_j=z[a_j]-z[b_j],\qquad \beta_j=[u_j>0],\qquad c=\sum_{j=0}^{n_\text{ap}-1}\beta_j 2^{ n_\text{ap}-1-j}$$

Reading: each probe asks which of two coordinates is larger ($[\cdot]$ is 1 if true, else 0); the bits, first probe most significant, form $c\in\{0,\dots,K-1\}$, $K=2^{n_\text{ap}}$. Anchors are integers drawn at random at creation (how: [section 1.2](#12-anchor-sampling-policies) below) and fixed in training.

**From address to row.** The table has $K$ rows; row $c$ is a learnable vector $W[c]\in\mathbb{R}^{d_\text{out}}$. The hard lookup returns $W[c(z)]$, so a table splits its inputs into $K$ classes and stores one vector per class.

*Example.* $n_\text{ap}=2$, $z=(0.9,0.2,0.3,0.4)$, pairs $(1,3),(2,4)$: $u=(0.6,-0.2)$, bits $10$, $c=2$.

**Margins and the neighbour row.** $\lvert u_j\rvert$ is how far the input is from the point where bit $j$ flips. The smallest, $u^\star=\min_j\lvert u_j\rvert$, is the **deciding margin**, attained at the deciding probe $j^\star=\arg\min_j\lvert u_j\rvert$. Inverting that bit gives the **neighbour address**

$$c'=c\oplus 2^{ n_\text{ap}-1-j^\star}$$

Reading: $c'$ is the row the input would select after the smallest change that alters the address ($\oplus$ is exclusive-or). For the hard read, the margins and $c'$ matter only for the gradient ([Gradient backpropagation](backpropagation.md#11-three-channels-and-the-row-score)).

A layer uses many such tables, each with its own anchors, and adds the rows they return (see [Readout mechanics](readout.md#11-hard-lookup-and-accumulate)). Tables are grouped into **heads**: a head is $\mathrm{tph}$ tables (tables per head) that look at the same slice of the input, and $G$ heads look at separate slices. Everything on this page concerns the address of a single table.

**Invariances.** A bit can flip only if the perturbation exceeds its margin. The hard address is unchanged by positive rescaling of $z$, by any increasing map applied to all coordinates, and by adding one constant to all coordinates.

**Cost.** One bit costs two reads, one subtraction and one comparison, and has no parameters. Distinct pairs require $n_\text{ap}\le\binom{d_\text{in}}{2}$.

**Code.** [`addressing.py`][ex-addr] (MSB-first packing), [`manifesto_base.py`][ex-mbase] (bits, deciding probe, neighbour). Earlier code: [`lut_fused`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lut_fused) ([ground-truth python][lf-gt-index]), [`lutorch`](https://github.com/anatoli-starostin/spiky/tree/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch) ([`anchor_pairs_lookup.py`][lt-apl]).

#### 1.2 Anchor sampling policies

*From the `lutorch_ex` code ([`anchors.py`][ex-anchors]).*

The only free choice is which pairs each table gets. The current policy, **canonical full coverage**, per head:

1. The pool is all pairs $a<b$ of the head's coordinates, $P=\binom{d_\text{in}}{2}$, which is 1,128 pairs for $d_\text{in}=48$.
2. Concatenate $\lceil \mathrm{tph}\cdot n_\text{ap}/P\rceil$ random permutations of the pool, truncate to $\mathrm{tph}\cdot n_\text{ap}$ slots, reshape to $[\mathrm{tph},n_\text{ap}]$.
3. Repair duplicates within a table by swapping with another table's slot.

Reading: pairs are drawn without replacement, so all pairs appear equally often (to within one) and every pair is covered once $\mathrm{tph}\cdot n_\text{ap}\ge P$. The ratio $\rho=n_\text{ap} \mathrm{tph}/d_\text{in}$ (comparison density) is bits spent per input coordinate.

The older code has other policies (twelve are listed in [`lut_helpers.py`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/lut_helpers.py#L11-L23)): *balanced* (tiled permutations for $a$ and $b$), *connected* (`b = roll(a,-1)`, a chain), *distinct* (distinct within a table only), *disjoint* (a table's pairs use $2n_\text{ap}$ different coordinates), *hierarchical* and *multiscale* (pairs at several distances), *triplets/quadruplets* (all pairs inside shared groups), *conv2d*, candidate pools, forbidden/allowed groups, a doubling-step prior, and in `lut_fused` a receptive field grown by a synapse-growth engine ([`anchor_sampler.py`][lf-sampler]). They did not make it into `lutorch_ex` because their benefit is not obvious; canonical full coverage seems to be the best choice in most cases.

#### 1.3 Mixed address widths

*Code: [`tiny_multi_nap_multi_head_lut.py`][cal-multinap] ([`TinyMultiNapMultiHeadLut`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp387_multinap_out_proj/tiny_multi_nap_multi_head_lut.py)).*

The address width $n_\text{ap}$ is normally one number per module. It can be mixed: a module holds several independent groups of tables over the same input, each group with its own $n_\text{ap}$ and its own number of tables, and the module output is the sum of the groups. Every group hashes the input with its own anchor pairs and reads one row per table, so nothing else changes. Narrow groups (small $n_\text{ap}$) have few rows, so every row is visited often and is trained densely. Wide groups have many rows, so they separate inputs finely but each row is visited rarely. The sum lets one module have both.

A coarser form of the same idea gives different $n_\text{ap}$ to different modules of one model, for example narrow tables for queries and keys and wide ones for values and the output projection.

Combined with input projections ([section 4](#4-input-projections)), mixed widths may give good results: the groups could share one learned projection or each have its own. This is worth trying. *Untested.*

<details>
<summary>Mixed address widths in the LUT SDPA models: shallow Q and K tables, deeper V</summary>

Evidence for mixed widths: in the LUT SDPA models the Q and K tables preferred many shallow tables (4 address bits) and the value and residual tables fewer deeper ones (6 bits), with a heavy-early schedule of tables across layers for the output projection ([LUT attention](../architectures/transformers/lut-attention.md#what-was-found-with-lut-sdpa-bpe-context-512-vocabulary-32768), notes with the old evaluation).

</details>

### 2. Single-anchor hashing

*Source: [`lutorch_ex` report][report-tex-onetable-single]; code [`anchors.py`][ex-anchors], [`lut_spec.py`][ex-spec], [test][ex-single-test].*

A probe on a **single anchor** $a_j$ tests one coordinate against zero:

$$u_j=z[a_j],\qquad \beta_j=[z[a_j]>0],\qquad c=\sum_j\beta_j 2^{ n_\text{ap}-1-j}$$

Reading: same address and neighbour rules as 1.1; the margin is the coordinate's magnitude. The pool is the $d_\text{in}$ coordinates, same sampling, $n_\text{ap}\le d_\text{in}$ (`anchor_mode="single"`); all tables of a layer share the probe kind.

On the raw input a single anchor tests only an axis-parallel plane, so pairs are much stronger. If a learned linear map is placed in front of the tables (the *input projection*, [section 4](#4-input-projections) below), both kinds test learned hyperplanes (at most $d_\text{in}$ for singles, up to $\binom{d_\text{in}}{2}$ for pairs). A single is cheaper (one read, no subtraction). A coordinate sign does not cancel an offset shared by all tokens, a pair difference does; this is why pairs win. Kept as a flag.

### 3. Hyperplanes

#### 3.1 Every probe tests a hyperplane

*Source: [`lutorch_ex` report][report-tex-inproj], subsection "The input projection", table "The hyperplane that one probe tests".*

A probe asks on which side of a hyperplane the input lies; which hyperplanes depends on the probe kind and on whether a learned linear map $z=W_cx+b_c$ (the *input projection*, [section 4](#4-input-projections) below) is placed in front. With $x$ the raw input, $z[a]=w_a^\top x+b_a$:

| | single anchors | anchor pairs |
| --- | --- | --- |
| no input projection | $x[a]>0$: plane along a coordinate axis | $x[a]-x[b]>0$: normal $e_a-e_b$, zero bias (the Manifesto form) |
| with input projection | $w_a^\top x+b_a>0$: arbitrary learned hyperplane | $(w_a-w_b)^\top x+(b_a-b_b)>0$: also arbitrary, up to $\binom{d_\text{in}}{2}$ planes per head |

Reading: a pair probe is a hyperplane test with normal $e_a-e_b$: one $+1$ and one $-1$. With such a projection in front, a single anchor could emulate a pair only if the row of $W_c$ became $e_a-e_b$; ordinary training has no reason to do that, and forcing it would need a straight-through estimator onto $\pm1$, which the library lacks. So both probe kinds are kept. A pair is the cheap special case (two reads, one subtraction, integer anchors); a general hyperplane is more expressive because every bit sees the whole input, at the price of a multiply-accumulate over all $d_\text{in}$ inputs.

#### 3.2 Free affine hyperplanes ([`HyperplaneMultiHeadLUT`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/hyperplane_multi_head_lut.py#L622-L891))

*Code [`hyperplane_multi_head_lut.py`][lt-hyp] (main); experiments on the unmerged branch [`hyperplane_ffn_next`][hyp-readme]. Not in `lutorch_ex`.*

Every bit gets its own learnable hyperplane instead of the fixed $e_a-e_b$: $w\in\mathbb{R}^{n_\text{tables}\times n_\text{ap}\times d_\text{in}}$, $b\in\mathbb{R}^{n_\text{tables}\times n_\text{ap}}$,

$$s_{t,j}=\langle w_{t,j},x\rangle+b_{t,j},\qquad \beta_{t,j}=[s_{t,j}>0],\qquad c_t=\sum_j\beta_{t,j} 2^{ n_\text{ap}-1-j}$$

Reading: $s$ plays the role of the margin $u$, so deciding probe and neighbour are as before; one matrix product $[B,d_\text{in}]\times[d_\text{in},n_\text{tables}n_\text{ap}]$. Initialisation `anchor_pairs` sets $w_j=e_{a_j}-e_{b_j}$, $b_j=0$ (reproduces the fixed model bit for bit in fp32); `random` uses Gaussian rows. Extra parameters $n_\text{tables}n_\text{ap}(d_\text{in}+1)$.

Two analysed runs ([anchor-pair initialisation][hyp-analysis], [random initialisation][hyp-geom]) show that learning refines the initial structure instead of discovering a new one. Rows initialised as anchor pairs stayed concentrated on about two coordinates and did not turn into arbitrary dense hyperplanes (the analysis cannot say whether each row kept its own initial pair). Rows initialised randomly stayed dense, and no pairwise structure emerged. This motivates the ternary hash ([section 5](#5-ternary-approach) below).

**Bridge to `lutorch_ex`.** The free-hyperplane module needs no separate implementation in the current library. Take [`ProjectionMHL`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/projection.py#L69-L273) ([section 4](#4-input-projections) below) with single anchors and set the per-head width $d_\text{in}=n_\text{ap}\cdot\mathrm{tph}$. A head then has exactly $n_\text{ap}\cdot\mathrm{tph}$ coordinates in $z=W_cx+b_c$, and canonical full coverage ([section 1.2](#12-anchor-sampling-policies)) draws $\mathrm{tph}\cdot n_\text{ap}$ single anchors without replacement from them, so every coordinate is used by exactly one bit. Bit $j$ of table $t$ is then $[z[a_{t,j}]>0]=[w_{a_{t,j}}^\top x+b_{a_{t,j}}>0]$: its own learnable hyperplane over the whole input, the row $a_{t,j}$ of $W_c$. That is the hyperplane of `HyperplaneMultiHeadLUT` with the same number of parameters.

### 4. Input projections

*Source: [`lutorch_ex` report][report-tex-wrapper], section "The ProjectionMHL wrapper"; code [`projection.py`][ex-proj] (current), [`compression_mhl.py`][lt-comp] (research predecessor).*

A table sees a narrow slice of $d_\text{in}$ values, while the model works on one vector of model width. The wrapper therefore puts one linear map in front of the tables, the **input projection** (`compress`):

$$z=W_c x+b_c,\qquad x\in\mathbb{R}^{D_\text{in}},\quad z\in\mathbb{R}^{h_\text{in}\cdot d_\text{in}}$$

Reading: $x$ is the model-width input vector, $z$ is split into $h_\text{in}$ slices of width $d_\text{in}$, and each head hashes its own slice, so normally $h_\text{in}=G$, the number of heads (defined in [section 1.1](#11-the-basic-construction-manifesto-math)). The same holds for a batch, row by row.

Every coordinate of $z$ is a learned linear function of $x$, so probes become arbitrary learned hyperplanes. One dense product per token serves all bits. $W_c$ is initialised $\mathcal{N}(0,0.02)$.

### 5. Ternary approach

Constrain each hyperplane's weights to $\{-1,0,+1\}$: an anchor pair is the ternary vector with one $+1$ and one $-1$, so ternary strictly generalises pairs, and after training the test is a sum of some inputs minus a sum of others, with no multiplication. The weights are learned through a straight-through estimator, with a per-table temperature setting the share of zeros. Code: [`ternary_hyperplane_multi_head_lut.py`][hyp-ternary] (subclass of the free-hyperplane module, branch `hyperplane_ffn_next`). The construction, the standardised variant and the sparsity experiments are on their own page, [Ternary routing](ternary.md); the idea is written up in *Ternary Anchoring Lookup Tables* ([LaTeX source][talut-tex], [PDF][talut-pdf]).

### 6. Other models

#### 6.1 LIF first-spike bucket digits

*Code [`lif_multi_head_lut.py`][lt-lif]; branch [`exp/lif-mhl-experiments`][lif-readme] with math note [`mhl_detectors_math.tex`][lif-math].*

*Status: probably a dead end. It may become relevant only if a real gain in capacity or expressivity is proven.*

Replace sign bits by a leaky integrate-and-fire (LIF) neuron: the address digit is the time bucket of its first spike. Input $x_i$ is latency-coded to a spike time $t_i=\mathrm{clamp}\big(T_w(0.5-3x_i/32),0,T_w\big)$ (larger fires earlier), arrival $a_i=t_i+\text{delay}_i$ with learnable delay. Sorting arrivals, the membrane after the arrival number $k$ is

$$V_k=e^{-a_k/\tau}\sum_{m\le k}w_m e^{a_m/\tau},\qquad w=w_\text{max} \sigma(w_\text{raw}),\quad \tau=\mathrm{softplus}(\tau_\text{raw})+1$$

Reading: earlier inputs add their weight, decayed by $\tau$. The first-spike time $t^\ast$ is the first arrival with $V_k\ge\theta=1$; if there is none, it is $T_w$. The digit $b$ counts the $M-1$ trainable boundaries $\le t^\ast$. With $D$ neurons per table the address is mixed-radix:

$$c=\sum_{d=0}^{D-1}b_d M^{ D-1-d},\qquad M^D\le 65{,}536\ \text{cells}$$

Reading: $M=2$ gives the bit address of [section 1](#1-pairwise-comparison-hashing). The address is hard in evaluation; training uses a straight-through surrogate for the first-spike time and the bucket boundaries.

Never tested for language models; sorting costs $O(N\log N)$ per table per token, far above one subtraction per bit.

#### 6.2 BH4 coordinate-sign addressing (LookupFFN-style)

*Code [`bh4_multi_head_lut.py`][frf-bh4], branch `research/ffn_replacement_fix`; reference [`lookup_ffn.py`][lffn-py] with [README][lffn-readme].*

LookupFFN ([Zeng et al., ICML 2023][lffn-pmlr]; [arXiv:2403.07221][lffn-arxiv]) hashes with a structured learned projection and takes coordinate signs. The projection is $R=\prod_{i=1}^{4}B_iH$, with $H$ a fixed normalised Walsh-Hadamard transform (additions only, width a power of two) and each $B_i$ a learnable block-diagonal matrix, at a cost of order $d\log d$ instead of a dense product. With $z=Rx$:

$$c=\sum_j[z_j>0] 2^{ j}$$

Reading: the nearest code on the hypercube is the sign pattern (classic hyperplane LSH), so no codebook is stored; the address is hard in training and evaluation. (The read score that gives $R$ its gradient is a readout matter.) A fixed Hadamard also precedes the learnable blocks.

*Never tested properly.*

#### 6.3 Winner-take-all address

*Code [`wta_lookup.py`][lt-wta] and [`multi_head_wta.py`][lt-mwta] ([`WTALookup`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/wta_lookup.py#L81-L185), [`WTA`](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/multi_head_wta.py#L15-L112)).*

The address is not a bit string but the position of the largest of $N$ candidates. A channel has $N$ input values $x_1,\dots,x_N$ and the address is

$$c=\arg\max_k x_k$$

so a channel's table has $N$ rows instead of $2^{n_\text{ap}}$. Channels are independent, each with its own table, which the `WTA` module reads with [`LProjection`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/l_projection.py#L248-L443). Reading: like the pair hash, the address is hard. The gradient goes through runner-ups: the code returns the next `n_alternatives` largest candidates together with their gaps to the winner, the counterpart of the margins and the neighbour address of [section 1.1](#11-the-basic-construction-manifesto-math). The backward pass pushes the weighted gradient to the winner and subtracts it from each runner-up, with the same uncertainty functions as the pair hash.

#### 6.4 Never tested

- **Structured input projection with the pair address:** the BH4-style projection $\prod_i B_iH$, at a cost of order $d\log d$, in place of the dense $W_c$ of [section 4](#4-input-projections), followed by anchor-pair comparisons instead of coordinate signs. BH4 (6.2) changed both the projection and the address at once, so this combination was not isolated.
- **Fixed random Hadamard rotation before the pair hash** ($D_1HD_2HD_3H$, random sign diagonals): a parameter-free rotation mixing all coordinates before pairs are drawn.
- **Hierarchical (coarse-to-fine) tables:** a short first address selects a sub-table refined by further bits (not the hierarchical sampling policy of 1.2, which only chooses pairs).

### 7. Trainable anchors

*Code: [`trainable_anchors_multi_head_lut.py`][cal-trainanch]; [`LightMHL`](https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/light_multi_head_lut.py#L151-L154) options `trainable_anchors`, `anchor_init` ([`light_multi_head_lut.py`][frf-light]).*

Keep the pair hash but learn *which* coordinates a bit compares, instead of a direction. Each probe $j$ has two selector logit vectors $g_j$ and $h_j$ with $d_\text{in}$ entries each. The anchors are the argmax coordinates:

$$a_j=\arg\max_d g_{j,d},\quad b_j=\arg\max_d h_{j,d},\quad u_j=z[a_j]-z[b_j]$$

Reading: forward uses the hard argmax; backward uses a straight-through softmax of the logits at temperature $T_\text{anchor}$, annealed from 1.0 to 0.001. Then anchors are snapped to integers. `anchor_init="warm"` starts from the fixed model's pairs: the logits are small noise of scale 0.02, plus $+C$ on the first coordinate of each pair and $-C$ on the second, with $C=8$ (`anchor_init_scale`). Any $C$ far above the noise works; it only has to make the argmax pick the intended coordinate, so step 0 reproduces the fixed model exactly. The sign of $C$ is there because the `LightMHL` implementation takes the argmax for the first anchor and the argmin for the second, from one logit vector.

It did not improve on random full-coverage anchors; the anchors barely moved from their start (the variants included a cold start). A long evolutionary search over the anchor comparison graph ([`evolve.py`](https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/transformer_exps/evolve.py)) improved only slightly over random pairs. Retry conditions, untested: more probes, no snap, freeze-then-train.

<details>
<summary>Full-coverage anchors against random anchors, byte level: a free 0.02 gain</summary>

Full coverage against random anchors in the byte-level series (about 5M parameters, 50k steps, old evaluation, single seeds): 1.6839 against 1.7060, a free gain of about 0.02 since the sampling policy costs nothing at run time ([LUT attention](../architectures/transformers/lut-attention.md#earlier-lut-attention-models-byte-level-march-to-april-2026)).

</details>

## Open ideas

- Matched-parameter comparison of learned hash directions (free and ternary hyperplanes) against pairs behind a dense projection, on the corrected evaluation. *Untested.*
- Ternary hyperplanes capped at two non-zeros, one $+1$ and one $-1$: learns pairs through a projection. *Untested.*
- Centred single anchors (subtract a running mean before the sign), to test the offset explanation of [section 2](#2-single-anchor-hashing). *Untested.*
- Mixed address widths (1.3) combined with input projections ([section 4](#4-input-projections)): shared or per-group projection. *Untested.*
- Trainable-anchor retry ([section 7](#7-trainable-anchors)); fixed random Hadamard rotation and hierarchical tables (6.4); LIF hash on a control task. *Untested.*

[report-pdf]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/lutorch_ex/lutorch_ex_report.pdf
[report-tex-onetable-single]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/lutorch_ex/lutorch_ex_report.tex#L111-L229
[report-tex-wrapper]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/lutorch_ex/lutorch_ex_report.tex#L473-L633
[report-tex-inproj]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/lutorch_ex/lutorch_ex_report.tex#L499-L556
[report-tex-onetable]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/lutorch_ex/lutorch_ex_report.tex#L111-L229
[manifesto]: https://arxiv.org/abs/2512.11843
[lffn-arxiv]: https://arxiv.org/abs/2403.07221
[lffn-pmlr]: https://proceedings.mlr.press/v202/zeng23a.html
[lf-sampler]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/anchor_sampler.py
[lf-gt-index]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lut_fused/tests/gt_lut_transformer.py#L111-L140
[lt-apl]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/anchor_pairs_lookup.py
[lt-hyp]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/hyperplane_multi_head_lut.py
[lt-comp]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/compression_mhl.py
[lt-wta]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/wta_lookup.py
[lt-mwta]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/multi_head_wta.py
[lt-lif]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch/lif_multi_head_lut.py
[ex-addr]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/addressing.py
[ex-anchors]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/anchors.py
[ex-spec]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/lut_spec.py
[ex-proj]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/projection.py
[ex-mbase]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/cartridges/manifesto_base.py
[ex-single-test]: https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex/tests/test_single_anchor.py
[cal-multinap]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/nanochat_exps/exp387_multinap_out_proj/tiny_multi_nap_multi_head_lut.py
[cal-trainanch]: https://github.com/anatoli-starostin/spiky/blob/e85a5ceca156e79d199b0f6b395aa8e7e4ee287a/src/spiky/lutorch/trainable_anchors_multi_head_lut.py
[hyp-readme]: https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/README.md
[hyp-ternary]: https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/src/spiky/lutorch/ternary_hyperplane_multi_head_lut.py#L206-L807
[talut-tex]: https://github.com/anatoli-starostin/spiky/blob/116a2190ab7d59939953cb33a810aec2bda3703f/doc/lutorch/ffn_replacement/ternary_anchoring.tex
[talut-pdf]: https://github.com/anatoli-starostin/spiky/blob/116a2190ab7d59939953cb33a810aec2bda3703f/doc/lutorch/ffn_replacement/ternary_anchoring.pdf
[frf-light]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/light_multi_head_lut.py
[frf-bh4]: https://github.com/anatoli-starostin/spiky/blob/c4a750b2f9ab65efe6a61254103a92d6fd8fc2a2/src/spiky/lutorch/bh4_multi_head_lut.py
[lffn-py]: https://github.com/anatoli-starostin/spiky/blob/93de410db25f3b139a0539c8a17b395e49043d5a/research/lookupffn/lookup_ffn.py
[lffn-readme]: https://github.com/anatoli-starostin/spiky/blob/93de410db25f3b139a0539c8a17b395e49043d5a/research/lookupffn/README.md
[lif-readme]: https://github.com/anatoli-starostin/spiky/blob/ccd0616c84e0efbe71c2177368f68b48512fc456/experiments/lif_detectors_mhl/README.md
[lif-math]: https://github.com/anatoli-starostin/spiky/blob/ccd0616c84e0efbe71c2177368f68b48512fc456/experiments/lif_detectors_mhl/mhl_detectors_math.tex
[ferns-2007]: https://doi.org/10.1109/CVPR.2007.383123
[ferns-2010]: https://doi.org/10.1109/TPAMI.2009.23
[hyp-analysis]: https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/exp010_hyperplane_full_model_16k/analysis/ANALYSIS.md
[hyp-geom]: https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/exp023_singlestream_linear_unembedder/analysis/geometry_anchor_vs_random.json
