# Ternary routing

Hyperplane routing whose weights are only $-1$, $0$ or $+1$: the address is then computed with adds and subtracts alone. The page tells the whole story: the construction, the standardised variant, and what we found about making the weights sparse. The place of ternary routing among the other address mechanisms is in [Hashing mechanics 5](hashing.md#5-ternary-approach); the write-up of the idea is the paper *Ternary Anchoring Lookup Tables* ([LaTeX source][talut-tex], [PDF][talut-pdf]). The evidence is summarised in short paragraphs that link to the [Architectures](../architectures/README.md) pages. All runs are single seeds with the old evaluation (see [`CONVENTIONS.md`](../../CONVENTIONS.md)).

## Contents

| Section | Covers |
| --- | --- |
| [1. Construction](#1-construction) | ternary weights with a straight-through estimator, options |
| [2. Standardised weights](#2-standardised-weights) | standardise, then ternarise |
| [3. Sparsity](#3-sparsity) | the three ways to force zeros (3.1), a penalty weight under Adam (3.2), results (3.3) |

## 1. Construction

*Code [`ternary_hyperplane_multi_head_lut.py`][hyp-ternary] (subclass of the free-hyperplane module, branch `hyperplane_ffn_next`). Write-up: *Ternary Anchoring Lookup Tables* ([LaTeX source][talut-tex], [PDF][talut-pdf]). Not in [`lutorch_ex`](https://github.com/anatoli-starostin/spiky/tree/3ac72cca40c63b77d509d4ccff33bb85d43e515e/src/spiky/lutorch_ex).*

Constrain each hyperplane's weights to $\{-1,0,+1\}$: an anchor pair is the ternary vector with one $+1$ and one $-1$, so ternary strictly generalises pairs, and after training the test is a sum of some inputs minus a sum of others, with no multiplication. Weights are learned through a straight-through estimator. With shadow weight $w$ and a per-table temperature $T$:

1. Smooth value: $s = \tanh(w/(2T))$.
2. Rounded value: $q = +1$ if $s > 1/2$, $q = -1$ if $s < -1/2$, otherwise $q = 0$.
3. Straight-through value: $q_\text{ST} = s + \mathrm{stopgrad}(q - s)$.

Reading: the forward pass uses the rounded $q$, the backward pass differentiates the smooth $s$. The weight is zero exactly when $\lvert w\rvert\le T\ln 3$ (dead zone). The bit is $[\langle q,x\rangle(+b)>0]$ with an optional trainable bias. At deployment `bake_ternary_weights()` stores $q$ in two bits per entry and the test is $\sum_{q_i=+1}x_i-\sum_{q_i=-1}x_i$ against $-b$: adds and subtracts only.

**Options.** `normalize_weights` standardises each hyperplane's weights so that $T$ alone sets the density ($T=0.392$ gives a third each of $-1,0,+1$ for Gaussian weights), see 2. A density penalty on the mean of $\lvert\tanh\rvert$ with a target number of non-zeros per hyperplane controls sparsity, see 3. Very sparse hyperplanes hurt; very dense ones approach continuous ones.

## 2. Standardised weights

This works in our runs, with the old evaluation. Each hyperplane weight vector is standardised to unit scale in every forward pass before it is ternarised, so only the direction trains, and the threshold, density and score divisor follow from the width with no tuned constants. The run is behind for most of training with heavy sign churn and overtakes [exp_g_0033](https://github.com/anatoli-starostin/spiky/tree/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/exp_g_0033_ternary_balanced_trainable_bias_4k) at the end, so a mid-run snapshot looks like a failure. Multiplication-free routing came within 0.0009 of learned continuous hyperplanes. [exp_g_0036](https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/exp_g_0036_ternary_normalize_w_4k/README.md), [exp_g_0037](https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/exp_g_0037_ternary_normw_decompress_heads_4k/README.md), [TERNARY_16K_SWEEP](https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/TERNARY_16K_SWEEP.md).

<details>
<summary>Ternary hyperplanes: one-axis-at-a-time sweep at 16k steps</summary>

The ternary one-axis-at-a-time sweep at 16k steps (fast test scale, old evaluation, single seeds, [sweep](https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/TERNARY_16K_SWEEP.md)): the baseline with 192 non-zero weights per hyperplane reached 1.1894 and the best stacked point (8 heads, no sparsity penalty) 1.1820, against 1.2014 for the dense baseline and 1.1915 for the anchor-pair compression anchor of the paper; every axis rewarded more capacity or freedom.

</details>

## 3. Sparsity

The zeros of a ternary hyperplane are free to skip at deployment, so the question was how many weights can be forced to zero before quality pays for it.

### 3.1 Three ways to force zeros

Three ways were tried, all on the same recipe. (1) A penalty on a smooth count of the non-zero weights, pushing the density down. (2) The same penalty as a hinge: it acts only while the density is above a chosen target, so the density settles near the target and stays there. (3) Hard top $m$ ("train what you deploy"): in every forward pass each hyperplane keeps only the $m$ components with the largest confidence, meaning how far $\lvert w\rvert$ lies beyond the dead zone, and sets all others to zero. The kept ones take their ternary sign. The gradient still reaches every component, the masked ones included, so a masked weight can grow and displace a weaker one later. The density is then exactly $m$ per hyperplane at all times, and training and deployment compute the same function. (The letter $m$ is used because $K$ is the number of table rows elsewhere in the wiki.)

### 3.2 A penalty weight has no effect under Adam

Setup: the sparsity penalty of 3.1 adds a term $\lambda \cdot \text{density}$ to the loss, where density is a smooth count of the non-zero ternary routing weights. Two runs differed only in $\lambda$ (100 and 50), everything else equal. The expectation was that halving $\lambda$ would halve the pressure and the density would fall about half as fast. It did not: the density followed the same trajectory in both runs.

The reason is the optimiser. Adam divides each parameter's update by a running estimate of that parameter's own gradient size. If the penalty were the only force on a weight, multiplying it by a constant would change the gradient and the divisor by the same factor, and the update would stay the same. With the task loss present the cancellation is only partial, but the sweep saw no visible effect from the factor of two. So under Adam $\lambda$ is not a usable knob for how sparse the weights become.

What does control density is a target. In 3.1 the penalty is applied as a hinge that switches on only while the density is above a chosen target, so the density reaches the target and stays there. A plain one-way penalty has no stopping point and kept pushing the density down for the whole run. This was observed for the sparsity penalty only, in one pair of runs; other penalties (the TV penalty of [Regularisation 2](../optimisation/regularisation.md#2-total-variation-penalties), the sine penalty of [Regularisation 3.2](../optimisation/regularisation.md#32-a-sine-penalty-pulling-weights-to-a-quantisation-grid)) were not checked for the same effect. [sparsity sweep summary](https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/SPARSITY_SWEEP_SUMMARY.md), [exp_g_0039](https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/exp_g_0039_ternary_sparsity_lam100_4k/README.md), [exp_g_0040](https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/exp_g_0040_ternary_sparsity_lam50_4k/README.md).

### 3.3 Results

None of these helped quality (old evaluation); sparsity is only a price paid for cheaper routing. At 4k steps a hinge on a target of 192 looked almost free, but over the full 16k anneal sparsity is a pure quality cost that grows as the target tightens, and no penalty was best. Hard top $m$ training beat pruning to the same density after training only at 64 non-zeros (it lost at 128), and the penalty with a target matched or beat it with fewer non-zero components. The penalty does work for density control, but only through a target, see 3.2. [SPARSITY_SWEEP_SUMMARY](https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/SPARSITY_SWEEP_SUMMARY.md), [TERNARY_16K_SWEEP](https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/TERNARY_16K_SWEEP.md), [exp_g_0045](https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/exp_g_0045_ternary_topk64_4k/README.md).

<details>
<summary>Ternary hyperplanes: sparsity target of 64 to 256 non-zero weights per hyperplane</summary>

Target number of non-zero weights per hyperplane of 64, 128, 192, 256 and no penalty gave 1.2094, 1.2046, 1.1894, 1.1885 and 1.1876 at the full 16k schedule (old evaluation, single seeds): the hinge is a pure cost in loss, and buys about 172 non-zeros per hyperplane against 269 without it ([sweep](https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/experiments/hyperplane_ffn/TERNARY_16K_SWEEP.md); [FFN replacement, section 2](../architectures/transformers/ffn-replacement.md#2-learned-and-ternary-routing-instead-of-anchor-pairs-july-to-september-2026-old-evaluation)). The earlier 4k-step reading that sparsity is free did not hold at the full schedule.

</details>

[talut-tex]: https://github.com/anatoli-starostin/spiky/blob/116a2190ab7d59939953cb33a810aec2bda3703f/doc/lutorch/ffn_replacement/ternary_anchoring.tex
[talut-pdf]: https://github.com/anatoli-starostin/spiky/blob/116a2190ab7d59939953cb33a810aec2bda3703f/doc/lutorch/ffn_replacement/ternary_anchoring.pdf
[hyp-ternary]: https://github.com/anatoli-starostin/spiky/blob/0c35e3be8deab30338681013158ecbd166548764/src/spiky/lutorch/ternary_hyperplane_multi_head_lut.py#L206-L807
