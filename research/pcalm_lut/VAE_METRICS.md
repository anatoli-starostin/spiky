# What the VAE literature reports on Fashion-MNIST, and whether our MSE can join it

Searched 2026-09-23. Every number below is quoted from the paper's own table via its full text, not
recalled. Where a number does not exist in the literature, this file says so rather than approximating.

## The short answer

**Our MSE cannot be placed on the scale the VAE literature uses, and the obstacle is not a missing
constant — it is a different data distribution.** The dominant Fashion-MNIST and MNIST likelihood
numbers are computed on **dynamically binarised** images under a **Bernoulli** likelihood. Those pixels
are 0 or 1, ours are continuous. No choice of Gaussian variance converts between them, because the two
models are not scoring the same random variable.

The one quantity that would be comparable — PSNR — is **not reported by any VAE paper found here**.

## 1. Which metrics the literature actually reports

**Dominant, by a wide margin:** test negative log-likelihood in **nats per image**, estimated with an
importance-weighted bound (IWAE with 100–5000 samples). This is the currency of the whole VAE
literature on MNIST-family data.

**Common:** bits per dimension (BPD), but almost exclusively for *colour* datasets — CIFAR-10, CelebA,
ImageNet 32×32. MNIST is reported in nats, not BPD, because it is binarised.

**Occasional:** FID, and the numbers are not trustworthy across papers — see §4.

**Rare, to the point of near-absence in papers:** plain per-pixel reconstruction MSE with a stated
latent dimension and a stated pixel scale. This is what we measure. It lives in blog posts and
tutorials, not in the papers.

**How much of the literature is Bernoulli/binarised:** all four of the primary sources below.
IWAE binarises by sampling (Salakhutdinov & Murray 2008); NVAE uses "dynamically binarized MNIST" with
a Bernoulli likelihood and switches to discretized logistic *for every other dataset*; VampPrior uses
Bernoulli for binary data; Exemplar VAE reports Fashion-MNIST under a Bernoulli likelihood. So the
Fashion-MNIST number that looks most citable (−228.70 nats) is a number about *binary* Fashion-MNIST.

## 2. The numbers, with the context that makes them mean anything

| paper | dataset | metric | value | latent | architecture | likelihood |
|---|---|---|---|---|---|---|
| Exemplar VAE | **Fashion-MNIST** | test NLL (IWAE-5000) | VAE **−228.70 ± 0.15**, VampPrior **−227.35 ± 0.05**, Exemplar **−226.75 ± 0.07** | **40** | MLP, 2×300 hidden | Bernoulli |
| Exemplar VAE | dynamic MNIST | test NLL (IWAE-5000) | VAE −84.45, VampPrior −82.43, Exemplar −82.09 | 40 | MLP, 2×300 | Bernoulli |
| Exemplar VAE | Omniglot | test NLL (IWAE-5000) | VAE −108.34, VampPrior −106.78, Exemplar −105.22 | 40 | MLP, 2×300 | Bernoulli |
| IWAE | binarised MNIST | test NLL | VAE k=1 **86.76**, k=50 86.35; IWAE k=5 85.54, **k=50 84.78** | 50 (1 layer) | MLP, tanh | Bernoulli |
| IWAE | binarised MNIST | test NLL, 2 stochastic layers | VAE k=1 85.33, k=50 84.78; IWAE k=50 **82.90** | 100 + 50 | MLP, tanh | Bernoulli |
| VampPrior | static MNIST | test LL | VAE (L=1)+VampPrior −85.57; HVAE (L=2)+VampPrior −83.19 | 40 (per layer) | MLP 2×300, gated | Bernoulli |
| VampPrior | dynamic MNIST | test LL | HVAE (L=2)+VampPrior **−81.24** | 40 | MLP 2×300 | Bernoulli |
| VampPrior | Omniglot / Caltech101 | test LL | −101.18 / −108.28 | 40 | MLP 2×300 | Bernoulli |
| NVAE | dynamic MNIST | test NLL | **78.01** (78.19 with flows) | hierarchical | deep conv hierarchy | Bernoulli |
| NVAE (table 1) | dynamic MNIST | test NLL, others | BIVA 78.41, IAF-VAE 79.10 | — | — | Bernoulli |
| NVAE | CIFAR-10 / CelebA-64 | BPD | 2.91 / 2.03 | hierarchical | — | discretized logistic mixture |

**No Fashion-MNIST in VampPrior, NVAE or IWAE.** Fashion-MNIST is simply not a standard benchmark in the
likelihood-based VAE line; MNIST and Omniglot are. Exemplar VAE is the exception that makes the table.

### The closest paper, read properly — and it is NOT the experiment it first looks like

*Stochastic Bottleneck: Rateless Auto-Encoder for Flexible Dimensionality Reduction* (arXiv:2005.02870,
Koike-Akino & Wang) is the only paper found reporting reconstruction MSE against latent dimension on
MNIST-family data. On a first reading it looks like our experiment. It is not, and the difference is
the whole point of the paper.

Their goal is a **rateless** autoencoder: ONE model, trained at M=64, that can be truncated to any
L ≤ 64 at inference without retraining. The mechanism is **TailDrop**, a non-uniform dropout that drops
a consecutive run of tail units, with the number dropped following a power CDF Pr(D < τM) = τ^β
(β = 0.67 for MNIST, 2.1 for CIFAR-10). Head units are almost never dropped and tail units usually are,
which forces a PCA-like importance ordering onto the latent coordinates.

Table 1 in full — both columns, because the second column alone is misleading:

| latent dim L | 4 | 14 | 24 | 34 | 44 | 54 | 64 |
|---|---|---|---|---|---|---|---|
| **Conv. AE** MSE (dB) | 8.18 | 6.91 | 5.03 | 3.01 | 0.56 | −2.33 | **−5.81** |
| **Prop. AE** (rateless) MSE (dB) | 5.16 | −0.05 | −3.00 | −4.35 | −5.00 | **−5.26** | −5.19 |

Caption: "MSE, SSIM, and SVM classification accuracy of SAE and RL-AE, optimized under MSE measure at
dimensionality of M=64 for MNIST datasets". MLP, three fully-connected layers, ReLU, 1024 hidden units
for MNIST; Adam lr 1e-3, batch 100, max 500 epochs with early stopping (patience 20); standard
normalisation, no binarisation; loss is MSE (or negative SSIM in the SSIM-optimised arm).

**This is a truncation-robustness table, not a capacity table.** Both columns are ONE model trained at
M=64, evaluated at truncated L. The conventional AE collapses when truncated (−5.81 → +8.18) because
nothing ordered its latent coordinates; the rateless one degrades gracefully. At the dimension both were
actually trained for, the conventional AE is **better** (−5.81 vs −5.19) — TailDrop costs 0.6 dB at full
width to buy graceful degradation everywhere below it.

So the non-monotonic dip (−5.26 at L=54, −5.19 at L=64) **says nothing about latent capacity ceasing to
help**. It is a property of the rateless arm, whose tail units are trained under heavy dropout. The
paper does not discuss the inversion at all — no explanation appears in the text.

**The dB reference is undefined.** No formula, no reference signal power, nothing in the caption or
surrounding text: the loss is given only as "distortion (e.g., MSE)". −5.19 dB is 0.30 of *something*.
Not convertible, so not numerically comparable to us.

**On PCA, the paper says the opposite of what a short quote suggests.** In full: "It should be noted
that the linear PCA dimensionality reduction performs surprisingly well, competitive to the proposed
nonlinear AE for CIFAR-10 datasets in Fig. 3(b). Because MNIST images are nearly binary bitmaps whose
statistics are far from the Gaussian distribution, PCA did not work well as shown in Fig. 3(a)." The
"surprisingly well" is about **CIFAR-10**; for MNIST they report PCA doing *badly*. That is contrary to
what we observe on Fashion-MNIST, where our linear baselines are strong — though note their MNIST claim
rests on MNIST being nearly binary, which Fashion-MNIST is markedly less so.

PCA numbers appear **only in figures** (Figs. 3 and 4), never in a table, so there are no per-dimension
PCA values to quote. Fashion-MNIST appears only in the supplementary experiments — MSE curves in
Fig. 8(a) and reconstruction snapshots in Fig. 9 — again with **no numerical table**.

## 3. Our numbers on every scale that is well defined

Computed from the saved checkpoints over the full 10 000-image test split by `convert_mse_scales.py`.
Conversion is exact: standardisation is affine, so MSE[0,1] = MSE_std × 0.3081² = MSE_std × 0.09492561,
and MSE[0,255] = MSE[0,1] × 255².

| model | compression | MSE (standardised) | MSE [0,1] | MSE [0,255] | PSNR dB | PSNR dB, clamped |
|---|---|---|---|---|---|---|
| ViT d128 h4 lat128 + aug | 6.125× | 0.02787 | 0.002646 | 172.0 | **25.77** | 25.78 |
| ViT d64 lat64 plain | 12.25× | 0.05193 | 0.004930 | 320.6 | 23.07 | 23.08 |
| linear 784-128-784 | 6.125× | 0.06877 | 0.006528 | 424.5 | 21.85 | 22.12 |
| linear 784-64-784 | 12.25× | 0.11026 | 0.010467 | 680.6 | 19.80 | 20.04 |

The clamped column matters more than it looks. Clamping to [0,1] before scoring moves the ViTs by 0.01 dB
and the linear baselines by 0.24–0.27 dB, because the linear maps overshoot outside the valid pixel range
and the ViTs essentially do not. Any PSNR we ever quote must say which of the two it is; an image-quality
paper would report the clamped one, our training loss minimises the unclamped one.

### Can the MSE be placed on the NLL scale? No.

Under a Gaussian likelihood with fixed σ, per-pixel NLL = MSE/(2σ²) + log(σ√(2π)), so MSE determines NLL
only once σ is fixed — and σ is a free parameter no one in the table above states, **because none of them
use a Gaussian likelihood on these datasets**. Worse than an undetermined constant: their pixels are
binary and ours are continuous, so the two likelihoods score different random variables. Quoting our
0.02787 next to −228.70 nats would be meaningless in both directions.

The honest comparison we can make is the one we already make: our own linear PCA-equivalent floor at the
same compression ratio. Nothing found here improves on it.

## 4. FID on Fashion-MNIST — reported, but do not trust cross-paper

Found values for Fashion-MNIST span **0.97 to 196.34** across sources (xAI-GAN reports ~1.0–1.16 for GAN
variants; VGrow reports 8.75–9.72; a generative autoencoder paper reports 196.34). A three-orders-of-
magnitude spread is not model quality, it is incompatible FID implementations — different feature
extractors, different reference statistics, different sample counts. FID is not usable as a
cross-paper number on this dataset, and we should not adopt it.

## 5. Conventional latent dimensions

From the sources above: **40** (VampPrior, Exemplar VAE), **50** (IWAE one-layer; also the Student's-t
prior VAE paper), **100+50** (IWAE two-layer), and sweeps over **4–64** (Stochastic Bottleneck).
Tutorials and blog sources commonly use 2 (for latent-space scatter plots), 10, 20, 32 and 64.

So the classic VAE operating point on these datasets is **40–50 dims**, and 2 is common for
visualisation. Our **64 is normal-to-large**; our **128 is above anything in this table** — larger than
every latent dimension found in the likelihood-based VAE literature for MNIST-family data. Worth knowing
when reading the 6.125× arm: it is a looser bottleneck than the VAE literature typically studies, not
just looser than our own 12.25× arm.

## Sources

- Exemplar VAE — Norouzi et al., NeurIPS 2020, arXiv:2004.04795
- IWAE — Burda, Grosse, Salakhutdinov, ICLR 2016, arXiv:1509.00519
- VampPrior — Tomczak & Welling, AISTATS 2018, arXiv:1705.07120
- NVAE — Vahdat & Kautz, NeurIPS 2020, arXiv:2007.03898
- Stochastic Bottleneck — Koike-Akino & Wang, arXiv:2005.02870
- Fashion-MNIST — Xiao, Rasul, Vollgraf, arXiv:1708.07747
