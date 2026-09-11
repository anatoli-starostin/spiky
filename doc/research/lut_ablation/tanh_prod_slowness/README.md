# Why exp_g_0244 (tanh_margin) took 1.721 h vs exp_g_0193's 0.92 h

Diagnosis only. Nothing was changed to fix it. Measured on gpustar (RTX 5090, torch 2.9.1+cu130),
with LD_LIBRARY_PATH set for NVRTC.

## Wall clock from the checkpoint timestamps

Minutes per 4K steps, uniform across all four blocks:

| run | min / 4K steps | hours |
|---|---|---|
| exp_g_0193 margin | ~14 | 0.92 |
| exp_g_0243 min_margin | ~15 | 0.955 |
| exp_g_0244 tanh_margin | ~26 | 1.721 |

exp_g_0243 used the same torch no-grad eval fallback as 0244 and was not slow.

## `bench_forms.py` (`bench_forms.log`)

The exp_g_0193 model rebuilt per form on CUDA, timing train.py's step (4×12×512 fwd+bwd, clip,
AdamW), the no-grad bs48 eval forward, and the isolated score:

| form | train step | eval fwd bs48 | score fwd+bwd (one call) | est. 16K run |
|---|---|---|---|---|
| margin | 187.6 ms | 38.1 ms (native kernel) | 1.50 ms | 0.867 h |
| min_margin | 195.1 ms | 62.3 ms (torch fallback) | 1.72 ms | 0.923 h |
| tanh_margin | **391.7 ms** | 61.2 ms (torch fallback) | **25.98 ms** | 1.795 h |
| sharp_margin γ=1.75 | 185.4 ms | 62.2 ms | 1.49 ms | 0.879 h |
| sharp_margin γ=3 | 186.2 ms | 62.1 ms | 1.39 ms | 0.883 h |

The estimates reproduce the actual runs (0.92 / 0.955 / 1.721 h). The eval fallback costs
~24 ms × 3,200 eval batches ≈ 0.02 h, which is negligible. The whole gap is in the training step.

## `diag_prod.py` and `diag_prod2.py` — the cause

On one micro-batch of one layer, d [6144, 4, 128, 8]:

* `torch.prod(-1)` forward costs 0.10 ms, so the NVRTC-JIT forward kernel is not the problem.
* torch.prod's **backward** is data-dependent. It checks the input for exact zeros (`aten::eq`
  plus an `item()` sync). With none, it takes the cheap division path; with any, it takes a
  safe-zeros path built from two `cumprod` scans with flips and cats over the whole tensor.
  `cumprod(-1)` alone costs 11.3 ms on this shape.
  * no zero: prod fwd+bwd **0.78 ms**
  * **one** exact zero among 25,165,824 factors: **25.71 ms** (33×)
* tanh(2m) is exactly 0 iff the margin d is exactly 0, i.e. z[a] == z[b] in float32. There are no
  a == b anchor pairs and no exact zeros in the code z, but coordinate collisions occur at about
  4e-8 per margin:
  * init model, one random-token micro-batch: 2, 1, 0, 1, 0, 1 exact-zero margins in L0–L5
  * trained exp_g_0244 checkpoint, 4 val rows: 1 in 50M

  That is of order one per layer per 25M-margin micro-batch, so a large share of the 24 score
  backward calls per step take the slow path.
* **Causal A/B on the real model, same process and data:** stock tanh_margin step 371.8 ms;
  tanh product replaced by exp(Σ log(clamp(t, 1e-30))) 201.9 ms; stock again 380.6 ms. Parameters
  stayed finite throughout.

  An earlier unclamped swap (in diag_prod.py) produced NaN parameters from log(0) at those same
  exact zeros. That removed the zeros and made the later "stock again" measurement fast, which is
  itself further evidence for the zero trigger.

**Cause:** autograd's torch.prod backward switches to its cumprod-based zero-safe path whenever any
factor in the tensor is exactly 0, and tanh_margin produces exact-zero factors from rare float32
coordinate collisions. It is not the eval fallback, not the NVRTC forward kernel, and not the analytic
prefix/suffix cumprod in `_confidence_score_and_dscore`, which only FastMultiHeadLut uses.
sharp_margin and min_margin do not use torch.prod and are unaffected. exp_g_0245 took 0.913 h.
