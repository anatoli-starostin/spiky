# exp_g_0047 — trained hard top-192 per hyperplane

**Result: `final_val_bpb = 1.3559372`, exactly 192.00 non-zeros per hyperplane,
76,373,004 params, 0.58 h.**

| run | K | nnz/hyperplane | val bpb @ 4k | vs no-penalty exp_g_0037 |
|---|---|---|---|---|
| exp_g_0045 | 64 | 64.00 | 1.366177 | +0.019718 |
| exp_g_0046 | 128 | 128.00 | 1.358295 | +0.011836 |
| **exp_g_0047** | **192** | **192.00** | **1.355937** | **+0.009478** |

"Train what you deploy": every forward keeps exactly the top-192 components of each
hyperplane, ranked by confidence `|normalized_weight()|` — how far past the dead-zone
boundary the component sits — and takes their ternary sign. The same mask applies in the
hard eval path, so train-mode and deploy-mode are the same function.

**Density is pinned exactly.** `nonzero_per_hyperplane` took a single distinct value
across all 21 evals: 192.0000, with `frac_zero` 0.500000 =
1 − 192/384 at every row.

## What it has to beat

| comparison | nnz/hyperplane | val bpb | exp_g_0047 − it | verdict |
|---|---|---|---|---|
| post-hoc prune to 192 | — | — | — | not applicable: 192 exceeds exp_g_0044's natural 161.58, so trimming to it removes nothing |
| exp_g_0044 (target 192) | 161.58 | 1.348412 | +0.007525 | **penalty wins, with fewer components** |

## The straight-through estimator is what makes this trainable

```python
return s + (self._q_hard_from(s) - s).detach()
```

The mask sits **inside** the `.detach()`, so it changes the forward VALUE and never
multiplies the backward. Every component — masked-out included — receives the full
gradient its position earns, can grow, and can displace a weaker one on a later step.
Masking the backward instead would freeze each excluded component at whatever value it
held when it dropped out, and the set could only ever shrink. Verified in the module's
sanity block (7c): 100% of masked-out components receive gradient, and 25 Adam steps moved
64 components into the top-K from outside.

## Does the top-K set settle?

The set is 192 × 24,576 = **4,718,592 components** of the 9,437,184 total.
Churn below is normalised by that set, not by all slots — a larger K can move more
components while replacing a smaller share of its own routing, so "% of all slots" would
rank the runs wrongly.

| step | entered the set | % of set replaced | % of set differing from init |
|---|---|---|---|
| 800 | 1,294,066 | 27.4% | 48.5% |
| 1,600 | 1,106,829 | 23.5% | 49.9% |
| 2,400 | 805,555 | 17.1% | 50.0% |
| 3,200 | 698,192 | 14.8% | 50.0% |
| 4,000 | 656,249 | 13.9% | 50.0% |

It falls from 32.8% to 13.9% of the set replaced per eval, and over the
last five evals it moved -6.0%: **still decaying**. The distance from the initial mask
plateaus, so the set is not travelling anywhere — it churns in place. For scale, the
hinge-trained exp_g_0042 ends at 0.08% turnover.

## A kept component is ±1 even inside the dead zone

That is deliberate, and it is the likeliest reason this family underperforms. The
alternative — intersecting the top-K with the dead-zone survivors — would give "at most K",
a count that drifts with T. Exactly-K was chosen so the deployed count is fixed.

The cost is that top-K **cannot emit zero as a decision**. It must nominate exactly K
components and give each of them full ±1 weight, however unconfident it is about them. The
penalty-trained runs keep the third state and can spend it: a component they are unsure
about simply quantises to 0. The zero is not merely absence, it carries information, and
top-K trades it away for a fixed count.

## Trajectory

| step | val bpb | vs exp_g_0037 | nnz/hp | score/T | T_mean |
|---|---|---|---|---|---|
| 800 | 1.8704 | +0.0001 | 192.00 | 1.7286 | 0.389097 |
| 1,600 | 1.6358 | +0.0041 | 192.00 | 1.5449 | 0.379467 |
| 2,400 | 1.5033 | +0.0148 | 192.00 | 1.2855 | 0.364205 |
| 3,200 | 1.4096 | +0.0131 | 192.00 | 1.0798 | 0.348012 |
| 4,000 | 1.3559 | +0.0095 | 192.00 | 0.9219 | 0.331242 |

## What is held fixed

A fork of **exp_g_0037** (1.346459 at 76,373,004 params) with the top-K as the only
change: `decompress_heads=True` with `inner_out=48`, `normalize_weights=True`,
`T = max_entropy` (0.392065, derived), divisor `sqrt_expected_nonzero` (16, derived),
`trainable_bias=True`, random init, n_heads 4, nap 8, tph 128 — same seed, same data, same
batch, same held 16,000-step LR schedule stopped at 4,000. **The sparsity penalty is off**
(lambda 0, target 0): top-K sets the density directly, so the hinge has nothing to do.
Parameter count unchanged at 76,373,004 — `topk_per_hyperplane` is a plain int and adds no
parameters and no `state_dict` entries.

## Caveat, as on every run on this board

The run stops at **94% of peak LR**, having traversed ~6% of the cosine — the schedule is
anchored to 16,000 steps and merely stopped at 4,000. exp_n_0121 goes on to 1.1915 by step
16,000. Every number here is an early-trajectory reading, fair *relative* to exp_g_0037,
which is identical but for the routing constraint.

![loss](loss.png)
