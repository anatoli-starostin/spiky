# Sparsity regularisation sweep — where routing sparsity stops being free

Six runs on the **exp_g_0037** recipe (ternary routing + per-head decompress, 76,373,004
params, 1.346459 at 4k), each differing from it only in the sparsity penalty. The question:
the ternary router uses ~269 of each hyperplane's 384 components — **how many can be forced
to zero before validation bpb pays for it?**

| run | regime | nnz/hyperplane | frac_zero | val bpb @ 4k | vs lambda 0 |
|---|---|---|---|---|---|
| exp_g_0037 | no penalty (reference) | 269.04 | 0.2994 | 1.346459 | — |
| exp_g_0044 | target 192 | 161.58 | 0.5792 | 1.348412 | +0.001953 |
| exp_g_0043 | target 128 | 84.71 | 0.7794 | 1.358134 | +0.011675 |
| exp_g_0042 | target 64 | 47.85 | 0.8754 | 1.365208 | +0.018749 |
| exp_g_0039 | one-way lambda 100 | 21.44 | 0.9442 | 1.361572 | +0.015113 |
| exp_g_0040 | one-way lambda 50 | 21.37 | 0.9443 | 1.362309 | +0.015850 |

## The frontier

**Sparsity is free down to ~162 non-zeros per hyperplane.** At 161.58 — 1.7x sparser than
unregularised — the cost is +0.0020, inside what a single eval moves (exp_g_0038 and
exp_n_0121 are the same configuration on two machines and differ by 0.0015). At 84.71 it is
+0.0117, six times that and unambiguously real. **The boundary lies between 85 and 162.**

From 48 non-zeros upward the frontier is monotone — more density, less cost. The single
inversion is at the low end: **exp_g_0042 holds 2.2x more non-zeros than the one-way runs
and still costs more** (+0.0187 against +0.0151).

![frontier](sparsity_frontier.png)

## Four findings

**1. Lambda is a no-op knob under AdamW.** exp_g_0040 (lambda 50) reproduced exp_g_0039's
(lambda 100) density trajectory to five decimal places while its penalty value was exactly
halved; the finals differ by 0.0007 bpb and 0.07 non-zeros of 384. Adam divides each
update by that parameter's own gradient magnitude, so wherever the penalty dominates,
lambda cancels in `m / sqrt(v)`. A planned lambda=10 rung was dropped as redundant.

**2. A target-density hinge has a stable fixed point; the one-way push does not.** Every
target reached its target by step 200–400, released, and then *held* — exp_g_0042 sat at
48.5 → 47.9 over 3,200 steps while the one-way penalty ground 30 → 21 over the same span.
Density has to be controlled by a target, not by lambda.

**3. The soft target undershoots the hard count, and not by a constant.** 192 → 161.58
(0.84x), 128 → 84.71 (0.66x), 64 → 47.85 (0.75x). The runs settle where the *surrogate*
equals the target, and the surrogate sits below the hard density by an amount that depends
on how the weights are distributed around the dead zone. A wanted hard count cannot be
obtained by scaling the soft target.

**4. The cost tracks how frozen the router is, better than it tracks density.** Final
routing changes per eval:

| run | regime | churn/eval | T drift | cost |
|---|---|---|---|---|
| exp_g_0037 | no penalty | 1,029,846 | 0.993x | — |
| exp_g_0044 | target 192 | 827,778 | 1.076x | +0.0020 |
| exp_g_0043 | target 128 | 166,796 | 1.031x | +0.0117 |
| exp_g_0042 | target 64 | 888 | 1.195x | +0.0187 |
| exp_g_0039 | one-way lambda 100 | 2,736 | 0.971x | +0.0151 |
| exp_g_0040 | one-way lambda 50 | 2,923 | 0.985x | +0.0158 |

exp_g_0042 is the most frozen run on the board and the most expensive, despite holding more
density than either one-way run. Its T also rises the most (1.195x): the penalty has T
detached and cannot widen the band, but after release the *task* gradient is the only thing
acting on T and nothing holds it, so part of the held sparsity is band-widening rather than
weight structure. **Letting T move after release is the obvious next variable.**

## Numbering

exp_g_0041 was to be the lambda=10 rung and was dropped before launch once exp_g_0040
showed the ladder answered nothing. The number is left as a deliberate gap; see
`exp_g_0041_DROPPED_lam10_redundant/DROPPED.md`.

## Caveat

The run stops at **94% of peak LR**, having traversed ~6% of the cosine — the schedule is
anchored to 16,000 steps and merely stopped at 4,000. exp_n_0121 goes on to 1.1915 by step
16,000. Every number here is an early-trajectory reading. It is fair *relative* to
exp_g_0037, which is identical in every respect but the penalty, but a full-anneal verdict
is not established by it.
