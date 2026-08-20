# exp_n_0054 — soft-forward annealing with adaptive handoff to the hard FastMHL, H8/d48/tph64, 16k

> **RESULT: final_val_bpb = 1.2573194 (best = final; 16k, 1.05 h). Soft-forward annealing + handoff HURTS —
> worse than plain hard/soft training.** vs the hard-forward control **exp_n_0052 (1.2285517): +0.0288**; vs
> dense (1.196646): +0.0607. Handoff fired exactly at step 8000 (top1_threshold; mean top-1 mass 0.9446, temp
> 0.171, soft→hard logit gap 0.050). The soft phase left the hard-deployable metric far behind (hard-eval bpb
> ~1.35 at handoff vs exp_n_0052's ~1.28 at the same step); the 8000-step hard phase recovered to 1.257 but
> never caught up. **Takeaway: the soft weighted-sum forward optimizes a blended objective that ≠ the hard
> argmax lookup used at eval, so the soft phase is ~wasted for the hard metric — it's better to train the real
> hard-forward/soft-backward (STE) path from step 0.** The gap trajectory (below) confirms the mismatch persists
> even at high top-1 mass (gap still 0.05 at top-1 0.94).

## Idea
`FastMultiHeadLut` normally does a **hard** gather on the forward and only uses the soft full-K surrogate on the
**backward** (STE). Here the forward IS the differentiable **softmax-weighted sum over all K=64 rows** per table
(plain autograd, no straight-through). Both temperatures anneal from a soft init toward a small floor, so the
blend sharpens over training. **But we do not ride the anneal to a dead floor:** each step we measure the mean
top-1 softmax mass (how concentrated the blend is = how close soft output is to the hard argmax lookup), and once
it crosses a threshold we **hand off** — flip every slot to the *real* FastMHL hard-forward/soft-backward path and
finish training with the standard STE surrogate. This gets differentiable, well-conditioned early optimization
from the soft path, then the exact discrete lookup (and its STE) for the sharp regime.

## Mechanism (soft forward, faithful to FastMHL's surrogate)
```
d         = x[:, anchor_a] - x[:, anchor_b]
soft_sign = d / (T_soft + |d|)                 # -> sign(d) as T_soft -> 0
score_k   = <bit_matrix[:,k], soft_sign>       # agreement with row k's ±1 cluster code
w         = softmax(score / T_sel, dim=K)      # -> one-hot(argmax) as T_sel -> 0
out_table = Σ_k w_k · weights[table, k, :]     # bag-summed over tables_per_head
```
For the argmax row FastMHL's pinned surrogate `p = d/(T_soft+|d|)` equals `soft_sign`, so this is the same math,
unpinned over all K and run on the forward.

## Adaptive handoff
`SoftAnnealLut` (experiment-local, torch.compile'd) **wraps** a real per-head FastMHL and **shares its weight /
anchor / bit_matrix tensors** — so the soft phase trains the FastMHL's own `weights`, and at handoff there is
nothing to copy: we just flip `mode` to delegate to `fmhl(x)` (real hard forward + STE backward), seeding the
FastMHL's learnable log-temps to the current annealed value for continuity. **Eval always reports the TRUE
hard-eval bpb** — every eval forces the FastMHL hard path (even during the soft phase), so val bpb is always the
deployable number.

**Handoff timing (full-run scale adjustment):** temps anneal exponentially `soft_anneal_temp_start=0.5 →
soft_anneal_temp_floor=0.02` over a **stretched horizon `soft_anneal_over_steps=24000`** (> the 16k training
steps), so the blend sharpens gradually and top-1 mass approaches the threshold near mid-run rather than early.
Handoff fires when mean top-1 mass ≥ `handoff_top1_threshold=0.85` **and** `step ≥ handoff_min_step=8000` (guard
against too-early handoff), and is **force-fired at `handoff_max_step=10000`** even if the threshold is not
reached. With this schedule temp≈0.17 (top-1≈0.93) at step 8000, so the handoff lands at ≈step 8000 (~50% of
training). A per-eval soft-vs-hard logit-gap trajectory is logged through the soft phase.

## No shared-module edits
`fast_multi_head_lut.py` / `compression_mhl.py` are untouched. `train.py` builds the standard CompressionMHL
model (per-head loop path) then wraps each `block.ffn.luts[h]` FastMHL in a `SoftAnnealLut`. LUT tables + temps
stay in the nodecay optimizer group (they live in the wrapped FastMHL). Init is bit-identical to exp_n_0052's
forward slot.

## Config
H8/d48/tph64/nap6, device_bs 48, grad_accum 1, 16000 steps, warmup 1600, seed 1, clean val 245,760 — the same
rung as **exp_n_0052 (1.2285517)**. λ n/a (no recon). Params **27,343,200** (standard rung; the wrapped FastMHL
keeps its per-head temps for the post-handoff STE phase).

## Smoke test (60 steps, full model size) — all 5 checks pass
- **(a)** soft forward is pure autograd, grads reach the LUT weights (no STE): `fmhl.weights.grad` norm 5.5e-4,
  nonzero_frac **1.0** (all K rows get gradient), `grad_fn=CompiledFunctionBackward` (not a custom STE Function).
- **(b)** temps anneal (0.47→0.21…) and mean top-1 mass climbs (0.316→0.851).
- **(c)** torch.compile builds cleanly.
- **(d)** handoff fired at step 16 (top-1 crossed 0.85); post-handoff FastMHL path trained and eval'd sanely
  (hard_bpb 2.36 at the compressed smoke schedule).
- **(e)** at handoff the soft→hard end-to-end logit gap was **0.00068** (soft ≈ hard).
