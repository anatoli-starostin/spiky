# exp033 — hybrid FFN: dense FFN **+** FastMHL (parallel-sum), 4k steps

A **matched 4096-step A/B against exp032** (fast vanilla baseline, val_bpb **1.39371**).
Everything is byte-identical to exp032 — MinimalGPT + RoPE, d384 / 6 layers / 6 heads /
seq 512, device_bs 48, total_bs 24576, n_steps 4096, lr 3e-4, wd 0.1, warmup 0.1,
eval_every 200, seed 1, vocab 32768, same data — **except the FFN block**.

## The design — augment, don't replace
Each transformer block keeps the standard vanilla structure; the FFN sub-block is turned
into **two parallel paths whose outputs are summed** into the residual:

```
h = ln2(x)
x = x + mlp(h) + fastmhl(h)          # was: x = x + mlp(h)
```

- **Path A — dense FFN** (unchanged): `Linear 384→1536 → GELU → Linear 1536→384`, exactly
  as in exp002/exp032 (incl. the zero-init on the second Linear).
- **Path B — plain FastMultiHeadLUT** (`src/spiky/lutorch/fast_multi_head_lut.py`): the
  **classic fixed-anchor-pair sign-test LUT** — `bit_i = 1[x[a_i] − x[b_i] > 0]` — **NOT**
  HyperplaneMHL, no learned hyperplane addressing. Operates on the same LayerNorm'd input
  `h`, reshaped `[B,T,384] → [B·T,384]`, output `[B·T,1,384] → [B,T,384]`.

Both paths are fully differentiable and trained **simultaneously**: the dense path is a
plain GEMM; the FastMHL soft surrogate backward emits gradients for its LUT tables and,
densely, back into `h`. (Smoke-verified: grad flows through both paths in all 6 layers.)

**Init behaviour:** FastMHL tables use near-zero init (`initial_weights_noise=1e-3`), so at
step 0 `fastmhl(h) ≈ 0` and the block ≈ the vanilla FFN — the LUT path only grows if it
earns its gradient. Clean augmentation.

## FastMHL module + chosen capacity
`FastMultiHeadLut(input_dim, n_heads, n_outputs, n_anchor_pairs, tables_per_head=1, *,
forward_mode="hard", weight_dtype=fp32, use_bf16=True, soft_score_temp=0.5,
select_temp=0.5, learnable_temps=False, random_seed=None, initial_weights_noise=0.001, …)`.
Forward `x:[B,input_dim] → [B,n_heads,n_outputs]`. Params `= n_heads·tph·2^NAP·n_outputs`.

Capacity chosen for path B (mirrors the proven exp007-scale "rich" geometry):

| knob | value | note |
|---|---|---|
| n_heads | 1 | single head, outputs the full 384 |
| n_outputs | 384 | `n_heads·n_outputs = 384 = n_embd` |
| n_anchor_pairs (NAP) | 6 | K = 2⁶ = 64 rows/table (decoder-like NAP≈6) |
| tables_per_head (tph) | 256 | main capacity lever |
| forward_mode | "hard" | the deployable single-lookup primitive |
| use_bf16 | **false** | exp032 actually runs **fp32** (its `compute_dtype:bf16` is unused — no autocast in train.py), so the LUT path is fp32 too, matching |
| init_weights_noise | 1e-3 | near-zero init |
| random_seed | 1000 + layer_idx | decorrelated anchors/init across depth |

Per-layer FastMHL params = 256 · 64 · 384 = **6,291,456**. Temps are buffers
(`learnable_temps=False`) → 0 params.

## Param count
| | params |
|---|---|
| exp032 (vanilla) | 35,792,640 |
| + FastMHL ×6 layers | +37,748,736 |
| **exp033 total** | **73,541,376** |

(Verified: FastMHL 6,291,456/layer ×6 = 37,748,736; total confirmed by build smoke.)

## One optimizer note (a decision, overridable)
The only change outside the block: the new **FastMHL table weights are routed to the
no-weight-decay AdamW group** (`weight_decay=0`), following the project's standing lesson
that LUT tables take no weight decay (near-zero init; wd fights the sparse table gradient).
**Every dense-path parameter is grouped exactly as in exp032** (2-D → wd 0.1, 1-D → wd 0);
same AdamW, same betas (0.9, 0.95), same warmup+cosine schedule. If you'd rather the LUT
tables also see wd 0.1 (strict "identical optimizer"), or use a Lion group like the earlier
hyperplane runs, say so on GO — both are one-line changes.

## Status
**Setup + committed, NOT launched.** Awaiting explicit GO. Compare final val_bpb against
**exp032's 1.39371** (same 4096-step budget): does adding a parallel fixed-anchor FastMHL
path to the dense FFN help at matched steps?

Est. runtime: a bit above exp032's ~8 min (extra FastMHL forward/backward per block; hard
fp32 LUT). Run (once GO'd), from this dir:
`sbox ~/projects/spiky/.venv/bin/python train.py`
Outputs alongside: `metrics.csv`, `summary.json`, `loss.png`, `checkpoint.pt` (checkpoint gitignored).
