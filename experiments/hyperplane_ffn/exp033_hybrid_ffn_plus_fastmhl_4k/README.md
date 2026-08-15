# exp033 — FFN slot = simple Linear **+** FastMHL (parallel-sum), **param-matched**, 4k steps

A **matched 4096-step A/B against exp032** (fast vanilla baseline, val_bpb **1.39371**),
**parameter-matched to exp032's 35,792,640 params** (actual 35,794,944, Δ +0.0064%).
Everything is byte-identical to exp032 — MinimalGPT + RoPE, d384 / 6 layers / 6 heads /
seq 512, device_bs 48, total_bs 24576, n_steps 4096, lr 3e-4, wd 0.1, warmup 0.1,
eval_every 200, seed 1, vocab 32768, same data — **except the FFN block**, which is
replaced by two parallel paths at (near-)identical param budget.

## The design — dense FFN REMOVED, replaced by (Linear ∥ FastMHL)
The standard 384→1536→384 GELU MLP is **gone**. Each block's FFN slot becomes **two
parallel paths whose outputs are summed** into the residual:

```
h = ln2(x)
x = x + linear(h) + fastmhl(h)          # was exp032: x = x + mlp(h)
```

- **Path A — a single plain `nn.Linear(384 → 384)` with bias.** No hidden expansion, no
  GELU, no second linear — just one linear map (the simple gradient-highway path). Its
  weight is **zero-init'd** (residual-branch identity start, exactly as vanilla did for the
  FFN's output projection).
- **Path B — a plain FastMultiHeadLUT** (`src/spiky/lutorch/fast_multi_head_lut.py`): the
  classic **fixed-anchor-pair sign-test LUT** (`bit_i = 1[x[a_i]−x[b_i] > 0]`) — NOT
  HyperplaneMHL. `forward_mode="hard"`: hard row-selection forward, but a **full
  soft-surrogate backward** — gradients reach the LUT tables AND flow densely back into `h`
  (so path A, the attention sub-block, and upstream layers all train through it).

Both paths train **simultaneously**. FastMHL tables use near-zero init (1e-3) and the
Linear weight is zeroed, so the whole second sub-block ≈ 0 at step 0 — each path grows only
on its own merit.

## FastMHL capacity (chosen to param-match exp032)
`FastMultiHeadLut(input_dim=384, n_heads=1, n_outputs=384, n_anchor_pairs=6,
tables_per_head=42, forward_mode="hard", use_bf16=False, initial_weights_noise=1e-3,
random_seed=1000+layer_idx)`. Forward `x:[B,384] → [B,1,384]`, reshaped to `[B,T,384]`.
Per-layer FastMHL params = 42·64·384 = **1,032,192** (temps are buffers → 0 params).
`use_bf16=False` because exp032 actually runs fp32 (its `compute_dtype:bf16` is unused —
no autocast in train.py), so the LUT path is fp32 to match.

**Why tph=42:** the removed FFN cost 1,179,648/layer; the replacement should match it.
Linear/layer = 147,840, so FastMHL should be ≈ 1,031,808/layer → tph = 1,031,808/(64·384)
= 41.98 → **tph=42** (FastMHL 1,032,192/layer). Any NAP with `tph·2^NAP = 2688` gives the
same budget; NAP=6 chosen (decoder-like modules want NAP≈6).

## Param count — matched
| component | per layer | ×6 |
|---|---|---|
| removed: dense FFN (384→1536→384) | −1,179,648 | −7,077,888 |
| added: Linear 384→384 + bias (path A) | +147,840 | +887,040 |
| added: FastMHL 42·64·384 (path B) | +1,032,192 | +6,193,152 |
| **net per-layer replacement** | **+1,180,032** | **+7,080,192** |

The replacement (1,180,032/layer) ≈ the FFN it replaces (1,179,648/layer) — off by exactly
+384/layer, the new Linear's bias.

| | params |
|---|---|
| exp032 (vanilla, incl. its 6× FFN) | 35,792,640 |
| exp033 (Linear ∥ FastMHL, tph=42) | **35,794,944** |
| Δ vs exp032 | **+2,304 (+0.0064%)** |

(Confirmed by a build smoke.)

## Optimizer
Unchanged AdamW + warmup/cosine from exp032. Param grouping: FastMHL LUT-table weights →
**no-weight-decay** group (project lesson: LUT tables take no wd); the new Linear **weight →
wd 0.1** group, **bias → no-wd**; all other params exactly as exp032.

## Status
Launched under the owner's GO. Matched 4096-step A/B: compare final val_bpb against
**exp032's 1.39371** — does a simple-Linear + hard-FastMHL FFN slot beat the dense-FFN
baseline at matched steps? Outputs alongside: `metrics.csv`, `summary.json`, `loss.png`,
`checkpoint.pt` (checkpoint gitignored).
