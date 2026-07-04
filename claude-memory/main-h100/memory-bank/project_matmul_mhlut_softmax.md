---
name: project_matmul_mhlut_softmax
description: "MatmulMultiHeadLut (dense gated-matmul LUT, no STE) — softmax routing reproduces exp444's soft-mixture win (1.4806, beats exp475) at 5.5x the speed; all other gates lose. Softmax IS the normalized exp-hamming kernel."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**New module `MatmulMultiHeadLut`** in `src/spiky/lutorch/tiny_multi_head_lut.py` (2026-05-22):
dense, fully-differentiable LUT — same front end as TinyMHLut (anchors, soft-sign
`p=d/(T_soft+|d|)`, `ts=einsum(p, bit_matrix)`), but routing is a `gate_mode` applied to
`ts` then a DENSE matmul `g @ W[2^NAP, n_out]`. Pure PyTorch + @torch.compile, NO STE/argmax.
Every weight gets gradient from every token (dense). `gate_mode ∈ {unit, signed, layernorm,
hamming, softmax}`, optional `use_bias`. Costs K×n_out matmul/table (NOT matmul-free).
Optimizer: weight tables routed into the unembedder's AdamW group (lr=adam_lr=3e-4, wd=0.1) —
LION's edge was sparse-gradient-specific, useless for dense.

**Gate sweep, all bs=16, vs exp475 (sparse argmax, 1.4962):**
- unit `0.5(1+ts/(T_sel+|ts|))` ∈(0,1): **+0.23** — common-mode (unnormalized positive gate →
  output dominated by Σ W[k], gradient can't specialize). exp489.
- signed `ts/(T_sel+|ts|)` ∈(-1,1): **+0.11** — common-mode fixed but soft/blurry sum over all
  rows < sharp single pick. exp490.
- layernorm(ts) + affine: **+0.15**. exp491.
- hamming (learnable per-shell weight `ts*ham_weight[h]`, h=(NAP-ts_hard)/2): **OOM at bs=16** —
  the int64 `h` index [B,nt,K]×24 modules ≈ 77GB (int64 = 4× bf16). exp492. (Fix if needed:
  polynomial in ts_hard, no int64 — not run.)
- **softmax `softmax(ts/T_sel)`: 1.4806 — BEATS exp475 by −0.0156, ≈ exp444 (1.4821).** exp493.

**KEY INSIGHT (user's): all the gates were reinventing softmax.** `softmax(ts/T)[k] ∝
exp(-2·h_k/T_sel)` — softmax IS the normalized exponential kernel over hamming distance.
It has the two properties every failed gate lacked: NORMALIZED (Σ=1, no common-mode) and
PEAKED (sharp-ish). The hamming idea = its free-shape generalization but loses normalization.

**exp493 = exp444-class quality at 5.5× SPEED:** 1.4806 @ **0.384 h** vs exp444 (same softmax
soft-mixture via `SoftMultiHeadLut`) 1.4821 @ 2.118 h. exp444's slowness was implementation
overhead (per-table machinery), NOT the softmax — the clean @torch.compile einsum path is
~5.5× faster. So the soft-mixture win is now cheap to train.
**Caveat:** softmax needs all-K soft inference (not matmul-free) — same deploy limitation as exp444.

**exp494 (running): softmax + temperature penalty** `loss = CE + λ·Σ T_sel` (λ=0.01) to push
learnable T_sel DOWN → sharper routing. Goal: HARDENABILITY — if T_sel small enough,
softmax≈argmax, so train-soft-deploy-hard (single-lookup matmul-free inference) with ~no loss.
Soft pressure on learnable temp (unlike exp483's failed forced anneal). Watch temperatures.csv +
argmax-hardened eval vs soft eval.

**BUG fixed (MatmulMHL forks):** lut_params is empty (weights on AdamW group) → `lut_optimizer`
never created, but end-of-run `torch.save` referenced `lut_optimizer.state_dict()` → NameError
AFTER training (exp493 lost its checkpoint; summary/metrics OK). Fix: `lut_optimizer=None` in the
empty branch + guard the save with `... if lut_optimizer is not None else None`. exp494 has the fix.

**Full gate sweep (all bs=16, MatmulMHL, vs softmax exp493=1.4806):** unit +0.23 (exp489),
signed +0.11 (exp490), layernorm +0.15 (exp491), hamming OOM (exp492), **softmax 1.4806 WINS
(exp493)**, relu +0.12 (exp496), relu_norm +0.085 (exp497), gelu_norm ≈ relu_norm +0.09 (exp498),
gelu unnorm ~+0.09 widening (exp499, best non-softmax early but reverts). **Softmax is uniquely
best across all 9 gates.** gate_relu_bias [n_tables,K] per-table threshold (no wd) shared by
relu/relu_norm/gelu_norm/gelu modes. Debugging the ReLU underperformance (instrumented):
- exp496 relu `ReLU(ts+b)`: unnormalized → g.sum≈40/table (softmax=1), output 16× too large →
  common-mode (same pathology as unit gate). NOT a bug — missing normalization.
- exp497 relu_norm `ReLU(ts+b)/Σ` (sparsemax-style): fixes common-mode but still +0.085 — ReLU
  hard-zeros ~half the rows → those get NO gradient (gradient-sparsity reintroduced).
- exp498 gelu_norm `GELU(ts+b)/Σ`: GELU smooth tail gives nonzero gradient everywhere, BUT after
  normalization a tail row's weight g_k≈0 so its WEIGHT gradient (∝g_k) is still ~0 → re-sparsified;
  ≡ relu_norm. Activation (ReLU vs GELU) is irrelevant.
**Why softmax uniquely wins:** not just normalized + dense-gradient — it's the EXPONENTIAL peaking
(sharply, smoothly concentrate on the best-matching row, learnable T). Linear-above-threshold gates
(relu/gelu) spread weight too evenly among firing rows and lose. `gate_mode` ∈ {unit,signed,layernorm,
hamming,softmax,relu,relu_norm,gelu_norm} all in the lib; **softmax is the one to use.**

Relates to [[project_lut_convergence_bottleneck]] (dense gradients alone don't beat sparse-sharp
unless the routing is softmax) and the soft-mixture line (exp444).
