# hyperplane_ffn

Research chapter for the **hyperplane_ffn** idea (see `claude/experiment-methodology.md`).
This README wraps up the **first pass** of the investigation — the one that produced
[PR #65](https://github.com/anatoli-starostin/spiky/pull/65).

> **Status: paused, not concluded.** The approach is *not* fully investigated. The
> most promising configuration (exp009) was still improving at its step budget, and
> the follow-up work continues in a **separate PR**. Nothing here is a final verdict.

- **Tracking issue:** [#61](https://github.com/anatoli-starostin/spiky/issues/61) — origin & status log.
- **Implementation spec:** [#64](https://github.com/anatoli-starostin/spiky/issues/64).
- **PR (this pass):** [#65](https://github.com/anatoli-starostin/spiky/pull/65) — for review, **not merged**.
- **Module:** [`src/spiky/lutorch/hyperplane_multi_head_lut.py`](../../src/spiky/lutorch/hyperplane_multi_head_lut.py).
- **Baseline anchor:** `exp001_untied_vanilla_baseline` — untied-vanilla MinimalGPT,
  **val_bpb 1.20144**, 35.79M params. Every experiment compares against it.

---

## 1. The idea

`FastMultiHeadLut` computes each of its `NAP` index bits from a **fixed anchor pair**
`(p1, p2)` picked at init:

```
bit_i = 1[ x[p1] − x[p2] > 0 ]
```

i.e. a sign test against the axis-aligned hyperplane with normal `e_p1 − e_p2` through
the origin. **hyperplane_ffn** generalizes each bit to a **fully learned affine
hyperplane**:

```
bit_i = 1[ ⟨w_i, x⟩ + b_i > 0 ]
```

with `w_i` trainable over *all* input components and `b_i` a trainable threshold. Weight
scope is one set of `NAP` hyperplanes **per table**: `w : [n_tables, NAP, d_model]`,
`b : [n_tables, NAP]` (`n_tables = n_heads · tables_per_head`). Everything downstream is
structurally unchanged — MSB-first bit packing, `F.embedding_bag(mode='sum')` reduce over
`tables_per_head`, both forward modes, and the always-soft surrogate backward.

## 2. What was built (PR #65, closing issue #64)

`HyperplaneMultiHeadLUT`, cloned from `FastMultiHeadLut`:

- **Soft backward ported to the affine pre-activation** `a_i = ⟨w_i, x⟩ + b_i` in place of
  the pairwise difference `d_i = x[p1] − x[p2]`. The input gradient now flows **densely
  through `w`** (one fused GEMM) instead of a two-coordinate scatter, and the surrogate
  additionally emits gradients for `w` and `b`. The full-K-softmax surrogate structure
  (pinned to the chosen index, learnable `T_soft` / `T_sel`) is preserved.
- **Anchor-pair-equivalent init** (`hyperplane_init="anchor_pairs"`: `w_i = e_p1 − e_p2`,
  `b_i = 0`) reduces to `FastMultiHeadLut` **bit-for-bit**, so the module is a strict
  generalization and can be A/B'd against it. `hyperplane_init="random"` (small-norm
  Gaussian rows, zero bias) is the learn-from-scratch path used by the experiments below.
- **Separate storage dtypes:** the module exposes `hyperplane_dtype` (default fp32) for
  `w`/`b`, *independent of* `weight_dtype` for the LUT tables — so the LUT can live in
  bf16 while the hyperplanes stay fp32.
- **Preserved from `FastMultiHeadLut`:** both `hard` and `hybrid_smooth` forward modes +
  runtime flip, the LUT-weight-grad backend auto-pick (fp32 `index_add` / bf16 sparse-S +
  bmm), and the `weight_dtype`-vs-autocast handling. Affine grads are gated on
  `needs_input_grad`, so a frozen hyperplane skips the extra GEMMs.
- **Hybrid optimizer (in the experiment harness, routed by parameter identity):**
  **Lion** on the LUT tables (lr 2e-4, betas (0.9, 0.95), wd 0); **Adam, no weight decay**
  on `hyperplane_weight` + `hyperplane_bias` + the temperatures; **AdamW** on the rest of
  the model (2-D → wd 0.1, 1-D → no wd) at base lr 3e-4. Shared warmup(0.1)+cosine(→0.1×)
  schedule.
- **Tests: 30/30** green on H100 (`torch 2.13.0+cu130`), `FastMultiHeadLut` suite 33/33
  (no regression): fp64 `gradcheck` on `w`/`b`/`x`; index-packing correctness; parity with
  `FastMultiHeadLut` under anchor-pairs init (forward + `x`/weight/temp grads); frozen-
  hyperplane grad gating; both forward modes + flip; dtype coverage incl. a **CPU fallback
  path** (the eager, non-`torch.compile` route the CPU test run exercises).

> **⚠️ bf16 parity caveat.** Parity with `FastMultiHeadLut` under anchor-pairs init is
> bit-exact **only in fp32 with autocast off**. The projection `a = x @ Wᵀ + b` is an
> autocast-eligible GEMM, so under bf16 a value near a decision boundary (`a_i ≈ 0`) can
> flip the sign bit vs fp32 — a *discrete* change (a different table row), not a tolerance
> diff. A handful of boundary-row flips under bf16 are expected; not a bug.

## 3. Experiment setup

All experiments are strict, **single-variable** clones of `exp001` (dense-FFN baseline):
MinimalGPT + RoPE, untied head, **d_model 384, 6 layers, 6 heads, seq 512**, dense MLP FFN
384→1536→384, device_bs 48 / total_bs 24576, **16 000 steps ≈ 393M tokens**, lr 3e-4,
wd 0.1, warmup 0.1, seed 1, bf16, vocab 32768 → **val_bpb 1.20144, 35.79M params**.

The FFN in every block is swapped for a **single-head** `HyperplaneMultiHeadLUT`
(`n_heads=1`, `n_outputs=384`), random hyperplane init, identity-ish at init
(`initial_weights_noise=0.001`). Everything else is held fixed; runs compare at the same
16 000 steps. LUT tables bf16, hyperplanes fp32.

## 4. Results

Final val_bpb vs the **1.20144** dense anchor (lower is better):

| exp | geometry (NAP / tph) | forward | val_bpb | Δ vs dense | params | wall (h) |
|-----|----------------------|---------|---------|-----------|--------|----------|
| exp001 (dense baseline) | — | — | **1.20144** | — | 35.8M | 0.47 |
| exp006 | NAP5 / tph128 | hard | 1.26451 | **+0.06307** | 39.6M | 0.59 |
| exp007 | NAP6 / tph256 | hard | 1.24045 | **+0.03901** | 70.0M | 0.68 |
| exp008 | NAP6 / tph512 | hard | 1.23255 | **+0.03111** | 111.3M | 1.07 |
| exp009 | NAP6 / tph512 | hybrid_smooth | **1.21393** | **+0.01248** | 111.3M | 4.08 |

(exp008 best = 1.23253; final reported. exp009 best = final = 1.21393.)

## 5. Findings

- **The dense MLP is still ahead at every configuration tested.** Best hyperplane result
  (exp009) is **+0.0125 bpb** off the dense anchor — closer than any earlier point, but not
  past it.
- **Scaling the LUT tables helped, with sharply diminishing returns.** exp006→exp007
  (NAP 5→6 *and* tph 128→256, ~1.8× params) bought −0.0241; the clean table doubling
  exp007→exp008 (tph 256→512, NAP fixed) bought only **−0.0079**. Capacity alone is a
  weakening lever.
- **The single biggest knob was the forward mode.** At *fixed* exp008 geometry (NAP6 /
  tph512), switching `hard → hybrid_smooth` (exp008→exp009) bought **−0.0186** — more than
  doubling the tables did. The top-2 smooth forward gives the surrogate a materially better
  training signal than the hard argmax.
- **exp009 looks step-limited, not capacity-limited.** Its best eval *is* its final eval
  (best = final = 1.21393), i.e. it was still descending at 16 000 steps — unlike exp008,
  whose best (1.23253) preceded its final (1.23255). The gap to dense may close further with
  more steps at the same geometry.
- **Cost.** `hybrid_smooth` is expensive per step: exp009 took **4.08 h** vs exp008's
  **1.07 h** at identical geometry/steps (~3.8× wall-clock), from the top-2 smooth forward
  plus its 2-row LUT-table-weight gradient. Front-end benchmark (`BENCH_NOTE.md`, H100): the
  projection GEMM is ~1.5× the anchor gather in isolation, hidden to 0.8–0.9× under the full
  hard train/eval path and showing through at ~1.5× only in the lean `hybrid_smooth` forward.

## 6. Open questions / next steps (follow-up PR)

- **Train the exp009 geometry longer** — since best = final, the run had not plateaued; how
  much of the +0.0125 gap is just step budget?
- **`hybrid_smooth` at the smaller/cheaper geometries** — it was the largest single lever;
  worth isolating from the tph=512 size.
- Whether learned hyperplanes ever **beat** fixed anchors on val bpb is still **open** —
  that A/B is what this primitive enables, and it is **not settled by this pass.** Work
  continues in a separate PR.
