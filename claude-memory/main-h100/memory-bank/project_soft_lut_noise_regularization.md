---
name: Soft-LUT noise injection replaces bf16 implicit regularization
description: TinyMHLut(soft) only matches SoftMHLut(hard=True) when consistent fwd+bwd low-confidence noise is injected; bf16 was doing this implicitly. Drop-in swap without noise costs ~0.013 bpb.
type: project
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
## Finding (validated 2026-05-11 by exp257)

Naive swap `SoftMultiHeadLUT(hard=True)` → `TinyMultiHeadLut(backward_mode='soft')`
loses ~0.013 bpb on nanochat LM. Root cause: `SoftMHLut(hard=True)` runs the
argmax inside bf16 autocast, which acts as **implicit Gumbel-style
regularization** on the ~0.7% of comparison positions where `|d_i|` is so
small that bf16 rounding flips the sign. `TinyMHLut` originally argmaxed in
fp32 (sign-bit-pack of `d > 0`), missing this regularization.

Fix: `argmax_noise_eps` flag on `TinyMultiHeadLut(backward_mode='soft')`.
In the fp32 sign-bit-pack path, randomly flip the bit at any position where
`|d_i| < eps`. **Critical**: backward must use the *same* flipped bits as
forward — implemented by reconstructing bits from the saved `index` via
`((index >> shifts) & 1)` rather than recomputing `d > 0`. Without this
consistency, the noise creates a fwd/bwd mismatch instead of an honest
regularizer.

## Results (exp251 = SoftMHLut(hard=True), bf16 path, 1.6026 baseline)

| Exp | Config | Final bpb | Δ vs exp251 |
|-----|--------|-----------|-------------|
| exp251 | SoftMHLut(hard=True), bf16 default | 1.6026 | — |
| exp252 | TinyMHLut(soft) naive, no noise | ~1.6156 | +0.013 |
| exp257 v3 | TinyMHLut(soft) + bf16 backward + fp32 fwd index + noise eps=0.002 (consistent fwd/bwd) | 1.6060 | **+0.003** (within noise) |
| exp258 | STE backward + noise eps=0.005 | failed (≈ exp234 1.6212 trajectory) | confirms soft bwd is required substrate |
| exp259 | TinyMHLut(soft) + fp32 everywhere + noise eps=0.001 | killed @ step 2600 | early trajectory matched exp257 within ±0.005, but fp32 path is 1.5x slower wall-clock — bf16 backward is preferable |

## 48K-horizon validation (exp260, 2026-05-11)

The exp257 recipe extended to 48K (6x training horizon, same config otherwise)
landed at **1.4655 bpb** — new 48K LUT-LM SOTA, beating prior best exp235
(1.4906, d_v=32 STE) by −0.025 bpb and exp229 (1.4958, e96) by −0.030 bpb.
Confirms the noise-regularization recipe scales gracefully past 8K.

## Why: implicit-regularization decomposition

The 0.019 bpb gap between exp234 (1.6212, vanilla TinyMHLut+STE) and exp251 (1.6026)
breaks down as roughly:
  ~0.013 bpb: bf16 implicit regularization on low-confidence argmax bits
  ~0.006 bpb: real algorithmic gains (V2D learnable T, learnable T_soft/T_sel,
              soft pipeline gradient surrogate beyond STE)

## How to apply

When using `TinyMultiHeadLut(backward_mode='soft')` in nanochat LM configs,
**always set `argmax_noise_eps=0.002`** (or similar) — without it you're
silently losing ~0.013 bpb. Default `use_bf16=True` is fine; `bf16_argmax=False`
is fine. The noise alone closes most of the gap.

For other models or different scales, eps may need tuning. The bf16 effective
threshold is data-dependent (concentrates at near-tied bits); a fixed eps is
a coarse approximation. Try eps ∈ {0.001, 0.002, 0.005} and pick whichever
matches SoftMHLut(hard=True, use_bf16=True) on a calibration run.

## Performance benefit

TinyMHLut(soft) + noise vs SoftMHLut(hard=True): ~12% faster training step,
~35% lower peak memory (no [B, n_tables, K] tensor materialized as saved
activation — recomputed in @torch.compile'd backward body).

## Code

- `src/spiky/lutorch/tiny_multi_head_lut.py`: `backward_mode='soft'`,
  `argmax_noise_eps`, `bf16_argmax`, `use_bf16` flags; @torch.compile'd
  `_soft_lut_fwd_body` and `_soft_lut_bwd_body` with consistent bit
  reconstruction in bwd.
- `src/spiky/lutorch/tests/test_tiny_multi_head_lut.py`: 11 tests covering
  the soft mode including the noise-vs-no-noise gradient finite check and
  the fp32 sign-pack ≡ argmax(soft ts) invariant for NAP ∈ {4, 6, 8}.
- Committed in `3800834` on `feature/lutorch-calibrate-output-normalize-weights`.
