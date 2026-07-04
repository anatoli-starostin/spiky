---
name: soft-topk-math-isolation
description: "soft_topk backward mode (mask-based) isolated the lever in soft-vs-STE gap: soft attribution math is ~60% of the gap, with only 4 rows out of K=256 needed. K row count saturates fast (Hamming-1 ball ≈ top-3 1-bit-flips). Remaining gap is Hamming-≥2 softmax mass."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# soft_topk math-isolation experiments (2026-05-16, exp401-exp403)

## Question
Soft mode (n_alt=N/A, K=256) beats STE n_alt=3 by ~0.05 bpb at bs=192. Is the lever:
(a) soft *attribution math* (softmax-based d_z chain rule), or
(b) the *number of alternatives* K=256 considered?

## Setup: added `backward_mode='soft_topk'` to TinyMultiHeadLut
File: `src/spiky/lutorch/tiny_multi_head_lut.py`
- New `_soft_lut_bwd_body_topk` function (compiled, mask-based "Approach B")
- Reuses soft forward path (identical sign-pack argmax)
- Backward: compute full-K `ts` einsum (tensor cores), then `mask` non-kept rows to `-inf` before softmax → softmax renormalizes over kept rows only. Rest of soft chain rule unchanged.
- Kept rows = chosen + `n_alternatives` 1-bit-flip neighbors (smallest |d|). If `n_alternatives >= NAP`, all 1-bit-flips kept (full Hamming-1 ball).
- New cfg: `soft_topk_n_alt: int | 'nap'` (the string `'nap'` resolves per-module to that module's NAP).

## Cost characterization (out_proj shape: NAP=8, tph=128, B=16384):
- soft (full): 4.0 ms / 3.2 GB
- soft_topk-1: 7.5 ms / 5.3 GB
- soft_topk-3: 9.3 ms / 5.3 GB  (~2.3× slower than soft)
- soft_topk-8 (NAP): 5.2 ms / 5.4 GB (~1.3× slower than soft)
- ste-n3: 2.4 ms / 0.3 GB

soft_topk is **slower AND uses more memory** than full soft at out_proj shape — the topk + scatter + masked_fill overhead exceeds the savings from sparser softmax. Real speed/memory win would require a native CUDA kernel.

## Math verification (NAP=4, K=16):
- forward bit-identical ✓
- weight grad bit-identical ✓ (both use hard `index_add` at chosen row)
- input grad: cosine sim 0.967 (gradient direction within 4° of full soft); ||Δ||/||a|| = 0.28
- Analytical softmax mass omitted (Hamming-≥2 rows): 6.2%
- The 28% norm error from 6.2% mass omitted is from softmax renormalization (kept rows' sel_soft scaled up) + shifted sum_term + d_p amplification through bit_matrix on omitted rows.

## Quality results (all bs=16, 8000 steps, exp365 arch matched, 43.06M params):

| Run | Mode | n_alt | Rows kept | Final bpb | Δ vs soft |
|---|---|---|---|---|---|
| exp365 | soft | N/A | K=256 (all) | 1.6215 | reference |
| exp403 | soft_topk | NAP | 9 (Hamming-1 ball) | **1.6359** | **+0.0144** |
| exp402 | soft_topk | 3 | 4 (top-3 1-bit-flips) | 1.6402 | +0.0188 |

Tracking through training (exp402 vs exp403 vs exp365):
- Early phase (step 200-1000): exp402 marginally LEADS exp403 — surprising; both within ±0.005 of exp365
- Mid phase (step 2000-5000): exp402 and exp403 track parallel ~+0.008-0.014 vs soft
- Late phase (step 6000-8000): exp403 pulls ~3 mb ahead of exp402; both ~+0.014-0.019 vs soft

## STE-n3 vs soft at bs=192 (exp401, killed early at step 1800):
- Same arch + recipe as exp364 (bs=192 SOTA, soft mode 1.3769 final)
- Switched all 4 modules to STE n_alt=3
- Gap **WIDENED monotonically** from +0.018 at step 800 → +0.047 at step 1800
- Killed and pivoted: this proved STE math has a real, scaling-resistant deficit vs soft

## Discrimination of the ~+0.05 bpb soft-vs-STE gap:
| Component | Contribution | Evidence |
|---|---|---|
| Soft attribution math vs STE uncertainty kernel | **~60% (+0.03)** | exp402 closes most of the ~+0.05 gap with just 4 rows |
| Top-3 1-bit-flips → full Hamming-1 ball (+5 rows) | ~8% (+0.004) | exp402 → exp403 = 4 mb |
| Hamming-≥2 rows (6.2% omitted softmax mass) | ~32% (+0.014) | exp403 → exp365 residual |

## Implications / How to apply

1. **Soft attribution math is the dominant quality lever** for LUT-LM training. Going STE→soft is the big win. The K=256 count is secondary.
2. **For inference / cheaper training**: top-K with K≈NAP or smaller captures most of the soft-mode benefit. Doesn't help current PyTorch impl (slower), but motivates a CUDA kernel that operates only on top-K rows.
3. **Closing the remaining +0.014 gap** would require either:
   - Sampling Hamming-≥2 rows stochastically (e.g., add top-K 2-bit-flips by |d| sum)
   - Full-K kernel (i.e., keep soft mode but make it cheaper)
4. **Don't use soft_topk in PyTorch as-is for production** — it's currently slower than soft. Only valuable as a knowledge tool to inform future kernel design or sparse-aware optimizers.
5. **Forward path is identical** between soft and soft_topk (sign-pack argmax). Only backward differs. Safe drop-in replacement.

## Code changes (persistent)
- `tiny_multi_head_lut.py`:
  - `_soft_lut_bwd_body_topk` (new compiled fn)
  - `_TinyMHLutSoft.forward/.backward` extended with `topk_n_alt` arg (default 0 = standard soft)
  - `_TinyMHLutSoft.ctx` saves `powers` and `topk_n_alt`
  - Module init: `backward_mode` accepts `"soft_topk"` value
  - Forward dispatch: `"soft_topk"` uses same `_soft_forward` path
- `nanochat_exps/bench_backward_modes.py`: benchmarks all 6 modes (soft, soft_topk-{1,3,8}, ste-{1,3})
- `nanochat_exps/exp402_soft_topk3_bs16/` and `exp403_soft_topk_nap_bs16/`: full training runs.
