---
name: lut-wall-leverboard
description: "Empirical wall-saving leverboard at exp760 scale (200-step bench): tphs ×0.5 = -44.7%, fwd hybrid_smooth->hard = -37 to -39%, E/d_v halving = only -4.3%. E/d_v is the wrong knob for wall."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# LUT-LM wall-saving leverboard at exp760 scale

Measured via 200-step bench (`bench_exp760_vs_E96.py`) with the exp760 stack
(FastMHL hard + dense_K + bf16 storage + master Lion + global clip(1.0)),
eff bs=48, seq 512. Bench within 4 % of clean exp756 actual.

## Result

| Lever | Wall saving | Param saving | Notes |
|---|---|---|---|
| **All tphs × 0.5** | **-44.7 %** | -44.6 % | linear in tph; touches every LUT |
| **forward_mode hybrid_smooth → hard** | **-37 to -39 %** | 0 % | already in exp760 |
| **E: 192→96, d_v: 32→16** | **-4.3 %** | -28.6 % | only touches v_lut + out_proj |

The tph knob saves wall roughly proportional to param savings (close to 1:1).
The forward-mode change saves wall without any param change (the historical big win,
hybrid_smooth → hard ~halves per-step wall by replacing K-row blend with single-row gather).
The E/d_v knob is the *worst* wall lever: cuts ~29 % of params for only 4 % wall.

## Why E/d_v shrink barely moves wall

The exp760 wall is dominated by modules whose compute is **E-insensitive**:
- `residual_lut`: input_dim is E but n_outputs = D = 384. Compute scales with n_outputs, not
  input_dim. Unchanged when E shrinks.
- `emb_resid_lut`: same — n_outputs = D = 384.
- `qk_lut`: n_outputs = 2·d_qk = 128 (independent of E).
- Unembedder: Linear(D=384, V=32768). Fixed.

Only **v_lut** (n_outputs = d_v) and **out_proj** (n_outputs = E) actually shrink with E and d_v.
Together they're a small slice of total wall when bf16 storage is in play.

Per-token LUT FLOPs at exp760 scale, by module:
- qk_lut       18.87 M/tok (E-insensitive)
- v_lut         9.44 M @ E=192 →  4.72 M @ E=96 (-50%)
- out_proj     37.75 M @ E=192 → 18.87 M @ E=96 (-50%)
- residual_lut 18.87 M (E-insensitive)
- emb_resid     3.15 M (E-insensitive)
- Total LUT    88.08 M  →  64.48 M  (-27 % compute, -4.3 % wall)
- Unembedder   12.58 M (E-insensitive)

The compute reduction is 27 %, but wall only drops 4.3 % — i.e., the E-sensitive modules are
nowhere near bandwidth-bound at these shapes; halving them mostly removes idle launch overhead.

## How to apply

- **Want to cut wall on a fork of exp760?** First lever: halve tphs. Second: shrink D
  (residual_dim) — touches residual_lut + emb_resid_lut + unembedder simultaneously. Third:
  shrink out_tph / out_input_nap specifically (out_proj is the largest LUT). E/d_v shrink is
  basically free quality-cut but doesn't save wall — only use it for explicit param cuts.
- **Want to grow capacity?** Same lever ordering inverted: bigger tphs > more NAP > wider D >
  wider E/d_v.
- **Don't conflate compute with wall**: at bf16 + dense_K, LUT modules are kernel-launch-bound
  more than memory-bound for E-sensitive modules. The E-insensitive modules (qk_lut, residual,
  emb_resid, unembedder) genuinely dominate the wall floor.

## Historical context — what actually drove the SOTA progression

The "big wall savings going from E=384 to E=192" memory (exp755 → exp756 mental model) is
**wrong**. Actual breakdown:

| Comparison | Wall delta |
|---|---|
| exp754 (E=384, hybrid_smooth) 6.30 h → exp755 (E=192, hybrid_smooth) 5.42 h | -14 % |
| exp752 (E=384, hard) 3.84 h → exp756 (E=192, hard) 3.40 h | -11 % |
| **exp754 (hybrid_smooth) → exp752 (hard) at E=384** | **-39 %** ← the big win |
| **exp755 (hybrid_smooth) → exp756 (hard) at E=192** | **-37 %** |

The composite exp755 → exp760 (E shrink + fwd mode flip + step extension) was dominated by the
fwd mode flip. The E shrink contributed only ~10 % of the wall saving.

## What was tested and confirmed

- exp764 (`exp764_tph_halved_24k_bs96`): tph-halved exp760 with eff bs 96 = 5.72 h actual,
  -44.6 % params, +6.8 mb val_bpb. Confirms the leverboard:
  - Wall savings from tph alone are real and large.
  - Quality cost from halving tphs is recoverable in part by 2× tokens.
  - See [[exp764-tph-halved-tiny-pareto]].

## Open extensions to test

- **D shrink** (residual_dim 384 → 256): would touch residual_lut, emb_resid_lut, AND
  unembedder simultaneously. Expected wall saving ~15-20 %, params saving ~10-15 %. Untested.
- **out_tph specifically halved** (just out_proj 512 → 256, keep others): out_proj is the
  largest LUT; this is the cheapest single-module tph cut.
- **NAP shrink on out_proj alone** (7 → 6): halves out_proj table dim from 128 to 64. Should
  give a measurable wall cut. exp761 tried this combined with E shrink and got swamped by the
  E-shrink dynamics.

Cross-refs: [[exp764-tph-halved-tiny-pareto]] (the experiment that confirmed the leverboard),
[[exp735-v-lut-nap7-sota]] (current quality SOTA, opposite direction — adds tph/NAP capacity),
[[exp731-fastmhl-hard-densek-sota]] (the recipe these benches use).
