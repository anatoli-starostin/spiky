---
name: project_lut_lm_residual_width_sota
description: "matmul-free LUT-LM @8K SoTA via residual width D + residual_tph; super-additive stacking, diminishing returns"
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

Residual-LUT shape sweep on the matmul-free LUT-LM (exp513 backbone: E=64, 6L, RoPE,
TinyMHLut soft, untied `Linear(D,V)` head, bs=16, 8K steps, 8192 tok/step, V=32768).
All single-knob forks of exp513.

**Reference grid** (val bpb @ 8K):
| exp | D (residual_dim) | residual_tph | bpb | params |
|--|--|--|--|--|
| exp513 | 384 | 128 | 1.4825 | 89.4M |
| exp530 | 384 | 256 | 1.4731 | 108.3M |
| exp562 | 512 | 128 | 1.4714 | 99.9M |
| exp564 | 512 | 256 | 1.4579 | 125.0M |
| **exp565** | **512** | **512** | **1.4513** | **175.4M** ← current matmul-free SoTA @8K |
| exp563 | 128 | 128 | 1.5570 | 68.4M |

**Findings:**
- **The two residual levers stack SUPER-additively.** tph alone (513→530) = −0.0094; width D alone (513→562) = −0.0111; both together (513→564) = −0.0246, *more* than the sum (−0.0205). Widening D gives the extra tables more useful dims to write into and vice versa.
- **Residual width (D) is more param-efficient than residual tables (tph)**: D=512 (+10M) beat exp513 by −0.0111; tph=256 (+18.9M) only −0.0094.
- **D-sweep at tph128**: D=64→128→384→512 = 1.617→1.5570→1.4825→1.4714. Knee at ~384; 384→512 buys only −0.011 for +10M. D=128 is a real quality drop (+0.0745 vs D=384).
- **tph-sweep at D=512 — sharp diminishing returns**: tph 128→256 = −0.0135 (+25M); 256→512 = −0.0066 (+50M). Each doubling ≈ half the gain at double the cost. tph is essentially tapped out by 512.
- These are all pure residual-LUT changes; SDPA + Linear unembedder untouched.

**Caveat / reframing (see [[project_lut_lm_pertoken_gap_vs_vanilla]]):** these gains chase *absolute* bpb by growing params — the opposite of the real goal (per-token convergence parity with vanilla at small size). The bloat barely dents the per-token gap. exp562 (D=512, 99.9M, 1.4714) is the cleanest "best small-ish backbone" to fork from.
