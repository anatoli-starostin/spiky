---
name: project_unembedder_head_explorations
description: "matmul-free unembedder head designs (VQ/ScoreLUT/BitReadout/Kendall) all ≤ trivial dot; gap was residual narrowing, not the head"
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

Arc (2026-05-25): the unembedder is one of the two matmul parts (other = SDPA); goal is a small, near-SOTA, matmul-free head. D=384 held fixed; reference = exp513 `Linear(384,V)` = 1.4825 @ 8K. Many head designs explored against trained-backbone targets / from scratch:

- **VQ VocabLUT** (`src/spiky/lutorch/vq_vocab_lut.py`): per-token latent → m sub-codebooks (STE + commitment), `logit(v)=x·concat C_j[code_v[j]]`. **Best matmul-free head** — near-Linear quality at 8–26× compression. Offline PQ of a trained Linear needs sub=2 (8×) for near-lossless.
- **ScoreLUT** (`score_lut_head.py`): `logit(v)=Σ_j w_v[j]·S[j,code_v[j]]` scalar gather. ~+0.2 vs Linear; near-uniform `assign` init was the symmetry trap (fixed with assign_init_std/weight_init_noise), tph not the limiter.
- **BitReadout** (`bit_readout_head.py`): binary popcount dot `sign(anchor-diffs)·sign(latent)`; needs learnable `logit_scale` init 1/√P (else bpb≈33 saturation). Plateau ~+0.28.
- **Mixture-of-Softmaxes** (`mixture_bit_readout.py`): cold-start pathology (uniform gate + 1/T responsibility) — failed.
- **Kendall-tau readout** (`kendall_readout.py`): `logit(v)=Σ_p sign(ê_i-ê_j)·sign(emb_{v,i}-emb_{v,j})`, tied to tok_emb. HARD mode only (soft was broken/worse). Full E(E-1)/2 pairs (K2016) = only positive lever (~−0.02) but NOT bandwidth-efficient (O(E²) bits = 33MB > Linear-128's 16.8MB; only K≈E·logE is). Rank E=64 ~+0.3 (1.78).
- **MultiKendall** (`multi_kendall_readout.py`): T weightless partial-order tables, MLP-aggregated τ. ≈ flat, and too slow.

**KEY FINDING:** the **trivial tied dot** (exp560 E=64 = 1.6283) BEATS every rank/Kendall head; tie≈untie (exp561 untied 1.6171). The +0.13–0.15 gap to Linear was the **residual narrowing (E=64 vs D=384)**, NOT the head design. A plain dot > any rank readout. Boundedness/[0,1]-normalize hurts (softmax-over-distances is already in CE; learnable logit_scale = temperature is what's needed).

LUT/codebook duality noted: backbone LUT = input-addressed static table; PQ head = output-addressed dynamic table; both = `embedding_bag(table, idx, mode='sum')`.

Next (queued): see [[project_todo_isolated_big_head]] — test whether a *bigger head* improves the backbone via better downstream gradients.
