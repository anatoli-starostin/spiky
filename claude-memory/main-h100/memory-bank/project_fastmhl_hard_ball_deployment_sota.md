---
name: fastmhl-hard-ball-deployment-sota
description: "FastMultiHeadLUT(hard, ball) trained natively in hard mode beats soft-trained + hard-deployed baselines (exp724/exp726) by 25-104 mb at matched arch. Hard-soft gap is the load-bearing metric."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# exp729 (FastMHL hard+ball) is the new hard-inference SOTA @ 16k bs=24×2

**Final results @ 2026-06-05, 16k steps, E=384 NAP-bump arch (276.83 M params):**

| Model | Train recipe | Soft val | Hard val | Hard-soft gap |
|---|---|---|---|---|
| **exp729** | FastMHL hard fwd + ball NAP+1 bwd (native hard) | n/a | **1.2360** | 0 by construction |
| exp724 | Tiny hybrid_smooth top-2 fwd + K-row dense bwd | 1.1936 | 1.2611 | +0.0676 |
| exp726 | Tiny (NAP+1)-row ball fwd + autograd bwd | 1.1997 | 1.3399 | **+0.1402** |

**Why:** exp726's autograd backward only updates the (NAP+1) rows the forward actually gathers. The other 87% of K=2^NAP rows never see useful gradient signal, so at hard inference (1-row argmax) they're noise. exp724's K-row dense soft surrogate backward distributes gradient across all K rows weighted by softmax — much better hard-mode generalization. exp729 sidesteps the soft↔hard distribution shift entirely by training in hard mode with rich ball backward.

**How to apply:** For LUT-LM runs targeting hard inference deployment:
- Use `FastMultiHeadLUT(forward_mode='hard', backward_mode='ball')` from `src/spiky/lutorch/fast_multi_head_lut.py`.
- Don't trust soft val_bpb when the backward path is sparse-row (autograd or (NAP+1)-row); always measure the hard-soft gap before claiming wins.
- See `nanochat_exps/exp729_FastMHL_hard_ball/` for the training recipe template; `/tmp/eval_exp726_hard.py` for the hard-eval flip pattern (set `module.backward_mode='soft'` on each Tiny module to trigger the embedding_bag hard inference path).

**Speed bonus:** FastMHL hard/ball measured ~21 ms/LUT on out_proj NAP=7 vs ~34 ms (Tiny K-row dense) and ~42 ms (Tiny autograd). exp729 total wall-clock ~5.5-6 h vs 7.3 h (exp724) and 6.6 h (exp726).

Cross-refs: [[fast-multi-head-lut-cherry-pick]], [[ball-backward-fast-mhl]].
