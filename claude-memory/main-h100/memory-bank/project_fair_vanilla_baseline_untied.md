---
name: fair-vanilla-baseline-is-untied-exp476
description: "Fair vanilla reference for LUT-LM is UNTIED (exp476=1.4143), not tied (exp328=1.3882). LUT-LM unembedder is structurally untied, so tied vanilla is an unfair target. Real LUT-LM gap = ~0.082, not ~0.108. 2026-05-21."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Fair vanilla baseline for LUT-LM is UNTIED (exp476), not tied (exp328) (2026-05-21)

exp476 = exp328 (vanilla MinimalGPT+RoPE, bs=16, 8K) with the token-embedding↔output-head weight tying REMOVED (single change: delete `self.head.weight = self.tok_emb.weight`; head gets its own normal(0,0.02) matrix). +12.58M params (32768×384), 23.21M → 35.79M.

## Result: untying HURTS vanilla by +0.0261 bpb
- exp328 vanilla **tied** = 1.3882 @ 23.21M
- exp476 vanilla **untied** = **1.4143** @ 35.79M (Δ +0.0261 despite +54% params)
- Trajectory: untied led early (−0.011 @ step200, dedicated head helps first ~800 steps), crossed at ~step1000, settled to stable +0.026–0.032 deficit. Classic small-model result: tying is a strong param-efficient regularizer (the tied matrix gets gradient from both input+output paths); untying's extra capacity isn't exploitable at this scale/token budget.

## Why this matters: exp476 is the FAIR LUT-LM reference
The LUT-LM unembedder is **structurally untied** — input is `tok_emb_E` (E=64), output is `Linear(D=384, V)` reading the D-stream; different shapes/streams → cannot share weights. So comparing LUT-LM against TIED vanilla (exp328) gives vanilla a +0.026 advantage the LUT-LM can never access. Use **untied vanilla (exp476=1.4143) as the apples-to-apples target.**

| reference | bpb | LUT-LM gap |
|---|---|---|
| vanilla tied (exp328) | 1.3882 | 0.108 (inflated) |
| **vanilla untied (exp476) — fair** | **1.4143** | **0.082** |

LUT-LM bests vs fair ref: exp453=1.4967 (gap 0.082), exp475 mean-abs=1.4962 (0.082), exp466 scatter=1.4937 (0.079). The tying advantage was inflating the apparent LUT deficit by ~24%; the real gap is ~0.082. **Supersedes the "vanilla bs=16 = 1.3882" framing for LUT-LM comparisons** ([[project_bs16_lut_lm_sota]] used the tied number).
