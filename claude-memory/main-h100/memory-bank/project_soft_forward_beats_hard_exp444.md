---
name: soft-forward-beats-hard-exp444
description: "exp444 — replacing TinyMHL(soft) hard-pick forward with SoftMultiHeadLUT(hard=False) genuine soft mixture forward beats exp428 by 16 mb @ matched arch + matched effective batch. New bs=16-effective LUT-LM SOTA at 1.4821 bpb. Curve-shape finding: hard pick lags early in soft regime, soft mixture wins post-step-3K and lead widens monotonically through step 8K."
metadata:
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Soft-mixture forward beats hard-pick forward at matched arch (2026-05-19, exp444)

## Headline
**exp444 = 1.4821 bpb @ 89.4 M, 2.12 h** vs exp428 (same arch, hard pick) = **1.4983 bpb @ 0.41 h**. Δ = **−16.2 mb** at matched architecture, matched effective batch (8 192 tokens/step via phys 8 + grad_accum 2), matched optimiser, matched LR schedule, matched seed.

## What changed (single knob)
All 4 LUT modules per layer (`qkv_lut`, `v_lut`, `out_proj`, `residual_lut`) swapped from `TinyMultiHeadLut(backward_mode='soft')` to `SoftMultiHeadLUT(hard=False)`.

| | hard pick (exp428) | soft mixture (exp444) |
|---|---|---|
| Forward output | `out_t = weights[t, argmax_k ts[t,k], :]` | `out_t = Σ_k sel_soft[t,k] · weights[t,k,:]` |
| Backward `dL/dweights` | hard `index_add` at picked row only | every row gets `sel_soft[k] · grad_out` |
| Backward chain (x, T_*) | identical | identical |

Implementation note: `TinyMHL(backward_mode='soft')` and `SoftMHL(hard=True)` are bit-equivalent at fp32 no-noise (parity-tested in `test_soft_backward_mode_gradients_match_softmhlut_fp32`). The only thing exp444 changes is the `hard=True → hard=False` switch on `SoftMHL` (no STE one-hot snap on forward).

## Trajectory shape — the load-bearing observation
| step | exp444 | exp428 | Δ (mb) |
|---|---|---|---|
| 200  | 2.2909 | 2.2589 | +32 |
| 600  | 1.9308 | 1.8963 | +34.5 |
| 1000 | 1.8189 | 1.7954 | +23.5 |
| 2000 | 1.6911 | 1.6772 | +13.9 |
| 2800 | 1.6307 | 1.6261 | +4.6 |
| **3400** | **1.5955** | **1.5975** | **−2.0 (crossover)** |
| 4000 | 1.5711 | 1.5744 | −3.3 |
| 5000 | 1.5335 | 1.5413 | −7.8 |
| 6000 | 1.5068 | 1.5202 | −13.4 |
| 7000 | 1.4915 | 1.5060 | −14.5 |
| 8000 | **1.4821** | **1.4983** | **−16.2** |

Shape: soft-mixture LAGS hard-pick by ~25–40 mb through warmup + early peak-LR, crosses over at ~step 3.4 K, and then opens a monotonically widening lead through step 8 K. Lead growth was still > 0 at the final eval — no plateau.

## Why
At bs=16-effective the LUT weight update is **per-row sparse**: only the chosen row of each table per token gets a gradient. With NAP=6, K=64 rows and ~8192 tokens/step ≈ 128 tokens/row on average if perfectly uniform — but routing is far from uniform, so cold rows starve. Soft mixture forward replaces `d_weights[chosen_row] += grad` with `d_weights[k] += sel_soft[k] · grad` for all K rows, giving every row a non-zero gradient on every token. The win is **purely** about gradient coverage, the same mechanism as bs-scaling — but achieved through forward-path continuity rather than more tokens.

Confirms: the bs=16-effective regime is **gradient-coverage-limited**, not capacity-limited. Hard pick's row collapse is the binding constraint; remove it and the model trains better.

## Cost
~5× wall-clock (0.41 h → 2.12 h at 8 K steps). The soft mixture adds:
- Forward: `K`-wide softmax + `[B·T, n_tables, K] @ [n_tables, K, n_out]` einsum per LUT call
- Backward: `sel_soft @ grad_out` for every row instead of hard `index_add`
- Activation memory: `O(K · n_outputs)` per LUT call per token — 6 layers × 4 LUT modules × bs=8 → forced grad_accum=2 on 80 GB H100 (won't fit at phys-bs=16)

## How to apply
**Default to `SoftMultiHeadLUT(hard=False)` for all LUT-LM training going forward**, at least when training compute is the bottleneck and we care about quality. Inference still wants hard pick, but exp444's tables can presumably be deployed via hard-pick argmax at convergence (untested — see Open Questions).

## Reframing
The prior intuition that "LUT models lose ground to vanilla late in training" was an artefact of hard-pick forward — gradient sparsity compounds over time as routing crystallises. With soft-mixture forward, exp444's trajectory **accelerates** mid-late training rather than plateauing. The "LUT-vs-vanilla" gap is now plausibly mostly a bs-scaling story; the hard-pick-vs-soft-mixture knob is independent and additive.

## Open question
**Hardened inference of an exp444 checkpoint** — argmax-snap forward at deploy. If the gap to soft-forward eval is small, this is the practical answer: pay 5× train, recover full bandwidth at inference. Worth a small ablation when the checkpoint is reused.
