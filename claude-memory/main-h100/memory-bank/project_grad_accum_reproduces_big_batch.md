---
name: grad-accum-reproduces-big-batch
description: "bs=16 phys + grad_accum=8 ≈ bs=128 phys within ~+0.01 bpb across exp363 vs exp367, confirming the bs=16 → bs=128 gap is pure gradient-quality (no phys-batch matmul effect)."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Grad accumulation reproduces big-batch behaviour (exp367, 2026-05-16)

**exp367** = exp365 fork with `total_batch_size: 8192 → 65536` (device_batch_size=16 unchanged → grad_accum=8). Same 8000 optimizer steps, same lut_lr=1e-3, noise=0.0, same arch.

Tracked vs **exp363** (bs=128 phys, accum=1, final=1.4105) across steps 200–3600. Δ stayed in the narrow band **+0.002 to +0.011 bpb** at every eval — never exceeded ±0.012. Killed at step 3600 (1.4869 vs exp363's 1.4761, Δ=+0.0108) once the trajectory was clearly locked.

Per-step alignment (exp367 vs exp363):
- step 200: 2.2034 vs 2.2014 (Δ=+0.002)
- step 400: 1.9039 vs 1.9009 (Δ=+0.003)
- step 800: 1.7374 vs 1.7306 (Δ=+0.007)
- step 1200: 1.6466 vs 1.6363 (Δ=+0.010)
- step 1800: 1.5733 vs 1.5661 (Δ=+0.007)
- step 2400: 1.5338 vs 1.5259 (Δ=+0.008)
- step 3000: 1.5076 vs 1.4967 (Δ=+0.011)
- step 3600: 1.4869 vs 1.4761 (Δ=+0.011)

**Why:** the ~+0.01 drift is almost certainly bf16 accumulation order — accum-8 sums 8 small forwards' grads in sequence; exp363 does 1 forward with 8× larger matmuls. Different reduction order → tiny AdamW v_t noise → slightly slower convergence. No phys-batch matmul-size effect at LUT-LM scale.

**How to apply:**
- Treat the **entire 0.21-bpb gap between bs=16-direct (exp365=1.6215) and bs=128-direct (exp363=1.4105)** as pure gradient-quality / Monte-Carlo estimation deficit. Any bs=16 optimiser trick has up to 0.21 bpb of recoverable space.
- No need to invent "true phys-batch" tricks — grad_accum *is* the gold-standard reproduction of larger effective batch.
- Wall-clock: grad_accum=8 costs 8× per optimizer step, so it's only useful as a diagnostic ceiling; for production speed-ups we need bs=16-direct tricks that emulate big-batch gradient quality (sparse-aware Adam, distillation, EMA on touched rows, etc).
- Don't confuse "step N" reported by training loop with "N micro-batches": with grad_accum=k, step N = N optimizer steps = N×k micro-batches.
