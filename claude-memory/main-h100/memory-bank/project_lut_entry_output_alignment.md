---
name: lut-entry-output-alignment
description: "Probe of LUT output=Σ selected entries: low-tph out_proj (exp365 tph=128) collapses to redundant rank-1 code (cos≈1); high-tph (exp454 tph=1024) stays distributed (cos≈0.15). Collapse is a selection phenomenon, not stored degeneracy. 2026-05-21."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Entry↔output alignment probe (2026-05-21)

Tool: `nanochat_exps/analysis_entry_output_corr/analyze.py` — execs an exp's train.py prefix (up to model creation, before file writes), loads checkpoint, hooks each TinyMultiHeadLut, recomputes selected rows bit-faithfully via `_soft_index_signpack`, decomposes `output = Σ_t selected_entry_t` over tph tables. Metrics per (token,head): cos(entry, output), R/tph=‖o‖²/(tph·mean‖e‖²) (coherence; ~1/tph=distributed, ~1=redundant), dominance, intrinsic per-table row similarity. merr (recomputed vs hooked output) ~1e-5 = faithful.

## Findings
- **Low-tph out_proj collapses to a redundant rank-1-ish code at depth.** exp365 (tiny, out_proj tph=128): L5 out_proj cos(entry,output)=0.9991, R/tph=0.976 — the 128 tables each emit ~the same direction. Early layers distributed (cos~0.3).
- **High-tph stays distributed.** exp454 (out_proj tph=1024): cos≈0.15, R/tph≈0.03 everywhere — output is a sum of many near-independent slivers; no collapse at any depth.
- **Collapse is a SELECTION phenomenon, not stored degeneracy.** intrinsic per-table row similarity ≈0 even where collapsed — the table's *stored* rows stay near-orthogonal; the *data* just lands on a coherent slice at low tph.
- qkv most coherent of the modules (cos~0.33–0.48) but never collapses.

## Why it matters
Mechanistically explains why out_proj benefits from MORE tables (cf. exp390 NAP=4, and the exp464/466 [[lut-scatter-specialization-sota]] win): more tables → forced distributed code → no rank-1 collapse → better capacity use. The low-tph collapse wastes tables.
