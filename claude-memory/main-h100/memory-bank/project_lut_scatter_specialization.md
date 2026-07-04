---
name: lut-scatter-specialization-sota
description: "Output-scatter + proportional tph (many narrow output-specialized tables vs fewer wide, equal params) lowers bpb: exp466 qkv+out_proj = 1.4937 (−0.0030 vs exp453) but at ~2x LUT lookups; set aside as too heavy. 2026-05-21."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Output-scatter specialization — real bpb win, costly (2026-05-21, exp462–466)

`sparse_scatter_n_outputs` in TinyMultiHeadLut: each table writes only `n_out/f` outputs, scattered (balanced) into the full `n_out`-wide output → each output dim is assembled from ~tph/f tables instead of all tph. At fixed output dim this is "f× fewer, narrower, output-specialized tables."

## Results (fork exp453, LION 0.9/0.95, bs=16, 8K, 1.4967)
| exp | restructure | params | final | Δ |
|---|---|---|---|---|
| exp462 | out_proj 4× scatter (tph=1024) | out_proj 25.2M→6.3M | 1.5120 | +0.0153 |
| exp464 | out_proj 4× scatter + **4× tph** (4096) | param-matched 25.2M | **1.4943** | **−0.0024** |
| exp465 | qkv 4× scatter + 4× tph (256) | param-matched | 1.4951 | −0.0016 |
| **exp466** | **qkv + out_proj** scatter + 4× tph | param-matched 89.4M | **1.4937** | **−0.0030** |
| exp460 | ALL LUTs 4× scatter (no tph bump) | — | +0.045 (killed) | confounded: v_lut 16→4 over-sparsified |

## Findings
1. **Many narrow output-specialized tables beat fewer wide tables at EQUAL params.** exp462→exp464: the +0.015 sparsity loss was pure capacity; restore it as 4× MORE tables (not wider) and you end up −0.0024 AHEAD of dense exp453. Confirmed on both out_proj and qkv.
2. **Wins stack sub-additively (~75%):** out_proj −0.0024 + qkv −0.0016 → combined −0.0030 (not −0.0040). out_proj dominates.
3. **Cost: 4× more lookups/token** on the restructured modules (read-bandwidth; params & write-bandwidth matched). exp466 ≈ 2× total LUT lookups. **Set aside — too heavy for the matmul-free goal; user prefers exp453 (lighter, ~equal).**
4. Mechanism ties to the alignment probe [[lut-entry-output-alignment]]: out_proj is the collapse-prone high-tph module; forcing each table to specialize on a few output dims uses capacity better than wide tables that all write everything.

Lowest bpb of the session, but not adopted (lookup cost). exp453 (and the exp475 mean-abs recipe) remain the lightweight reference. Code: `out_scatter_factor`/`qkv_scatter_factor` config + `_scatter_kw` in the exp46x train.py.
