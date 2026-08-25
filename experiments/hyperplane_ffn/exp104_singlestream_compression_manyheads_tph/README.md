# exp104 — many routing-heads + RAISED tph (table-capacity boost)

Clone of exp103 (48 routing mini-heads grouped 8→1 into 6 attention heads, full-rank
routing) with the LUT **tph raised** to add pure table capacity at fixed routing rank:
- q_lut, k_lut: tph 4 → **16**
- v_lut: tph 8 → **32**
- out_proj: tph 128 (unchanged)

Everything else identical to exp103: CompressionMultiHeadLUTMH, 48 routing heads, inner_in=48,
`batched_multi_head_input=True` all sites, single-stream, no FFN, 6L, E=384, batch 48 seq
(24,576 tok/step), 16k steps, Lion-on-LUT-tables / AdamW split. Shared lutorch NOT modified.

**Rationale:** the compress projection W_c is a fixed per-site cost independent of tph, so
raising tph adds table CAPACITY without changing routing RANK. This lifts q/k/v to
**exact cell-parity with exp023** while keeping full-rank 48-mini-head routing.

**Verified build (smoke + fwd/bwd):** TOTAL **108,973,872** (~109M). Per-site (×6 layers):
- q_lut 10,040,844 (compress 5,322,240 / table 4,718,592)
- k_lut 10,040,844 (same)
- v_lut 43,070,988 (compress 5,322,240 / table 37,748,736)
- out_proj 20,648,460 (compress+decompress 1,774,080 / table 18,874,368)
- rest (non-LUT: tok_emb + untied unembed + norms) 25,172,736
- LUT A(routing/compress) 17,740,800 · B(table) 66,060,288 · A:B 0.269

vs exp103: A (routing) UNCHANGED (17,740,800, compress is tph-independent); B (tables) grew
30,670,848 → 66,060,288 (2.15×). Cells now match exp023 exactly at every attention site
(total 1,130,496 ×6). No duplicate-key bug (effective tph verified q/k=16, v=32, out=128).

**Targets:** exp023 (Hyperplane, 276.5M) 1.20632; exp103 (many-heads, tph4/8) final.
Outputs: metrics.csv, summary.json, loss.png, checkpoint.pt.
