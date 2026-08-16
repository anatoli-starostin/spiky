# exp098_D5_H12_d32_tied_4k — Sweep D (fixed throughput H*d=384): H=12, d=32

Fixed-throughput head/dim split. Hold H*d=384 constant (compress+decompress matmul
cost 2*H*D*d = 294,912 MACs/layer identical for every run, ~4x cheaper than the dense
FFN's 1,179,648) and vary ONLY the H/d split. Param-matched to the 2x FFN budget.

- H (lut_n_heads) = 12, d (inner_in=inner_out) = 32, nap=6, gamma=0, tied, AdamW two-group.
- fixed (compress 147,840 + decompress 147,840) = 295,680
- tables budget = 2,063,616 -> tph = 84 (tables 2,064,384)
- per-layer = 2,360,064 -> total = 30,292,224 (target 30,287,616, +4,608)

D3 (H=6,d=64) is C1/exp075 = 1.36613 (reused, not rerun). Refs: tied dense exp055 = 1.35543.
