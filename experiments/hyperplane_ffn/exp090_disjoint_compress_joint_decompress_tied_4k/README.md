# exp090 — disjoint per-head compress + JOINT (shared) decompress, tied, 4k

Forked from the best 6-head tied short config **exp060 (B5: 6h, 64/64, nap6, gamma0, tied,
4096 steps, val_bpb 1.38111)**. Tests replacing the **per-head (disjoint) decompress** with a
**single shared (joint) decompress**, param-matched to exp060.

## Architecture (Anatoli's spec)
```
x_norm = ln2(x)
per head h:  z_h = compress_h · Linear(384 → inner_in)     # DISJOINT compress (joint_head_compression=False)
             y_h = LUT_h(z_h)                                # per-head hard LUT, inner_in → inner_in
SUM over heads → [N, inner_in]                               # heads summed in the shared inner-width space
out = joint_decompress · Linear(inner_in → 384)             # ONE shared decompress
x = x + out                                                 # gamma=0, no parallel Linear
```
Implemented with the shared module (no fork): `CompressionMultiHeadLUT(input_dim=384,
output_dim=inner_in, inner_in_dim=inner_in, inner_out_dim=-1, joint_head_compression=False)`
produces the summed [N, inner_in] vector; a local `nn.Linear(inner_in → 384)` (zero-init'd) is
the joint decompress. (Verified the module supports output_dim≠384 with inner_out_dim=-1; the
shared module + its tests are untouched.)

## Param-match to exp060 (both = 23,214,336)
| component | exp060 (disjoint+disjoint, tph36) | exp090 (disjoint+joint, tph41) |
|---|---|---|
| compress (6× Linear 384→64) | 147,840 | 147,840 |
| per-head LUTs | 6·36·64·64 = 884,736 | 6·41·64·64 = 1,007,616 |
| decompress | 6× Linear 64→384 = 147,840 | **1× Linear 64→384 = 24,960** |
| FFN slot/layer | 1,180,416 | 1,180,416 |
| **total (tied)** | **23,214,336** | **23,214,336** |

The joint decompress saves 122,880/layer vs the 6 disjoint decompressors; that budget goes
into more LUT tables (tph 36→41). Same total, so a clean A/B for "does sharing the decompress
help or hurt at fixed params?"

n_heads=6, inner_in=64, nap=6, **tph=41**, tied, 4096 steps, **AdamW** (standard trainer, LUT
tables in the no-wd group — not Lion). gamma=0.

## Status
Queued in the Sweep C rolling orchestrator's tail (launches after the Sweep C runs free a
slot — no oversubscription). On completion: report final val_bpb + delta vs exp060 (1.38111);
negative = joint decompression helps.
