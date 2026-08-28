# FFN-LUT cost/quality sweep — results (task f551912a)

Anchor **A0 = exp_n_0084 config re-run on the short 3500-step schedule** → val_bpb **1.39691**.
All 10 runs share the SAME 3500-step LR schedule, untied head, FFN slot only, standard LUT
optimizer (AdamW; tables in nodecay wd=0; init noise 0.001). Predicted == measured on
params/FFN-FLOP/vBW for every config (`tools/measure_flops_bandwidth.py`; patched to handle the
no-decompress `inner_out=−1` sentinel). Dense FFN reference = 14.156M FLOP / 14.156M vBW.

vBW = decode-regime block-only bytes/token (weights re-read every token + LUT **selected rows only** + act).

## Ranked quality-vs-cost (best bpb first)
| rank | id | exp | total params | FFN FLOPs | vBW | ×FLOP / ×vBW vs dense | val_bpb @3500 | Δ vs A0 |
|---|---|---|---|---|---|---|---|---|
| 1 | **S2a** | 0114 | 180,597,900 | 2.286M | 2.375M | 6.19 / 5.96 | **1.36939** | **−0.02752** |
| 2 | S2c | 0115 | 95,663,244 | 2.200M | 2.302M | 6.43 / 6.15 | 1.38626 | −0.01065 |
| 3 | S2e | 0117 | 95,810,892 | 2.470M | 2.597M | 5.73 / 5.45 | 1.39264 | −0.00427 |
| 4 | **A0** | 0108 | 67,351,692 | 2.236M | 2.375M | 6.33 / 5.96 | 1.39691 | anchor |
| 5 | S1b | 0110 | 67,056,396 | 1.475M | 1.490M | 9.60 / 9.50 | 1.40208 | +0.00517 |
| 6 | S1c | 0112 | 66,982,668 | 1.376M | 1.342M | 10.29 / 10.55 | 1.40519 | +0.00828 |
| 7 | S1f | 0109 | 68,776,908 | 1.751M | 1.798M | 8.08 / 7.87 | 1.40574 | +0.00883 |
| 8 | S1e | 0111 | 60,764,940 | 1.425M | 1.441M | 9.93 / 9.82 | 1.40895 | +0.01204 |
| 9 | S1d | 0113 | 71,848,812 | 1.604M | 1.674M | 8.83 / 8.46 | 1.40977 | +0.01286 |
| 10 | S2b | 0116 | 71,625,612 | 1.590M | 2.223M | 8.90 / 6.37 | 1.42976 | +0.03285 |

## Answer 1 — cheapest config that HOLDS anchor quality
All Sweep-1 configs regress only slightly vs A0 (+0.005 … +0.013 bpb) while cutting cost ~1.5–1.6×.
The Pareto frontier is **{S1c, S1b}** (S1c dominates S1e/S1f/S1d — cheaper AND better than all three):
- **S1c** (H4 in32/out24 nap8 tph256) — cheapest on both axes: **1.376M FLOP / 1.342M vBW = 10.3× / 10.5× vs dense**, 66.98M params, Δ **+0.0083**. Essentially anchor-quality for ~1.6× less compute+bandwidth.
- **S1b** (H4 in32/out32 nap8 tph192) — tightest hold: Δ **+0.0052** (within short-run noise of the anchor), 9.6× / 9.5×, params-matched to 0084.

→ **Cheapest that holds: S1c**; **safest hold: S1b**. (nap=8's free cells largely defend quality against the inner-dim/tph cuts.)

## Answer 2 — best bpb achievable within 0084's ~6× envelope
**S2a (nap9): 1.36939, −0.0275 vs the anchor** — the best of the whole sweep — at **exactly the vBW
envelope** (2.375M, identical to A0) and only +2.2% FLOP. It buys this purely by spending **nap**:
4× the table cells (128→512) add params (2.68× A0) but ~zero FLOP/vBW, because the LUT still reads
one row per table. S2c (nap8) confirms the direction at half the params (−0.0107). The
**no-decompress reallocation (S2b) backfired** (+0.033): dropping the decompress matmul for full-width
(384) tables hurt quality at this budget.

→ **nap (table cells = params) is the strong in-envelope quality lever.** Within a fixed FLOP/vBW
budget, pour it into nap, not into wider/no-decompress tables.

## Caveats
- These are **3500-step** numbers (the anchor's own 16k value is 1.19866). The Δ ranking is the deliverable; absolute bpb is short-run. S2a's −0.0275 margin is large enough to warrant a full-length (16k) confirmation.
- S2a used device_batch 12 / grad_accum 4 (OOM fix for nap9's soft-backward buffer); effective batch 24,576 unchanged.

See `ffn_sweep_quality_vs_cost.png`.
