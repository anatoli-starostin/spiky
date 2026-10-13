# lutorch_ex performance snapshot ADDENDUM: #162 re-measured (c8e48c36), QuantisedConfidenceLUT n=1

**ADDENDUM, NOT PART OF THE v2 SESSION.** These cells come from a later session than the v2 body
(`snapshot_*_v2.json`). They are comparable with each other (one session, interleaved) but **must not be differenced
against v2 cells**: gaps drift by several percent between sessions.

**Why it exists:** the audit of (class × read_top_n × dtype) found one runnable configuration bench_snapshot never
measured: the quantised reference at read_top_n=1. bench_snapshot now has the row (bd4e5dcb). Rather than mix sessions
inside the snapshot of record, the row was measured on its own on all three trees.

**Session.** 2026-10-13, #162's code (c8e48c36), #160's code (a811390e) and #164 HEAD (bd4e5dcb), interleaved cell by
cell in a rotating order, one fresh process per cell. Each cell ran the tree's OWN `bench_snapshot.measure()` with its
own `_conf(QuantisedConfidenceLUT, 1)` constructor (the per-cell path its CLI runs; the #160 / #162 cell lists predate the
row, so their CLI could not name it): Dynamo reset, 10 warm-up + 30 timed iterations, median, peak above the pre-step
allocation. Champion geometry h16 tph64 nap8 d48 at 8,192 (`champion`) and 32,768 (`large`) vectors.

**Conditions:** SM up to 3.06 GHz, 35–42 C. **No throttle or power-cap flag on any cell.**

**A real regression, re-measured before believing it.** HEAD's fp32 cells are slower than both baselines (eval ×1.27
at 8k, ×1.29 at 32k; train total +11–12% at 8k, +6–11% at 32k). Three further interleaved rounds on all three trees
reproduced it (HEAD eval 1.651–1.654 ms vs 1.302–1.304 at 8k; 6.19–6.52 vs 4.79–5.13 at 32k), so the row was NOT
replaced. A bisect on scratch trees puts the step at 41df6f25 (Stage 3): the n=1 reference forward now computes k' with
`_pow2.single_exponent` (the deployed kernel's integer order) instead of reusing its logsigmoid pass. See the note.

**bf16:** refused on every tree (the quantised reference rejects low precision by design).

Comparisons: `figures_162_remeasured_vs_164_v2_addendum/`, `figures_160_vs_164_v2_addendum/` (bench_compare.py --same-session).

| geometry | cartridge | dtype | status | eval forward ms | GiB | train forward ms | GiB | train backward ms | GiB | train total ms | GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | QuantisedConfidenceLUT n=1 | float32 | ok | 1.300 | 0.165 | 1.371 | 0.228 | 5.065 | 0.555 | 6.500 | 0.555 |
| champion | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
| large | QuantisedConfidenceLUT n=1 | float32 | ok | 4.796 | 0.520 | 5.198 | 0.770 | 21.041 | 1.932 | 26.152 | 1.932 |
| large | QuantisedConfidenceLUT n=1 | bfloat16 | refused: QuantisedConfidenceLUT does not support low precision | - | - | - | - | - | - | - | - |
