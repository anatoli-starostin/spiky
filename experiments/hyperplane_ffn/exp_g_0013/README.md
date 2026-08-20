# exp_g_0013 — gradient-free row revival on the lean tph64 hard arm

Tracking issue: **#108**. Exploratory prototype.

Control arm: **`exp_g_0006`** (H8/d48/tph64 hard, 1.228335 @ 27,343,200 params). This config is
`exp_g_0006` plus six `lut_revival_*` keys and nothing else.

## 1. How the routing actually works

Read from `src/spiky/lutorch/fast_multi_head_lut.py`. Per `FastMultiHeadLut` instance, in
INDEPENDENT mode (`joint_head_compression: false`) there is one instance **per head per layer** —
48 of them here (6 layers × 8 heads), each with `n_heads=1`, `tables_per_head=64`.

| tensor | shape (this config) | kind |
|---|---|---|
| `weights` | `[64 tables, 64 rows, 48 outputs]` | **Parameter** (learnable) |
| `soft_anchor_a_long` | `[64 tables, 6]` | buffer — **not** learnable |
| `soft_anchor_b_long` | `[64 tables, 6]` | buffer — **not** learnable |
| `soft_powers` | `[32,16,8,4,2,1]` | buffer (MSB-first) |
| `log_soft_score_temp`, `log_select_temp` | scalar each | **Parameter** |

Routing, per table `t`, per token:

```
d_i   = x[a_{t,i}] - x[b_{t,i}]                    i = 0..NAP-1   (NAP = 6)
index = sum_i  (d_i > 0) * 2^(NAP-1-i)             MSB-first  ->  0..63
```

then the chosen row is gathered from each of the 64 tables and **summed** across the
`tables_per_head` axis (`F.embedding_bag(mode='sum')`).

**This is a hash, not a nearest-neighbour lookup.** Which row a token reaches depends only on the
*fixed* anchor pairs and the token's own coordinates — never on the row's contents. So no
modification of a row's values can attract traffic to it. Any revival scheme that assumes
"make the row better and tokens will come" is wrong here.

Learnable: table values, the two temperatures, and the CompressionMHL `compress`/`decompress`
projections. Fixed: the anchors, hence the partition.

### The asymmetry the mechanism exploits

- **Forward (hard)** reads exactly one row per table per token → a zero-traffic row does not
  affect the output.
- **Weight gradient** is a **1-row scatter** at the chosen row (`grad_w_flat.index_add_`) → a
  starved row barely trains.
- **Input gradient** is a full-K softmax surrogate:
  `d_sel_soft = einsum("bto,tko->btk", grad_pt, weights)` — it contracts against **all 64 rows**,
  so *every* row's value feeds the gradient of *every* token.

A starved row is therefore **untrained yet fully present in everything else's gradient** — a
stale, near-random vector injecting noise into `grad_x`. That is the only channel through which
this intervention can act, and it is what the experiment tests.

**Rejected alternative — perturbing anchors.** An anchor pair is shared by all 64 rows of its
table, so changing one flips an index bit for every token and re-partitions the whole table at
once, scrambling all its learned rows. That is a *table restart*, not a row revival, and it is
strictly more destructive.

## 2. Is there anything to revive? (measured first)

Before implementing, per-row traffic was measured on a **trained** checkpoint
(`exp_g_0009`, H8/d48, 147,456 real validation tokens, 98,304 rows):

```
rows with EXACTLY zero traffic  :      7 / 98,304  ( 0.01%)
rows below 0.01x uniform        :    306 / 98,304  ( 0.31%)
rows below 0.1x  uniform        :  7,226 / 98,304  ( 7.35%)
rows below 0.25x uniform        : 20,250 / 98,304  (20.60%)

top-8-of-64-rows traffic share  : mean 0.419  (uniform would be 0.125)
row-share p100                  : 19.6x uniform
dead rows per table             : mean 0.00 / 64, max 2
```

**Traffic is skewed but almost nothing is truly dead.** This matters for interpretation: the
mechanism is not reclaiming dead capacity (there is ~none). At the default `0.1x` threshold it
rewrites ~7% of rows, essentially all of which *do* receive some tokens.

## 3. Mechanism implemented

Gradient-free, **zero added model parameters**, fully gated by `lut_row_revival`.

- **Traffic tracking** — a forward hook on each of the 48 modules recomputes the routing index
  (identical math to `_soft_lut_fwd_body`) on the first `lut_revival_probe_rows` (4,096) rows of
  each batch and folds the per-row share into an EMA (`lut_revival_ema` = 0.99). Held in a plain
  dict of tensors, not module state — no parameters, no checkpoint change.
- **Trigger** — every `lut_revival_every_n` (500) steps, **after `optimizer.step()`** so a fresh
  value is not immediately overwritten by an update computed against the value it replaced.
- **Revival** — a row whose EMA share is below `lut_revival_threshold` × uniform (0.1 × 1/64) is
  overwritten with the values of the **busiest row of the same table** plus Gaussian noise scaled
  to `lut_revival_noise` × that donor row's RMS (0.1). Anchors are never touched.
- **Adam state** — `exp_avg` and `exp_avg_sq` are zeroed for every rewritten row. The values change
  discontinuously, so carrying stale momentum would apply updates computed for the vector that was
  replaced.
- **Anti-thrash** — a revived row's traffic estimate is reset to the uniform share so it is not
  re-revived on the very next trigger.

All of it lives in this experiment's `train.py`; **`src/spiky/lutorch/` is untouched**, so no other
experiment's behaviour can change. With `lut_row_revival: false` this file is behaviourally
identical to `exp_n_0040`'s `train.py`, which it was copied from.

## 4. Smoke test

`SMOKE=1 python train.py` → **`Params: 27,343,200`** — identical to `exp_g_0006`; revival adds
zero parameters. 6 CompressionMHL ✓, 48 `FastMultiHeadLut` ✓, `forward_mode == "hard"` live on all
48 ✓, LUT tables 9,437,184 reconciled against `depth·H·tph·2^nap·d_out` ✓.

A 12-step live run (real data, real optimizer) then verified the mechanism end-to-end:

```
traffic tracked for 48 / 48 modules; per-table shares sum to 1.0000
revive_rows fired: 123,166 / 196,608 rows (62.65%)
module0: 12 starved rows detected, 12 rows actually rewritten  (exact match)
all weights finite after revival: True
losses finite before and after; training continues cleanly post-revival
```

### Two honest caveats from that check

1. **62.65% is a 12-step artefact, not the steady state.** The EMA had seen 12 batches, and at
   step 12 the activation distribution is still degenerate, so most rows had genuinely not been
   visited yet. The real first trigger is at step 500 with a 500-step EMA (≈2M sampled tokens).
   The trained-checkpoint measurement above suggests ~7%. Every trigger logs its own count, so the
   run will report the truth rather than this estimate — **if the first in-run trigger is still
   very large, that is a red flag worth stopping for.**
2. **The refresh is *not* free-by-construction, and my first draft of the code said it was.**
   Rewriting a row with *exactly* zero traffic cannot move the forward output — but "starved"
   means low, usually **nonzero**, traffic, so some revived rows are read by some tokens. Measured
   on an identical batch: loss `8.8328227997` → `8.8329210281`, a jump of **+9.8e-05** (~1e-5
   relative). Small, but real. The code comment has been corrected to state this.

## 5. What would count as a result

Control is `exp_g_0006` = **1.228335**. This is an exploratory prototype with a mechanism whose
addressable mass was measured at ~7% of rows and whose action is indirect (gradient-surrogate
noise, not capacity). A neutral outcome is a perfectly plausible and still-useful result; the
occupancy measurement above already argues against a large effect.

## Status

Built, code-studied, occupancy-probed, smoke-tested, revival verified live, committed before run.
