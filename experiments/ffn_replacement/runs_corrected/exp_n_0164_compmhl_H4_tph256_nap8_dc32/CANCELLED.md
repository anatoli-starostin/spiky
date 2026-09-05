# CANCELLED — superseded by the short proxy sweep

Stopped deliberately on 2026-09-05, not a failure. Anatoly's call: rather than spend ~4.7 h
on one 16k-step point, spend a comparable budget on **11 short 4k-step proxy runs** that
sweep the shape of the compression FFN. This run's configuration is reproduced exactly as
**S1** in that sweep (`sweep_s01_tied_H4_tph256_c256_din32_dout32`), which doubles as the
regression check that the untied `d_in`/`d_out` path matches the tied one.

**Step reached: 1,700 of 16,000** (10.6%). The loss and the val curve were healthy
throughout — no spikes, no NaNs:

| step | train loss (ema) | val bpb (fixed protocol) |
|---|---|---|
| 200 | 9.383217 | 2.598185 |
| 400 | 7.606691 | 2.118845 |
| 600 | 6.639104 | 1.931579 |
| 800 | 6.183895 | 1.834670 |
| 1000 | 5.885968 | 1.757988 |
| 1200 | 5.650139 | 1.695074 |
| 1400 | 5.450853 | 1.637774 |
| 1600 | 5.293622 | 1.591431 |

The trainer (PID 742595) was sent SIGTERM and exited cleanly; GPU memory returned to the
desktop baseline (2,083 MiB) with no orphaned python or dataloader workers. No checkpoint was
written — the trainer saves only at the end — so there is nothing to resume from.

## Config, for the record

| | |
|---|---|
| params | **79,639,308** (confirmed in the trainer log via `SMOKE=1`) |
| | = 50,331,648 tables (6·4·256·256·32) + 592,908 projections + 28,714,752 base |
| shape | H=4, tph=256, nap=8 (256 cells/table), `d_in` = `d_out` = 32 |
| batch | device_batch **12** / grad_accum 4 → effective batch 24,576 tokens (48 sequences) |
| throughput | 0.952 steps/s blended, including the fixed-protocol evals every 200 steps |
| projection FLOPs | `H*384*d_c` = 49,152 vs vanilla `384*384*4` = 589,824 → **0.0833×** |

device_batch 12 fits here (unlike `exp_n_0162`/`0163`) because the nap8 soft-backward buffer
is `[tokens, H*tph=1024, 2^8=256]` fp32 = 6.4 GiB — half the nap9 buffer.

## Comparability warning

The numbers above are on the **long-run** budget (16k steps, effective batch 48 sequences)
and are directly comparable to the other 16k anchors — vanilla `exp_n_0135` 1.165147, naive
LUT `exp_n_0136` 1.192926, `exp_n_0118` 1.164939, `exp_n_0129` 1.170961 — except that this run
never reached 16k. **The proxy-sweep numbers in `sweep_s*/` are on a different budget
(4k steps, effective batch 24 sequences) and are comparable only to each other**, never to
these anchors or to the partial curve above.
