# exp_g_0022 — v_lut output widened 16 → 64

Tracking issue: **#108**. Run on **gpustar (RTX 5090)**. Code commit `046dcb30`.

`exp_g_0021_smallE_concat_readout_noutdecomp` with **one** change: `d_v` 16 → 64, so each of the
6 attention heads carries a 64-wide value again (q/k were already 64 — attention is symmetric
once more). Config-only; `train.py` is byte-identical to exp_g_0021's.

`out_proj`'s `input_dim` is `H*d_v` in code, so it grew 96 → 384 automatically. The **readout path
is unaffected**: each block still emits `E=64`, so the 6-layer concatenation stays 384 and the
readout LUT keeps `input_dim = output_dim = 384`. v width is internal to attention.

## Result

**final = best = `1.3104001316627083`** · 57,097,522 params · **2.028 h**. Converged (last four
evals 1.312688 / 1.310804 / 1.310914 / **1.310400**).

### vs exp_g_0021 — a clean, flat win

```
  step      0022 (v64)   0021 (v16)      delta
  2000       1.617221     1.646070    -0.028849
  4000       1.479503     1.508976    -0.029473
  6000       1.412550     1.442771    -0.030221
  8000       1.373446     1.402826    -0.029380
 10000       1.347768     1.376342    -0.028574
 12000       1.328925     1.358312    -0.029387
 14000       1.316710     1.346877    -0.030167
 16000       1.310400     1.340616    -0.030216
```

**Ahead at 80/80 evals.** Mean delta −0.028857 over the first third and −0.030071 over the last;
the whole run spans −0.032209 to −0.015223.

This is unusual for this line: **no crossover, and the gap is essentially constant.** Almost every
other comparison in this sweep has involved an early lead that decayed or reversed (exp_g_0018 led
2,200 steps then lost; exp_g_0016b led to step 6,000 then reversed; exp_g_0020/0021 converged to a
tie). Here the two curves are parallel from the first eval — a **level shift**, not a transient.
That makes it the most unambiguous single-variable result of the small-E series.

### What it cost

```
params   49,356,082 -> 57,097,522   (+7,741,440, +15.7%)
gain     -0.030216 bpb
rate      0.003903 bpb per M params
wall      1.877 h -> 2.028 h  (+8%)
```

For scale against the exp_g_0017 attention sweep, which measured both axes on the same footing:

| lever | bpb per M params |
|---|--:|
| `inner_in` 24 → 96 (exp_g_0017) | 0.00587 |
| **v width 16 → 64 (this run)** | **0.00390** |
| `tph` 32 → 64 (exp_g_0017) | 0.00135 |

So widening v is a *good* buy — roughly 3× more parameter-efficient than buying tables, though
still short of widening `inner_in`.

### But the gap to baseline dominates everything

```
exp024 single-stream   1.2034   ->  +0.107000
exp010 dual-stream     1.1940   ->  +0.116400
```

A −0.030 gain is real, but the small-E concat line is still **+0.107 behind exp024** — the win
closes about a fifth of the deficit at a 15.7% parameter cost. On this trajectory, widening
individual sites will not reach baseline; the E=64 + concat-readout choice is what sets the level.

## Status

Complete. Single seed. Results committed.
