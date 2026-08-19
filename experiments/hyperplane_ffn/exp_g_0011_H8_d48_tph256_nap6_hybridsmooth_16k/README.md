# exp_g_0011 — H8 / d48 / tph256 / nap6, `hybrid_smooth` forward

Tracking issue: **#108**.

## Intent

The **`hybrid_smooth` twin of `exp_n_0040`**. A true single-variable fork: config differs from
`exp_n_0040` in exactly `exp_name` and `lut_forward_mode` (`hard` → `hybrid_smooth`), and
`train.py` is **byte-identical** to `exp_n_0040`'s (`cmp` clean). `hybrid_smooth` adds **zero
parameters**, so the two arms are exactly param-matched at 55,654,752.

This is the *largest* table budget in the sweep (37,748,736 LUT params, 67.8% of the model) and
tests whether the smoothing benefit **survives at scale** or is a small-table phenomenon.

## The question, framed by what is already known

Smoothing has been measured at two table budgets, both at H8/d48, both single-variable:

| tph | hard | smooth | benefit of smoothing |
|--:|--:|--:|--:|
| 32 | — | 1.227724 (`exp_g_0009`) | — |
| 64 | 1.228335 (`exp_g_0006`) | 1.217592 (`exp_n_0044`) | **−0.010743** |
| 256 | 1.204266 (`exp_n_0040`) | **this run** | ? |

Two readings are live and this run separates them:

1. **Smoothing is a fixed-size win.** Another ~−0.0107 on top of `exp_n_0040` would land near
   **1.1935** — *below the 1.196646 dense baseline*, and essentially at the 1.19 target.
2. **Smoothing substitutes for table budget** and therefore *diminishes* as tables grow. Evidence
   for this reading already exists: `exp_n_0044` (tph64 smooth, 1.217592) lands within 0.0002 of
   `exp_n_0004` (tph128 hard, 1.217377) — smoothing bought exactly one doubling of `tph`. If that
   is the mechanism, tph256-smooth should land near a hypothetical tph512-hard, and the gain over
   `exp_n_0040` should be *smaller* than −0.0107, not equal to it.

Reading 2 is the better-supported prior. Reading 1 would be the first result in this sweep to
reach the target.

## Config

Single-variable fork of `exp_n_0040`:

```
  lut_forward_mode:  "hard"  ->  "hybrid_smooth"
```

Held: H8, `inner_in`/`inner_out` 48, nap6, **tph256**, tied unembedder, learnable temps, 16,000
steps, seq 512, device_bs 48, total_bs 24,576, lr 3e-4, wd 0.1, warmup 0.1, seed 1,
`lut_base_seed` 1000, bf16, `eval_every` 200.

> `train.py` here is `exp_n_0040`'s, which differs from `exp_n_0033`'s (used by `exp_g_0006`–
> `exp_g_0010`) by a single **additive diagnostic print** in `setup_optimizer` reporting the
> decay/no-decay split. It changes no model, optimizer, or data behaviour. Taking
> `exp_n_0040`'s makes "fork `exp_n_0040`, change only the forward mode" literally true and puts
> an independent LUT-table count in the run log.

## Smoke test

`SMOKE=1 python train.py` → **`Params: 55,654,752`** — matches `exp_n_0040` **exactly**, as it
must, since `hybrid_smooth` is a forward-path change with no parameters of its own.

| component | params | share |
|---|--:|--:|
| **LUT tables** | **37,748,736** | 67.83% |
| tok_emb (tied to head) | 12,582,912 | 22.61% |
| attention (qkv+proj) | 3,538,944 | 6.36% |
| compress.weight | 884,736 | 1.59% |
| decompress.weight | 884,736 | 1.59% |
| block LayerNorms | 9,216 | 0.02% |
| compress.bias | 2,304 | 0.00% |
| decompress.bias | 2,304 | 0.00% |
| ln_f | 768 | 0.00% |
| LUT temps (log_soft_score_temp) | 48 | 0.00% |
| LUT temps (log_select_temp) | 48 | 0.00% |
| **TOTAL** | **55,654,752** | 100.00% |

### Structural checks

- 6 `CompressionMultiHeadLUT` modules (= depth) ✓
- 48 `FastMultiHeadLut` modules (= depth × H, INDEPENDENT mode) ✓
- `forward_mode == "hybrid_smooth"` **live on all 48 instances**, not merely in `config.json` ✓
- weight tensors `(256, 64, 48)` each, reconciling to 37,748,736 against the closed form
  `depth · H · tph · 2^nap · d_out` = 6·8·256·64·48 ✓
- per-component sum reconciles to the reported total by assertion ✓

## Budget note

At 55.65M this is **2.4× the dense baseline's 23,209,728 params** — far outside the "under the
vanilla budget" framing of #108. It is a *mechanism* probe (does smoothing scale?), not a
candidate architecture. `exp_g_0009`/`exp_g_0010` remain the on-budget arms.

## Status

Built, cross-checked, smoke-tested, committed before launch. 16,000-step run launched on the 5090.
