# exp_g_0009 — H8 / d48 / tph32 / nap6, `hybrid_smooth` forward

Tracking issue: **#108**.

## Intent

A **super-small `hybrid_smooth`** point: *half* the table budget of `exp_g_0006`, asking whether
top-2 boundary-smoothing holds quality when the tables are cut.

`hybrid_smooth` replaces the hard row lookup with a **top-2 soft blend of the main row and its
Hamming-1 alternative** (`src/spiky/lutorch/fast_multi_head_lut.py`), so a point near a partition
boundary reads a mixture of the two rows it sits between rather than committing to one. The
hypothesis: if the loss cost of shrinking the table budget is mostly *boundary* error — inputs
landing near a partition edge and getting the wrong row — then smoothing the boundary should
recover much of what halving `tph` gives up, and a cheap LUT could stay competitive.

## Config

Config-only experiment: `train.py` is **byte-identical** to `exp_n_0033`'s (`cmp` clean).

Two fields differ from `exp_g_0006` (H8/d48/tph64/nap6, hard forward, tied):

```
  lut_tables_per_head:  64  ->  32
  lut_forward_mode:  "hard" -> "hybrid_smooth"
```

Everything else is held: H8, `inner_in`/`inner_out` 48 (H·d = 384), nap6, tied unembedder,
16,000 steps, seq 512, device_bs 48, total_bs 24,576, lr 3e-4, seed 1, `eval_every` 200,
learnable temps.

> **Note on provenance.** `exp_g_0006` was never committed and its folder was lost in a scratchpad
> wipe, so this config was reconstructed from `exp_n_0033` by applying `exp_g_0006`'s documented
> deltas (`inner_in`/`inner_out` 24→48, `n_heads` 16→8) and then the two changes above. The
> reconstruction is confirmed independently: this config diffs against nebius's
> `exp_n_0044_H8_d48_tph64_nap6_hybridsmooth_16k` in *exactly* `exp_name` and
> `lut_tables_per_head` (64→32) — two independent derivations agreeing.

## Its natural A/B partner is exp_n_0044

`exp_n_0044` (H8/d48/**tph64**/nap6/`hybrid_smooth`) is the same configuration at *double* the
tables. So **exp_g_0009 vs exp_n_0044 is a clean single-variable `tph` 32-vs-64 comparison at
fixed forward mode** — cleaner than exp_g_0009 vs exp_g_0006, which would confound `tph` with the
hard→smooth switch. When both have run, read them as the pair.

## Smoke test

`SMOKE=1 python train.py` → **`Params: 22,624,608`**

| component | params | share |
|---|--:|--:|
| tok_emb (tied to head) | 12,582,912 | 55.62% |
| **LUT tables** | **4,718,592** | 20.86% |
| attention (qkv+proj) | 3,538,944 | 15.64% |
| compress.weight | 884,736 | 3.91% |
| decompress.weight | 884,736 | 3.91% |
| block LayerNorms | 9,216 | 0.04% |
| compress.bias | 2,304 | 0.01% |
| decompress.bias | 2,304 | 0.01% |
| ln_f | 768 | 0.00% |
| LUT temps (log_soft_score_temp) | 48 | 0.00% |
| LUT temps (log_select_temp) | 48 | 0.00% |
| **TOTAL** | **22,624,608** | 100.00% |

**Table budget halves exactly, projections do not move.** LUT tables 9,437,184 → **4,718,592**
(exactly ×0.5), while compress/decompress stay at 1,774,080 — `tph` scales only the tables, since
the projections depend on `H·d`, which is unchanged at 384. Total drops 27,343,200 → 22,624,608,
i.e. **−4,718,592**, entirely the tables.

Verified against the closed form `depth · H · tph · 2^nap · d_out` = 6·8·32·64·48 = 4,718,592 ✓,
reconciled against the actual weight tensors (each FastMHL holds `(32, 64, 48)`).

### Structural checks

- 6 `CompressionMultiHeadLUT` modules (= depth) ✓
- 48 `FastMultiHeadLut` modules (= depth × H, INDEPENDENT mode) ✓
- `forward_mode == "hybrid_smooth"` **live on all 48 instances**, not merely present in
  `config.json` ✓
- Per-component sum reconciles to the reported total by assertion ✓

## Budget context

At 22.62M this is **below the 23,209,728-param dense tied baseline** (`exp073`, val_bpb
1.196646) — the first CompressionMHL arm in this sweep that fits under the vanilla budget rather
than merely near it. If `hybrid_smooth` holds quality here, that matters for the #108 goal, which
is explicitly "≥1.20 val_bpb *under* the vanilla param budget".

## Status

Built, cross-checked, smoke-tested, committed before launch. 16,000-step run launched on the 5090.
