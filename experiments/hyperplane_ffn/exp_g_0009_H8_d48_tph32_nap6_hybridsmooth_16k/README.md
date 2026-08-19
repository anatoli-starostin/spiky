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

## Result — completed 16,000 steps

**final = best = `1.2277236319109346`** · 22,624,608 params · **0.779 h** on the 5090.
(Final and best coincide at step 16,000; the curve was still descending at the budget.)

| reference | val_bpb | params | Δ 0009 | Δ params |
|---|--:|--:|--:|--:|
| target | 1.190000 | — | +0.037724 | — |
| dense `exp073` (tied vanilla) | 1.196646 | 23,209,728 | +0.031078 | **−585,120** |
| `exp_n_0040` H8/d48/tph256 | 1.204266 | 55,654,752 | +0.023458 | −33,030,144 |
| `exp_n_0004` H8/d48/tph128 | 1.217377 | 36,780,288 | +0.010347 | −14,155,680 |
| `exp_g_0006` H8/d48/tph64 **hard** | 1.228335 | 27,343,200 | **−0.000611** | −4,718,592 |
| `exp_n_0033` H16/d24/tph64 **hard** | 1.228762 | 27,343,296 | **−0.001038** | −4,718,688 |
| `exp_g_0007` recursive | 1.236749 | 21,743,280 | −0.009025 | +881,328 |

### The finding: smoothing substitutes for table budget

`exp_g_0009` matches-or-beats **both** hard-forward tph64 arms while using **half their tables**
(4,718,592 vs 9,437,184), **4.7M fewer total params**, and **39% less wall time** (0.779 h vs
1.279 h). Against its nearest hard-forward sibling `exp_g_0006` — same H8/d48, double the tables —
it is 0.000611 *better*, not worse.

So the hypothesis holds in the direction it was posed: much of what a bigger table budget buys is
**boundary accuracy**, and top-2 blending recovers it without paying for the tables.

### The crossover is the interesting part

```
matched-step vs exp_n_0033:
   2000:  1.577130  vs  1.567088   +0.010042    <- behind
   4000:  1.367813  vs  1.365938   +0.001875
   6000:  1.311042  vs  1.309116   +0.001926
   8000:  1.280820  vs  1.280246   +0.000574
  10000:  1.256982  vs  1.257420   -0.000438    <- crosses over
  12000:  1.241781  vs  1.242677   -0.000896
  14000:  1.231732  vs  1.232928   -0.001196
  16000:  1.227724  vs  1.228762   -0.001038    <- ahead
```

Half the tables costs real ground early (+0.010 at step 2,000) and smoothing pays it back
gradually. On the full 200-step grid the transition is not a single clean crossing: the delta
**first dips negative at step 6,800**, then oscillates about zero through step 9,000 (last
positive eval: +0.000066 at 9,000), and is **negative at every one of the 35 evals from step
9,200 to 16,000**. So the honest description is *first crossing at 6,800, durable from 9,200* —
not the "around step 10,000" a coarse every-2000 sampling suggests. Mean delta over the last
quarter (steps 12,200–16,000): **−0.000879**.

**A short-budget run would have concluded the opposite.** Anything reading this line at ≤6k steps
gets the wrong sign, and anything in 6.8k–9k reads noise.

### How strongly to hold it

The **capacity and speed savings are unambiguous**: −4.7M params, −39% wall, under the dense
budget, at no loss cost. That part needs no statistics.

The **−0.001038 win over `exp_n_0033` should be held loosely.** It is a single seed. It is
~2.4× the ±0.0006 band used to dismiss earlier deltas, and unlike the `exp_g_0008`/`exp_n_0004`
comparison it is *sign-stable and monotone* over the last four evals rather than oscillating —
which is what makes it worth reporting at all. But "≤ hard forward at half the tables" is the
claim the data supports; "better than" needs a second seed.

Still **+0.031078 short of the dense baseline** and +0.037724 short of the 1.19 target, so this
does not yet answer #108 — it moves the efficient frontier, not the ceiling.

## Status

Complete. Results committed and pushed. `checkpoint.pt` deliberately not committed (gitignored
under `experiments/**/*.pt`; results are reproducible from `config.json`).
