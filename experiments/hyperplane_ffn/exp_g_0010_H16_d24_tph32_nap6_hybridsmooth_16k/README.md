# exp_g_0010 — H16 / d24 / tph32 / nap6, `hybrid_smooth` forward

Tracking issue: **#108**.

## Intent

A **pure width-axis test under smoothed routing**. `exp_g_0010` (H16/d24) against
`exp_g_0009` (H8/d48) at *identical* table budget and *identical* projections — the only thing
that moves is how the fixed inner width `H·d = 384` is split between heads and per-head
dimension.

Width-neutrality has been shown twice under **hard** forward:

- `exp_g_0006` (H8/d48) vs `exp_n_0033` (H16/d24): −0.000427, inside noise.
- `exp_g_0008` (H16/d48) vs `exp_n_0004` (H8/d48/tph128): delta oscillating inside ±0.0006 over
  steps 2,400–5,600.

The open question this run asks: **does width-neutrality survive `hybrid_smooth`?** There is a
concrete reason it might not. `hybrid_smooth` blends the main row with its Hamming-1 alternative,
so its behaviour depends on the geometry of the partition *within a head*. Sixteen narrow d24
heads and eight wide d48 heads carve up the same 384-d inner space very differently, and a
boundary-smoothing rule could plausibly be worth more (or less) in one regime than the other.
If neutrality holds here too, the head/dim split is confirmed as a free parameter — pick it for
speed. If it breaks, smoothing interacts with head geometry, which is a new mechanism.

## Config

Config-only experiment: `train.py` is **byte-identical** to `exp_n_0033`'s (`cmp` clean).

Exactly the head/dim split differs from `exp_g_0009`:

```
  lut_n_heads:       8  ->  16
  lut_inner_in_dim:  48 ->  24
  lut_inner_out_dim: 48 ->  24        (H·d held at 384)
```

Everything else held: `tph` 32, nap6, `forward_mode` `hybrid_smooth`, tied unembedder, 16,000
steps, seq 512, device_bs 48, total_bs 24,576, lr 3e-4, seed 1, `eval_every` 200, learnable temps.

## Smoke test

`SMOKE=1 python train.py` → **`Params: 22,624,704`**

| component | exp_g_0010 (H16/d24) | exp_g_0009 (H8/d48) | Δ |
|---|--:|--:|--:|
| tok_emb (tied to head) | 12,582,912 | 12,582,912 | 0 |
| **LUT tables** | **4,718,592** | **4,718,592** | **0** |
| attention (qkv+proj) | 3,538,944 | 3,538,944 | 0 |
| compress.weight | 884,736 | 884,736 | 0 |
| decompress.weight | 884,736 | 884,736 | 0 |
| block LayerNorms | 9,216 | 9,216 | 0 |
| compress.bias | 2,304 | 2,304 | 0 |
| decompress.bias | 2,304 | 2,304 | 0 |
| ln_f | 768 | 768 | 0 |
| LUT temps (log_soft_score_temp) | 96 | 48 | **+48** |
| LUT temps (log_select_temp) | 96 | 48 | **+48** |
| **TOTAL** | **22,624,704** | **22,624,608** | **+96** |

### The two arms are NOT param-identical — they differ by exactly 96

Tables and projections *are* exactly invariant under the H↔d trade, as expected:

- tables: `depth · H · tph · 2^nap · d_out` — 6·8·32·64·48 = 6·16·32·64·24 = **4,718,592** ✓
- projections: `compress` is `Linear(384 → H·d)` and `decompress` is `Linear(H·d → 384)`, and
  `H·d` = 384 either way, so both are **unchanged** ✓

But the **learnable temps scale with `H`, not with `H·d`**: each `FastMultiHeadLut` owns one
`log_soft_score_temp` and one `log_select_temp`, so the count is `2 · depth · H` — 192 at H16
against 96 at H8, a difference of **+96**.

> **This corrects a claim made earlier in this sweep** that parameter count is "exactly invariant
> under the H↔d trade at fixed H·d". It is invariant in the tables and the projections, but not in
> total. The already-recorded numbers show it: `exp_n_0033` (H16) 27,343,296 vs `exp_g_0006` (H8)
> 27,343,200 differ by the same **96**. The discrepancy was always in the data; the claim was
> simply too strong.

96 params on 22.6M is 4 parts per million and cannot plausibly move val_bpb — but "param-identical"
is the wrong description, and the arms should be reported as *matched on tables and projections*,
which is what the experiment actually controls.

### Structural checks

- 6 `CompressionMultiHeadLUT` modules (= depth) ✓
- 96 `FastMultiHeadLut` modules (= depth × H = 6 × 16, INDEPENDENT mode) ✓
- `forward_mode == "hybrid_smooth"` **live on all 96 instances**, not merely present in
  `config.json` ✓
- LUT weight tensors are `(32, 64, 24)` each, reconciling to 4,718,592 against the closed form ✓
- Per-component sum reconciles to the reported total by assertion ✓

## Status

Built, cross-checked, smoke-tested, committed before launch. **Queued** behind `exp_g_0009` —
auto-launches on the 5090 the moment 0009 finishes.
