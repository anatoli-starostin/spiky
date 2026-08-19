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

## Result — completed 16,000 steps

**final = `1.2267103801621473`** (best 1.2266303077139210) · 22,624,704 params · **1.026 h**.

### Answer: width-neutrality does NOT survive `hybrid_smooth`

Against `exp_g_0009` (H8/d48, identical tables and projections, identical everything else):

```
exp_g_0010 (H16/d24) MINUS exp_g_0009 (H8/d48)

final delta @ step 16,000              -0.001014
evals where 0010 is better             78 / 80
last eval where 0010 was worse         step 11,000
  -> better at all 25 evals thereafter (from step 11,200)
mean delta, last quarter (12,200+)     -0.000931
min / max delta, last quarter          -0.001750 / -0.000512   (never touches zero)

   2000:  1.570207  vs  1.577130   -0.006923
   4000:  1.365297  vs  1.367813   -0.002516
   6000:  1.308277  vs  1.311042   -0.002765
   8000:  1.279376  vs  1.280820   -0.001444
  10000:  1.256327  vs  1.256982   -0.000655
  12000:  1.240713  vs  1.241781   -0.001068
  14000:  1.231167  vs  1.231732   -0.000565
  16000:  1.226710  vs  1.227724   -0.001014
```

**Many-narrow heads beat few-wide heads under smoothed routing**, and the ordering is the
*reverse* of what hard forward showed:

| forward mode | H16/d24 vs H8/d48 | verdict |
|---|--:|---|
| **hard** (`exp_n_0033` vs `exp_g_0006`) | +0.000427 (H16 worse) | indistinguishable — oscillating, inside noise |
| **hybrid_smooth** (`exp_g_0010` vs `exp_g_0009`) | **−0.001014 (H16 better)** | consistent — 78/80 evals, last quarter never crosses zero |

So the head/dim split is **not** a free parameter once routing is smoothed. This is the
interaction the experiment was posed to look for: `hybrid_smooth` blends the main row with its
Hamming-1 alternative, so its value depends on the partition geometry *within* a head — and
sixteen d24 partitions apparently give the blend more to work with than eight d48 partitions.

### Strength of the claim

Stronger evidence than the earlier width comparisons, but still one seed:

- **What supports it:** the sign is consistent across 78 of 80 evals and across every eval in the
  last quarter, where the delta ranges −0.001750 to −0.000512 and *never touches zero*. That is
  qualitatively unlike `exp_g_0008` vs `exp_n_0004`, which oscillated in sign inside ±0.0006.
- **What weakens it:** adjacent evals of a single run are autocorrelated, so 78/80 is **not** 78
  independent trials. The magnitude (−0.001014) is the same order as `exp_g_0009`'s −0.001038 over
  `exp_n_0033`. A second seed would settle it.

### It costs wall time

1.026 h against `exp_g_0009`'s 0.779 h — **+32%** — for 16 per-head LUT invocations per layer
instead of 8. That direction is consistent with the invocation-count ordering heuristic (which
under-predicted *magnitude* on `exp_g_0008`, but has been right about *order* within this host).

So the two arms are a genuine trade, not a free win: **−0.001014 bpb for +32% wall time** at
identical parameters.

### Standing on the leaderboard

| experiment | val_bpb | params | hours |
|---|--:|--:|--:|
| target | 1.190000 | — | — |
| dense `exp073` | 1.196646 | 23,209,728 | 1.391 |
| `exp_n_0040` H8/d48/tph256 | 1.204266 | 55,654,752 | 2.132 |
| `exp_n_0004` H8/d48/tph128 | 1.217377 | 36,780,288 | 1.323 |
| **`exp_g_0010` H16/d24/tph32 smooth** | **1.226710** | **22,624,704** | 1.026 |
| `exp_g_0009` H8/d48/tph32 smooth | 1.227724 | 22,624,608 | 0.779 |
| `exp_g_0006` H8/d48/tph64 hard | 1.228335 | 27,343,200 | 0.927 |
| `exp_n_0033` H16/d24/tph64 hard | 1.228762 | 27,343,296 | 1.279 |

Best CompressionMHL result *under the dense parameter budget*, and −0.002052 against
`exp_n_0033` at 4.7M fewer params. Still **+0.030064 short of the dense baseline** and +0.036710
short of the 1.19 target — the frontier moves, the ceiling does not.

> Note `best` (1.226630) and `final` (1.226710) differ slightly here — `best` occurs before step
> 16,000. Use **final** for endpoint comparisons across arms; mixing best-vs-final would
> manufacture an artefact.

## Status

Complete. Results committed and pushed. `checkpoint.pt` not committed (gitignored under
`experiments/**/*.pt`).
