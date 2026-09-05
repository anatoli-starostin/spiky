# Short proxy sweep — 11 runs over the shape of the compression FFN

Eleven cheap runs that vary the shape of `CompressionMultiHeadLUT` on one shared budget, so
the shapes can be ranked against each other for roughly the cost of one long run.

## ⚠️ These numbers are comparable ONLY to each other

Every run here is **4,000 steps at an effective batch of 24 sequences** (12,288 tokens) —
one eighth of the training budget of the 16k / batch-48 line. Their bpb therefore sits far
above the long-run anchors, and comparing a sweep number with `exp_n_0135` (1.165147),
`exp_n_0136` (1.192926), `exp_n_0118` (1.164939) or `exp_n_0129` (1.170961) is meaningless.

**S0 (`sweep_s00_vanilla_dense`) is this sweep's own zero-line.** Read every other run as a
delta against it. The warning is repeated in each run's `config.json` `_arch_note` and in its
`corrected_score.json` `comparability_warning`, so a number lifted out of context still
carries it.

## What IS shared exactly

* **Eval is not scaled down.** Every run is scored with the corrected `evaluate_bpb_fixed` —
  bs48 × 100 batches, leading 12 rows skipped, 2,451,456 tokens of the held-out
  `shard_06542.parquet`, batch-size independent. Only the *frequency* dropped, to every 500
  steps. So the runs are mutually comparable at full eval precision.
* **Recipe** is `exp_n_0118`'s otherwise: `ffn_type=compression`, independent per-head
  compression, untied unembedder, hard forward with learnable temps, 6 layers d=384 on
  ClimbMix, lr 3e-4, warmup_frac 0.1, cosine.
* **Effective batch is exactly 24 sequences** everywhere. `device_batch` × `grad_accum` is
  12 × 2, except the two runs with `H*tph*cells = 524,288` (S6, S10), where the soft-backward
  buffer `[tokens, H*tph, cells]` fp32 is 12.9 GiB at bs12 and OOMs the 5090 — those use
  6 × 4. Same effective batch either way, and eval is decoupled from `device_batch_size`.

## d_in / d_out were already untied

The brief asked for the LUT's input width and its table-row width to be separated. They
already are, and have been since PR #109:

```python
CompressionMultiHeadLUT(input_dim=384, output_dim=384,
                        inner_in_dim=d_in,    # width the sign comparisons are computed on
                        inner_out_dim=d_out,  # width of the table rows that get summed
                        nap=..., tph=..., n_heads=...)
```

`model_build.py` passes `lut_inner_in_dim` / `lut_inner_out_dim` straight through, and
`CompressionMultiHeadLUT.param_count` computes the table budget as
`n_heads * tph * 2**nap * eff_out` — **tables scale with `cells × d_out` only; `d_in` touches
nothing but the compress projection**, exactly as required. No library change was needed. S1
sets `d_in == d_out == 32` and is the tied control that reproduces `exp_n_0164`, which is
also the regression check that the untied path is a strict generalisation of the tied one.

## The runs, in execution order

| # | run | H | tph | cells (nap) | d_in | d_out | params | dev vs brief | dev batch | what it isolates |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S0 `s00_vanilla_dense` | — | — | dense 4×MLP | — | — | 35,792,640 | −0.02% | 12×2 | the sweep's zero-line |
| 2 | S1 `s01_tied_…din32_dout32` | 4 | 256 | 256 (8) | 32 | 32 | 79,639,308 | +0.05% | 12×2 | tied control = exp_n_0164 |
| 3 | S2 `s02_din64_…` | 4 | 256 | 256 (8) | 64 | 32 | 79,934,988 | −0.33% | 12×2 | d_in ladder 2/3 |
| 4 | S3 `s03_din96_…` | 4 | 256 | 256 (8) | 96 | 32 | 80,230,668 | −0.70% | 12×2 | d_in ladder 3/3 |
| 5 | S9 `s09_pure_H1_tph1024_…` | 1 | 1024 | 256 (8) | 128 | 32 | 79,418,124 | −0.60% | 12×2 | pure-FastMHL end of the head trade |
| 6 | S8 `s08_H2_tph512_…` | 2 | 512 | 256 (8) | 64 | 32 | 79,491,852 | −0.39% | 12×2 | midpoint of the head trade |
| 7 | S6 `s06_isoparam_deep_c512_dout16` | 4 | 256 | 512 (9) | 32 | 16 | 79,491,852 | −0.14% | **6×4** | iso-param, deep/narrow |
| 8 | S7 `s07_isoparam_shallow_c128_dout64` | 4 | 256 | 128 (7) | 32 | 64 | 79,934,220 | −0.83% | 12×2 | iso-param, shallow/wide |
| 9 | S4 `s04_dout16_…` | 4 | 256 | 256 (8) | 32 | 16 | 54,326,028 | −0.14% | 12×2 | d_out ladder, low |
| 10 | S5 `s05_dout48_…` | 4 | 256 | 256 (8) | 32 | 48 | 104,952,588 | +0.05% | 12×2 | d_out ladder, high |
| 11 | S10 `s10_scaled_H2_tph512_c512_…` | 2 | 512 | 512 (9) | 64 | 32 | 129,823,500 | −0.21% | **6×4** | scaled candidate |

Every built count was checked against the brief before any training started (`make_sweep.py`
builds each model and compares); all 11 land within 1%, the residual being the projection
biases the brief's formula omits. `H*tph ≤ 1024` and `d_in ≥ 32` hold for all.

Compress-projection FLOPs, `H*384*d_in` against vanilla's `384*384*4` = 589,824:

| run | FLOPs | ratio |
|---|---|---|
| S1, S6, S7, S4, S5 (d_in 32, H4) | 49,152 | 0.0833× |
| S8, S10 (d_in 64, H2) | 49,152 | 0.0833× |
| S2 (d_in 64, H4) | 98,304 | 0.1667× |
| S9 (d_in 128, H1) | 49,152 | 0.0833× |
| S3 (d_in 96, H4) | 147,456 | 0.2500× |

Note S1 / S8 / S9 are iso-FLOPs as well as iso-table-budget — that is what makes the head
trade a clean comparison.

## Files

| file | what |
|---|---|
| `make_sweep.py` | generates the 11 run folders and verifies every param count |
| `sweep_manifest.json` | the run order, shapes, param counts and FLOPs ratios |
| `run_sweep.sh` | runs them sequentially: SMOKE param check → train → score |
| `score_sweep.py` | writes each run's `corrected_score.json` on the corrected protocol |
