# exp032 — fast vanilla baseline (4k steps)

A **fast, low-wait dense-transformer yardstick** for A/B-ing future LUT / hyperplane
experiments without paying the ~0.54 h full-baseline cost. Byte-for-byte clone of
**exp002_untied_vanilla_baseline_nebius** — pure `MinimalGPT` + RoPE, untied head,
standard dense GELU FFN (384→1536→384). **No LUT / hyperplane modules at all**;
`train.py` is copied verbatim (fully config-driven).

## What it is (starting point = exp002)
exp002 is the on-box untied-vanilla anchor: MinimalGPT+RoPE, d384 / 6 layers / 6 heads /
seq 512, device_bs 48 / total_bs 24576 (grad_accum 1), 16 000 steps ≈ 393M tokens,
lr 3e-4, wd 0.1, warmup 0.1, seed 1, bf16, vocab 32768 → **val_bpb 1.20144, 35.79M params**,
0.539 h. Every hyperplane_ffn experiment compares against that number.

## Only change vs exp002 — n_steps
| field | exp002 | **exp032** |
|---|---|---|
| `n_steps` | 16000 | **4096** |

**Everything else is byte-identical to exp002:** depth 6, n_embd 384, n_head 6, seq_len 512,
device_batch_size 48, total_batch_size 24576, lr 3e-4, wd 0.1, warmup 0.1, eval_every 200,
eval_steps 10, seed 1, bf16, vocab 32768, same base_data_climbmix subset + tokenizer.
Params unchanged: **35,792,640**. (With eval_every 200 over 4096 steps that's ~20 eval
points + the final.)

## Estimated runtime
4096 / 16000 × 0.539 h ≈ **0.138 h ≈ 8.3 min** on the H100 (throughput is per-token identical
to exp002 — same batch, same everything but step count).

## Status
**Setup + committed, NOT launched.** Awaiting explicit GO (methodology: agree → commit →
GO → launch). Trains only 25.6% of exp002's steps, so it will land **higher** than 1.20144;
it is a fast *relative* baseline, **not** a replacement for exp002's reproducibility anchor.

Run (once GO'd), from `experiments/hyperplane_ffn/exp032_fast_vanilla_baseline_4k/`:
`sbox ~/projects/spiky/.venv/bin/python train.py`
Outputs alongside: `metrics.csv`, `summary.json`, `loss.png`, `checkpoint.pt` (checkpoint gitignored).
