# exp032 — fast vanilla baseline (bs64, 4k steps)

A **fast, low-wait dense-transformer yardstick** for A/B-ing future LUT / hyperplane
experiments without paying the ~0.5 h full-baseline cost. Byte-for-byte clone of
**exp002_untied_vanilla_baseline_nebius** — pure `MinimalGPT` + RoPE, untied head,
standard dense GELU FFN (384→1536→384). **No LUT / hyperplane modules at all**;
`train.py` is copied verbatim (it is fully config-driven — the only edits are in
`config.json`).

## What it is (starting point = exp002)
exp002 is the on-box untied-vanilla anchor: MinimalGPT+RoPE, d384 / 6 layers / 6 heads /
seq 512, device_bs 48 / total_bs 24576 (grad_accum 1), 16 000 steps ≈ 393M tokens,
lr 3e-4, wd 0.1, warmup 0.1, seed 1, bf16, vocab 32768 → **val_bpb 1.20144, 35.79M params**,
0.54 h. Every hyperplane_ffn experiment compares against that number.

## Only changes vs exp002
| field | exp002 | **exp032** | why |
|---|---|---|---|
| `device_batch_size` | 48 | **64** | "batch size = 64" — 64 sequences/step |
| `total_batch_size` | 24576 | **32768** | = 64 × 512 → grad_accum = 1 (single micro-batch, effective batch = exactly 64 seq) |
| `n_steps` | 16000 | **4096** | short run |
| `eval_every` | 200 | **50** | 200 × 4096/16000 = 51.2 → 50; keeps ~82 eval points (exp002 had ~80) |

Everything else byte-identical: depth 6, n_embd 384, n_head 6, seq_len 512, lr 3e-4,
wd 0.1, warmup 0.1, eval_steps 10, seed 1, bf16, vocab 32768, same base_data_climbmix
subset + tokenizer. Params unchanged: **35,792,640**.

## Interpretation note
"batch size = 64" is read as **64 sequences per optimizer step** (grad_accum = 1). This is
a modest step *up* from exp002's 48-sequence batch. If you meant 64 in some other unit
(e.g. keep exp002's token budget and only change grad-accum), say so on GO and it's a
one-line config edit.

## Status
**Setup + committed, NOT launched.** Awaiting explicit GO (methodology: agree → commit →
GO → launch). This short run will land **higher** than 1.20144 (far fewer steps/tokens); it
is a fast *relative* baseline, **not** a replacement for exp002's reproducibility anchor.

Run (once GO'd), from `experiments/hyperplane_ffn/exp032_fast_vanilla_baseline_bs64_4k/`:
`sbox ~/projects/spiky/.venv/bin/python train.py`
Outputs alongside: `metrics.csv`, `summary.json`, `loss.png`, `checkpoint.pt` (checkpoint gitignored).
