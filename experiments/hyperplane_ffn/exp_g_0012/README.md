# exp_g_0012 — exp_g_0006 at 1.5× batch

Tracking issue: **#108**. **Built and smoke-tested; not yet launched.**

## Intent

Reconstruct the lost `exp_g_0006` config (H8/d48/tph64, **hard** forward, tied — 1.228335 at
27,343,200 params) and change **only the batch size**: `total_batch_size` 24,576 → **36,864**
(1.5×), with `n_steps` held at 16,000 so the run sees 1.5× the tokens, and lr held at 3e-4.

Tests whether more tokens per step moves this line toward 1.19.

## Config

`train.py` byte-identical to `exp_n_0040`'s — the same base `exp_g_0011` uses.

LUT knobs are `exp_g_0006`'s, reconstructed and **independently cross-checked**: this config
diffs against `exp_n_0044` (= `exp_g_0006` with `hybrid_smooth`) in exactly `exp_name`,
`forward_mode`, and the three batching fields below. Reverting `forward_mode` to `hard` recovers
`exp_g_0006` exactly — confirmed by the smoke test landing on 27,343,200 params, `exp_g_0006`'s
number to the digit.

| field | exp_g_0006 | exp_g_0012 | why |
|---|--:|--:|---|
| `total_batch_size` | 24,576 | **36,864** | the change under test (1.5×) |
| `device_batch_size` | 48 | **24** | forced — see below |
| `eval_steps` | 10 | **20** | to hold the validation set constant — see below |

### Why `device_batch_size` could not stay at 48

`train.py` computes

```python
tokens_per_step = DEVICE_BS * SEQ_LEN
grad_accum      = max(1, TOTAL_BS // tokens_per_step)   # INTEGER division
```

With `DEVICE_BS = 48`, `tokens_per_step` = 24,576 and `grad_accum = 36864 // 24576 = 1`, so the
**effective batch would be 24,576, not 36,864** — the run would be a silent duplicate of
`exp_g_0006` while its `config.json` claimed otherwise. There is no warning; the number is simply
truncated.

`DEVICE_BS · 512` must therefore divide 36,864 exactly, i.e. `DEVICE_BS` must divide 72:

| device_bs | tokens/micro | grad_accum | effective | exact? |
|--:|--:|--:|--:|:--|
| 48 | 24,576 | 1 | 24,576 | **NO — silently wrong** |
| 72 | 36,864 | 1 | 36,864 | yes |
| 36 | 18,432 | 2 | 36,864 | yes |
| **24** | **12,288** | **3** | **36,864** | **yes (chosen)** |

Gradient accumulation is mathematically equivalent to one large batch here: every micro-batch has
identical token count and the loss is `mean`-reduced then divided by `grad_accum`, so
mean-of-means equals the overall mean.

### Why `eval_steps` went from 10 to 20

This is a deliberate addition to the requested change, and it exists to *protect* the comparison.

`evaluate_bpb` consumes `EVAL_STEPS` batches from a loader built with `DEVICE_BS`, so the
validation set size is `eval_steps · device_bs · seq_len`. Every prior arm in this sweep used
48 × 10 × 512 = **245,760 tokens**. Dropping `device_bs` to 24 while leaving `eval_steps` at 10
would have measured `val_bpb` on **122,880 tokens — half the validation data** — making this run's
numbers noisier and not directly comparable to `exp_g_0006` or anything else in the sweep.

`eval_steps = 20` restores it exactly: 20 × 24 × 512 = **245,760**, identical to the baseline.
(Packing may group those tokens into sequences slightly differently, but the measurement covers
the same amount of data rather than half.)

Chosen so that the **one intended variable — batch size — is the only thing that changes in the
measurement**. If a strict "change nothing but the two batch fields" reading is preferred, set
`eval_steps` back to 10 and treat every `val_bpb` here as measured on half the usual sample.

## Smoke test

`SMOKE=1 python train.py` → **`Params: 27,343,200`** — matches `exp_g_0006` exactly, as it must:
batch size adds no parameters.

| component | params | share |
|---|--:|--:|
| tok_emb (tied to head) | 12,582,912 | 46.02% |
| LUT tables | 9,437,184 | 34.51% |
| attention (qkv+proj) | 3,538,944 | 12.94% |
| compress.weight | 884,736 | 3.24% |
| decompress.weight | 884,736 | 3.24% |
| block LayerNorms | 9,216 | 0.03% |
| compress.bias | 2,304 | 0.01% |
| decompress.bias | 2,304 | 0.01% |
| ln_f | 768 | 0.00% |
| LUT temps (log_soft_score_temp) | 48 | 0.00% |
| LUT temps (log_select_temp) | 48 | 0.00% |
| **TOTAL** | **27,343,200** | 100.00% |

### Checks

- 6 `CompressionMultiHeadLUT` modules (= depth) ✓
- 48 `FastMultiHeadLut` modules (= depth × H) ✓
- `forward_mode == "hard"` **live on all 48 instances** ✓
- weight tensors `(64, 64, 48)`, reconciling to 9,437,184 against `depth·H·tph·2^nap·d_out` ✓
- **effective batch resolves to exactly 36,864** (24 × 512 × 3), re-deriving `train.py`'s own
  formula ✓
- **validation tokens 245,760**, identical to every prior arm ✓

Compute: 589,824,000 training tokens over 16,000 steps, against `exp_g_0006`'s 393,216,000.

## Memory

`device_batch_size` 24 is **half** the 48 that `exp_g_0006` ran at, so per-micro-batch activation
memory is strictly below an already-proven footprint and cannot OOM where `exp_g_0006` did not.
No empirical memory check was run: `exp_g_0011` holds ~21.7 GB of the 5090 at build time, and
probing a large allocation alongside it risks OOM-ing a 4.5-hour run. The arithmetic guarantee is
enough to commit; a real step will confirm it at launch.

`device_bs = 72 / grad_accum = 1` would avoid accumulation overhead and is worth trying once the
card is free, but it is 1.5× the proven per-micro-batch memory and would need `eval_steps` to
change again to hold the validation budget (72 does not divide 480).

## Status

Built, cross-checked, smoke-tested, code committed and pushed. **Not launched** — `exp_g_0011`
(pid 165968) is training on the 5090 and must not be disturbed. Awaiting the go.
