# exp_g_0195 — how to run it on nebius

Built and gated on gpustar, **never run there** (reassigned before it started). Runs off the
branch with no hand-editing.

## Command

```bash
cd ~/projects/<checkout>            # any checkout of research/ffn_replacement_fix
git pull
cd experiments/ffn_replacement/runs_corrected/exp_g_0195_B16k_light_margin_blend_n2_tau_learn0p5_seed1
python -u train.py > train.log 2>&1
```

Run it **from inside the repo checkout** — `train.py` locates the spiky source four levels
above itself. It does not care which checkout, only that it is one, so a worktree is fine.

## What must already exist on the machine

| thing | how it is found | note |
|---|---|---|
| **spiky source** | `<checkout>/src`, resolved from `__file__` | **not** the installed/editable `spiky`; see below |
| **nanochat** | `$NANOCHAT_ROOT`, default `~/projects/nanochat` | set the env var if it lives elsewhere |
| **tokenizer** | `<nanochat base_dir>/tokenizer` | gpustar: `~/.cache/nanochat/tokenizer` |
| **data** | 5 parquet shards; last one (`shard_06542.parquet`) is the held-out val split | the corrected eval scores rows [12, 4800) of it |
| **GPU** | `torch.cuda.is_available()`, single device | no multi-GPU or device-count assumption; falls back to CPU (unusably slow, but it will not crash) |

torch on gpustar is 2.9.1+cu130. Nothing in this run is version-pinned beyond
`F.embedding_bag(..., per_sample_weights=...)`, which is long-standing.

## The spiky-source override — why it is there

nebius's editable `spiky` points at `~/projects/spiky/src` on branch
`hyperplane_ffn_next`, which **predates the top-n blend entirely**. `train.py` therefore
prepends `<checkout>/src` to `sys.path`, drops editable finders from `sys.meta_path`, and then
**asserts** that the imported module really came from the checkout.

Every failure raises at import with a message naming what was sought and where. It never
falls through to a stale source — training against one would silently produce a model
*without* the blend and report it as a result.

Verified on gpustar against a simulated stale editable install (a copy with the blend removed,
exposed through a finder at `sys.meta_path[0]`, i.e. more adversarial than the real appended
one): a bare import resolved to the stale copy, while `train.py` resolved to the checkout and
built correctly.

## A correct start — the first log lines

```
[spiky-source] /…/<checkout>/src/spiky/lutorch/light_multi_head_lut.py
FIXED EVAL: bs48 x 100 steps, skip 12 rows (val window independent of device_batch_size=12)
MinimalGPT depth=6 dim=384 heads=6 seq=512 | ffn=compression tie=False | params=67,351,686
Tokens/micro-batch: 6,144 | grad_accum: 4 | effective batch: 24,576 tokens
```

Check all four:

- **`[spiky-source]` must point inside the checkout you just pulled.** If absent, an old copy
  of `train.py` is being run.
- **`params=67,351,686`.** 67,351,680 (six fewer) means `lut_read_tau_learnable` did not take
  and tau is a frozen buffer — that would be `exp_g_0194`, not this run.
- **`bs48 x 100 steps, skip 12 rows`.** Anything else is the old batch-coupled eval.
- **`24,576 tokens`** effective batch.

## Expected

- **16,000 steps**, eval every 500, seed 1.
- **67,351,686 params** (= exp_g_0193's 67,351,680 **+6**; the six learnable tau scalars). The
  comparison against `exp_g_0193`/`exp_g_0194` is therefore **not** param-matched.
- **~0.30 s/step** on a 5090 → ~1.35 h. An H100 should be quicker; `exp_g_0194` (same shape,
  frozen tau) took 1.664 h on the 5090.
- Writes `metrics.csv`, `summary.json`, `loss.png`, `checkpoint.pt`, plus step-tagged
  checkpoints every 4,000 steps (`CKPT_EVERY` env var to change).
- `metrics.csv` carries **`tau_L0..tau_L5`** at every eval — the point of the run. tau starts
  flat at 0.5; the measured margin scale Delta_m is 0.033–0.108, so the question is whether
  tau descends toward it. In a 20-step gpustar smoke test it had already moved to
  0.49877 / 0.49928 / 0.49971 / 0.49991 / 0.49997 / 0.50002 by layer — descending fastest at
  layer 0, which is where Delta_m is smallest.

## Comparing back

`runs_corrected/cmp_vs_0193.py` does exact-step matched deltas and takes an arbitrary
reference via `delta_table_vs(run_dir, ref_dir)`. It needs `metrics.csv` for whichever
reference you compare against — `exp_g_0193` and `exp_g_0194` are both committed on this
branch, so a fresh clone has them. Steps align exactly (same seed, same schedule); a step with
no counterpart prints `NO EXACT COUNTERPART` rather than being interpolated.
