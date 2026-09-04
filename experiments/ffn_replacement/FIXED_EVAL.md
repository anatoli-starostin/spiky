# Fixed validation protocol

This branch (`research/ffn_replacement_fix`) corrects the validation procedure used by the
`ffn_replacement` runs. It adds a single, batch-size-independent eval set and wires it into
both the training-time eval curve and final scoring. **No training was launched by this
branch** — it is the protocol change plus its verification only.

## The bug

The historical per-run trainers built the validation loader at the *training*
`device_batch_size` and scored a fixed `eval_steps`:

```python
val_loader_factory = lambda: ...loader(tokenizer, DEVICE_BS, SEQ_LEN, split='val', ...)
bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)   # EVAL_STEPS=10
```

The val loader walks the held-out shard **deterministically from token 0**, so the number
of validation tokens scored was `DEVICE_BS × SEQ_LEN × EVAL_STEPS` — a function of the
*training* batch size. LUT runs (`device_batch_size` 12) were scored on the first ~61,440
tokens; vanilla baselines (`device_batch_size` 48) on the first ~245,760 — different,
nested slices of the val stream. Comparing those numbers compared models on different data.
That coupling is the protocol confound behind the earlier "LUT matches vanilla" reads.

## The fix

Evaluation is decoupled from training and identical for every run:

- **Always batch size 48 × 100 eval steps.** The val window is a property of the eval set,
  never the training batch. `device_batch_size` no longer influences it in any way.
- **Skip the leading 12 rows.** Rows 0 and 8 are the two easiest 512-token spans in the
  shard; the first 12-row block is anomalously easy/contaminated and is dropped from the
  metric. The scored set is rows `[12, 4800)` of the from-token-0 val stream — 4,788 rows,
  2,451,456 target tokens (minus special/ignored) — the same for every model.
- **One eval set, two call sites.** The training loop's eval curve and the standalone
  final scorer call the *same* function, so a run's curve and its final number are the
  identical measurement.

Implemented in [`tools/fixed_eval.py`](tools/fixed_eval.py) as `evaluate_bpb_fixed(...)`
with constants `EVAL_BATCH_SIZE=48`, `EVAL_STEPS=100`, `EVAL_SKIP_ROWS=12`. Overridable
per-run only through an explicit `"fixed_eval": {…}` block in `config.json`; the legacy
top-level `eval_steps` key is **deliberately ignored** so no old config can silently
re-couple the window.

### Reused-buffer safety

The nanochat loader yields views into a single GPU buffer that the next yield overwrites
via a `non_blocking` HtoD copy. Interleaving forward passes with further `next()` calls
lets that async copy race the buffer reuse and silently corrupts later batches. `fixed_eval`
therefore **drains all eval batches with `.clone()` first, then scores** the private
snapshots — so the metric is stable and correct. (Verified: without this, repeated calls
drifted 1.147→1.153; with it, the number is bit-stable and matches the reference harness.)

## Files

| path | what |
|---|---|
| `tools/fixed_eval.py` | `evaluate_bpb_fixed(...)` — THE eval set (bs48×100, skip 12), batch-size independent. |
| `tools/model_build.py` | Config-driven `build_model(cfg, vocab)` shared by trainer + scorer (dense / compression / `fastmhl_raw`). |
| `train_fixed.py` | Corrected trainer template — fork into a run folder as `train.py`; uses `fixed_eval` for the eval curve. |
| `tools/score_checkpoint.py` | Standalone final scorer — same `fixed_eval`, rebuilds from a run's `config.json`. |

The historical `runs/*/train.py` files are left untouched (they are the record of what was
actually run); new runs on this line use `train_fixed.py`.

## Hold-out (verified)

`nanochat.dataset.list_parquet_files()` → `parquet_paths[:-1]` = train, `[-1:]` = val.

```
all shards : shard_00000  shard_00001  shard_00002  shard_00003  shard_06542
train      : shard_00000  shard_00001  shard_00002  shard_00003
val        : shard_06542            (the last shard)
disjoint (val ∉ train): True
```

Validation is a genuine file-level hold-out: the val shard (`shard_06542.parquet`) is never
in the train set.

## Sanity check (0151 vanilla)

Rebuild `exp_n_0151_long48k_untied_vanilla` from its `config.json`, load its `checkpoint.pt`
(0 missing / 0 unexpected keys), and score with `evaluate_bpb_fixed`:

| window | bpb | note |
|---|---|---|
| bs48 × 100, **skip 0** | **1.11508095** | reproduces the reference `nanochat.loss_eval.evaluate_bpb` over the same window **to 6 dp** → harness is sound |
| bs48 × 100, **skip 12** (the protocol) | **1.11541948** | bit-stable across repeated calls; slightly higher than skip-0, as expected when the anomalously-easy leading rows are dropped |

`evaluate_bpb_fixed(skip_rows=0)` matches the reference harness exactly at 10/50/100 steps
(1.151444 / 1.117280 / 1.115081), so the only behavioural change from the reference is the
intended leading-row skip.
