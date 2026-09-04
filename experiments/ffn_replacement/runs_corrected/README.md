# runs_corrected

Re-runs of the `ffn_replacement` experiments under the **fixed validation protocol** on
branch `research/ffn_replacement_fix`. See [`../FIXED_EVAL.md`](../FIXED_EVAL.md) for the
protocol and why it was needed.

Every run here is evaluated with the batch-size-independent eval set — **bs48 × 100 eval
steps, leading 12 rows skipped** (`tools/fixed_eval.py`) — used identically for the
training-time eval curve and final scoring, so results are directly comparable across runs
regardless of training `device_batch_size`.

Convention (per `claude/experiment-methodology.md`): one folder per run, each self-contained
with at least `config.json`, `metrics.csv`, `summary.json` (plus `loss.png`); fork the
`../train_fixed.py` trainer template into a run folder as `train.py`. The original
uncorrected runs stay in `../runs/` untouched as the historical record.
