# Launching `train_multi.py`

`train_multi.py` builds the GPT skeleton with the research branch's `model_build`, which constructs the **old**
library's layers (`spiky.lutorch`, e.g. `CompressionMultiHeadLUT(backward_topk=…)`), and then replaces every
feed-forward layer with a `ProjectionMHL` around a **new** `spiky.lutorch_ex` cartridge. One process therefore needs
both libraries, from two different checkouts:

| package | comes from |
| --- | --- |
| `spiky.lutorch`, `spiky.util`, `spiky.lut_fused`, `spiky.spnet` | `research/ffn_replacement_fix` @ `c4a750b2` (main's old library lacks `backward_topk`) |
| `spiky.lutorch_ex` | a **copied** snapshot of main's `src/spiky/lutorch_ex` (nebius: `e45fd05e`; identical at `7e19f0d7`) |
| `model_build`, `fixed_eval`, `wandb_tracking` | `TOOLS_DIR` = `<research checkout>/experiments/ffn_replacement/tools` |
| `nanochat.*` | `NANOCHAT_ROOT` (nanochat `da32e1d`) |

## The import overlay

One synthetic `spiky/` package made of symlinks, put on `PYTHONPATH` (`build_xspiky_overlay.sh` does exactly this):

```bash
XT=<overlay dir>                     # e.g. .../xspiky_quant_n1
rm -rf "$XT"; mkdir -p "$XT/spiky"
for e in "$RESEARCH"/spiky/*; do ln -s "$e" "$XT/spiky/$(basename "$e")"; done    # RESEARCH = <research checkout>/src
ln -sfn "$STABLE/spiky/lutorch_ex" "$XT/spiky/lutorch_ex"                          # STABLE = copied lutorch_ex snapshot
export PYTHONPATH=$XT
```

- **No `__init__.py` in `$XT/spiky`**: `spiky` must stay a namespace package, or the cross-checkout submodules stop
  resolving.
- Snapshot `lutorch_ex` by copying (`git archive <commit> src/spiky/lutorch_ex`), not by symlinking into a live
  checkout, so switching that checkout's branch cannot change a running job's library.
- `PYTHONPATH` carries only the overlay. `train_multi.py` itself inserts `NANOCHAT_ROOT` and `TOOLS_DIR` at the front
  of `sys.path`, so the effective order is `[NANOCHAT_ROOT, TOOLS_DIR, overlay, …]`.

## Environment

`PYTHONPATH=$XT`, `TOOLS_DIR`, `NANOCHAT_ROOT`, `ABL47_CONFIG` (the champion `config.json` of the arm being
reproduced), `CART` (one of the keys of `_BUILDERS`), `OUT_DIR` (the run folder: `ckpt.pt`, `metrics.csv`,
`summary.json`, `run_tag.txt`), `CKPT_EVERY=1000`. The p2_int8 JIT op needs `CUDA_HOME=/usr/local/cuda` and `nvcc` on
`PATH`. Smoke-test with `SMOKE_STEPS` / `SMOKE_EVAL_EVERY`. A complete, working example:
`lutorch_ex_abl47_quant_n1_gs_1006_1611/launch.sh`.

## Checkpoints and resume

`ckpt.pt` (model, optimizer, step, ema, best, run tag) is written atomically every `CKPT_EVERY` steps and at the end;
re-running the same command resumes from it (tested: SIGKILL mid-run, then resume, on gpustar 2026-10-06). Resume
does not restore the data-loader position or RNG state, so the data order after a resume differs from an
uninterrupted run; and an eval that ran after the last checkpoint is repeated, leaving a duplicate `metrics.csv` row
for that step (keep the last).
