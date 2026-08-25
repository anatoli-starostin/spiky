# exp101_sweep_outproj — out_proj CompressionMHL grid sweep

Sweeps ONLY the `out_proj` CompressionMHL of the exp101 single-stream, separate-q/k/v
CompressionMHL-attention model, holding q/k/v fixed at the exp101 baseline
(`input_dim=384, output_dim=64, inner_in_dim=48, inner_out_dim=-1, nap=6, tph=32,
n_heads=6, multihead_output=True`). Shared lutorch is NOT modified — uses the local
`mh_compression.py` subclass, same as exp101.

**Grid (12 runs, 4k steps each):**
```
out_proj = CompressionMultiHeadLUT(input_dim=384, output_dim=384,
             inner_in_dim=<IN>, inner_out_dim=48, nap=6, tph=<TPH>,
             n_heads=8, forward_mode="hard", use_bf16=True)
<IN>  in {24, 48, 96}     <TPH> in {16, 32, 64, 128}     (nap, inner_out, n_heads held)
```

**Naming scheme:** one shared trainer `train_sweep.py`, env-parameterized
(`OUT_IN`, `OUT_TPH`, `RUN_TAG`, `N_STEPS`). Each run writes ALL its outputs to a per-run
subdir **`in{IN}_tph{TPH}/`** (metrics.csv, summary.json, loss.png, checkpoint.pt), so 3
concurrent runs never collide. Per-run `exp_name = exp101_sweep_outproj_in{IN}_tph{TPH}`.

**Orchestration** (`run_sweep.sh`): 4 waves × 3 concurrent (each run its own
`TRITON_CACHE_DIR` to avoid kernel-cache races). Wave N+1 starts only after wave N's 3
runs all finish. Otherwise exp101-equivalent config (single-stream, Lion on LUT tables /
AdamW elsewhere, seed 42, 24,576 tok/step, eval every 200).

**Progress:** ONE consolidated Slack bar (`sweep_progress.py`) — done-count /12, current
best run, in-flight steps; finalizes with a top-3 ranking. (Not 12 separate bars.)

**Baseline:** exp101's out_proj was `inner_in=48 / tph=32` → 1.30213 at 16k (compare only
WITHIN this sweep; these are 4k). Reports the full `run → inner_in / tph → final val_bpb`
table ranked best-first when all 12 complete.
