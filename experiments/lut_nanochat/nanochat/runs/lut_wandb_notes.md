Dense d24 nanochat baseline (research/lut_nanochat), 1x H100 80GB, nanochat vendored at 92d63d4.
Recipe = runs/speedrun.sh (d24, ratio 8, fp8, ClimbMix) with nproc_per_node 1: global batch 2^20 tokens,
5,568 steps, 5.84B tokens; only grad_accum changes (32 at dbs 16, 64 at dbs 8). Seed 42 (hardcoded).
Target: standalone base_eval CORE ~0.2626 (8xH100 record), must beat GPT-2 0.256525; noise ~+-0.008.

Metric legend
- train/loss: EMA-smoothed (beta 0.9, debiased) training loss, nats/token, last micro-batch of each step
- train/loss_raw: unsmoothed last micro-batch loss of the step
- train/tokens_seen: total_batch_size x (step+1)
- train/lr_matrix: current Muon (matrix) learning rate, after warmup/warmdown and batch scaling
- train/lrm: LR multiplier of the schedule (warmup 40 steps, warmdown 65%, final 0.05)
- train/dt: wall-clock seconds of the optimizer step (all micro-steps)
- train/tok_per_sec: tokens per second over the step
- train/mfu: model FLOPs utilisation vs the BF16 peak of the GPU (fp8 can exceed nominal shares)
- train/epoch: dataloader position (epoch, parquet idx, row group)
- val/bpb: bits per byte on the val shard (shard_06542), 80 x 524,288 tokens, every 250 steps, fp8 disabled
- core_metric (in-training history): CORE on 500 examples/task every 2000 steps; a subsample that differs from the full set (#819: 0.2803 in-training vs 0.2702 standalone); not the reported number
- total_training_flops, total_training_time: cumulative (time excludes eval and the first 10 steps)
- summary core_metric / core_standalone_base_eval: full CORE from the standalone base_eval: THE reported number
- summary val_bpb_base_eval: val bpb from the standalone base_eval (40 x 524,288 tokens)
