# exp11_lut-anchor-pair-t64

Anchor-pair LUT actor (FastMultiHeadLut), tph=64, + MLP critic.

## Config

- algo: `ppo`  arch: `fastlut`  tables_per_head: `64`
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 3e-05  logstd_min: -1.897  ent_coef: 0.0  target_kl: 0.02  norm_returns: True
- params: 95,239

## Headline result

- PPO best (mean+/-std over seeds): **5761.2 +/- 270.1**
- PPO final (mean+/-std over seeds): **5755.3 +/- 272.2**
- throughput: ~156,797 env-steps/s  wall: ~0.357 h/seed

## Files

- `config.json` / `summary.json` / `metrics.csv` - convention metadata (generated from the raw runs).
- `ppo_s{0,1,2}.json` - raw per-seed run records (full per-update history).
- `*.gpu` / `agg.gpu` - GPU utilization traces.
- provenance: originally `bench12/t64/`; delegated tasks in the walker2d-lut programme.
