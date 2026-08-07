# exp12_lut-anchor-pair-t128

Anchor-pair LUT actor (FastMultiHeadLut), tph=128, + MLP critic. Anchor-pair beats hyperplane at every tph and is ~20-27% faster.

## Config

- algo: `ppo`  arch: `fastlut`  tables_per_head: `128`
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 3e-05  logstd_min: -1.897  ent_coef: 0.0  target_kl: 0.02  norm_returns: True
- params: 119,815

## Headline result

- PPO best (mean+/-std over seeds): **6125.3 +/- 156.9**
- PPO final (mean+/-std over seeds): **6077.7 +/- 171.7**
- throughput: ~137,458 env-steps/s  wall: ~0.407 h/seed

## Files

- `config.json` / `summary.json` / `metrics.csv` - convention metadata (generated from the raw runs).
- `ppo_s{0,1,2}.json` - raw per-seed run records (full per-update history).
- `*.gpu` / `agg.gpu` - GPU utilization traces.
- provenance: originally `bench12/t128/`; delegated tasks in the walker2d-lut programme.
