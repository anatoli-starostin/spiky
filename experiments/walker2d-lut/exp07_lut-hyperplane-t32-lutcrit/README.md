# exp07_lut-hyperplane-t32-lutcrit

Hyperplane-LUT actor (tph=32) + hyperplane-LUT critic (tph=32) - fully-LUT actor+value.

## Config

- algo: `ppo`  arch: `hyperlut2`
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 3e-05  logstd_min: -1.897  ent_coef: 0.0  target_kl: 0.02  norm_returns: True
- params: 21,254

## Headline result

- PPO best (mean+/-std over seeds): **4208.0 +/- 210.4**
- PPO final (mean+/-std over seeds): **4204.9 +/- 211.6**
- throughput: ~103,894 env-steps/s  wall: ~0.538 h/seed

## Files

- `config.json` / `summary.json` / `metrics.csv` - convention metadata (generated from the raw runs).
- `ppo_s{0,1,2}.json` - raw per-seed run records (full per-update history).
- `*.gpu` / `agg.gpu` - GPU utilization traces.
- provenance: originally `bench9/`; delegated tasks in the walker2d-lut programme.
