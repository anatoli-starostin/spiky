# exp08_lut-hyperplane-t64-lutcrit

Hyperplane-LUT actor (tph=64) + hyperplane-LUT critic (tph=64) - fully-LUT, wider tables.

## Config

- algo: `ppo`  arch: `hyperlut2_t64`
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 3e-05  logstd_min: -1.897  ent_coef: 0.0  target_kl: 0.02  norm_returns: True
- params: 42,502

## Headline result

- PPO best (mean+/-std over seeds): **4661.8 +/- 211.5**
- PPO final (mean+/-std over seeds): **4645.1 +/- 219.4**
- throughput: ~82,309 env-steps/s  wall: ~0.679 h/seed

## Files

- `config.json` / `summary.json` / `metrics.csv` - convention metadata (generated from the raw runs).
- `ppo_s{0,1,2}.json` - raw per-seed run records (full per-update history).
- `*.gpu` / `agg.gpu` - GPU utilization traces.
- provenance: originally `bench10/`; delegated tasks in the walker2d-lut programme.
