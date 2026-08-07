# exp09_lut-hyperplane-t64-mlpcrit

Hyperplane-LUT actor (tph=64) + MLP critic. Isolates the LUT-critic deficit vs exp08.

## Config

- algo: `ppo`  arch: `hyperlut_t64`
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 3e-05  logstd_min: -1.897  ent_coef: 0.0  target_kl: 0.02  norm_returns: True
- params: 102,151

## Headline result

- PPO best (mean+/-std over seeds): **5500.5 +/- 209.0**
- PPO final (mean+/-std over seeds): **5496.3 +/- 211.8**
- throughput: ~123,768 env-steps/s  wall: ~0.452 h/seed

## Files

- `config.json` / `summary.json` / `metrics.csv` - convention metadata (generated from the raw runs).
- `ppo_s{0,1,2}.json` - raw per-seed run records (full per-update history).
- `*.gpu` / `agg.gpu` - GPU utilization traces.
- provenance: originally `bench11/`; delegated tasks in the walker2d-lut programme.
