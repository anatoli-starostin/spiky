# exp06_lut-hyperplane-t32-mlpcrit

Hyperplane-LUT actor (learned per-bit sign-tests W.x+b>0, decoupled straight-through), tables_per_head=32, + MLP critic. First LUT policy on the stabilized PPO recipe.

## Config

- algo: `ppo`  arch: `hyperlut`
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 3e-05  logstd_min: -1.897  ent_coef: 0.0  target_kl: 0.02  norm_returns: True
- params: 86,407

## Headline result

- PPO best (mean+/-std over seeds): **4920.0 +/- 191.7**
- PPO final (mean+/-std over seeds): **4890.9 +/- 231.9**
- throughput: ~137,050 env-steps/s  wall: ~0.408 h/seed

## Files

- `config.json` / `summary.json` / `metrics.csv` - convention metadata (generated from the raw runs).
- `ppo_s{0,1,2}.json` - raw per-seed run records (full per-update history).
- `*.gpu` / `agg.gpu` - GPU utilization traces.
- provenance: originally `bench8/`; delegated tasks in the walker2d-lut programme.
