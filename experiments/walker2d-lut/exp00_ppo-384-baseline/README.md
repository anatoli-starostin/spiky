# exp00_ppo-384-baseline

Early PPO baseline: 3 seeds x 384 updates (~101M env-steps), constant LR 3e-4, MLP actor-critic. Superseded by the 768-update stabilization series (exp02-exp05); kept as the earlier reference point.

## Config

- algo: `ppo`  arch: `mlp`
- envs: 8192  rollout: 32  updates: 384  seeds: [0, 1, 2]
- lr_schedule: None  lr_min: None  logstd_min: None  ent_coef: None  target_kl: None  norm_returns: None
- params: 142,605

## Headline result

- PPO best (mean+/-std over seeds): **4909.5 +/- 52.9**
- PPO final (mean+/-std over seeds): **4908.3 +/- 51.6**
- throughput: ~164,680 env-steps/s  wall: ~0.17 h/seed

## Files

- `config.json` / `summary.json` / `metrics.csv` - convention metadata (generated from the raw runs).
- `ppo_s{0,1,2}.json` - raw per-seed run records (full per-update history).
- `*.gpu` / `agg.gpu` - GPU utilization traces.
- provenance: originally `bench3/`; delegated tasks in the walker2d-lut programme.
