# exp01_ppo-sac-baseline

PPO-vs-SAC equal-data baseline. GPU-resident PPO (768 upd) vs batched SAC (UTD 4, ~equal env-steps), MLP arch, 3 seeds each. PPO dominates in the 8192-env massively-parallel regime; SAC is high-variance and underperforms.

## Config

- algo: `ppo+sac`  arch: `mlp`
- envs: 8192  rollout: 32  updates: 384  seeds: [0, 1, 2]
- lr_schedule: None  lr_min: None  logstd_min: None  ent_coef: None  target_kl: None  norm_returns: None
- params: 142,605

## Headline result

- PPO best (mean+/-std over seeds): **4753.1 +/- 141.7**
- PPO final (mean+/-std over seeds): **4419.4 +/- 393.0**
- SAC final (mean+/-std over seeds): **1759.0 +/- 1175.3**
- throughput: ~492,136 env-steps/s  wall: ~0.057 h/seed

## Files

- `config.json` / `summary.json` / `metrics.csv` - convention metadata (generated from the raw runs).
- `ppo_s{0,1,2}.json` - raw per-seed run records (full per-update history).
- `*.gpu` / `agg.gpu` - GPU utilization traces.
- provenance: originally `bench2/`; delegated tasks in the walker2d-lut programme.
