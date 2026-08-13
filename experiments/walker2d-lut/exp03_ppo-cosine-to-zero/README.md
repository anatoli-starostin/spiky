# exp03_ppo-cosine-to-zero

PPO stabilization step 2 - cosine LR decay 3e-4->0. Partial fix (1/3 collapse) but peak lowered.

## Config

- algo: `ppo`  arch: `mlp`
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: None  logstd_min: None  ent_coef: None  target_kl: None  norm_returns: None
- params: 142,605

## Headline result

- PPO best (mean+/-std over seeds): **5409.1 +/- 537.0**
- PPO final (mean+/-std over seeds): **4638.2 +/- 1393.1**
- throughput: ~166,863 env-steps/s  wall: ~0.335 h/seed

## Files

- `config.json` / `summary.json` / `metrics.csv` - convention metadata (generated from the raw runs).
- `ppo_s{0,1,2}.json` - raw per-seed run records (full per-update history).
- `*.gpu` / `agg.gpu` - GPU utilization traces.
- provenance: originally `bench5/`; delegated tasks in the walker2d-lut programme.
