# exp04_ppo-cosine-logstd-floor

PPO stabilization step 3 - cosine LR 3e-4->1e-5 + log_std floor (std>=0.15) + entropy 0.005. Improves reliability, caps peak.

## Config

- algo: `ppo`  arch: `mlp`
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 1e-05  logstd_min: -1.897  ent_coef: 0.005  target_kl: None  norm_returns: None
- params: 142,605

## Headline result

- PPO best (mean+/-std over seeds): **5149.2 +/- 314.5**
- PPO final (mean+/-std over seeds): **4634.3 +/- 782.0**
- throughput: ~166,200 env-steps/s  wall: ~0.336 h/seed

## Files

- `config.json` / `summary.json` / `metrics.csv` - convention metadata (generated from the raw runs).
- `ppo_s{0,1,2}.json` - raw per-seed run records (full per-update history).
- `*.gpu` / `agg.gpu` - GPU utilization traces.
- provenance: originally `bench6/`; delegated tasks in the walker2d-lut programme.
