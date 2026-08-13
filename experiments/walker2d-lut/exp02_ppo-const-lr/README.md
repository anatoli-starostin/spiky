# exp02_ppo-const-lr

PPO stabilization step 1 - constant LR 3e-4, 768 updates. Reveals late-training instability: 2/3 seeds partially collapse in the last quarter (final 5278+/-1281 vs best 6281+/-585).

## Config

- algo: `ppo`  arch: `mlp`
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: None  lr_min: None  logstd_min: None  ent_coef: None  target_kl: None  norm_returns: None
- params: 142,605

## Headline result

- PPO best (mean+/-std over seeds): **6281.3 +/- 584.6**
- PPO final (mean+/-std over seeds): **5278.2 +/- 1280.8**
- throughput: ~166,750 env-steps/s  wall: ~0.335 h/seed

## Files

- `config.json` / `summary.json` / `metrics.csv` - convention metadata (generated from the raw runs).
- `ppo_s{0,1,2}.json` - raw per-seed run records (full per-update history).
- `*.gpu` / `agg.gpu` - GPU utilization traces.
- provenance: originally `bench4/`; delegated tasks in the walker2d-lut programme.
