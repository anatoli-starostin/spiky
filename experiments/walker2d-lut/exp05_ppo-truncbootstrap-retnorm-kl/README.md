# exp05_ppo-truncbootstrap-retnorm-kl

PPO stabilization step 4 (WINNER) - cosine 3e-4->3e-5 + log_std floor + exact truncation bootstrap + return normalization + KL early-stop guard, entropy 0. Collapse eliminated (0/3) and peak reclaimed; final approx best (gap ~33).

## Config

- algo: `ppo`  arch: `mlp`
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 3e-05  logstd_min: -1.897  ent_coef: 0.0  target_kl: 0.02  norm_returns: True
- params: 142,605

## Headline result

- PPO best (mean+/-std over seeds): **5985.3 +/- 423.9**
- PPO final (mean+/-std over seeds): **5952.1 +/- 415.9**
- throughput: ~164,705 env-steps/s  wall: ~0.34 h/seed

## Files

- `config.json` / `summary.json` - convention metadata (generated from the raw runs).
- Not carried on this branch, kept with the full run on `research/walker2d-lut`: `metrics.csv`,
  the raw per-seed records `ppo_s{0,1,2}.json` (full per-update history), and the `*.gpu` /
  `agg.gpu` utilization traces.
- provenance: originally `bench7/`; delegated tasks in the walker2d-lut programme.
