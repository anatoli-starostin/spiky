# exp10_lut-anchor-pair-t32

Anchor-pair LUT actor (FastMultiHeadLut, fixed sign(x[a]-x[b]) addressing, only tables train), tph=32, + MLP critic.

## Config

- algo: `ppo`  arch: `fastlut`  tables_per_head: `32`
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 3e-05  logstd_min: -1.897  ent_coef: 0.0  target_kl: 0.02  norm_returns: True
- params: 82,951

## Headline result

- PPO best (mean+/-std over seeds): **5551.0 +/- 175.9**
- PPO final (mean+/-std over seeds): **5488.4 +/- 179.9**
- throughput: ~168,258 env-steps/s  wall: ~0.332 h/seed

## Files

- `config.json` / `summary.json` / `metrics.csv` - convention metadata (generated from the raw runs).
- `ppo_s{0,1,2}.json` - raw per-seed run records (full per-update history).
- `*.gpu` / `agg.gpu` - GPU utilization traces.
- provenance: originally `bench12/t32/`; delegated tasks in the walker2d-lut programme.
