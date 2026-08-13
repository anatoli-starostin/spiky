# exp14_lut-anchor-pair-lutcrit-t64

Anchor-pair LUT actor + anchor-pair LUT critic (arch `fastlut2`), tph=64. Fully-LUT counterpart of exp11.

## Config

- algo: `ppo`  arch: `fastlut2`  tables_per_head: `64`  nap: 6
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 3e-05  logstd_min: -1.897  ent_coef: 0.0  target_kl: 0.02  norm_returns: True
- params: 28,678  (vs 95,239 for the MLP-critic exp11)

## Result (3 seeds)

- final ep-return: **3425 ± 676**
- best  ep-return: **3455 ± 700**
- vs MLP-critic (exp11, tph 64): final 5755 ± 272  → Δ = -2330
- params: 28,678 vs 95,239  → 70% fewer
- throughput: ~144,442 env-steps/s  wall: ~23.2 min/run
- avg_epochs_per_update: 3.99 (4.0 = KL-stop never fired)

## Files

- `config.json` / `summary.json` / `metrics.csv` — convention metadata.
- `ppo_s{0,1,2}.json` — raw per-seed run records.  `agg.gpu` — GPU trace.  `curve.png` — learning curves.
