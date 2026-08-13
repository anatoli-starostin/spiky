# exp15_lut-anchor-pair-lutcrit-t128

Anchor-pair LUT actor + anchor-pair LUT critic (arch `fastlut2`), tph=128. Fully-LUT counterpart of exp12.

## Config

- algo: `ppo`  arch: `fastlut2`  tables_per_head: `128`  nap: 6
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 3e-05  logstd_min: -1.897  ent_coef: 0.0  target_kl: 0.02  norm_returns: True
- params: 57,350  (vs 119,815 for the MLP-critic exp12)

## Result (3 seeds)

- final ep-return: **3344 ± 1401**
- best  ep-return: **3372 ± 1414**
- vs MLP-critic (exp12, tph 128): final 6078 ± 172  → Δ = -2734
- params: 57,350 vs 119,815  → 52% fewer
- throughput: ~117,828 env-steps/s  wall: ~28.5 min/run
- avg_epochs_per_update: 3.99 (4.0 = KL-stop never fired)

## Files

- `config.json` / `summary.json` / `metrics.csv` — convention metadata.
- `ppo_s{0,1,2}.json` — raw per-seed run records.  `agg.gpu` — GPU trace.  `curve.png` — learning curves.
