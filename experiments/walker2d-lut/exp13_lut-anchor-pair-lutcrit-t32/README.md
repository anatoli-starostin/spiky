# exp13_lut-anchor-pair-lutcrit-t32

Anchor-pair LUT actor (FastMultiHeadLut) + anchor-pair LUT critic (arch `fastlut2`), tph=32. Fully-LUT counterpart of exp10 (same actor, MLP critic -> anchor-pair LUT critic).

## Config

- algo: `ppo`  arch: `fastlut2`  tables_per_head: `32`  nap: 6
- envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 3e-05  logstd_min: -1.897  ent_coef: 0.0  target_kl: 0.02  norm_returns: True
- params: 14,342  (vs 82,951 for the MLP-critic exp10)

## Result (3 seeds)

- final ep-return: **2359 ± 879**
- best  ep-return: **2543 ± 753**
- vs MLP-critic (exp10, tph 32): final 5488 ± 180  → Δ = -3130
- params: 14,342 vs 82,951  → 83% fewer
- throughput: ~162,406 env-steps/s  wall: ~20.7 min/run
- avg_epochs_per_update: 4.00 (4.0 = KL-stop never fired)

## Files

- `config.json` / `summary.json` / `metrics.csv` — convention metadata.
- `ppo_s{0,1,2}.json` — raw per-seed run records.  `agg.gpu` — GPU trace.  `curve.png` — learning curves.
