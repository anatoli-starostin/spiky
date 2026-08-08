# exp15b_anchor-t128-actor_hyperplane-t64-critic

MIXED LUT arch: anchor-pair LUT actor (FastMultiHeadLut, fixed anchors, tph=128) + HYPERPLANE LUT critic (HyperLUTHead, learned sign-test addressing via STE, tph=64, scalar V). Tests whether a learned-addressing hyperplane critic beats the fixed-anchor LUT critic (exp15) for an anchor-pair actor. Compare vs exp12 (MLP critic) and exp15 (anchor LUT critic), all t128.

## Config

- actor: anchor-pair FastMultiHeadLut, tph=128  |  critic: hyperplane HyperLUTHead, tph=64
- arch: `fastlut_hypcrit`  nap: 6  envs: 8192  rollout: 32  updates: 768  seeds: [0, 1, 2]
- lr_schedule: cosine  lr_min: 3e-05  logstd_min: -1.897  ent_coef: 0.0  target_kl: 0.02  norm_returns: True
- params: 60,166

## Result (3 seeds)

- final ep-return: **5053 ± 186**
- best  ep-return: **5062 ± 197**

## vs other t128 critics (anchor actor held fixed)

| critic | final | params |
|---|---:|---:|
| MLP (exp12) | 6078 ± 172 | 119,815 |
| **hyperplane-LUT t64 (exp15b)** | **5053 ± 186** | 60,166 |
| anchor-LUT t128 (exp15) | 3344 ± 1400 | 57,350 |

- throughput ~93,100 env-steps/s  wall ~36.0 min/run  avg_epochs_per_update 3.99

## Files

- config.json / summary.json / metrics.csv / README.md + ppo_s{0,1,2}.json + agg.gpu + curve.png
