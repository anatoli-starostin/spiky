# aux_runs — auxiliary conf_n1 runs (seed replicates + an init ablation)

These are supporting runs for `../06_conf_n1` (the confirmed reference fused conf_n1 run). All of them train on
the **NEW full 170-shard ClimbMix corpus** (see the corpus note below), so they are directly comparable to
`06_conf_n1` and to each other, but **not** to any abl47-family run from 2026-10-04..06.

## What each run is
- **06_conf_n1_s2** — seed replicate of `06_conf_n1` (random_seed 2; train.py byte-identical to 06_conf_n1,
  only `random_seed`/`exp_name` differ). Final val_bpb **1.0968218702576613** (48000 steps).
- **06_conf_n1_s3** — seed replicate (random_seed 3). Final val_bpb **1.0945992497236325** (48000 steps).
- **06_conf_n1_oldbias** — the OLD decompress-bias-init ablation: identical to `06_conf_n1` seed 1 plus one
  block that re-initialises every `ffn.proj.decompress.bias` to the OLD `nn.Linear` default `U(±1/sqrt(384))`
  from a dedicated `Generator(seed=777)` (global RNG preserved, so the dropout stream matches seed 1 exactly).
  Stopped early at **step 36000, val_bpb ~1.115693** (trajectory comparison only — not a valid final number).
  Result: NEGATIVE — restoring the old bias did not reproduce abl47's deficit; it tracked the new zero-bias
  seeds. See `06_conf_n1_oldbias/RESULT.md`.

## 3-seed stats (06_conf_n1 seed 1 + s2 + s3, all NEW corpus, 48000 steps)
finals = 1.0969795755941025 / 1.0968218702576613 / 1.0945992497236325 ; mean ≈ 1.09613 ; sample sigma ≈ 0.001331.
This ~0.0013 is the same-geometry seed-noise floor on the new corpus.

## CORPUS NOTE (why abl47 is not comparable)
abl47 (run 2026-10-04) trained ~**7 epochs over only 4 training shards** (~170M tokens), because at that time
`~/.cache/nanochat/base_data_climbmix` held just shards 00000-00003 + the val shard 06542 (the original
5-shard subset). On **2026-10-07** a download (`python -m nanochat.dataset -n 170` → unpinned HF
`karpathy/climbmix-400b-shuffle`, which **skips files that already exist**) added shards 00004-00169, giving 170
training shards. Every run from 2026-10-13 onward (06_conf_n1 and everything here) therefore trains **<1 epoch
on ~27 unique shards** — a completely different (much larger, non-repeated) data regime. Consequently the old
0.021 "fused better than abl47" gap was a **corpus artefact, not a library effect**, and abl47-family numbers
are NOT comparable to anything from 2026-10-13 onward. (abl47's original 5 shards are still intact on disk, so a
faithful abl47 reproduction — 5-shard dir so the loader sees 4 train shards and loops — is still possible if
ever wanted.)
