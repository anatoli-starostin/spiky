# lut_ablation — the 21 prepared runs of the LUT core-module ablation

These runs fill every cell of Table B in `doc/research/lut_ablation/lut_ablation_table_v4.tex`. There is one folder per
run, laid out like `runs_corrected/`:
- `config.json` is the parent's config with only the flags below changed, plus a new `exp_name` and `_arch_note`.
- `train.py` is copied with `tools/fork_trainer.py`, plus the two-line W&B group edit (below).

**Nothing here has been launched.** Launching needs Anatoli's go.

## The 21 runs

The ablation is **21 runs**: 10 LUT rows (1.1, 1.2, 2.1–2.4, 3.1–3.4) × {plain, +TV} = 20, plus the vanilla untied
baseline V, plain only. V's +TV is n/a because there are no LUT tables, and **V is one of the 21**. All 21 are prepared.

This directory holds **22 folders**: the 21 cells plus **abl_06**, a seed replicate of 3.1 that sits outside the 21.

| cell | folder | host | trainer | status | expected bpb |
|---|---|---|---|---|---|
| V | `exp_g_abl_10_B16k_vanilla_dense_untied_seed1` | gpustar | train_fixed.py | prepared (rerun of `exp_n_0135`) | ≈ 1.165147 |
| 1.1 | `exp_n_abl_11_B16k_gen1_hard_nap8_tph128_seed1` | nebius | train_fixed.py | prepared | — |
| 1.1 +TV | `exp_n_abl_12_B16k_gen1_hard_tv10_nap8_tph128_seed1` | nebius | train_fixed.py | prepared | — |
| 1.2 | `exp_n_abl_13_B16k_gen1_smooth_nap8_tph128_seed1` | nebius | train_fixed.py | prepared | — |
| 1.2 +TV | `exp_n_abl_14_B16k_gen1_smooth_tv10_nap8_tph128_seed1` | nebius | train_fixed.py | prepared | — |
| 2.1 | `exp_n_abl_07_B16k_fast_hard_topk0_nap8_tph128_seed1` | nebius | train_fixed.py | prepared (rerun of `exp_n_0121`) | ≈ 1.180622 |
| 2.1 +TV | `exp_n_abl_15_B16k_fast_hard_topk0_tv10_nap8_tph128_seed1` | nebius | train_fixed.py | prepared | — |
| 2.2 | `exp_n_abl_01_B16k_fast_hybridsmooth_topk0_nap8_tph128_seed1` | nebius | train_fixed.py | prepared | — |
| 2.2 +TV | `exp_n_abl_16_B16k_fast_hsmooth_topk0_tv10_nap8_tph128_seed1` | nebius | train_fixed.py | prepared | — |
| 2.3 | `exp_n_abl_02_B16k_fast_hard_topk1_nap8_tph128_seed1` | nebius | train_fixed.py | prepared | — |
| 2.3 +TV | `exp_n_abl_17_B16k_fast_hard_topk1_tv10_nap8_tph128_seed1` | nebius | train_fixed.py | prepared | — |
| 2.4 | `exp_n_abl_03_B16k_fast_hybridsmooth_topk1_nap8_tph128_seed1` | nebius | train_fixed.py | prepared | — |
| 2.4 +TV | `exp_n_abl_18_B16k_fast_hsmooth_topk1_tv10_nap8_tph128_seed1` | nebius | train_fixed.py | prepared | — |
| 3.1 | `exp_g_abl_08_B16k_light_lmfrozeng_tph128_seed1` | gpustar | exp_g_0249 lineage | prepared (rerun of `exp_g_0248`) | ≈ 1.169328 |
| 3.1 +TV | `exp_g_abl_09_B16k_light_lmfrozeng_tv10_tph128_seed1` | gpustar | exp_g_0249 lineage | prepared (rerun of `exp_g_0249`) | ≈ 1.163698 |
| 3.2 | `exp_g_abl_04_B16k_light_lmfrozeng_n2_tau0p5_tph128_seed1` | gpustar | exp_g_0249 lineage | prepared | — |
| 3.2 +TV | `exp_g_abl_05_B16k_light_lmfrozeng_n2_tau0p5_tv10_tph128_seed1` | gpustar | exp_g_0249 lineage | prepared | — |
| 3.3 | `exp_g_abl_19_B16k_light_lmfrozeng_hard_tph128_seed1` | gpustar | exp_g_0249 lineage | ran 2026-09-14: **DIVERGED** (final 3.0159), superseded by `…_seed1_fix` | — |
| 3.3 +TV | `exp_g_abl_20_B16k_light_lmfrozeng_hard_tv10_tph128_seed1` | gpustar | exp_g_0249 lineage | ran 2026-09-14: **DIVERGED** (final 3.0262), superseded by `…_seed1_fix` | — |
| 3.3 (rerun) | `exp_g_abl_19_B16k_light_lmfrozeng_hard_tph128_seed1_fix` | gpustar | exp_g_0249 lineage | prepared, runs on code ≥ b5b0f2c3 | — |
| 3.3 +TV (rerun) | `exp_g_abl_20_B16k_light_lmfrozeng_hard_tv10_tph128_seed1_fix` | gpustar | exp_g_0249 lineage | prepared, runs on code ≥ b5b0f2c3 | — |

**3.3 superseded runs.**
- The original 3.3 pair (abl_19 / abl_20) learned to ~step 1000 and then diverged between steps 1000 and 2000. Train loss rose above its initial value, and both stayed near chance (~3.0 bpb) to step 16000.
- Cause: LightMultiHeadLUT `_hard_read`'s straight-through gradient scaled with the unbounded learned-margin score, and nothing in the forward value corrected it.
- Their artefacts stay committed as evidence of the bug.
- `b5b0f2c3` (nebius) rescales the surrogate's score by its detached per-token mean. The forward value is unchanged (the plain hard read); only the input / β / γ / τ gradients are bounded.
- The `_fix` folders are byte-identical copies of the originals apart from `exp_name`, so they run the same config on the fixed code.
- The same fix applies to row 3.4 (abl_21 / abl_22).
| 3.4 | `exp_n_abl_21_B16k_light_lmfrozeng_hard_n2_tau0p5_tph128_seed1` | nebius | exp_g_0249 lineage | prepared | — |
| 3.4 +TV | `exp_n_abl_22_B16k_light_lmfrozeng_hard_n2_tau0p5_tv10_seed1` | nebius | exp_g_0249 lineage | prepared | — |
| (extra) 3.1 seed 2 | `exp_g_abl_06_B16k_light_lmfrozeng_tph128_seed2` | gpustar | exp_g_0249 lineage | prepared, outside the 21 | — (a new sample) |

**Hosts, applied once with all 21 present:** 14 nebius, 7 gpustar (plus abl_06 on gpustar).
- Nebius gets everything heavy or unmeasured: the Gen-2 full-K pairs 2.1 and 2.2, which build several `[tokens, 512, 256]` fp32 buffers per layer backward (≈ 3.0 GiB each per micro-batch, roughly 20 GiB peak, not measured); all Gen-1 runs, whose speed and memory at this size have not been measured; the sparse Gen-2 pairs 2.3 / 2.4; and the 3.4 pair, to reach 14.
- Gpustar gets the Gen-3 pairs 3.1–3.3, whose parents ran on its 5090 at this geometry, and V, a 35.8M dense model.
- The host is the `exp_<n|g>_` prefix.

## What differs from each parent

The architecture is held fixed: E=384, 6 layers, 16K steps, device_batch 12 × grad_accum 4 (V: 48 × 1, as its parent), clip 1.0, CompressionMultiHeadLUT FFN H=4 × tph 128, n=8, d_in = d_out = 48.
- **+TV** always adds `lut_cell_smoothness: 10.0`.
- **Gen 1** (abl_11–14): `exp_n_0121` + `lut_impl: gen1`, `lut_gen1_smooth: false` (1.1) / `true` (1.2), `lut_gen1_n_alternatives: 1`, `lut_gen1_weights_init: uniform` (tables drawn exactly as Fast's and Light's).
- **Gen 2**, all from `exp_n_0121`:
  - 2.1 (abl_07 / 15): no flag;
  - 2.2 (abl_01 / 16): `lut_forward_mode: hybrid_smooth`, `lut_backward_topk: 0`;
  - 2.3 (abl_02 / 17): `lut_backward_topk: 1`;
  - 2.4 (abl_03 / 18): `hybrid_smooth` + `lut_backward_topk: 1`.
- **Gen 3**, all from `exp_g_0248`:
  - 3.1 (abl_08 / 09): no flag; abl_09's parent is `exp_g_0249` = `exp_g_0248` + TV;
  - 3.2 (abl_04 / 05): `lut_read_top_n: 2`, `lut_read_tau: 0.5` (τ itself, learnable), `lut_read_tau_learnable: true`;
  - 3.3 (abl_19 / 20): `lut_light_forward_mode: hard`;
  - 3.4 (abl_21 / 22): `hard` + the 3.2 read-out flags.
- **V** (abl_10): `exp_n_0135` verbatim.
- **abl_06:** `exp_g_0248` with `random_seed: 2`, `lut_base_seed: 2000`.

## Trainers

- **train_fixed.py (current)** — all Gen 1, all Gen 2 (abl_01/02/03/07 were refreshed so every Gen-2 run shares it), and V. It provides:
  - the corrected fixed eval;
  - the `lut_cell_smoothness` switch: `(λ·lut_tv_penalty()).backward()` once per step, before clipping;
  - `lut_tv` columns;
  - no weight decay on LUT tables, including the Gen-1 `LProjection` tables;
  - the W&B shim.
- **exp_g_0249 lineage** — all Gen 3 and abl_06: `exp_g_0248`'s trainer plus the same TV switch, read-only `lut_tv` columns and the shim.
- Every `train.py` has, immediately before `from wandb_tracking import Tracker`:
  `import wandb_tracking` / `wandb_tracking.GROUP = 'lut_ablation'`. The shim passes its GROUP constant as an explicit `group=`, so neither a config key nor `WANDB_RUN_GROUP` can set it.

**W&B config.** The W&B run config is `config.json`, less the notes keys and plus the tracker's extras. So every config spells out the keys that tell rows apart, including where the value is the default: `lut_impl`, `lut_forward_mode` / `lut_backward_topk` (Gen 2), `lut_light_forward_mode` / `lut_read_top_n` (Gen 3), `lut_gen1_smooth` / `lut_gen1_weights_init` (Gen 1), `lut_cell_smoothness`, `random_seed`, `lut_base_seed`. No row depends on an absent key. Writing the defaults out is build-neutral: 22/22 state_dicts are torch.equal to those built from the implicit configs.
- `eval_every` is 500 in all 22. The last eval is step 16,000 and produces `final_val_bpb`.
- The Light configs still carry the Fast-only `lut_forward_mode: hard` from their parents. It does nothing on the light path; `lut_light_forward_mode` sets Light's forward.
- **Launch: source `~/.wandb_env` in the trainer's own environment first**, or the tracker is OFF and the run trains with no W&B record.
  - The tracker needs `WANDB_BASE_URL`, and the entity comes from `WANDB_ENTITY`. Each host needs these two exports in `~/.wandb_env`, mode 600 and outside the repo; the key stays in `~/.netrc`. If the file is missing on a host, create it before launching.
  - **gpustar:** source it inside the cage command, e.g. `sbox --net tailnet -- bash -c '. ~/.wandb_env && python -u <folder>/train.py'`. With `--net tailnet` the run logs online. Plain `sbox` has no network, so the run logs offline and needs a later `wandb sync`.
  - **nebius:** `. ~/.wandb_env` in the launching shell.
  - Check the trainer's `[wandb]` start-up line: it says online, offline or off. `off: WANDB_BASE_URL not set` means the file was not sourced.

The code these runs need is all on the branch:
- #121 (c327aebe): FastMHL hard + `backward_topk`;
- 985c7707: TV for Gen 2 and the build-time guard;
- 6e782759: LightMHL `forward_mode='hard'` via `lut_light_forward_mode`;
- 2683f090: Gen 1 as `lut_impl='gen1'` and `MultiHeadLut.cell_tv`.

## Reproductions (V, 2.1, 3.1, 3.1 +TV)

Config and seeds are identical to the parent, apart from two build-neutral changes: `eval_every` 500, and explicit values for the row-identity keys, which the parent left at their defaults (see "W&B config" below). Initial weights are torch.equal to the parent trainer's on every tensor:
- abl_10 vs `exp_n_0135`: 52 tensors;
- abl_07 vs `exp_n_0121`: 112;
- abl_08 vs `exp_g_0248` and abl_09 vs `exp_g_0249`: 130 each.

Data loader, seed, schedule, AdamW grouping, clip and fixed eval match.

They are **not bit-identical**:
- no trainer enables deterministic algorithms, and the LUT backwards use CUDA-nondeterministic ops;
- abl_07 and abl_10's parents trained on nebius with older code and their own inline-model trainers.

Expect close to the published value. A gap around the vanilla 16K two-seed spread (0.00335) or larger is a signal.

## Verified for all 22, without a training step

- `config.json` builds, and the **built** model holds the row's module:
  - Gen 1: 6 × `MultiHeadLut` (smooth as the row, n_alternatives 1, INVERSE_L1, `weights_init='uniform'`, every table torch.equal to the per-head Uniform rule);
  - Gen 2: 6 × `FastMultiHeadLut` with the row's `forward_mode` / `backward_topk`;
  - Gen 3: 6 × `LightMultiHeadLUT` with the row's `forward_mode` (scored / hard) and `read_top_n`, τ 0.5 learnable at n=2;
  - V: 0 LUT modules and 6 dense FFNs.
- **+TV** runs (10): the build-time guard does not fire; `lut_tv_penalty()` on the built model is > 0 with a gradient; and the trainer source reads `lut_cell_smoothness` and backprops it after the micro-batch backwards and before clipping. **Plain** runs (12): `lut_cell_smoothness` is 0.
- `fork_trainer` check, the five shim calls, and exactly one group line before the `Tracker` import.
- W&B: executing each trainer's own tracker statements against a stub `wandb.init` gives `project='Spiky'`, `group='lut_ablation'`, `name = id = folder name`.
- `SMOKE=1 python <run>/train.py`: `SMOKE OK` for all 22; the Gen-1/2 +TV runs also print their `[tv]` setup line; nothing is written.

## Judgement calls worth checking

- **LightMHL hard forward key.** It is switched by its own key `lut_light_forward_mode`, not `lut_forward_mode`. The latter is a FastMHL key that reads `hard` in every existing light config, so reusing it would have flipped all of them.
- **Gen-1 wiring.** Gen 1 is wired only on the independent per-head path. Each head's tables draw anchors from its own compressed slice, mirroring Fast's `multi_head_input`.
- **Gen-1 table init — resolved (Anatoli overruled the N(0, 1e-3²) default).** abl_11–14 set `lut_gen1_weights_init: uniform`, which applies the Fast/Light table rule to `MultiHeadLut`: head h draws Uniform[−noise, +noise] from `Generator(seed + h + 1)`. The layer-0 tables are torch.equal to Fast's and Light's at the same seed. Std is 5.77e-04 and TV at init is 3.20e-05 in all three generations. `MultiHeadLut`'s own default is unchanged, and so is every config without the key: 179/179 committed configs build identical state_dicts and optimiser groups.
- **Names.** Kept at 61 characters or fewer: abl_16/18 say `hsmooth`, and abl_22 drops `tph128`.
- **τ logging.** The Gen-3 trainer does not log τ per eval; abl_04/05/21/22 keep the final per-layer `log_tau` in `checkpoint.pt` only.
- **abl_06.** Kept as a prepared extra (3.1 seed 2), outside the 21.
