# lut_ablation — prepared runs for the LUT core-module ablation table

These runs fill the rows of `doc/research/lut_ablation/lut_ablation_table_v4.tex` (Tables A/B). There is one folder per
run, laid out like `runs_corrected/`:
- `config.json` is the parent's config with only the flags below changed, plus a new `exp_name` and `_arch_note`.
- `train.py` is copied with `tools/fork_trainer.py`, plus the one W&B group line (below).

**Nothing here has been launched.** Launching needs Anatoli's go, one run at a time.

## Naming

`exp_<host>_abl_<NN>_<descriptor>`:
- `<host>` keeps the repo's convention: `n` = nebius, `g` = gpustar.
- `abl_<NN>` replaces the shared four-digit counter, so no name here can collide with an `exp_[gn]_NNNN` run.
- Names are at most 61 characters, the longest `exp_name` already used as a W&B run id.

## The nine runs

| run | Table B row | parent | config change vs parent | expected host | status | expected bpb |
|---|---|---|---|---|---|---|
| `exp_n_abl_01_B16k_fast_hybridsmooth_topk0_nap8_tph128_seed1` | 2.2 | `exp_n_0121` | `lut_forward_mode` hard → hybrid_smooth; `lut_backward_topk` (absent) → 0 | **nebius** | prepared | — |
| `exp_n_abl_02_B16k_fast_hard_topk1_nap8_tph128_seed1` | 2.3 | `exp_n_0121` | `lut_backward_topk` (absent) → 1 | either | prepared | — |
| `exp_n_abl_03_B16k_fast_hybridsmooth_topk1_nap8_tph128_seed1` | 2.4 | `exp_n_0121` | `lut_forward_mode` hard → hybrid_smooth; `lut_backward_topk` (absent) → 1 | either | prepared | — |
| `exp_g_abl_04_B16k_light_lmfrozeng_n2_tau0p5_tph128_seed1` | 3.2 | `exp_g_0248` | + `lut_read_top_n` 2, + `lut_read_tau` 0.5, + `lut_read_tau_learnable` true | 5090 | prepared | — |
| `exp_g_abl_05_B16k_light_lmfrozeng_n2_tau0p5_tv10_tph128_seed1` | 3.2 +TV | `exp_g_0248` | as abl_04, + `lut_cell_smoothness` 10.0 | 5090 | prepared | — |
| `exp_g_abl_06_B16k_light_lmfrozeng_tph128_seed2` | 3.1, seed replicate | `exp_g_0248` | `random_seed` 1 → 2; `lut_base_seed` 1000 → 2000 | 5090 | prepared | — (a new sample) |
| `exp_n_abl_07_B16k_fast_hard_topk0_nap8_tph128_seed1` | 2.1, reproduction | `exp_n_0121` | none | **nebius** | prepared | **≈ 1.180622** |
| `exp_g_abl_08_B16k_light_lmfrozeng_tph128_seed1` | 3.1, reproduction | `exp_g_0248` | none | 5090 | prepared | **≈ 1.169328** |
| `exp_g_abl_09_B16k_light_lmfrozeng_tv10_tph128_seed1` | 3.1 +TV, reproduction | `exp_g_0249` | none | 5090 | prepared | **≈ 1.163698** |

Parents live in `../runs_corrected/`.
- Trainers: abl_01–03 and abl_07 use `train_fixed.py`; abl_04–06, abl_08 and abl_09 use `exp_g_0249`'s `train.py`.
- The expected values are the parents' published numbers. `exp_n_0121`'s is the corrected re-score of its checkpoint.

**τ in abl_04 / abl_05 is a deliberate choice: 0.5 at initialisation, learnable per layer.**
- `lut_read_tau` is τ itself: `LightMultiHeadLUT` stores `log_tau = ln 0.5` and reads `τ = exp(log_tau)`.
- Checked at build: every layer has τ = 0.500000000.
- The n=1 runs (abl_06, 08, 09) never use τ.

**Seeds:** abl_06 moves both seeds. `lut_base_seed` + layer index seeds each layer's anchors and tables, so 2000..2005 doesn't overlap 1000..1005.

## Reproductions (abl_07–09): how close to expect

**Same as the parent, checked:**
- Config and seeds are identical: the config diff is empty apart from `exp_name` and `_arch_note`.
- Initial weights are identical. Running each parent's trainer and the reproduction's trainer with `SMOKE=1` on CPU gives `torch.equal` on every tensor:
  - abl_07 vs `exp_n_0121`: 112 tensors, even though its trainer builds the model inline and `train_fixed.py` uses `model_build`;
  - abl_08 vs `exp_g_0248` and abl_09 vs `exp_g_0249`: 130 tensors each.
  Both trainers initialise on CPU before `.to(device)`, so the same holds on GPU.
- The code after init matches:
  - the data loader (`tokenizing_distributed_data_loader_bos_bestfit` at device_batch 12), `torch.manual_seed(random_seed)`, grad_accum 4, clip 1.0;
  - the LR schedule (`get_lr_scale`) and the AdamW groups (FastMHL / Light tables and 1-D parameters without decay);
  - the fixed eval.
  The reproduction trainers only add read-only work: LayerNorm / learned-margin / TV columns at evals, the W&B tracker, and keeping the returned grad norm.

**Not bit-identical: expect close, not the last digit.**
- No trainer enables deterministic algorithms, and the LUT backwards use CUDA ops that are not deterministic (`embedding_bag` with per-sample weights, `index_add_`, `scatter_add_`). Even the same code on the same GPU diverges at float level and drifts over 16K steps.
- abl_07 has more distance still. `exp_n_0121` trained on an H100 on 2026-09-03 with its own trainer and that day's `FastMultiHeadLut` / `CompressionMultiHeadLUT`; both have changed since, in commits documented as bit-identical for existing configs but not re-checked against that date.
- abl_08/09's parents ran on this 5090 on 2026-09-12. Since then the only library change is #121, which touches Fast only.
- **How close counts as "close":** the vanilla 16K two-seed spread is 0.00335, and that is between different seeds. A same-seed rerun should land nearer than that; a gap of that size or more is a signal to investigate.

## W&B group

Every run reports to the group **`lut_ablation`**.
- The shim (`tools/wandb_tracking.py`) passes its module constant `GROUP = 'ffn_replacement'` as an explicit `group=` argument. The package only falls back to `WANDB_RUN_GROUP` when no argument is given, so neither a config key nor the environment can change it.
- Each `train.py` therefore gets one line just before `from wandb_tracking import Tracker`:
  `import wandb_tracking` / `wandb_tracking.GROUP = 'lut_ablation'`. It is the same in all nine; no library code changed.
- Checked by executing each trainer's own tracker statements against a stub `wandb.init`: every run gives `project='Spiky'`, `group='lut_ablation'`, `name = id = folder name`, and `lut_ablation` is also the first tag.

## Where to run: nebius vs the local 5090

- **abl_01 (2.2) and abl_07 (2.1) need nebius.**
  - Their full-K soft backward builds several `[tokens, 512, 256]` fp32 buffers in each layer's LUT backward (Table B note a), and the parent ran on an H100.
  - Rough, unmeasured estimate: one buffer per 6,144-token micro-batch is ≈ 3.0 GiB. About six alive at once, plus weights, optimiser state and activations, gives a peak of roughly 20 GiB.
- **abl_02 / abl_03** use the sparse backward, which touches only R = 2 rows. Expected a few GiB and fine on either host (not measured).
- **abl_04–06, 08, 09 (Gen 3)** have the same geometry as `exp_g_0248` / `exp_g_0249`, which ran on the 5090.

## Verified without a training step

- Each config differs from its parent in exactly the flags above (script diff), and `exp_name` equals the folder name.
- `tools/fork_trainer.check_trainer` passes on every `train.py`. Each contains all five shim calls and exactly one group line, placed before the `Tracker` import.
- `model_build.build_model` on CPU gives 6 LUT modules with the intended class and settings:
  - Gen 2: `FastMultiHeadLut` with `forward_mode` / `backward_topk`, learnable temperatures;
  - Gen 3: `LightMultiHeadLUT`, learned_margin with g a frozen 0 buffer, `read_top_n`, τ as above.
- `SMOKE=1 python <run>/train.py` builds the tokenizer, data loader and model, prints `SMOKE OK`, and writes nothing; every folder holds only `config.json` and `train.py`.
- The reproductions' initial weights and the W&B group are checked as described above.

## Blocked on code (not prepared)

| rows | why not | what it needs |
|---|---|---|
| 1.1, 1.2 (and +TV) | Gen 1 (`MultiHeadLut`) is not wired into `CompressionMultiHeadLUT` (`lut_impl` ∈ {fast, light, bh4}) | a `lut_impl` for `MultiHeadLut` in `compression_mhl.py` and `model_build.py` |
| 3.3, 3.4 (and +TV) | `LightMultiHeadLUT` has no hard forward: every path returns the score-scaled read | a hard-forward mode (plain read + f − sg(f)), plumbed through `CompressionMultiHeadLUT` / `model_build` |
| 2.1–2.4 +TV | see below | see below |

Why the Gen-2 +TV runs are blocked, checked on a Fast config:
- `model_build.lut_tv_penalty` collects `LightMultiHeadLUT` tables only. On a `FastMultiHeadLut` model it returns 0.0 with no gradient.
- `train_fixed.py` never reads `lut_cell_smoothness`, so a +TV config on it would silently train with no penalty.
- `exp_g_0249`'s trainer would crash: `(λ·penalty).backward()` raises "does not require grad", and its `tv_stats` raises AttributeError (no `ffn.lut_light`).

To unblock them:
1. extend `lut_tv_penalty` to `FastMultiHeadLut` tables (a `cell_tv` over its `weights`);
2. make the trainer's `tv_stats` implementation-agnostic;
3. fork a trainer that carries the `lut_cell_smoothness` switch.
