# lut_ablation — prepared runs for the LUT core-module ablation table

These runs fill the rows of `doc/research/lut_ablation/lut_ablation_table_v4.tex` (Tables A/B). There is one folder per
run, laid out like `runs_corrected/`:
- `config.json` is the parent's config with only the flags below changed, plus a new `exp_name` and `_arch_note`.
- `train.py` is copied with `tools/fork_trainer.py`, which refuses trainers without the corrected eval.

**Nothing here has been launched.** Launching needs Anatoli's go, one run at a time.

## Naming

`exp_<host>_abl_<NN>_<descriptor>`:
- `<host>` keeps the repo's convention: `n` = nebius, `g` = gpustar.
- `abl_<NN>` replaces the shared four-digit counter, so these names can never collide with an `exp_[gn]_NNNN` run, including anything taken on nebius and not yet pushed.
- Names are at most 61 characters, the longest `exp_name` already used as a W&B run id (the tracker uses `exp_name` as the run name and id).

## Prepared

| run | Table B row | parent | config change vs parent | trainer | expected host |
|---|---|---|---|---|---|
| `exp_n_abl_01_B16k_fast_hybridsmooth_topk0_nap8_tph128_seed1` | 2.2 | `exp_n_0121` | `lut_forward_mode` hard → hybrid_smooth; `lut_backward_topk` (absent) → 0 | `train_fixed.py` | **nebius** |
| `exp_n_abl_02_B16k_fast_hard_topk1_nap8_tph128_seed1` | 2.3 | `exp_n_0121` | `lut_backward_topk` (absent) → 1 | `train_fixed.py` | either |
| `exp_n_abl_03_B16k_fast_hybridsmooth_topk1_nap8_tph128_seed1` | 2.4 | `exp_n_0121` | `lut_forward_mode` hard → hybrid_smooth; `lut_backward_topk` (absent) → 1 | `train_fixed.py` | either |
| `exp_g_abl_04_B16k_light_lmfrozeng_n2_tau0p5_tph128_seed1` | 3.2 | `exp_g_0248` | + `lut_read_top_n` 2, + `lut_read_tau` 0.5, + `lut_read_tau_learnable` true | `exp_g_0249`'s `train.py` | 5090 |
| `exp_g_abl_05_B16k_light_lmfrozeng_n2_tau0p5_tv10_tph128_seed1` | 3.2 +TV | `exp_g_0248` | as abl_04, + `lut_cell_smoothness` 10.0 | `exp_g_0249`'s `train.py` | 5090 |
| `exp_g_abl_06_B16k_light_lmfrozeng_tph128_seed2` | 3.1, seed replicate | `exp_g_0248` | `random_seed` 1 → 2; `lut_base_seed` 1000 → 2000 | `exp_g_0249`'s `train.py` | 5090 |

Parents live in `../runs_corrected/`. Rows already run are not re-prepared: 2.1 = `exp_n_0121`, 3.1 = `exp_g_0248`, 3.1 +TV = `exp_g_0249`.

**τ in abl_04 / abl_05 is a deliberate choice: 0.5 at initialisation, learnable per layer.**
- `lut_read_tau` is τ itself. `model_build` passes it through as a float; `LightMultiHeadLUT` stores `log_tau = ln 0.5` as a parameter and reads `τ = exp(log_tau)`.
- Checked at build: every layer has τ = 0.500000000 (`log_tau` = −0.693147182).
- abl_06 is n=1, so its τ (the default 0.1, a frozen buffer) is never used.

**Seeds:** the replicate moves both seeds. `lut_base_seed` + layer index seeds each layer's anchors and tables, so 2000..2005 doesn't overlap 1000..1005.

## Where to run: nebius vs the local 5090

- **abl_01 (2.2) needs nebius.**
  - Its full-K soft backward builds several `[tokens, 512, 256]` fp32 buffers in each layer's LUT backward (Table B note a), and the parent `exp_n_0121` ran on an H100.
  - Rough, unmeasured estimate: one micro-batch is 12 × 512 = 6,144 tokens, and one such buffer is 6,144 × 512 × 256 × 4 B ≈ 3.0 GiB.
  - The body holds about six at once, ≈ 18 GiB transient. On top of that: ≈ 1 GiB of weights, gradients and AdamW state for 67.35M parameters, plus the transformer activations. That puts the peak at roughly 20 GiB.
- **abl_02 / abl_03 (2.3 / 2.4)** use the sparse backward, which touches only R = 2 rows, so no K-sized tensor is built. Expected to be a few GiB and fine on either host (not measured). The `n` prefix only follows the parent.
- **abl_04–06 (Gen 3)** have the same geometry as `exp_g_0248` / `exp_g_0249`, which ran on the 5090. The n=2 read adds one gathered row per table.

## Trainers and the W&B shim

- **Gen 2:** `train_fixed.py`. It has the corrected fixed eval and the shim (`tools/wandb_tracking.py`). `exp_n_0121`'s own corrected trainer has no shim, so it was not copied. `train_fixed.py` uses the same `model_build`, LR schedule, optimiser grouping and seeds.
- **Gen 3:** `exp_g_0249`'s `train.py`. That is `exp_g_0248`'s trainer plus:
  - the `lut_cell_smoothness` switch (0 gives `exp_g_0248`'s math);
  - read-only `lut_tv` columns;
  - the shim.
  It does not log τ per eval (`exp_g_0195`'s trainer did); abl_04/05 keep the final per-layer `log_tau` only in `checkpoint.pt`.
- All six `train.py` call `Tracker.start` / `train_step` / `eval_step` / `finish`. The shim is off unless `WANDB_BASE_URL` is set, and it is never fatal.

## Verified without a training step (re-run after the rename)

- Each config differs from its parent in exactly the flags above (script diff), and `exp_name` equals the folder name.
- `tools/fork_trainer.check_trainer` passes on every `train.py`, and each contains all five shim calls; `wandb_tracking` imports with the package available.
- `model_build.build_model` on CPU gives 6 LUT modules with the intended class and settings:
  - Gen 2: `FastMultiHeadLut` with `forward_mode` / `backward_topk`, learnable temperatures;
  - Gen 3: `LightMultiHeadLUT`, learned_margin with g a frozen 0 buffer, `read_top_n`, τ as above.
- abl_04/05 have 67,351,698 parameters, which is `exp_g_0248`'s 67,351,692 plus the six per-layer τ.
- `SMOKE=1 python <run>/train.py` builds the tokenizer, data loader and model, prints `SMOKE OK`, and writes nothing.

## Not prepared

| rows | why not | what it needs |
|---|---|---|
| 1.1, 1.2 (and +TV) | Gen 1 (`MultiHeadLut`) is not wired into `CompressionMultiHeadLUT` (`lut_impl` ∈ {fast, light, bh4}) | a `lut_impl` for `MultiHeadLut` in `compression_mhl.py` and `model_build.py` |
| 3.3, 3.4 (and +TV) | `LightMultiHeadLUT` has no hard forward: every path returns the score-scaled read | a hard-forward mode (plain read + f − sg(f)), plumbed through `CompressionMultiHeadLUT` / `model_build` |
| 2.1–2.4 +TV | see below | see below |

Why the Gen-2 +TV runs are not prepared, checked on the abl_01 config:
- `model_build.lut_tv_penalty` collects `LightMultiHeadLUT` tables only. On a `FastMultiHeadLut` model it returns 0.0 with no gradient.
- `train_fixed.py` never reads `lut_cell_smoothness`, so a +TV config on it would silently train with no penalty.
- `exp_g_0249`'s trainer would crash: `(λ·penalty).backward()` raises "does not require grad", and its `tv_stats` raises AttributeError (no `ffn.lut_light`).

To make them runnable:
1. extend `lut_tv_penalty` to `FastMultiHeadLut` tables (a `cell_tv` over its `weights`);
2. make the trainer's `tv_stats` implementation-agnostic;
3. fork a trainer that carries the `lut_cell_smoothness` switch.
