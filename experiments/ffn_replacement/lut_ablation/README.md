# lut_ablation — prepared runs for the LUT core-module ablation table

These runs fill the rows of `doc/research/lut_ablation/lut_ablation_table_v4.tex` (Tables A/B). There is one folder per
run, laid out like `runs_corrected/`:
- `config.json` is the parent's config with only the flags below changed, plus a new `exp_name` and `_arch_note`.
- `train.py` is copied with `tools/fork_trainer.py`, which refuses trainers without the corrected eval.

**Nothing here has been launched.** Launching needs Anatoli's go, one run at a time.

## Prepared

| run | Table B row | parent | config change vs parent | trainer |
|---|---|---|---|---|
| `exp_n_0250_B16k_fast_hybridsmooth_topk0_nap8_tph128_seed1` | 2.2 | `exp_n_0121` | `lut_forward_mode` hard → hybrid_smooth; `lut_backward_topk` (absent) → 0 | `train_fixed.py` |
| `exp_n_0251_B16k_fast_hard_topk1_nap8_tph128_seed1` | 2.3 | `exp_n_0121` | `lut_backward_topk` (absent) → 1 | `train_fixed.py` |
| `exp_n_0252_B16k_fast_hybridsmooth_topk1_nap8_tph128_seed1` | 2.4 | `exp_n_0121` | `lut_forward_mode` hard → hybrid_smooth; `lut_backward_topk` (absent) → 1 | `train_fixed.py` |
| `exp_g_0253_B16k_light_learnedmargin_frozeng_n2_tau0p5learn_tph128_seed1` | 3.2 | `exp_g_0248` | + `lut_read_top_n` 2, + `lut_read_tau` 0.5 ⚠, + `lut_read_tau_learnable` true ⚠ | `exp_g_0249`'s `train.py` |
| `exp_g_0254_B16k_light_learnedmargin_frozeng_n2_tau0p5learn_tv10_tph128_seed1` | 3.2 +TV | `exp_g_0248` | as 0253, + `lut_cell_smoothness` 10.0 | `exp_g_0249`'s `train.py` |
| `exp_g_0255_B16k_light_learnedmargin_frozeng_tph128_seed2` | 3.1, seed replicate | `exp_g_0248` | `random_seed` 1 → 2; `lut_base_seed` 1000 → 2000 | `exp_g_0249`'s `train.py` |

Parents live in `../runs_corrected/`. Rows already run are not re-prepared: 2.1 = `exp_n_0121`, 3.1 = `exp_g_0248`, 3.1 +TV = `exp_g_0249`.

⚠ **UNCONFIRMED:**
- `lut_read_tau` 0.5 and `lut_read_tau_learnable` true in 0253/0254 are copied from `exp_g_0195`, which is n=2 with the plain margin score. Anatoli has not confirmed them for the learned-score form.

**Chosen here, not given:**
- Run numbers 0250–0255 continue past the highest number in the repo (0249). Work on nebius that hasn't been pushed could collide.
- The prefixes follow the parents: `exp_n_` for the Gen-2 forks of `exp_n_0121` (an H100 run), `exp_g_` for the Gen-3 forks.
- The seed replicate moves both seeds. `lut_base_seed` + layer index seeds each layer's anchors and tables, so 2000..2005 doesn't overlap 1000..1005.
- 2.2's full-K backward allocates several `[tokens, 512, 256]` fp32 buffers (Table B note a). `exp_n_0121` ran full-K at device_batch 12 on an H100; whether it fits on the 5090 is not checked.

## Trainers and the W&B shim

- **Gen 2:** `train_fixed.py`. It has the corrected fixed eval and the shim (`tools/wandb_tracking.py`). `exp_n_0121`'s own corrected trainer has no shim, so it was not copied. `train_fixed.py` uses the same `model_build`, LR schedule, optimiser grouping and seeds.
- **Gen 3:** `exp_g_0249`'s `train.py`. That is `exp_g_0248`'s trainer plus:
  - the `lut_cell_smoothness` switch (0 gives `exp_g_0248`'s math);
  - read-only `lut_tv` columns;
  - the shim.
  It does not log the n=2 read temperature per eval (`exp_g_0195`'s trainer did); 0253/0254 keep the final per-layer `log_tau` only in `checkpoint.pt`.
- All six `train.py` call `Tracker.start` / `train_step` / `eval_step` / `finish`. The shim is off unless `WANDB_BASE_URL` is set, and it is never fatal.

## Verified without a training step

- Each config differs from its parent in exactly the flags above (script diff).
- `tools/fork_trainer.check_trainer` passes on every `train.py`, and each contains all five shim calls; `wandb_tracking` imports with the package available.
- `model_build.build_model` on CPU gives 6 LUT modules with the intended class and settings:
  - Gen 2: `FastMultiHeadLut` with `forward_mode` / `backward_topk`, learnable temperatures;
  - Gen 3: `LightMultiHeadLUT`, learned_margin with g a frozen 0 buffer, `read_top_n`, τ 0.5 learnable.
- 0253/0254 have 67,351,698 parameters, which is 0248's 67,351,692 plus the six per-layer τ.
- `SMOKE=1 python <run>/train.py` builds the tokenizer, data loader and model, prints `SMOKE OK`, and writes nothing.

## Not prepared

| rows | why not | what it needs |
|---|---|---|
| 1.1, 1.2 (and +TV) | Gen 1 (`MultiHeadLut`) is not wired into `CompressionMultiHeadLUT` (`lut_impl` ∈ {fast, light, bh4}) | a `lut_impl` for `MultiHeadLut` in `compression_mhl.py` and `model_build.py` |
| 3.3, 3.4 (and +TV) | `LightMultiHeadLUT` has no hard forward: every path returns the score-scaled read | a hard-forward mode (plain read + f − sg(f)), plumbed through `CompressionMultiHeadLUT` / `model_build` |
| 2.1–2.4 +TV | see below | see below |

Why the Gen-2 +TV runs are not prepared, checked on the `exp_n_0250` config:
- `model_build.lut_tv_penalty` collects `LightMultiHeadLUT` tables only. On a `FastMultiHeadLut` model it returns 0.0 with no gradient.
- `train_fixed.py` never reads `lut_cell_smoothness`, so a +TV config on it would silently train with no penalty.
- `exp_g_0249`'s trainer would crash: `(λ·penalty).backward()` raises "does not require grad", and its `tv_stats` raises AttributeError (no `ffn.lut_light`).

To make them runnable:
1. extend `lut_tv_penalty` to `FastMultiHeadLut` tables (a `cell_tv` over its `weights`);
2. make the trainer's `tv_stats` implementation-agnostic;
3. fork a trainer that carries the `lut_cell_smoothness` switch.
