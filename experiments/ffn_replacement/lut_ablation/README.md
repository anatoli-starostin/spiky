# lut_ablation — prepared runs for the LUT core-module ablation table

These runs fill the rows of `doc/research/lut_ablation/lut_ablation_table_v4.tex` (Tables A/B). There is one folder per
run, laid out like `runs_corrected/`:
- `config.json` is the parent's config with only the flags below changed, plus a new `exp_name` and `_arch_note`.
- `train.py` is copied with `tools/fork_trainer.py`, plus the one W&B group line (below).

**Nothing here has been launched.** Launching needs Anatoli's go, one run at a time.

## The 21-run ablation, and what is prepared here

The ablation is **21 runs**:
- 10 LUT rows (1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4) × {plain, +TV} = 20;
- plus the vanilla untied baseline, plain only. Its +TV is n/a because there are no LUT tables to regularise. **The baseline is one of the 21.**

This directory holds **10 prepared runs**:
- **9 of the 21:** 2.1, 2.2, 2.3, 2.4, 3.1, 3.1 +TV, 3.2, 3.2 +TV, and the vanilla baseline.
- **abl_06:** a seed replicate of 3.1, outside the 21.

Still to prepare, 12 of the 21:
- 2.1–2.4 +TV: the code now supports it (below), but the runs aren't prepared yet;
- 1.1, 1.2, 3.3, 3.4 and their +TV: blocked on code.

Hosts will be allocated across the full 21 (roughly 14 nebius / 7 gpustar) once all folders exist, so nothing is renamed twice. The "expected host" column below is the current technical expectation, not that allocation.

## Naming

`exp_<host>_abl_<NN>_<descriptor>`:
- `<host>` keeps the repo's convention: `n` = nebius, `g` = gpustar.
- `abl_<NN>` replaces the shared four-digit counter, so no name here can collide with an `exp_[gn]_NNNN` run.
- Names are at most 61 characters, the longest `exp_name` already used as a W&B run id.

## The ten prepared runs

| run | Table B row | in the 21? | parent | config change vs parent | expected host | status | expected bpb |
|---|---|---|---|---|---|---|---|
| `exp_n_abl_01_B16k_fast_hybridsmooth_topk0_nap8_tph128_seed1` | 2.2 | yes | `exp_n_0121` | `lut_forward_mode` hard → hybrid_smooth; `lut_backward_topk` (absent) → 0 | **nebius** | prepared | — |
| `exp_n_abl_02_B16k_fast_hard_topk1_nap8_tph128_seed1` | 2.3 | yes | `exp_n_0121` | `lut_backward_topk` (absent) → 1 | either | prepared | — |
| `exp_n_abl_03_B16k_fast_hybridsmooth_topk1_nap8_tph128_seed1` | 2.4 | yes | `exp_n_0121` | `lut_forward_mode` hard → hybrid_smooth; `lut_backward_topk` (absent) → 1 | either | prepared | — |
| `exp_g_abl_04_B16k_light_lmfrozeng_n2_tau0p5_tph128_seed1` | 3.2 | yes | `exp_g_0248` | + `lut_read_top_n` 2, + `lut_read_tau` 0.5, + `lut_read_tau_learnable` true | 5090 | prepared | — |
| `exp_g_abl_05_B16k_light_lmfrozeng_n2_tau0p5_tv10_tph128_seed1` | 3.2 +TV | yes | `exp_g_0248` | as abl_04, + `lut_cell_smoothness` 10.0 | 5090 | prepared | — |
| `exp_g_abl_06_B16k_light_lmfrozeng_tph128_seed2` | 3.1, seed replicate | **no** (extra) | `exp_g_0248` | `random_seed` 1 → 2; `lut_base_seed` 1000 → 2000 | 5090 | prepared | — (a new sample) |
| `exp_n_abl_07_B16k_fast_hard_topk0_nap8_tph128_seed1` | 2.1, reproduction | yes | `exp_n_0121` | none | **nebius** | prepared | **≈ 1.180622** |
| `exp_g_abl_08_B16k_light_lmfrozeng_tph128_seed1` | 3.1, reproduction | yes | `exp_g_0248` | none | 5090 | prepared | **≈ 1.169328** |
| `exp_g_abl_09_B16k_light_lmfrozeng_tv10_tph128_seed1` | 3.1 +TV, reproduction | yes | `exp_g_0249` | none | 5090 | prepared | **≈ 1.163698** |
| `exp_n_abl_10_B16k_vanilla_dense_untied_seed1` | V (vanilla baseline), reproduction | yes | `exp_n_0135` | none | either | prepared | **≈ 1.165147** |

Parents live in `../runs_corrected/`.
- Trainers:
  - abl_01–03 and abl_07: `train_fixed.py` as it was before the Gen-2 TV change;
  - abl_10: the current `train_fixed.py`, which carries the TV switch (off here);
  - abl_04–06, 08, 09: `exp_g_0249`'s `train.py`.
- The expected values are the parents' published numbers. `exp_n_0121`'s and `exp_n_0135`'s are corrected re-scores of their checkpoints.
- The vanilla second seed, `exp_n_0176`, scores 1.161798.

**τ in abl_04 / abl_05 is a deliberate choice: 0.5 at initialisation, learnable per layer.**
- `lut_read_tau` is τ itself: `LightMultiHeadLUT` stores `log_tau = ln 0.5` and reads `τ = exp(log_tau)`.
- Checked at build: every layer has τ = 0.500000000.
- The n=1 runs (abl_06, 08, 09) never use τ.

**Seeds:** abl_06 moves both seeds. `lut_base_seed` + layer index seeds each layer's anchors and tables, so 2000..2005 doesn't overlap 1000..1005.

## Reproductions (abl_07–10): how close to expect

**Same as the parent, checked:**
- Config and seeds are identical: the config diff is empty apart from `exp_name` and `_arch_note`.
- Initial weights are identical. Running each parent's trainer and the reproduction's trainer with `SMOKE=1` on CPU gives `torch.equal` on every tensor:
  - abl_07 vs `exp_n_0121`: 112 tensors;
  - abl_08 vs `exp_g_0248` and abl_09 vs `exp_g_0249`: 130 tensors each;
  - abl_10 vs `exp_n_0135`: 52 tensors.
  That holds even where the parent's trainer builds the model inline (0121, 0135) and the reproduction uses `model_build`. Both initialise on CPU before `.to(device)`, so the same holds on GPU.
- The code after init matches:
  - the data loader (`tokenizing_distributed_data_loader_bos_bestfit` at the config's device batch), `torch.manual_seed(random_seed)`, grad_accum, clip 1.0;
  - the LR schedule and the AdamW groups;
  - the fixed eval.
  The reproduction trainers only add read-only work (LayerNorm / learned-margin / TV columns at evals, the W&B tracker) plus, in `train_fixed.py`, a TV switch that is off at `lut_cell_smoothness` 0.

**Not bit-identical: expect close, not the last digit.**
- No trainer enables deterministic algorithms, and the LUT backwards use CUDA ops that are not deterministic. Even the same code on the same GPU drifts at float level over 16K steps.
- abl_07 and abl_10's parents trained on nebius with their own inline-model trainers and older code.
- abl_08/09's parents ran on this 5090 on 2026-09-12.
- **How close counts as "close":** the vanilla 16K two-seed spread is 0.00335 (`exp_n_0135` vs `exp_n_0176`), and that is between different seeds. A same-seed rerun should land nearer than that; a gap of that size or more is a signal.

## W&B group

Every run reports to the group **`lut_ablation`**.
- The shim (`tools/wandb_tracking.py`) passes its module constant `GROUP = 'ffn_replacement'` as an explicit `group=` argument. The package only falls back to `WANDB_RUN_GROUP` when no argument is given, so neither a config key nor the environment can change it.
- Each `train.py` therefore gets one line just before `from wandb_tracking import Tracker`:
  `import wandb_tracking` / `wandb_tracking.GROUP = 'lut_ablation'`. It is the same in all ten; no library code changed.
- Checked by executing each trainer's own tracker statements against a stub `wandb.init`: every run gives `project='Spiky'`, `group='lut_ablation'`, `name = id = folder name`.

## Where to run: nebius vs the local 5090

- **abl_01 (2.2) and abl_07 (2.1) need nebius.**
  - Their full-K soft backward builds several `[tokens, 512, 256]` fp32 buffers in each layer's LUT backward (Table B note a), and the parent ran on an H100.
  - Rough, unmeasured estimate: one buffer per 6,144-token micro-batch is ≈ 3.0 GiB. About six alive at once, plus weights, optimiser state and activations, gives a peak of roughly 20 GiB.
- **abl_02 / abl_03** use the sparse R = 2 backward. Expected a few GiB; either host (not measured).
- **abl_04–06, 08, 09 (Gen 3)** have the same geometry as `exp_g_0248` / `exp_g_0249`, which ran on the 5090.
- **abl_10 (vanilla)** is a 35.8M-parameter dense model; either host.

## Verified without a training step

- Each config differs from its parent in exactly the flags above (script diff), and `exp_name` equals the folder name.
- `tools/fork_trainer.check_trainer` passes on every `train.py`. Each contains all five shim calls and exactly one group line, placed before the `Tracker` import.
- `model_build.build_model` on CPU gives the intended modules:
  - Gen 2: 6 × `FastMultiHeadLut` with `forward_mode` / `backward_topk`, learnable temperatures;
  - Gen 3: 6 × `LightMultiHeadLUT`, learned_margin with g a frozen 0 buffer, `read_top_n`, τ as above;
  - abl_10: **0 LUT modules** (no `FastMultiHeadLut` / `LightMultiHeadLUT` / `BH4MultiHeadLUT` / `MultiHeadLut` / `CompressionMultiHeadLUT`) and 6 × dense FFN `Linear(384→1536, no bias)`, GELU, `Linear(1536→384, no bias)`.
- `SMOKE=1 python <run>/train.py` builds the tokenizer, data loader and model, prints `SMOKE OK`, and writes nothing; every folder holds only `config.json` and `train.py`.

## Not prepared yet

| cells | state | what it needs |
|---|---|---|
| 2.1, 2.2, 2.3, 2.4 +TV | **code ready, runs not prepared.** Since 985c7707 `lut_tv_penalty` covers `FastMultiHeadLut` tables, `build_model` refuses a TV value it cannot deliver, and the current `train_fixed.py` applies `lut_cell_smoothness` | fork the Gen-2 configs + `lut_cell_smoothness: 10` onto the **current** `train_fixed.py`. The abl_01–03/07 trainers predate the switch, so a TV key there would be ignored. |
| 1.1, 1.2 (and +TV) | blocked on code: Gen 1 (`MultiHeadLut`) is not wired into `CompressionMultiHeadLUT` (`lut_impl` ∈ {fast, light, bh4}) | a `lut_impl` for `MultiHeadLut` in `compression_mhl.py` / `model_build.py`; for +TV also a `cell_tv` for its `LProjection` tables (2^nap · n_buckets rows), with the adjacency verified through its read path |
| 3.3, 3.4 (and +TV) | blocked on code: `LightMultiHeadLUT` has no hard forward; every path returns the score-scaled read | a hard-forward mode (plain read + f − sg(f)), plumbed through `CompressionMultiHeadLUT` / `model_build` |
