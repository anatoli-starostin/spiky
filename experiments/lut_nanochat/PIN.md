# PIN / ENVIRONMENT: lut_nanochat d24 dense baseline

`pins.json` is the machine-readable copy. The launcher merges it, plus the resolved runtime versions and input hashes, into the wandb config.

## Code
- **nanochat:** [karpathy/nanochat](https://github.com/karpathy/nanochat), commit **`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`** (2026-07-03, "clean up fragile code"), tree `c6d965b5d89fe9e3d673d2c899c505b11f843772`.
- **Vendored, not a submodule.** The tree is copied into `nanochat/`, so the run cannot drift to another commit. Commit `67ded4a9` on this branch is the pristine copy (`git archive` of upstream). Every later change to `nanochat/` is ours and is listed below.

### Local changes to the vendored tree (all tagged `[lut_nanochat]` in the code)
| File | Change | Effect on the maths |
|---|---|---|
| `nanochat/flash_attention.py` | If `NANOCHAT_FA3_REVISION` is set, load `varunneal/flash-attention-3` at that Hub revision. | None (pins the kernel build). |
| `scripts/base_train.py` | Wandb `--wandb-project/--wandb-group/--wandb-tags/--wandb-notes-file`, `--pin-config` (merged into the config), `--log-every` (upstream hardcodes 100; we use 1). Adds the logged fields `train/loss_raw`, `train/tokens_seen`, `train/lr_matrix`. The derived run shape goes into the config. A wandb init failure warns instead of aborting. Prints peak *reserved* memory. | None. Logging only. |
| `runs/baseline_d24_1xh100.sh`, `runs/probe_memory_d24.sh`, `runs/stage_data.sh`, `runs/lut_env.sh`, `runs/env_smoke_test.py`, `runs/make_pin_config.py`, `runs/report_results.py`, `runs/lut_wandb_notes.md` | New files. | n/a |

Verify the full diff with: `git diff 67ded4a9 -- experiments/lut_nanochat/nanochat`.

## Python environment
- `uv sync --extra gpu` from the vendored `pyproject.toml` / `uv.lock`, exactly as `speedrun.sh` does.
- **torch `2.9.1+cu128`**, from the `pytorch-cu128` index. `kernels` 0.11.7 (from `uv.lock`). Python ≥ 3.10.
- **torchao is NOT a dependency.** fp8 is nanochat's own `nanochat/fp8.py`: tensorwise scaling via `torch._scaled_mm`.
- **CUDA:** the cu128 runtime ships in the torch wheel. Driver **≥ R570** (CUDA 12.8). This is UNCERTAIN: it is the general CUDA 12.8 requirement, not something nanochat states.

## FA3 kernel
- Hub repo `varunneal/flash-attention-3` (the Hopper path in `flash_attention.py`).
- Revision **`de87b9b5af06dd9984df595bef90b2eba44b181a`**. This is the repo's `main` sha from the HF API on 2026-10-07 (lastModified 2026-03-20). The expected build variant is `torch29-cxx11-cu128-x86_64-linux`.
- UNCERTAIN until `runs/env_smoke_test.py` passes:
  - the sha was read through a summarising web fetch;
  - whether `kernels.get_kernel(..., revision=...)` accepts it was not tested here (no Hopper GPU on our side).
  If it fails, see RUNBOOK troubleshooting.

## Data and evaluation inputs
- **ClimbMix:** [karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle).
  - Train shards `shard_00000`–`shard_00169` (170, as in `speedrun.sh`). The run consumes ~5.84B tokens; #819 used 131 shards.
  - **Val = `shard_06542`** (pinned as the last shard by `nanochat/dataset.py`).
  - nanochat downloads from `resolve/main`. On 2026-10-07 `main` pointed at `915333b4f8b8684f39aeaafea600fea6f43fb703` (HF API; UNCERTAIN, same caveat as above).
  - Reference sha256 for 5 shards (from our workstation) are in `pins.json`. `stage_data.sh` stops if they differ.
- **Tokenizer:** trained in-run by `scripts.tok_train` (vocab 32,768, 2B characters). Its sha256 is recorded in `results/<run>/manifest.json`. Whether it is byte-identical across machines is UNCERTAIN.
- **CORE eval bundle:** `https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip`. Its sha256 is recorded in the manifest; no reference hash is known in advance.

## Recipe (identical to the 8×H100 speedrun except `nproc_per_node`)
| Setting | Value |
|---|---|
| Flags | `--depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8` |
| Params | 1,384,122,122 total; 729,810,624 scaling |
| Global batch | 1,048,576 tokens (2^20) |
| Steps | 5,568 |
| Tokens | 5,838,471,168 |
| Gradient accumulation | 32 (dbs 16); fallback 64 (dbs 8, identical maths) |
| Seed | 42, hardcoded in `compute_init`; `s0` = first seed |
| Checkpoints | `--save-every=250`, newest 2 kept |

## Hardware
- **1× NVIDIA H100 80GB, Hopper `sm_90`.** SXM is preferred; PCIe works but is slower.
- **Not Blackwell (B200/RTX 5090).** FA3 has no `sm_100`/`sm_120` kernel, so nanochat falls back to SDPA, which has no sliding windows. That is why the RTX 5090 result in [#819](https://github.com/karpathy/nanochat/discussions/819) is not comparable.
- **Disk:** ≥ 100 GB free under `$NANOCHAT_BASE_DIR`. Data is ~15 GB; each checkpoint is ~11 GB, ×2 kept; plus the venv.
