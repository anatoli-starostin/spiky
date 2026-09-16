# exp_n_abl_47 — 48K steps, quantised arm (p2_int8), for nebius

**Run.**
- **Config:** row 3.2 +TV (`exp_g_abl_05` config, seed 1), with `n_steps` 48000 and `lut_quant_mode: "p2_int8"`.
- **Schedule:** warmup and cosine derive from `n_steps`, so both stretch to 48K.
- **Data:** the same 4 train shards as the 16K runs, so ~8.4 passes. Report the whole val curve.
- **Code:** spiky at this folder's commit (code = `4ad02234`); nanochat `da32e1d6`.
- **Files:** fill `run.env` from `run.env.template`. `preflight.py` checks the pins, the 5 shards and the tokenizer by sha256.

## A. The new code: p2_int8, and the kernel gate

**What it is.** This run uses the **quantised path, not the default arm**.
- **Training:** LUT tables train as float masters through a straight-through power-of-two read over int8-quantised tables (two cells per table, power-of-two weights, int32 shift-add).
- **Per-table integers:** computed by a CUDA kernel (`spiky.lutorch.pow2_int8`, JIT-built) when enabled, otherwise by the torch definition in `pow2_read`. The integers are bit-identical either way; the kernel is just faster.

**Gate — run before launching:**
```bash
set -a; . ./run.env; set +a; "$PYTHON" validate_kernel.py
```
- **`KERNEL GATE: PASS`** → train on the CUDA kernel. Leave `SPIKY_P2_CUDA_DISABLE` unset.
- **`KERNEL GATE: FAIL`, or the extension won't build/load** → set `SPIKY_P2_CUDA_DISABLE=1` in `run.env` and train on the torch fallback. **That is expected, not a bug.** Report the failing tests (`kernel_gate.json`).

**Background:**
- The kernel has only ever been validated on **sm_120 (RTX 5090)**. The sm_90 allowlist entry is **untested on real H100 hardware**; this gate is its first test.
- `preflight.py` prints which implementation serves (`CUDA KERNEL` / `TORCH FALLBACK`) and refuses the kernel without a PASS. `train.log` repeats it on a `[p2_int8]` line.
- **Wall clock:** the default arm takes ~3.5 h at 48K on an H100. Kernel speed on H100 is untuned. **Fallback speed is unmeasured, so the ETA is unknown if you land there.**

**Then:**
```bash
bash launch.sh
```
`launch.sh` requires `kernel_gate.json`, runs `preflight.py`, then starts `train_launch.py` detached.

## B. W&B

- **Target:**
  - endpoint `https://api.wandb.ai`;
  - entity `anatoli-starostin-relocation`, project `Spiky`, group `lut_ablation`;
  - run name = folder name.
  `run.env` / `launch.sh` set the endpoint; `~/.wandb_env` now also points at the cloud, but nothing relies on it.
- **Credential:** already in your `~/.netrc` (`machine api.wandb.ai`), verified 2026-09-16. `preflight.py` re-checks it.
- **On auth failure:** stop, report it to Anatoli, and never paste a key into this folder, `run.env`, a log or a commit.
- **Mandatory:** after the step-500 eval, run `"$PYTHON" verify_wandb.py`. It must print **`VERIFY_WANDB PASS`**, meaning train and val metrics are actually arriving on wandb.ai. **Do not walk away without it.**
  - On FAIL: `kill $(cat train.pid)` and relaunch from a clean copy of the folder.
  - Never delete the W&B run.

## Report to Anatoli

- The gate verdict (and failures), and the serving implementation.
- The W&B run name and the PASS.
- `final_val_bpb` / `best_val_bpb` to full precision, and `training_time_hours`.
- The val_bpb-by-step curve and where it bottoms out.
- Anything abnormal.
