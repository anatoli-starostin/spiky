# exp_n_abl_47 — 48K quantised run (p2_int8), prepared for nebius

This folder is a prepared **run definition for another machine**. It holds a config, a launcher, checks and a README. It is not code that participates in any build.

## ⚠️ Two decisions to veto before anything runs

**1. LR schedule: warmup and cosine decay STRETCH to 48K.**
- **Why:** `train.py` has no schedule constants. `get_lr_scale` derives warmup = `int(lr_warmup_fraction × n_steps)` and a cosine from lr down to a 0.1×lr floor, ending exactly at `n_steps`.
- **At 48K:** warmup is 4,800 steps (it was 1,600 at 16K) and the floor is reached at step 48,000 (it was 16,000). Peak lr is 3e-4 and the floor 3e-5, as before.
- **Why this is correct:** it is what the repo's recipe does whenever the step count changes, and decay horizon = total steps is the standard choice. It is also the only reading in which 48K is a scaled copy of 16K.
- **The alternative, not chosen:** keeping the 16K warmup and horizon would spend 32K steps at the floor and needs a trainer change.

**2. Data: the SAME 4 train shards, so ~8.4 passes over the training data (the 16K runs do ~2.8).**
- **What "identical" pins:** "identical except step count" pins the dataset. The 16K runs train on 4 ClimbMix train shards: ~215M raw tokens, ~140M used after the loader's best-fit cropping. nanochat's loader loops over them.
- **At 48K:** the run consumes 1.18B tokens, about 8.4 passes. These are estimates; see `data_manifest.json`.
- **Why it matters:** heavy repetition can flatten or reverse the val curve late in the run, so report the whole curve.
- **The alternative:** more shards for ~single-pass training would change the dataset and make it a different experiment.

## What this run is and what it tests

**The model:** row 3.2 +TV of the LUT ablation (`abl_45` geometry):
- a 6-layer, E=384 GPT whose FFNs are light LUT layers;
- H=4 heads × 128 tables, 8 anchor pairs, d_in = d_out = 48;
- `learned_margin` confidence with g frozen, a two-cell blended read (tau 0.5 learnable), cell-TV λ=10.

**The arm:** **`quant_mode="p2_int8"`**. Training is straight-through on float master tables. The forward is the power-of-two two-cell read over int8-quantised tables, and the integers are read as an int32 shift-add. This is the library feature merged in PR #126.

**The config:** `exp_g_abl_05_B16k_light_lmfrozeng_n2_tau0p5_tv10_tph128_seed1` with only `n_steps` 16000 → 48000 and `lut_quant_mode: "p2_int8"` changed. `preflight.py` checks every protocol key.

**Its 16K twin** runs on gpustar: `exp_g_abl_48_…_p2int8_postmerge126`, next to a default-path 16K rerun `exp_g_abl_46_…_postmerge126`. This run gives the first 48K number for the quantised arm. If the kernel gate passes, it is also the first training run of the p2_int8 CUDA kernel on a non-5090 GPU.

## Pinned code and data (do not use "latest")

| what | pin |
|---|---|
| spiky | `research/ffn_replacement_fix` @ **`4ad02234900413fa7555e4c4048de80fae4346f5`** (the PR #126 merge dd8c271e + the sm_90 allowlist gate). The commit that adds this folder comes after it; see the handoff message |
| nanochat | **`da32e1d657b62cc1110f20f86fbc301d6189726e`** |
| data | exactly the 5 ClimbMix shards in `data_manifest.json` (sha256 listed); train = shard_00000..00003, val = shard_06542 |
| tokenizer | `nanochat_tokenizer/` in this folder (sha256 in the manifest). It was trained locally, so copy it; do not retrain |

Run from the spiky checkout **that contains this folder**. The spiky package in your venv must import from that checkout (an editable install of `$SPIKY_ROOT`); preflight checks HEAD and that library files are unmodified.

## W&B — live cloud logging (read this; it is not boilerplate)

- **Where it logs:**
  - **endpoint `https://api.wandb.ai`** (wandb.ai cloud);
  - **project `Spiky`**, **group `lut_ablation`**;
  - **run name = run id = `exp_n_abl_47_B48k_light_lmfrozeng_n2_tau0p5_tv10_tph128_seed1_p2int8`** (= `exp_name`);
  - **entity `anatoli-starostin-relocation`** (the key's default entity).
- **How it is set:** `run.env` sets `WANDB_BASE_URL=https://api.wandb.ai`; `launch.sh` unsets `WANDB_MODE`. The project comes from the trainer's W&B shim, the group from `train.py`.
- **Do NOT source `~/.wandb_env`.** It points at the self-hosted W&B server on nucstar, which is **down** (Docker inactive, and the box has known-bad RAM). That dead endpoint already cost a restart: the gpustar 16K run launched with W&B off, nobody could watch it, and it had to be stopped and relaunched. With that file sourced, the tracker logs offline or not at all, and this run would be invisible.
- **The credential is already in your `~/.netrc`** (`machine api.wandb.ai`), added and verified to authenticate against wandb.ai from this machine on 2026-09-16. You do not need to obtain one. `preflight.py` re-checks that it authenticates, without printing it.
- **If it fails to authenticate:**
  - do not launch;
  - do not paste a key into `run.env`, the folder, a log or a commit;
  - report "W&B cloud auth failed on nebius" to Anatoli and wait for a fixed credential in `~/.netrc`.
- **Verification is mandatory:** after the first eval (step 500), `verify_wandb.py` must print **`VERIFY_WANDB PASS`**. It checks that the run exists on wandb.ai AND that train and val metrics are actually arriving and match `metrics.csv`. **Do not walk away from the run without that PASS.**
  - On FAIL: `kill $(cat train.pid)`, fix, and relaunch from a fresh copy of the folder.
  - Never delete the W&B run: the server reserves deleted ids.

## The kernel validation GATE (H100) — mandatory, before any launch

**Why a gate.**
- The p2_int8 CUDA kernel serves the quantised path wherever it is enabled. It is enabled only on compute capabilities in `pow2_int8.VALIDATED_ARCHES`.
- **The kernel has only been validated on sm_120 (RTX 5090). Nothing has been validated on an H100.** `(9, 0)` is on the build allowlist (commit 4ad02234) only so the extension can build and be tested here.
- **If the extension will not build or load on the H100, or any test fails, the torch fallback is the expected route, not a bug.**
- A kernel is trusted on an architecture only after `tests/test_pow2_int8.py` is fully green **on that hardware**. That file covers:
  - D = 48/40/52/8/128 × block sizes 32/64/128 × both load styles × garbage stride padding;
  - the fused kernel bit-exact against both the cells reference and `int8_blend_read`;
  - drift tests at the q rounding thresholds ±1 ulp;
  - train == forward_int, artefact fused == torch read, and the backward and fallback tests.
- **This is a gate, not a formality.**

```bash
set -a; . ./run.env; set +a; unset SPIKY_P2_CUDA_DISABLE
"$PYTHON" validate_kernel.py            # builds the extension for this GPU, runs the matrix, writes kernel_gate.json
```

**Branch A — `KERNEL GATE: PASS`** (0 failed, 0 errors, 0 skipped):
- Leave `SPIKY_P2_CUDA_DISABLE` **unset** in `run.env`. The run trains on the CUDA kernel.
- Preflight must then print `IMPLEMENTATION SERVING THE QUANTISED PATH: CUDA KERNEL`.

**Branch B — `KERNEL GATE: FAIL`** (any failure, error or skip, or the build fails):
- **Do NOT train on the kernel.** Set `SPIKY_P2_CUDA_DISABLE=1` in `run.env`.
- The run then trains on the torch definition in `pow2_read`. Its integers are bit-identical by definition, but it is slower (see wall clock).
- Preflight must then print `IMPLEMENTATION SERVING THE QUANTISED PATH: TORCH FALLBACK`.
- **Report the gate failures to Anatoli** (the `[matrix]` lines and `kernel_gate.json`) along with the run.

Preflight refuses kernel-serving without a PASS gate, and refuses an unexplained fallback after a PASS gate. The implementation line is written to `preflight_implementation.txt`, and `train_launch.py` writes a `[p2_int8] IMPLEMENTATION: …` line into `train.log`, so the run record always says which implementation produced the result.

**L2 residency** (the design premise) holds on the H100:
- H100 L2 is 50 MB; the per-layer int8 table is 4 × 128 × 256 × 48 bytes = **6.29 MB**.
- What is **untuned** for the H100 are the kernel's block-size constants (tuned on the 5090). Throughput there is not optimised; correctness is what the gate checks.

## Prerequisites (on your machine; none were checked from gpustar)

1. **Code:** check out the pinned commit (see the handoff message; it includes this folder), and nanochat at `da32e1d6…`.
2. **Native extension:** rebuild `lutorch_cuda` from the pinned `native/lutorch` if your install predates it, with your GPU's arch (e.g. `TORCH_CUDA_ARCH_LIST=9.0`):
   ```bash
   uv pip install --no-build-isolation ./native/lutorch
   ```
   The p2_int8 extension itself is JIT-built by `validate_kernel.py`; it needs `nvcc` and `ninja` on PATH.
3. **Data:**
   ```bash
   cd $NANOCHAT_ROOT && NANOCHAT_BASE_DIR=<dir> python -m nanochat.dataset -n 4
   ```
   This fetches train shards 0–3 plus the pinned val shard 06542. The data dir must hold exactly those 5 files.
4. **Tokenizer:**
   ```bash
   mkdir -p <dir>/tokenizer && cp nanochat_tokenizer/* <dir>/tokenizer/
   ```
5. **W&B:** already in `~/.netrc` (see the W&B section).

## Exact commands

```bash
cd $SPIKY_ROOT/experiments/ffn_replacement/lut_ablation/exp_n_abl_47_B48k_light_lmfrozeng_n2_tau0p5_tv10_tph128_seed1_p2int8
cp run.env.template run.env && $EDITOR run.env        # fill every FILL; leave SPIKY_P2_CUDA_DISABLE commented for now

set -a; . ./run.env; set +a
"$PYTHON" validate_kernel.py                           # THE GATE -> Branch A (PASS) or Branch B (FAIL: set SPIKY_P2_CUDA_DISABLE=1)
set -a; . ./run.env; set +a
"$PYTHON" preflight.py                                 # must end "PREFLIGHT OK"; note the IMPLEMENTATION line

bash launch.sh                                         # re-runs preflight, starts train_launch.py detached, writes train.pid
grep -E '\[wandb\]|\[p2_int8\]' train.log              # ~2 min in: "online: run exp_n_abl_47_..." and the implementation line
"$PYTHON" verify_wandb.py                              # after step 500: must end "VERIFY_WANDB PASS" -- do not walk away without it
```

- **Progress outside W&B:**
  - `train.log`: a step line every 100 steps, `[VAL]` every 500;
  - `metrics.csv`: a row per eval;
  - `checkpoint_step{8000,16000,…}.pt`: every 8,000 steps.
- **Single GPU only:** the trainer has no DDP.

## Expected wall clock

Not measured for this arm; there is no timing from an H100 or for the quantised path at 48K.

- **Orientation only:** the default (unquantised) arm took ~1.1–1.2 h per 16K steps on both an RTX 5090 and nebius's H100, i.e. ~3.5 h for 48K.
- **Kernel (Branch A):** expect somewhat longer than the default arm. The kernel's block sizes were tuned on the 5090, not the H100.
- **Torch fallback (Branch B):** expect slower still, by an unmeasured factor.

Record `training_time_hours` from `summary.json` in the report.

## What to report back, and to whom

Report to **Anatoli**, in the thread or channel this task came from. Put numbers in text:

1. The commits used (`git rev-parse HEAD` for spiky and nanochat), and the GPU model.
2. **The gate:** the `KERNEL GATE` verdict and `[matrix]` counts. On FAIL, the failing test names.
3. **The implementation that served the run:** the `preflight_implementation.txt` line and the `[p2_int8]` line from `train.log`.
4. **The W&B run name, and the `VERIFY_WANDB PASS` output.**
5. `summary.json`: **`final_val_bpb` and `best_val_bpb` to full precision**, `training_time_hours`.
6. The val curve: every `metrics.csv` row's `step,val_bpb`, plus the step of the minimum (decision 2).
7. Any divergence, NaN, restart or W&B gap, stated plainly.

Do not commit checkpoints. Commit run artefacts only when Anatoli says so.

## What a good result looks like

- **Completion:** 48,000 steps completed, loss finite throughout, W&B live and verified, implementation recorded, and `summary.json` written.
- **Correctness:** Branch A with the gate green is the stronger outcome: the kernel is validated on H100 AND trained a full run. Branch B is still a valid result; it just says the H100 kernel needs work.
- **There is no pre-recorded bar for this arm at any length.** For orientation only (not a target):
  - The 16K quantised twin on gpustar will report its number.
  - A post-hoc int8 export of the 16K *float* model scores 1.159149093631757; an STE-trained run is expected to do at least as well.
  - The 16K default-path reference for this row and seed is 1.1547389676188025.
- **Curve shape:** 48K should improve on the 16K figures overall. A val curve that bottoms out before 48K and turns up is a real finding about data repetition (decision 2), not a failure. Report it.

## Files

| file | purpose |
|---|---|
| `README.md` | this document |
| `config.json` | the run config (abl_05 + `n_steps` 48000 + `lut_quant_mode` p2_int8; `_arch_note` records provenance) |
| `train.py` | the trainer, byte-identical to abl_05's and the 16K runs' (forked with `tools/fork_trainer.py`) |
| `train_launch.py` | runs `train.py` unchanged and logs which p2_int8 implementation serves (`[p2_int8] IMPLEMENTATION: …`) |
| `validate_kernel.py` | **the gate**: builds the extension for this GPU, runs the kernel test matrix, writes `kernel_gate.json`, prints PASS / FAIL |
| `preflight.py` | checks pins, data / tokenizer sha256, one GPU, config, smoke build, quantised path active, **which implementation serves**, gate consistency, W&B cloud auth |
| `launch.sh` | requires `kernel_gate.json`, runs preflight, then a detached launch of `train_launch.py` (`train.log`, `train.pid`) |
| `verify_wandb.py` | confirms the W&B run is online AND metrics arrive and match `metrics.csv` |
| `run.env.template` | every machine-specific value, as FILL placeholders (paths, GPU, W&B endpoint, the fallback switch) |
| `data_manifest.json` | exact shard list with sha256, sizes and token estimates; nanochat pin |
| `nanochat_tokenizer/` | `tokenizer.pkl` (412 KB), `token_bytes.pt` (133 KB): the exact tokenizer |
