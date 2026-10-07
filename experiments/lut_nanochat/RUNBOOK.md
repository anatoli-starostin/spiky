# RUNBOOK: dense d24 nanochat baseline on 1×H100 (`d24-dense-1xh100-s0`)

**Who this is for.** A Claude Code agent on a fresh Nebius VM with no prior context. Follow it **verbatim and in order**.

**Shell conventions.**
- Every command runs from `experiments/lut_nanochat/nanochat` (call it `NC`) unless stated otherwise.
- Do not change the recipe, flags or code. If something here does not work, stop and report; do not improvise a workaround that changes the maths.
- Scope is in `README.md`: one run, nothing else.

**Checklist (in order; no step may be skipped):**
§4 env + smoke test → §5 data staging → §6 memory probe → **§7 REQUIRED smoke run (`SMOKE RUN: PASS`)** → §8 launch → §9 monitor → §11 eval and report.

## 1. Goal and success criterion
- **Goal:** train upstream nanochat's d24 speedrun recipe (commit `92d63d4`) on one H100 80GB, then measure it with the standalone `base_eval`.
- **Success:** standalone `base_eval` **CORE above 0.256525** (the GPT-2 bar), ideally within ±0.016 (2× run noise) of the 8×H100 record, **0.2626**.
- **Also record:**
  - val bpb (record: 0.718);
  - total training time;
  - peak memory;
  - the number of resumes.
- **The in-training CORE is NOT the result.** It uses only 500 examples per task. In #819 it read 0.2803 against 0.2702 for the standalone eval.

## 2. Hardware requirement
- **1× NVIDIA H100 80GB, Hopper (`sm_90`).** SXM is preferred; PCIe works but is slower.
- **Not B200, not an RTX 5090, not any Blackwell card.** The FA3 kernel does not exist for them, and the run would silently fall back to slower, non-comparable attention.
- **Disk:** ≥ 150 GB on the volume that holds `$NANOCHAT_BASE_DIR`.
- **CPU:** ≥ 16 vCPU recommended, because the dataloader tokenises on the fly. This is UNCERTAIN; upstream states no requirement.
- **Driver:** ≥ R570 (CUDA 12.8). UNCERTAIN, as in PIN.md.

## 3. Provisioning
1. Create a VM with 1× H100 80GB SXM, Ubuntu 22.04 or 24.04 with the NVIDIA driver preinstalled, and ≥ 150 GB of disk. Prefer **on-demand**. On spot, the run survives pre-emption only through §10 (resume); expect to lose up to `--save-every` = 250 steps (~35 min) each time.
2. Check the GPU and driver: `nvidia-smi`. Expect "H100 80GB HBM3" and driver ≥ 570.
3. Check free disk: `df -h ~`. If the big volume is mounted elsewhere, use `export NANOCHAT_BASE_DIR=/path/on/big/volume/nanochat` **in every shell** (add it to `~/.bashrc`).
4. Get the code:
   ```bash
   git clone https://github.com/anatoli-starostin/spiky.git && cd spiky
   git checkout research/lut_nanochat
   cd experiments/lut_nanochat/nanochat        # = NC
   ```
5. Run everything long inside `tmux` (`tmux new -s nc`), so an SSH drop does not kill it.

## 4. Environment setup and verification
```bash
mkdir -p ../results/d24-dense-1xh100-s0
if ! command -v uv; then curl -LsSf https://astral.sh/uv/install.sh | sh; source ~/.local/bin/env; fi
uv venv && uv sync --extra gpu                 # installs torch 2.9.1+cu128 etc. from the vendored uv.lock
export WANDB_API_KEY=...                       # the operator provides it; NEVER write it into any file in the repo
source runs/lut_env.sh && source .venv/bin/activate
python runs/env_smoke_test.py
```
**Proceed only if the last line is `SMOKE TEST: OK`.** It checks:
- torch is `2.9.1+cu128`;
- the GPU is an `sm_90` card with 80 GB;
- FA3 loads from the pinned revision and runs a sliding-window forward;
- an fp8 `_scaled_mm` runs.

On a failure, see §12.

## 5. Data staging with hash verification
```bash
bash runs/stage_data.sh 2>&1 | tee ../results/d24-dense-1xh100-s0/stage_data.log
```
- **What it does:**
  - downloads 170 ClimbMix train shards plus the val shard `shard_06542` (~15 GB; UNCERTAIN ~5–20 min);
  - trains the tokenizer (a few minutes);
  - downloads and unzips the CORE eval bundle;
  - writes `../results/d24-dense-1xh100-s0/manifest.json` with the sha256 of every input.
- **Proceed only if it prints `reference shard hashes: PASS`.**
  - If it stops with `STOP: ClimbMix shards differ`, the dataset changed upstream. **Do not run**; report the mismatching hashes.
  - It can be re-run; it skips completed downloads.

## 6. Memory probe and the batch-16 vs batch-8 decision
```bash
bash runs/probe_memory_d24.sh                  # ~5 min (compile + 4 real steps at device batch 16)
```
Read the `VERDICT:` line:
- `batch 16 OK` → launch with the default (`DBS=16`, 32 grad-accum steps). This gives exact micro-batch parity with the 8×H100 speedrun.
- `fall back to batch 8` (OOM, or peak reserved > 75 GiB) → run `DBS=8 bash runs/probe_memory_d24.sh` once to confirm, then launch with `DBS=8` (64 grad-accum steps). **The maths is identical**: same global batch, LR, weight decay and steps. Only the fp8 scale granularity per micro-batch differs, which is negligible.
- `probe FAILED for a non-memory reason` → stop and report the log `../results/d24-dense-1xh100-s0/probe_dbs16.log`.

Also note the `step_time` it prints; §9 uses it to sanity-check the ETA.

## 7. REQUIRED smoke run (before the long run)
**What it is.** A short real training run on the exact pinned configuration: d24, fp8, FA3 at the pinned revision, the `DBS` chosen in §6, 32 accumulation steps at batch 16 (64 at batch 8). It goes through the real launcher, so the long run's code path is exercised end to end. Expect ~20–30 min (UNCERTAIN: two `torch.compile` warm-ups plus about 40 steps of ~8.5 s).

```bash
bash runs/smoke_run_d24.sh 2>&1 | tee ../results/d24-dense-1xh100-s0-smoke.out        # batch 16
#   or, if §6 chose the fallback:  DBS=8 bash runs/smoke_run_d24.sh 2>&1 | tee ../results/d24-dense-1xh100-s0-smoke.out
```

**What it does.** `runs/smoke_run_d24.sh` reuses `runs/baseline_d24_1xh100.sh` unchanged through its default-off hooks.
- **Phase A:** a 40-step horizon with `--save-every 20`. The wrapper **kills** it (SIGTERM to the whole process group) as soon as step 21 starts, i.e. right after the step-20 checkpoint is written.
- **Phase B:** the **same launcher command** again. It must auto-resume from step 20, finish at step 39, then run a reduced standalone `base_eval` (16 examples per task, 2M tokens per split) and `report_results.py`.
- It then runs `runs/smoke_check.py` and deletes the smoke checkpoints.
- **Everything is separate from the real run:**
  - run name `d24-dense-1xh100-s0-smoke`;
  - model tag `d24_1xh100_smoke`;
  - results in `../results/d24-dense-1xh100-s0-smoke/`;
  - its own wandb run id;
  - the extra wandb tag `smoke`.

**PASS criteria.** `smoke_check.py` checks these mechanically and prints one `PASS`/`FAIL` line each, then `SMOKE RUN: PASS` or `SMOKE RUN: FAIL (n): …`. "(choice)" marks our thresholds, not upstream's.
1. No OOM at the chosen batch size, and peak *reserved* memory is reported and ≤ 75 GiB (choice; same rule as §6).
2. The global batch is 1,048,576 with 32 accumulation steps (64 at `DBS=8`).
3. **fp8 is active:** `FP8 training enabled (tensorwise scaling) - converted 145/158` appears in both phases.
4. **FA3 is active:** `Using Flash Attention 3` appears in both phases, and there is no "not available" line. The FA3 revision in the run's pin config equals `pins.json`.
5. **Loss** is finite at every step and has dropped by > 1 nat between step 0 and step 39 (choice; initialisation is ≈ ln 32768 ≈ 10.4).
6. **Checkpoint and resume:**
   - `Resuming optimization from step 20` appears in `train.log`;
   - `resuming from step 20` appears in `resumes.log`;
   - training reached step 39;
   - the reduced `base_eval` printed a CORE value. The value itself is meaningless after 40 steps.
7. **Step time.** The median steady-state step time is extrapolated to 5,568 steps, and the line `INFO median step time … -> pretraining … h … ~$…` is printed. Projected pretraining must be ≤ 17 h (choice; the estimate is 13–15 h, i.e. 15–18 h end to end and ~$48 at $3/GPU-h). Set `GPU_HOURLY_USD` for your actual rate.
8. **wandb:**
   - the run exists in project `spiky-nanochat`, group `nanochat_baseline`;
   - it is named `d24-dense-1xh100-s0-smoke` and carries the `smoke` tag;
   - its config holds the pin set: nanochat commit, input hashes and the derived run shape;
   - phases A and B landed in **one** run, up to the last step.

**STOP rule.** If the last line is not `SMOKE RUN: PASS`, **do not start the long run.** Report the full `smoke_check.txt` and the relevant log lines, then wait. Do not tweak flags to make it pass.

**Marking and cleanup.**
- The smoke checkpoints are deleted automatically.
- The smoke wandb run is **kept, clearly marked** by its `-smoke` name suffix and `smoke` tag, as evidence. It must never be cited as the baseline. Anatoly may delete it in the wandb UI once the report is accepted.
- Commit `../results/d24-dense-1xh100-s0-smoke/` (`git add -f`, small text files only) together with the final results (§11).
- Re-running the smoke run requires removing its leftovers first; the wrapper refuses otherwise and prints the `rm` command.

## 8. Launch
**Only after §7 printed `SMOKE RUN: PASS`.**
```bash
# in tmux, from NC, with WANDB_API_KEY exported (and NANOCHAT_BASE_DIR if not the default)
bash runs/baseline_d24_1xh100.sh 2>&1 | tee -a ../results/d24-dense-1xh100-s0/launcher.out          # batch 16
#   or:  DBS=8 bash runs/baseline_d24_1xh100.sh 2>&1 | tee -a ../results/d24-dense-1xh100-s0/launcher.out
```
- The script does, in order: base_train (5,568 steps, checkpoint every 250, newest 2 kept), then the standalone `base_eval`, then `report_results.py`.
- **wandb:** project `spiky-nanochat` (override with `WANDB_PROJECT`), group `nanochat_baseline` (override with `WANDB_GROUP`), run name `d24-dense-1xh100-s0`. The config carries the full pin set and the input hashes.

## 9. What to monitor
- **Log:** `../results/d24-dense-1xh100-s0/train.log`.
  - Read the header once: `Total batch size 1,048,576 => gradient accumulation steps: 32` (64 at DBS=8) and `Calculated number of iterations …: 5,568`. **If either differs, stop: the run is not the baseline.**
  - `✓ Using Flash Attention 3` must appear; a `WARNING: Flash Attention 3 not available` means stop (§12).
  - `✓ FP8 training enabled … converted 145/158 linear layers`.
- **Expected speed** (UNCERTAIN; scaled from 8×H100, no upstream 1×H100 timing exists):
  - `dt` ≈ 8–9 s per step and `tok/sec` ≈ 115–130k on H100 SXM; the `eta:` field should read ~13–15 h;
  - `bf16_mfu` should be roughly in the 50s.
  - Much slower (> 12 s per step) suggests a stall or FA3 fallback (§12).
- **Loss:** smoothly decreasing.
- **val bpb:** logged every 250 steps; it should end near **0.72**.
- **In-training CORE:** at steps 2000, 4000 and 5568. These are indicative only.
- **Total wall clock:** ≈ 13–15 h of training plus evals, ≈ 15–18 h end to end.
- **Disk:** `du -sh $NANOCHAT_BASE_DIR/base_checkpoints/d24_1xh100` should stay ≈ 22–33 GB.

## 10. On interruption (pre-emption, crash, OOM mid-run)
- **Re-run exactly the same launch command as §8, with the same `DBS`.** The script finds the newest *complete* checkpoint, logs the resume to `resumes.log`, and continues the same wandb run.
- **Every resume must be reported.** The dataloader resume is *approximate*, so a resumed run is not bit-identical to an uninterrupted one.
- If OOM happens mid-run at DBS=16, re-launch with `DBS=8`. Resuming across a DBS change is UNCERTAIN (untested). If it errors, report it and stop.
- If training already finished (a step-5568 checkpoint exists), the same command skips straight to `base_eval`.

## 11. Final evaluation and reporting back
- **The launcher runs the eval itself:** `torchrun --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=$DBS --model-tag=d24_1xh100` (full CORE, all examples). Then `report_results.py` writes `summary.json` and puts the standalone CORE into the wandb summary.
- **To re-run only the eval/report:** `bash runs/baseline_d24_1xh100.sh` again (it detects a finished run).
- **Report these numbers** (all in `summary.json`):
  - `core_standalone_base_eval` (**the result**);
  - `val_bpb_base_eval`;
  - `val_bpb_final_in_training`;
  - `min_val_bpb_in_training`;
  - `core_in_training_last`;
  - `total_training_time_min`;
  - `peak_memory_reserved_mib`;
  - `grad_accum_steps`, `num_iterations`;
  - `resumes`;
  - `verdict`.
  Also give the wandb run URL and the GPU model from `nvidia-smi`.
- **Commit back** to the branch (do NOT open a PR; do NOT commit checkpoints or the API key; `-f` because the repo ignores `*.log`):
  ```bash
  cd ..   # experiments/lut_nanochat
  git add -f results/d24-dense-1xh100-s0/{summary.json,manifest.json,pin_config.json,train.log,base_eval.log,stage_data.log,launcher.out,probe_dbs*.log,wandb_run_id,base_model_*.csv}
  git add -f results/d24-dense-1xh100-s0/resumes.log 2>/dev/null || true
  git add -f results/d24-dense-1xh100-s0-smoke/ results/d24-dense-1xh100-s0-smoke.out   # smoke evidence (§7)
  git commit -m "lut_nanochat: d24-dense-1xh100-s0 results (CORE <value>)" && git push origin research/lut_nanochat
  ```
- Keep the final checkpoint (`$NANOCHAT_BASE_DIR/base_checkpoints/d24_1xh100/*005568*`) on the VM until told otherwise, and report its path and size.

## 12. Troubleshooting
| Symptom | Action |
|---|---|
| **OOM** in the probe | Use `DBS=8` (§6). |
| **`SMOKE RUN: FAIL`** | STOP (§7). Report `smoke_check.txt`. Do not launch the long run. |
| **OOM** mid-run | Re-launch with `DBS=8` (§10) and report it. Do not reduce the total batch, and do not touch any other flag. |
| **FA3 unavailable** (`SMOKE TEST` FAIL on FA3, or `WARNING: Flash Attention 3 not available` in the log) | 1. Check the GPU really is `sm_90` (`nvidia-smi`). 2. Check the VM reaches huggingface.co: the kernel is downloaded at first import into `~/.cache/huggingface`. 3. Try `NANOCHAT_FA3_REVISION= python runs/env_smoke_test.py` (empty = upstream behaviour, Hub `main`). If that passes, the pinned revision or the `revision=` argument is the problem: **report it and wait.** Do not launch on an unpinned kernel without approval. **Never train on the SDPA fallback**: it has no sliding window, and the result would not be comparable. |
| **Dataloader stall** (`dt` spikes, `tok/sec` drops, GPU utilisation falls in `nvidia-smi`) | Check CPU load (`top`): tokenisation runs on the CPU. Check that disk is not full (`df -h`) and that all shards are present (re-run `stage_data.sh`; it is idempotent). A one-off slow step around evals or checkpoints (every 250 steps) is normal. |
| **wandb auth** (`WARNING: wandb.init failed …`) | Training continues without wandb (by design). Fix `WANDB_API_KEY` for the next resume. If the VM has no outbound access to wandb, set `WANDB_MODE=offline` and later run `wandb sync` on the run directory. The authoritative numbers are in `summary.json` either way. |
| **`STOP: ClimbMix shards differ`** | Do not run. Report the hashes. |
| **Header shows other than 32 (or 64) accumulation steps or 5,568 iterations** | Stop. Something changed the recipe. Report it. |
| **HF download errors or rate limits during staging** | Re-run `stage_data.sh` (the downloader retries with backoff). Optionally `export HF_TOKEN=…`. |
