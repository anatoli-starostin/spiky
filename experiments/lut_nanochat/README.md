# lut_nanochat: LUT models on the nanochat stack

## Scope
This branch is the long-lived sandbox for the whole line of work on LUT models in nanochat. The work has three stages, listed below.
- **Stage 1, the dense baseline / target model, is the CURRENT stage.** It is built and ready to run.
- **Stages 2 and 3 are not yet implemented here.**

**Keep the baseline arm clean:**
- The baseline run must not be contaminated with LUT or control code.
- The vendored nanochat at `92d63d4` stays diffable against upstream. Our changes are marked `[lut_nanochat]` and listed in `PIN.md`.
- The baseline run config must not drift once it has been launched.

Later stages go in alongside it, without touching the baseline arm.

## Roadmap
1. **Dense baseline / target model (current).**
   - The upstream speedrun recipe at `92d63d4` on 1×H100: d24, data:param ratio 8, fp8, ClimbMix, 2^20-token global batch, 5,568 steps.
   - Target: standalone `base_eval` CORE ≈ 0.2626, above the GPT-2 bar of 0.256525.
   - It serves both as the paper baseline and as the teacher for stage 2.
2. **Distillation (not yet implemented).** Train LUT models on the baseline model's output logits, with an output-distribution KL objective only.
   - Matching per-block hidden states or attention maps was considered and **rejected**.
   - The reason: LUT models reach the same bpb by reallocating compute, using cheap LUT FFNs and more sophisticated attention patterns. Any objective that pins internal representations to the teacher fights that mechanism.
3. **LUT models from scratch (not yet implemented).**
   - The LUT arm is **not** shape-matched to d24 with its FFNs swapped. The honest comparison is at equal compute or equal cost, with the model shape left free. Forcing the LUT model into the dense shape handicaps it if it wants more attention capacity.
   - The restructuring itself is a result worth logging from the first LUT run: attention entropy, window usage, head specialisation and induction behaviour.

## Files (stage 1)
| File | What it is |
|---|---|
| `RUNBOOK.md` | Step-by-step instructions for the operator (a Claude Code agent on a Nebius VM). Follow it verbatim. |
| `PIN.md`, `pins.json` | The pinned environment: nanochat commit, torch/CUDA, FA3 revision, data, hardware, wandb. |
| `nanochat/` | karpathy/nanochat vendored at `92d63d4`, plus a small marked patch (see PIN.md). |
| `nanochat/runs/baseline_d24_1xh100.sh` | The launcher, with resume-from-latest. |
| `nanochat/runs/probe_memory_d24.sh` | A ~5-minute probe that decides between device batch 16 and 8. |
| `nanochat/runs/smoke_run_d24.sh`, `smoke_check.py` | The REQUIRED short smoke run before the long run, and its PASS check (RUNBOOK §7). |
| `nanochat/runs/stage_data.sh` | Stages the data, tokenizer and eval bundle, and records their hashes. |
| `results/<run>/` | Written by the run and committed back by the operator. |
| `docs/` | The nanochat state report and the baseline decision brief (md + pdf), with the d24 architecture confirmed from code. |
