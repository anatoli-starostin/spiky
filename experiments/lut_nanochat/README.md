# lut_nanochat: dense d24 baseline on 1×H100

**Scope (deliberately narrow; do not expand it).** This branch holds exactly one thing: a single dense d24 nanochat baseline run on one H100 80GB.
- It reproduces the upstream speedrun recipe at commit `92d63d4`: d24, data:param ratio 8, fp8, ClimbMix. The global batch is 2^20 tokens, run for 5,568 steps.
- The target is standalone `base_eval` CORE ≈ 0.2626 (above the GPT-2 bar of 0.256525).
- There is **no LUT code, no fp8-off control, no multi-seed harness, and no stubs for future arms**. They were cut on purpose. Add them in a separate, agreed step, not here.

| File | What it is |
|---|---|
| `RUNBOOK.md` | Step-by-step instructions for the operator (a Claude Code agent on a Nebius VM). Follow it verbatim. |
| `PIN.md`, `pins.json` | The pinned environment: nanochat commit, torch/CUDA, FA3 revision, data, hardware. |
| `nanochat/` | karpathy/nanochat vendored at `92d63d4`, plus a small marked patch (see PIN.md). |
| `nanochat/runs/baseline_d24_1xh100.sh` | The launcher, with resume-from-latest. |
| `nanochat/runs/probe_memory_d24.sh` | A ~5-minute probe that decides between device batch 16 and 8. |
| `nanochat/runs/stage_data.sh` | Stages the data, tokenizer and eval bundle, and records their hashes. |
| `results/<run>/` | Written by the run and committed back by the operator. |
