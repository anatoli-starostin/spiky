# lut_nanochat: LUT models on the nanochat stack

## Scope
This branch is the long-lived sandbox for the whole line of work on LUT models in nanochat. The work has three stages, listed below; stage 2 has two parts, 2a and 2b.
- **Stage 1, the dense baseline / target model, is the CURRENT stage.** It is built and ready to run.
- **Stages 2a, 2b and 3 are not yet implemented here.**

**Keep the baseline arm clean:**
- The baseline run must not be contaminated with LUT or control code.
- The vendored nanochat at `92d63d4` stays diffable against upstream. Our changes are marked `[lut_nanochat]` and listed in `PIN.md`.
- The baseline run config must not drift once it has been launched.

Later stages go in alongside it, without touching the baseline arm.

## Roadmap
### Stage 1: dense baseline / target model (current)
- The upstream speedrun recipe at `92d63d4` on 1×H100: d24, data:param ratio 8, fp8, ClimbMix, 2^20-token global batch, 5,568 steps.
- Target: standalone `base_eval` CORE ≈ 0.2626, above the GPT-2 bar of 0.256525.
- It serves both as the paper baseline and as the teacher for stage 2b.

### Stage 2a: compressing the value embeddings (not yet implemented)
Numbers are marked *(code)* when they were confirmed from code or by instantiating the module, and *(estimate)* otherwise.

**Motivation.**
- The value embeddings are `nn.Embedding(32768, 1536)` on each of the 12 odd layers: **603,979,776 params, 43.6% of d24's 1,384,122,122**, the largest single component *(code; `docs/`, long report §3.9)*.
- They cost ~0% of FLOPs, because they are a gather.
- They are bf16 ([`gpt.py` L262–268](nanochat/nanochat/gpt.py#L262-L268)) and outside the fp8 set, since `nn.Embedding` is not `nn.Linear` *(code)*.
- Resident training state is ≈ 3.6 GB: 1.21 GB of weights plus 2.42 GB of bf16 Adam m and v *(estimate from the counts)*.
- Each value-embedding layer has a 144-param gate, `3·sigmoid(...)` in (0, 3), that sets the mix-in per head ([`gpt.py` L81–97](nanochat/nanochat/gpt.py#L81-L97)) *(code)*.
- Launch-era d20 had no value embeddings at all: its 560,988,160 params are reproduced exactly without them *(code)*.

**Why this is a better first LUT experiment than swapping the FFNs.**
- It targets the largest parameter block.
- It replaces a lookup with a cheaper lookup: the same kind of object, with no FFN-style expressivity confound.
- The existing gate lets the model dial down a poor approximation, so we get a graded signal rather than a collapse.
- The diff touches only the 12 value-embedding modules. Attention, the FFNs and the fp8 path stay as they are, so **fp8 stays on** and the comparison to the baseline is at matched precision. The fp8-off control problem does not arise. (One caveat, from the code: new `nn.Linear`s are picked up by nanochat's fp8 filter if both dims are ≥ 128, so a compress or decompress at r ≥ 128 must be excluded explicitly. At r ≤ 64 the filter skips them.)

**What `lutorch_ex` actually provides** (read from code). There is no `ProjectionLUT`. The wrapper is `ProjectionMHL` ([`projection.py` L59–189](../../src/spiky/lutorch_ex/projection.py#L59-L189)). It applies a dense `compress` (`nn.Linear(input_dim, h_in·d_in)`, bias on by default, init N(0, 0.02)), then a cartridge on `[B, h_in, d_in]`, then a dense `decompress` (`nn.Linear(h_out·d_out, output_dim)`, **zero-initialised**, so a fresh module outputs 0).
- **Addressing is not nearest-codebook.** Each of `G·tph` tables reads its head's `d_in` slice. It forms `nap` sign bits from **fixed, frozen anchor pairs**, `[z[a_j] − z[b_j] > eps]`, and packs them MSB-first into a cell index in `[0, 2^nap)` ([`cartridges/manifesto_base.py` L304–332](../../src/spiky/lutorch_ex/cartridges/manifesto_base.py#L304-L332)).
  - There are no distances, no argmin over entries, no codebook keys, no EMA, and no commitment loss.
  - The output is the sum of the addressed rows over a head's `tph` tables ([`cartridges/manifesto_base.py` L334–345](../../src/spiky/lutorch_ex/cartridges/manifesto_base.py#L334-L345)).
  - The only learned parts are the tables, the two projections, and (Gen-2/3) 2–3 scalars.
- **Gradient through the discrete step** (`ManifestoHardLUT` / `FusedManifestoHardLUT`): the value is the hard read. The input gradient is a straight-through surrogate through a two-cell blend with the neighbour that flips the least-confident bit, weighted by `U = 0.5/(1+|u*|)`. Only the addressed cell receives a table gradient, and the index is stop-gradient ([`cartridges/manifesto_hard.py` L49–57](../../src/spiky/lutorch_ex/cartridges/manifesto_hard.py#L49-L57)).
  - The Gen-3 `ConfidenceLUT` instead returns a score-weighted read, and differentiates that value directly ([`README.md` L120–166](../../src/spiky/lutorch_ex/README.md#L120-L166)).
- **Parameter count** *(code, verified by instantiation)*: `P = r·(2·1536 + 1 + tph·2^nap) + 1536` for 1536 → r → 1536 with `r = h·d`, plus 0 scalars for the Manifesto cartridges and 3 for Gen-3.
  - The tables (`h·tph·2^nap·d`) dominate.
  - With the library's reference geometry h = 8, tph = 64, nap = 8 ([`README.md` L38](../../src/spiky/lutorch_ex/README.md#L38); there `r` = 384 = the model width, i.e. no bottleneck):

| r | geometry | per instance | ×12 independent | vs 603,979,776 |
|---|---|---|---|---|
| 32 | h8 tph64 nap8 is **refused** (`nap` > C(4,2) anchor pairs, [`anchors.py` L124–151](../../src/spiky/lutorch_ex/anchors.py#L124-L151)); h4 tph64 nap8 d8 | 624,160 | 7,489,920 | 81× |
| 32 | h8 tph64 nap6 d4 | 230,944 | 2,771,328 | 218× |
| 64 | h8 tph64 nap8 d8 | 1,246,784 | 14,961,408 | **40×** |
| 128 | h8 tph64 nap8 d16 | 2,492,032 | 29,904,384 | 20× |
| 256 | h8 tph64 nap8 d32 | 4,982,528 | 59,790,336 | 10× |

- **FLOPs per input vector** *(estimate from the code path, r = 64)*: compress + decompress 2·(2·1536·64) ≈ 393k. Addressing is a subtract, compare and pack over `G·tph·nap` = 4,096 bits, ≈ 12k ops. The hard read is `G·tph·d` = 4,096 adds. Total ≈ 0.41 MFLOP forward, 96% of it in the projections.
- **Sharing and caching.** There is no dedicated API.
  - Instances are ordinary modules and can be shared by reference.
  - `compress=False` lets one external down-projection feed several cartridges.
  - A fan-out spec (`h_in = 1`) feeds one input to many heads ([`lut_spec.py` L1–30](../../src/spiky/lutorch_ex/lut_spec.py#L1-L30)).
  - Addresses are recomputed every forward; nothing caches them.
- **dtype.** The plain and Gen-3 cartridges are fp32/fp64 only and **raise on bf16** ([`cartridges/manifesto_base.py` L259–270](../../src/spiky/lutorch_ex/cartridges/manifesto_base.py#L259-L270)). The `Fused…` twins accept bf16, with fp32 addressing and fp32-accumulated reads.
- **torch.compile.** Cartridges lazily compile their own forward on CUDA (eval always; train only for Gen-3). The native lprojection kernels are not registered as `torch.library` ops, so behaviour inside nanochat's whole-model `torch.compile` is **untested** (UNCERTAIN). `LUTORCH_EX_NO_CUDA_EXT=1` forces the pure-torch path.

**The approach ladder**, cheapest to most expressive:
1. **One shared table plus the per-layer gate:** 50,331,648 params instead of 603,979,776 (12×) *(code)*.
   - A trivial diff.
   - It is also the ablation that tells us whether the 604M ever earned its keep.
2. **Low-rank:** a shared 32,768 × r code table plus a per-layer r → 1536 readout. At r = 256 that is 13,107,200 params (46× smaller) *(arithmetic)*.
   - It is a matmul, not a lookup.
   - All 12 layers become linear re-readings of one code.
3. **Product-quantised codebooks keyed by token id:** split the 1536 dims into g groups, with one codebook of K entries per layer per group.
   - At K = 256: 393,216 params per layer, 4,718,592 for all 12 (128× smaller) *(arithmetic)*. Pure gathers.
   - The per-token codes are not trainable parameters, but they are stored state (32,768 · g · 12 indices) and need an assignment procedure. Learning them is the hard part.
4. **`ProjectionMHL` on the token embedding (Anatoly's proposal, the one to try first).**
   - The key is the token's existing wte row: compress 1536 → r, LUT, decompress r → 1536.
   - No new 32,768-row table is needed, because wte is reused, so per-token storage is zero.
   - Real numbers, using the table above: at r = 64 with the reference geometry, **≈ 1.25M per layer, 15.0M for 12 independent layers, 40× smaller**. With one shared compress (all 12 layers read the same wte row) it is 13.9M (44×) *(arithmetic)*.
   - The earlier estimate, "~1.3M plus codebooks at r = 64, 100–400× cheaper", was **wrong**. The tables are 84% of the 1.25M, not an extra on top. Reaching 100–400× needs smaller or shared tables, e.g. tph 16 at r = 64 → 5.5M, 109× *(arithmetic)*.
   - It is **not** "strictly more expressive than low-rank at the same rank". Option 2 has a *free* code per token, while option 4 is a function of the wte row. Its map is piecewise-constant in the projected space, not a superset of rank-r linear maps.

**Three caveats for option 4.**
- **The output is a pure function of the token id**, because `wte[t]` is deterministic. The function class is capped at what is recoverable from a wte row through rank r plus the LUT, and at inference it could be materialised back into a full table. The win is parameters and training memory, not inference lookup cost.
- **The key is continuous.** The address is the sign pattern of fixed anchor-pair differences of `compress(wte[t])`. It is computed exactly every forward, but the compress weights that shape it learn only through the straight-through surrogate gradient (or the Gen-3 score gradient). Over the 32,768 known keys, all addresses can be enumerated cheaply, for example to monitor how many tokens collide per cell.
- **wte gains 12 new gradient paths.** The embedding table must now serve both the residual stream and 12 value readouts. This is observable, not predictable.

**Cross-cutting notes.**
- **This saves bytes, not FLOPs.** The value embeddings are ~0% of compute; the replacement *adds* ≈ 0.4 MFLOP per token per layer forward *(estimate)*. The honest claim is "same quality at a fraction of the parameters in the dominant block": a footprint win, not a compute win.
- **Rare tokens.** Quantising makes tokens share value vectors, and rare tokens are plausibly where the value path matters most (copying unusual strings is the induction-head job). Aggregate bpb can look fine while rare-token behaviour degrades, so measure it specifically.
- **Report both axes**, params and FLOPs, for every comparison in this stage, consistent with the equal-compute, shape-free decision in stage 3.

**First step: no training run needed.** Once the baseline checkpoint exists, fit the chosen compression post hoc to reproduce the 12 trained value-embedding tables from wte.
- This is a regression over 32,768 known input/output pairs.
- Then measure CORE and val bpb degradation against the uncompressed model.
- The result is a per-layer expressivity verdict and a rank/geometry sweep, in minutes of GPU time *(estimate)*.
- Only if post-hoc compression holds up do we train from scratch with the compressed form.
- **The bounding ablation:** d24 with the value embeddings removed entirely, cheap to run at d12 on the 5090. If that barely hurts, the 604M was not buying much, and the whole compression result is less interesting.

### Stage 2b: distillation (not yet implemented)
Train LUT models on the baseline model's output logits, with an output-distribution KL objective only.
- Matching per-block hidden states or attention maps was considered and **rejected**.
- The reason: LUT models reach the same bpb by reallocating compute, using cheap LUT FFNs and more sophisticated attention patterns. Any objective that pins internal representations to the teacher fights that mechanism.

### Stage 3: LUT models from scratch (not yet implemented)
- The LUT arm is **not** shape-matched to d24 with its FFNs swapped. The honest comparison is at equal compute or equal cost, with the model shape left free. Forcing the LUT model into the dense shape handicaps it if it wants more attention capacity.
- The restructuring itself is a result worth logging from the first LUT run: attention entropy, window usage, head specialisation and induction behaviour.

## Stage 2 entry gate: the distillation screen every LUT variant must pass

**Purpose.** A cheap, early go/no-go screen *before* committing a full LUT run. Distill a LUT student from the trained dense d24 teacher (`d24_1xh100`, step 5568) and compare its val-bpb-vs-step curve against the dense-student reference curve below. If the LUT student **tracks the reference through the early steep region**, it has enough capacity to justify a full LUT run. Where it **peels away** from the reference — the plateau height — is that LUT form's expressivity shortfall, and is itself a useful number to report.

**Protocol — must be byte-identical to the dense distillation run so the curves are comparable.**
- **Same d24 architecture depth.** Do NOT prelim at d12 or rescale: the reference curve is specific to this teacher, depth and batch, so a smaller model has no valid reference.
- **Same 2²⁰ global token batch** (device batch 8 × 64 grad-accum).
- **Teacher** = `d24_1xh100` frozen at step 5568.
- **T = 1.0, alpha = 0.0** for the gate.
  - *(Note: alpha > 0 — a hard-CE blend — is likely the right default for a REAL full LUT run, because a LUT student cannot match a dense teacher exactly and the data term lets it find its own route to the same bpb. But alpha > 0 changes the objective and breaks comparability with this reference curve, so it is NOT used for the gate. Clean split: alpha = 0 to pass the gate; alpha > 0 as a knob for the full run.)*
- **LR schedule (critical).** Use the FULL 5,568-step LR schedule and simply truncate/kill the run at the gate step. Do NOT schedule a short horizon (e.g. 1,000 steps): a short schedule cools the LR at the end and makes a truncated run look artificially better for reasons unrelated to architecture. The dense reference points were read *mid-flight* from a 5,568-step schedule at high LR, so the LUT gate must be read the same way — **identical truncation on both arms**.
- **Harness.** Reuse `nanochat/runs/distill_d24_1xh100.sh` with the LUT student swapped in for the dense student; everything else (teacher, batch, T, alpha, 5,568-step schedule) stays as above. The dense early-stop (`--distill-early-stop-*`) won't fire in the gate window and is irrelevant to the gate.

**Reference curve — dense d24-from-d24 student, val bpb vs step.** Teacher val bpb = **0.719**.

| step | dense-student val bpb |
|---|---|
| 250 | 0.932 |
| 500 | 0.837 |
| 750 | 0.804 |
| 1000 | 0.787 |
| 1250 | 0.779 |

*These are mid-flight points from the in-progress dense distill run `distill_d24_from_d24_1xh100` (T=1.0, alpha=0.0, DBS=8). **To be finalized and extended** (and the exact per-step values re-pinned) once that run completes / early-stops; treat the table as provisional until then.*

**Pass criterion — a band, not a point.** The LUT student passes if its val bpb is **within ~0.01 of the dense student at steps 500 / 750 / 1000**. Rationale: single seed, and val bpb has its own wobble, so anything under ~0.01 bpb is not signal in either direction. A LUT arm *beating* the dense student would be surprising enough to require a **second seed** before believing it.

**Interpretation.**
- **Tracks the first ~1,000 steps within band → PASS**: the form has the capacity; proceed to a full LUT run (where alpha > 0 becomes available as a knob).
- **Plateaus ~0.03 above the reference → FAIL**: the LUT form is short that much capacity. Report the plateau height — it is the expressivity shortfall of that form, a useful result in its own right even on a fail.

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
| `results/<run>/` | Written by the run and committed back by the operator: logs, `summary.json`, pins, manifest, eval CSV. |
| `results/<run>/checkpoints/` | The run's model, optimizer and meta files, **git-ignored**. nanochat's `$NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>` is a symlink to it. Shared inputs (data shards, tokenizer, eval bundle) stay in `$NANOCHAT_BASE_DIR`. |
| `docs/` | The nanochat state report and the baseline decision brief (md + pdf), with the d24 architecture confirmed from code. |
