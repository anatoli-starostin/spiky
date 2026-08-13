# Known engine bugs (spiky spnet), found from the steady-state work

## 1. Nondeterministic hang / error in `create_forward_groups` at build time — OPEN

**Symptom.** `build_pool` → `SpikingNet.add_connections` → the `create_forward_groups`
kernel misbehaves **nondeterministically on byte-identical input**. Three outcomes from the
same fixed-seed genomes in fresh processes:

1. clean build (the common case),
2. `ValueError: some error happened inside create_forward_groups kernel` — host-side raise
   off the kernel's error counter, context intact, **recoverable by retrying**,
3. **hang** — a GPU kernel spins forever. GPU pinned at 100%, memory flat, process alive
   burning CPU, no fault, no output. Not recoverable in-process: a running CUDA kernel
   cannot be interrupted or killed from the host.

Same input producing different outcomes across processes ⇒ this is a **race / missing
synchronisation in the growth kernels**, not a bad genome or a bad structure. Bisecting to
an "offending genome" is meaningless and was not attempted.

**Measured rate** (build only, fixed seed, one fresh process per attempt):

| pool | ok | error | hang |
|------|----|-------|------|
| K=128 | 9/10 | 0/10 | **1/10 (~10% per build)** |
| K=32 | 15/15 | 0/15 | 0/15 (too rare to show in 15) |

Field data agrees: a K=32 run hung at round 136 (≈0.7% per build) and K=128 runs hung at
round 3, every time. Since `build_pool` runs **every round**, P(a 300-round K=128 run
finishing) = 0.9^300 ≈ 0 — K=128 simply cannot complete unsupervised.

**Standalone reproducer** (needs no mutation, no STDP, no episodes — plain seed genomes):

```
python probe_build_flaky.py --k 128 --tries 10 --timeout 90
```

Roughly one hang per ten attempts, ~10s each. Supporting artifacts: `probe_hang_k128.py`
(instrumented round loop that localises the stall to `build_pool`), and
`results/hang_genomes_round0.npz` (the exact 128 genomes).

**Localisation.** Definitely the growth engine, not the runtime. Instrumented logging shows
the last line is always `calling build_pool ...`; `BUILD_POOL RETURNED` never prints, and no
training episode ever starts. (An earlier guess that maturation was responsible was wrong —
`build_pool` runs at the *start* of every round, before maturation.)

**Mechanism (probable): an unguarded chain walk.** Both `while(true)` loops in
`connections_manager_kernels_logic.proto` — the capacity estimator (~line 36) and
`create_forward_groups` (~line 294) — walk a source's block chain with the single exit
`if (header.shift_to_next_group == 0) break;`. There is no visited set, no iteration cap.
`shift_to_next_group` is a *signed* offset to an arbitrary block (chained blocks demonstrably
do not follow their root), so a **cycle in the chain spins forever** — exactly the observed
symptom. A visited/iteration bound would turn a silent infinite hang into a diagnosable
error, independent of whatever creates the cycle.

**Not caused by `forward_group_size=2`** (hypothesis tested and refuted). On identical
genomes, 3 fresh processes each: gs=2 → ok 3/3; gs=4 and 8 → `cudaErrorIllegalAddress` 3/3;
gs=16 and 32 → 2/3 error + 1/3 hang. gs=2 is the only value that builds at all and was
hang-free throughout. **There is no group size that is both hang-free and crash-free.**

**Not reproducible from the round-6 checkpoint.** An earlier note here claimed
`results/hang_repro_round6_k128.npz` was a deterministic reproducer because a supervised run
stalled at round 6 three times. That was wrong: on those genomes `build_pool` succeeds 4/4,
32 maturation episodes run 32/32, and the whole round replays in 17.3s (`probe_round6.py`).
The three supervisor stalls are unexplained — either the ~10% build flake three times running
(~0.1%) or a bug in the watchdog itself. **The only confirmed reproducer is
`probe_build_flaky.py --k 128` (seed genomes, ~1/10).**

**Likely the same defect as the group-size fault.** `create_forward_groups` is the same
kernel that dies with `cudaErrorIllegalAddress` at `connections_manager.cu:126` whenever
`synapse_group_size > 2` on multi-meta explicit wiring (see bug 2). A build that sometimes
trips this kernel's error counter and sometimes spins is plausibly the same underlying
defect caught at different moments.

**Workaround in place** (not a fix): `steady_state.build_pool_retry` retries case 2, and
`supervise_run.py` + per-round checkpointing recovers case 3 by killing and resuming the
process. Resume is verified bit-identical to an uninterrupted run (RNG state is
checkpointed too).

## 2. Multi-meta explicit wiring requires ENGINE `synapse_group_size == 2` — WORKED AROUND

`SynapseGrowthEngine(synapse_group_size=...)` must be **2** for explicit wiring spanning
several metas; 4, 8, 16 and 128 all die with `cudaErrorIllegalAddress` at
`connections_manager.cu:126` (the `create_forward_groups` launch), and `1` is rejected
upstream ("single_block_size must be even").

**Precision note.** An earlier version of this entry said "every forward group size except 2",
conflating two independent knobs — my sweeps had moved them together. Decomposed at K=128,
6 fresh processes each:

| engine gs | exc meta gs | inh meta gs | result |
|---|---|---|---|
| 2 | 2 | 2 | ok 5/6 (1 hang — bug 1) |
| 2 | 8 | 128 | ok 5/6 |
| 2 | 8 | 8 | ok 5/6 |
| 2 | 32 | 32 | ok 5/6 |
| **128** | 2 | 2 | **error 6/6** |

Only the ENGINE size matters. The meta `_forward_group_size` / `_backward_group_size` are a
**free parameter** for the build — which is what lets bug 3's fix (backward=32) work.

This was the original stage-two build crash: `stage2_metas` hardcoded 8 while the growth
engine used `GROUP_SIZE=2`. Reproducers: `test_stdp.py`, `test_stdp_bisect2..7.py`,
`test_multimeta.py`, `probe_ipynb_cfg.py`.

**spnet.ipynb's config does not transfer.** Its `synapse_group_size=((M+31)//32)*32` = 128
with exc meta 8 / inh meta 128 fails 0/12 here (9 error, 3 hang): the engine size crashes the
build, and its `_backward_group_size=8` crashes the runtime 3/3 (bug 3). It works in the
notebook because that is a two-meta *spatial-growth* net whose single excitatory meta spans
delays 0..19 by positional split — it never does multi-meta explicit wiring.

## 3. Backward/STDP structure overflows past ~8 incoming plastic synapses per target — WORKED AROUND

Independent of bug 2 and of the forward size. With `_backward_group_size=2`,
`process_ticks(do_train=True)` dies at `spnet_runtime.cu:1030` once a target neuron receives
more than ~8 incoming *plastic* synapses (exc→exc fanout 2/4/8 fine; 16/32/64/80 crash).
Sweeping the backward size alone with forward pinned at 2: 2/4/8 crash, 16/32/64/128 pass.
Our nets put ~80 incoming plastic on every excitatory neuron and ~100 on every output
neuron, so `stage2_metas` now uses `backward_group_size=32`. Note `_forward_group_size` and
`_backward_group_size` are independent fields — the forward one must stay 2 (bug 2), the
backward one must not.

## 4. Per-synapse delays are not expressible — DESIGN LIMIT, not a bug

`_grow_explicit` takes `(meta_index, src, tgt)` triples only; there is no delay input.
Delay is a **per-group** property stamped from the meta, and a meta with `min<max` spreads
its targets evenly across the range by position. Weights *are* genuinely per-synapse
(`input_weights[input_cursor]`), delays are not. So arbitrary inherited per-synapse delays
require one `min==max` meta per distinct delay — which is what `delay_metas()` does.
