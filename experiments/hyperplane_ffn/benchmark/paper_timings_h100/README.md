# Paper timings (H100) — FFN-slot inference numbers and the phase-breakdown figure

The **H100** counterpart of `../paper_timings/` (which is the RTX 5090 deliverable). Same
spirit: per-phase FFN-slot timings + a phase-breakdown figure for the paper, on the three
trained CompressionMHL grid cells (0126/0127/0128) vs the untied vanilla dense baseline.

The H100 best inference path is the fused routing+gather kernel `../gather_fused_v2_h100.cu`
(v2 routing — column-major z-slice in shared + token-inner warp, conflict-free — with the
index kept in shared and fed straight into the gather; **fp32, bit-exact**). See
`../H100_OPT_NOTES.md` for the full optimization sweep (what won, what was dead).

## Files

| file | what |
|---|---|
| `phase_split_h100.py` | the measurement harness: builds the three models on their trained checkpoints, times each phase of the fused_v2 slot + the vanilla split + eager-vs-compile, writes `results.json`. Uses the JIT kernels in `..` (`-std=c++20` required). |
| `make_figure.py` | portable figure generator — reads `results.json`, writes `ffn_phase_split_h100.png`. No hardcoded numbers, no absolute paths (`python paper_timings_h100/make_figure.py`). |
| `results.json` | the committed measured run (the numbers below). |
| `ffn_phase_split_h100.png` | the figure. |

## Citable conditions

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (sm_90) |
| torch / CUDA | 2.13.0+cu130 / CUDA 13.0 |
| workload | **batch 48 × seq 512 = 24,576 tokens** per slot call |
| dtype | routed: **bf16** compress/decompress Linears + **fp32** fused routing+gather (the fused kernel is **bit-exact** vs the fp32 reference — NOT bf16-approximate like the 5090 cuda-fused path). Vanilla baseline **bf16** throughout. |
| gather path | `gather_fused_v2_h100.fused_v2` — routing+gather in one kernel, index kept in shared (never materialized to HBM). Routing is v2 (bank-conflict-free) internally for all three. |
| timing | `torch.no_grad()`, CUDA events, **50 iters × median of 15 rounds**, 20 warmup iters. Steady-state. |
| clock | warmed / boosted, unlocked (H100 boosts under load; max SM 1980 MHz). Numbers are steady-state after warmup; absolute ms depend on clock/driver — **the routed/vanilla RATIO is the citable quantity**. |
| checkpoints | trained (`--load-checkpoint`); val bpb 1.20694 / 1.19471 / 1.20228 |

## Measured (this repo, the run committed in `results.json`)

**Vanilla dense FFN (bf16), ms/call:** up 384→1536 `0.0526` · GELU `0.0579` · down 1536→384
`0.0424` · **phase-sum 0.153** (whole-slot eager 0.180, torch.compile 0.157).

**Routed best path — fused_v2, per-phase ms/call (bit-exact):**

| model | compress 384→192 | routing+gather (fused) | decompress 192→384 | **slot** | **vs vanilla** |
|---|---|---|---|---|---|
| 0126 nap7/tph64  | 0.0129 | 0.1411 | 0.0146 | **0.169** | **1.10×** |
| 0127 nap7/tph128 | 0.0120 | 0.2534 | 0.0131 | **0.278** | **1.82×** |
| 0128 nap8/tph64  | 0.0126 | 0.1489 | 0.0131 | **0.175** | **1.14×** |

**Phases 2+3 are ONE kernel on the fused path** and cannot be split from a single run. For a
conceptual phase-2 vs phase-3 number, measured *separately* (standalone routing + Triton
gather; this is NOT how the fused path runs): routing `0.0615 / 0.1112 / 0.1146`, gather
`0.1122 / 0.2260 / 0.1198` for 0126/0127/0128.

**eager vs torch.compile** (routed full slot, fp32 reference path): `0.808/0.806`,
`1.492/1.490`, `0.949/0.948` — **no change** (6 graph breaks each: Dynamo can't trace the
custom routing kernel). Vanilla: eager 0.180 / compile 0.157 — the dense GEMMs are already
cuBLAS-optimal, so compile does not help either path. (torch.compile is not the tool here;
the fused_v2 numbers above are the fast path.)

### Takeaway

On H100 the fused_v2 routed FFN slot is within **~10–14%** of vanilla dense for the tph64
configs (0126, 0128) and ~1.82× for tph128 (0127, double the gather work). It does **not**
overtake vanilla dense — the dense baseline is compute-bound and extremely fast on H100
tensor cores — but ncu-guided kernel work (bank-conflict-free routing + fusing away the
50 MB HBM index round-trip) closed the gap from the 1.6–2.6× of the naive two-kernel path.
