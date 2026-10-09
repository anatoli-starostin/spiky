# LUT-FFN d24 block profile — gpustar (RTX 5090), 2026-10-09

**Provisional host.** This GPU is not the H100 the d24 runs train on. The absolute ms here are provisional. The
per-op *ranking*, the compile/memory structure and the harness are the durable part. Re-run on the H100 with
`experiments/lut_nanochat/profiling/profile_lut_ffn.sh` and diff `results.json` field by field.

Labels: **[HW-dep]** must be re-measured on the H100; **[HW-indep]** transfers as-is.

## Setup

- **Hardware** [HW-dep]
  - RTX 5090: cc 12.0, 31.35 GiB, 170 SMs, driver 595.84.
  - torch 2.9.1+cu130, CUDA 13.0, Triton 3.5.1.
  - Repo at cc46cd21.
- **Measured peaks** [HW-dep]
  - TF32 GEMM 120 TFLOP/s, bf16 241, fp8 474.
  - Copy bandwidth 1.52 TB/s.
  - fp8 is usable here: a functional `torch._scaled_mm` probe passed.
- **Configuration under test** [HW-indep]
  - The locked `--lut-ffn` geometry: h=16, d=48 (r=768), tph=64, nap=8, pairs, read_top_n=1, β/γ=2/1, table dropout 0.2, seed 1, fp32 island.
  - Training forward, whole-module `torch.compile(dynamic=False)` with the cc46cd21 dynamo flags.
  - TF32 matmuls, as base_train sets them.
- **Dense reference:** the nanochat d24 MLP, ReLU², 1536→6144→1536.
- **Statistics:** medians of 20 CUDA-event-timed repeats after 5 warm-ups. IQRs are ≤0.3 ms (in `results.json`).
- **Attention / FA3 not involved:** `fa3_loader_imported: false`.

## Headline: one block, fwd+bwd, 32,768 tokens (16×2048)

| variant | MACs/token [HW-indep] | fwd ms | fwd+bwd ms | Mtok/s | 24 layers × one 2^20-token step (FFN only) |
|---|---|---|---|---|---|
| dense-fp32 (TF32) | 18.87M | 11.70 | 35.00 | 0.94 | 26.9 s |
| dense-bf16 | 18.87M | 5.80 | 17.63 | 1.86 | 13.5 s |
| **dense-fp8** (baseline as trained) | 18.87M | 3.59 | **10.98** | 2.98 | **8.4 s** |
| **lut-fp32** (status quo) | 2.41M | 7.16 | **33.06** | 0.99 | **25.4 s** |
| lut-bf16 (narrowed island, experimental) | 2.41M | 6.16 | 30.43 | 1.08 | 23.4 s |
| lut-fp8 (narrowed island, experimental) | 2.41M | 6.44 | 30.72 | 1.07 | 23.6 s |

[HW-dep] At 8,192 tokens the per-token throughput is the same within ~5%: the block has no small-batch inefficiency,
and its time scales linearly with tokens.

**The FLOP-vs-wall-clock gap:**
- [HW-indep] The LUT block does **7.8× fewer MACs** than dense.
- [HW-dep] Yet it is **3.0× slower** than dense-fp8 and 1.9× slower than dense-bf16.
- [HW-dep] At 24 layers per 2^20-token step it costs **+17 s/step** over dense-fp8 on this GPU. Nebius observed
  about +18 s/step (35 vs 17 s), so the FFN swap plausibly explains essentially all of the step-time ratio.

## Per-op ranking, lut-fp32 at 32k tokens

[HW-dep] ms. [HW-indep] kernel structure and ranking.

| rank | piece | fwd | bwd | total | share |
|---|---|---|---|---|---|
| 1 | `embedding_bag`, per-sample-weight (score) grad kernel | — | 10.15 | 10.15 | 30% |
| 2 | `embedding_bag`, table weight grad: radix sort 5.7 + `compute_grad_weight_bags` 2.9 + scatter/unique 0.6 | — | 9.23 | 9.23 | 27% |
| 3 | `embedding_bag` forward (the gather itself) | 4.29 | — | 4.29 | 13% |
| 4 | fused Triton addressing + sign-bit pack + confidence score (one kernel each way) | 1.43 | 3.95 | 5.39 | 16% |
| 5 | compress + decompress GEMMs (TF32 cutlass sm80 `s1688gemm`) | 1.40 | 2.64 | 4.04 | 12% |
| 6 | other (casts, arange, …) | — | 0.48 | 0.48 | 1% |
| | **profiler total** | 7.13 | 26.46 | 33.6 | (timed: 33.06) |

- **`embedding_bag` (fwd + bwd) ≈ 23.7 ms ≈ 70%.**
- **The fp32 island (both projection GEMMs) ≈ 4.0 ms ≈ 12%.**
- Addressing and score ≈ 16%.
- [HW-indep] There are no transposes, `.contiguous()` copies, d2d copies or per-head kernel launches. Every op is
  batched over all 16 heads, and the casts cost about 1 ms.
- **Cross-check against the manual sub-module breakdown** (`components` in results.json): compress 2.17 +
  cartridge 29.66 + decompress 2.28 + casts ≈ 1.0 = 35.1 ms isolated, vs 33.06 ms for the compiled block (some
  cross-module fusion is lost when pieces run alone). The cartridge's `embedding_bag` alone is 23.5 ms
  fwd+bwd, matching the profiler's 23.7.
- The isolated "score" timings in `components` are **not** representative: run alone, it materialises the
  `[N,16,64,8]` margins that the in-graph fused kernel never stores. Use the profiler's fused-Triton row instead.

## Is the gather near roofline? (testing Anatoli's claim)

- **Forward: yes** [HW-dep]. It moves 6.85 GB at 32k tokens (an fp32 48-wide row + int64 index + fp32 weight per
  token × 1024 tables) in 4.29 ms, which is **≈1.6 TB/s against a measured 1.52 TB/s copy bandwidth**: at the
  roofline. Only moving fewer bytes would help.
- **Backward: no** [HW-dep magnitudes, HW-indep mechanism].
  - The per-sample-weight grad re-reads the same gathered rows (6.4 GB) at only **≈0.64 TB/s (~42%)**. It is the
    single biggest kernel.
  - The table weight-grad goes through a full **radix sort of 33.5M int64 keys** per layer per micro-batch.
  - The backward is ~4.6× the forward, and that is where the headroom is.

## fp32 island vs gather: how much of the slowdown is which

- [HW-dep] Both projection GEMMs together are **12%** of the LUT block (4.0 of 33 ms). Even at zero cost they would
  take the block only to ~29 ms, still 2.6× slower than dense-fp8.
- [HW-dep] **The slowdown is the read (embedding_bag, 70%), mostly its backward (57%).**

## Narrowed fp32 island (experimental, harness-only; lutorch_ex and training untouched)

### Speed [HW-dep]

| | block fwd+bwd | change |
|---|---|---|
| lut-fp32 (status quo) | 33.06 ms | — |
| lut-bf16 projections | 30.43 ms | −8.0% |
| lut-fp8 projections | 30.72 ms | −7.1% |

At 24 layers this saves ~2 s/step. fp8 is no better than bf16 at these GEMM sizes: compress_fp8 alone is 3.4 ms
fwd+bwd vs 1.1 ms in bf16, because the amax/scale/cast overhead dominates at K=768/1536. Re-check fp8 on the H100.

### Numerics [mostly HW-indep]

The comparison uses the same params, input and dropout mask, at 8k tokens.

| | output max abs dev | output mean abs dev | input-grad max dev | addresses flipped |
|---|---|---|---|---|
| bf16 projections | 4.0e-3 (13% of max\|y\|) | 2.0e-4 | 23% of max | **0.58%** |
| fp8 projections | 7.6e-3 (25% of max\|y\|) | 9.4e-4 | 41% of max | **9.13%** |
| *baseline: TF32 status quo vs exact fp32* | | | | *0.052%* |

The max deviations are dominated by flipped addresses, where a flip reads a different table row. bf16 flips 11×
more addresses than the status quo already flips vs exact fp32, and fp8 flips 175× more. The measurement is
meaningful (the variants compute the same function up to precision), but the effect on training quality is
unknown. These numbers are at random init: trained compress margins differ, so a bpb check is needed before adopting
bf16. fp8 projections are not recommended.

## Memory, 32k tokens

[HW-indep] shapes, [HW-dep] peaks.

All columns come from the `memory` pass. Reserved includes the eager warm-up's allocator cache.

| variant | held after fwd | peak alloc | peak reserved |
|---|---|---|---|
| lut-fp32 | 1.38 GiB | 3.34 GiB | 5.09 GiB |
| lut-bf16 | 1.15 GiB | 3.16 GiB | 4.73 GiB |
| dense-fp8 | 0.72 GiB | 2.77 GiB | 4.40 GiB |
| dense-bf16 | 0.88 GiB | 2.39 GiB | 2.52 GiB |

Nothing pathological. The largest saved LUT tensors are:
- the **int64** flat cell index `[N·1024]`;
- the fp32 copy of the block input;
- the score `[N,16,64]`;
- the per-sample weights.

The `[N,16,64,8]` margin tensor is recomputed in the fused backward, not stored.

## torch.compile [HW-indep]

`torch._dynamo.explain`: **1 graph, 0 graph breaks, 25 ops**. After 3 fwd+bwd there is still `unique_graphs=1` and
no recompiles. The cartridge's own inner `torch.compile` is inlined. Only `embedding_bag` runs as an ATen kernel,
the standard Inductor fallback. The cc46cd21 fix works, and compile is not the bottleneck.

## Ranked candidate optimisations

| # | change | evidence | expected payoff on this GPU | where / cost |
|---|---|---|---|---|
| 1 | **int32 flat cell indices** for the train read | `read_path_probe.json`: read fwd+bwd 24.3→19.3 ms at 32k (−20%); **gradients bit-identical** (max \|Δ\| = 0) | ≈ −5 ms/block (−15%), ≈ −3.7 s/step; halves the saved-index memory | **lutorch_ex library**, small (dtype in `_global_cells` / `_scored_read`) + a test. [HW-dep magnitude, HW-indep exactness] |
| 2 | **custom fused read backward**: per-sample-weight grad at bandwidth, weight grad by atomics (no sort, no materialised rows) | the psw-grad kernel runs at 42% of bandwidth; the weight grad is sort-bound; pure-PyTorch rewrites lose (`index_add_` 8.3 ms vs 9.2 for the whole sort path; gather-dot 28.8 ms) | estimate, not measured: −10 to −11 ms/block (~30%), ≈ −8 s/step | **library, high cost**: a Triton/CUDA kernel + oracle tests; the weight grad becomes non-deterministic |
| 3 | **bf16 compress/decompress** (narrowed island) | −8.0% block, measured; 0.58% of addresses flip | ≈ −2 s/step | **nanochat-side** (`lut_ffn.py`), small, but needs a bpb validation run first |
| 4 | fewer bytes per token: smaller tph or d_out, or bf16 table storage | the read is bandwidth-bound and ∝ G·tph·d_out = 196 KB/token/layer | ∝ bytes cut | config change (changes the model; a research decision) / library change for bf16 tables |
| — | ✗ skip table-dropout-dropped entries via `padding_idx` | `read_path_probe.json`: fwd+bwd **2× slower** (24.3→48.7 ms) | negative | do not do |
| — | per-step fixed costs | cell-TV 61 ms/step, fused AdamW over 358.7M LUT params 7.5 ms/step | <0.3% of a step | not worth touching |

## Must wait for the H100

- All absolute ms and tokens/s, and the LUT/dense ratio.
- Whether the H100's ~2× bandwidth shrinks the read cost, and whether its stronger fp8 GEMMs widen the gap.
- The fp8-projection conclusion.
- The int32-index gain.
- Peak memory.
- A whole-model step time: only FFN blocks were measured here.
