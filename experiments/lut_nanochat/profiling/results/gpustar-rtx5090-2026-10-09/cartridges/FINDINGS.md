# ConfidenceLUT vs QuantisedConfidenceLUT (p2_int8): per-op profile — gpustar (RTX 5090), 2026-10-09

**Provisional host:** an RTX 5090 (cc 12.0), not the H100. Absolute ms are provisional; the code-level facts and the
kernel structure transfer. Every finding is labelled **[HW-dep]** or **[HW-indep]**. Re-run with
`experiments/lut_nanochat/profiling/profile_cartridges.sh` and diff `cartridges.json`.

**Scope.** One cartridge (the LUT core of one FFN sub-layer), at the locked d24 geometry:
- h_in=h_out=16, d_in=d_out=48, tph=64, nap=8, pairs;
- β/γ 2/1, learnable score, table dropout 0.2, seed 1, fp32.

Token counts:
- 32,768 tokens (16×2048, nebius's live micro-batch), plus 8,192.
- **Note: the main table is the 32k column; the profiler ran at 32k only.**

Timing method:
- Seeded, CUDA-event, median of 20 after 5 warm-ups; IQRs ≤0.23 ms.
- Train-backward = (fwd+bwd) − fwd.
- Profiler: warm-up session burned first, the `CompiledFxGraph` wrapper excluded, empty traces detected.

The compress/decompress GEMMs are identical for both cartridges and are profiled in `../FINDINGS.md` (they are
~12% of the full block).

## Existing tooling: where the report's numbers came from [HW-indep]

- `src/spiky/lutorch_ex/bench.py` is the library's own harness. It times `forward_eval`, `forward_train` and
  `backward` for any cartridge, median/min, with **no per-op breakdown and no memory measurement**. Its CLI demo times
  only ManifestoHard/Soft at a toy geometry.
- The lutorch_ex report's §6.1 table (ConfidenceLUT n=1: 11.5 ms step, 1.1 GB) was produced by
  `doc/research/lutorch_ex_report/bench_report.py` on branch `feature/lutorch_ex_report`. It drives
  `bench.benchmark()` and adds a separate peak-memory probe, at h=8 (not d24's h=16) and B=24,576.
- `profile_cartridges.py` reuses `bench.benchmark()` for its headline (`bench` in `cartridges.json`). On top of it,
  the profiler, the manual breakdown, the bytes-per-token figures, the saved-tensor inventory and the deploy-only
  object are new.

## Verdict: quantised training does NOT move fewer bytes

### From the code [HW-indep]

- **Fake-quant every forward:** `QuantisedConfidenceLUT._fake_quant_tables()` → `_pow2.ste_tables(W)`.
  - It computes per-(head, channel) exponents over the WHOLE master table (`head_chan_exponents`: abs-amax).
  - It quantises (`round_half_up / 2^e`, clamp to int8 range, × 2^e) into a **new fp32 tensor**.
  - It returns `Wq + (W − W.detach())`: value W_hat·2^e, gradient the identity to the fp32 master.
- **The gather:** the read is the same `F.embedding_bag(gc, W2, per_sample_weights=psw, mode="sum")` as
  ConfidenceLUT, over that fp32 copy. The gathered rows are **float32**.
- **Score:** at read_top_n=1 the score is rounded to a power of two by an STE (`_ste_score_weight`:
  `val + (st − st.detach())`). The native `p2_scalars` CUDA op is n=2-only and is not used at n=1.

### From the profile [HW-indep structure, HW-dep ms]

Train-mode backward at 32k is the same `embedding_bag` backward, kernel for kernel:

| kernel | ConfidenceLUT n=1 | QuantisedConfidenceLUT n=1 |
|---|---|---|
| `_embedding_bag_per_sample_weights_backward_kernel<float,long>` | 10.14 ms | 10.12 ms |
| cub radix sort (onesweep) | 5.68 | 5.70 |
| `compute_grad_weight_bags<float,long>` | 2.70 | 2.69 |
| `sum_and_scatter<float,long>` | 0.43 | 0.43 |

### Bytes the read moves per token, per layer [HW-indep]

| path | rows | per token | plus |
|---|---|---|---|
| ConfidenceLUT n=1 (train/eval) | fp32 | **208,896 B** (1024 × (192 row + 8 idx + 4 weight)) | — |
| QuantisedConfidenceLUT n=1 (train/eval) | **fp32** (fake-quant copy) | **208,896 B — identical** | a fixed full-table fake-quant pass per call, ≈151 MB, independent of tokens |
| ConfidenceLUT / Quantised n=2 | fp32 | 417,792 B | — |
| deploy-only DeployedQuantisedConfidenceLUT n=2 | **int8** | 114,688 B (2048 × (48 + 8)) | — |

**Conclusion:** quantisation does not reduce the dominant read cost in training. The hope that int8 halves or
quarters the read for the gate run does not hold for the training path. Only the deploy-only n=2 object reads int8.

## Headline timings, one cartridge, median ms [HW-dep]

| cartridge | tokens | eval fwd | train fwd | train fwd+bwd | train bwd (diff) |
|---|---|---|---|---|---|
| ConfidenceLUT n=1 | 8,192 | 0.98 | 1.44 | 7.58 | 6.14 |
| QuantisedConfidenceLUT n=1 | 8,192 | 1.45 | 1.49 | 7.81 | 6.32 |
| **ConfidenceLUT n=1** | **32,768** | **3.80** | **5.53** | **29.52** | **24.00** |
| **QuantisedConfidenceLUT n=1** | **32,768** | **5.41** | **5.56** | **30.51** | **24.94** |
| ConfidenceLUT n=2 | 32,768 | 14.30 | 9.77 | 56.57 | 46.79 |
| QuantisedConfidenceLUT n=2 | 32,768 | 10.96 | 11.14 | 59.46 | 48.32 |
| **deploy-only** DeployedQuantisedConfidenceLUT n=2 | 8,192 | 3.86 | — | — | — |
| **deploy-only** DeployedQuantisedConfidenceLUT n=2 | 32,768 | **35.02** | — | — | — |

- At n=1 the quantised cartridge trains **+3.3% slower** (30.51 vs 29.52 ms) and is **never faster**.
- `bench.benchmark()` (the library harness) agrees: n=1 backward 25.15 vs 25.53 ms, train fwd 5.46 vs 5.57 ms at 32k.

## Per-op breakdown at 32k, n=1 — TRAIN FORWARD [HW-dep ms; HW-indep kernel structure]

From the profiler (3 iterations; the profiler total runs ~1–1.5 ms above the CUDA-event timing).

| op | ConfidenceLUT | QuantisedConfidenceLUT |
|---|---|---|
| addressing (margins + sign-bit pack) + confidence score + table-dropout mask, one fused Triton kernel | 1.12 | 1.26 (also computes the power-of-two score exponent k' and the STE weight) |
| fake-quant tables: `abs_amax` reduction + quantise/scale elementwise | — | 0.036 + 0.025 = **0.06** |
| gather: `EmbeddingBag_updateOutputKernel<float>` | 4.31 | 4.28–5.79 (*) |
| casts / misc | <0.01 | <0.01 |
| **profiler total** | 5.43 | 7.12 (*) |

(*) The quantised gather read 5.79 ms in one 3-iteration profile but 4.28 ms in its eval profile, and the 20-repeat
CUDA-event train forward is 5.56 vs 5.53 ms. Trust the timing table: the training forwards are equal within noise.

## Per-op breakdown at 32k, n=1 — TRAIN BACKWARD [HW-dep ms]

| op | ConfidenceLUT | QuantisedConfidenceLUT |
|---|---|---|
| gather bwd: per-sample-weight (score) grad | 10.14 | 10.12 |
| gather bwd: table weight grad — radix sort | 5.68 | 5.70 |
| gather bwd: table weight grad — `compute_grad_weight_bags` + `sum_and_scatter` + index_put | 3.39 | 3.39 |
| score / addressing backward (fused Triton) | 3.01 + 1.01 + 0.35 = 4.37 | 3.41 + 1.30 + 0.34 = **5.05** (STE adds ~0.7 ms) |
| fake-quant backward | — | none (identity STE: the gradient goes straight to the fp32 master) |
| **total** | 24.7 | 23.9 (profiler) / 24.94 (timing) |

### Manual sub-module breakdown

Each op is timed alone at 32k in the `components` pass of `cartridges.json`; numbers are fwd / fwd+bwd in ms.

| op | ConfidenceLUT | QuantisedConfidenceLUT |
|---|---|---|
| `embedding_bag` read, fp32 rows | 4.26 / 22.09 | 4.26 / 22.09 |
| fake-quant tables, compiled | — | 0.06 / 0.47 |
| table-dropout mask | 0.46 | 0.46 |
| cell-TV, once per optimizer step, not per micro-batch | 0.31 / 2.51 | 0.31 / 2.54 |
| cast bf16→fp32 `[N,768]` | 0.11 / 0.21 | 0.11 / 0.21 |

- The cell-TV regulariser runs on the fp32 master for both.
- The isolated "addressing" and "score" rows in the JSON are **not representative**: run alone, they materialise the
  `[N,16,64,8]` margins that the fused in-graph kernel never stores. Use the profiler rows above.

## What the STE adds

- **Kernels** [HW-indep]:
  - forward: one abs-amax reduction and one quantise/scale elementwise kernel over the full table (≈0.06 ms) [HW-dep];
  - the existing fused score kernel grows to also produce the power-of-two exponent and STE weight;
  - backward: a slightly heavier fused score backward (+~0.7 ms at 32k) [HW-dep].
  - No extra `embedding_bag` work, and no fake-quant backward kernel (identity).
- **Saved tensors** [HW-indep]: the same count (16). The table saved for the `embedding_bag` backward is a **new
  48 MiB fp32 fake-quant copy** `[262144, 48]` instead of a view of the master parameter.
- **Memory** [HW-dep peaks, HW-indep shapes]:
  - +48 MiB per layer held until backward (×24 layers ≈ +1.1 GiB per micro-batch);
  - peak 3.04 vs 2.94 GiB per cartridge at 32k;
  - **quantised training uses slightly MORE memory, not less.** Memory is saved only by the deploy object
    (params + buffers 12.1 MiB int8 vs 48.1 MiB fp32 per layer).

## Eval / deploy

### Eval-mode forward [HW-dep]

- **ConfidenceLUT n=1 eval is the fastest read: 3.80 ms.** Its compiled eval path uses a fused Triton gather +
  weighted sum (`triton_per_fused_add_arange_exp_index_mul_sum`, 2.90 ms), not `embedding_bag`.
- **QuantisedConfidenceLUT n=1 eval is 5.41 ms (+42%).** It stays on `embedding_bag` over the fake-quant copy, and
  there is no integer read at n=1 (`forward_int` is n=2-only).

### Deploy-only n=2 (`DeployedQuantisedConfidenceLUT`)

This is the true packed-int8 object: no fp32 master, int8 rows, int32 shift-add, scale folded into decompress.
**It is deploy-only**, read_top_n=2 only, and **not used in training.**

- [HW-indep] The library refuses deployment for n=1. So the n=1 configuration in the live d24 run has no int8
  deploy form at all.
- [HW-dep] **At 32k it is 3.2× slower than its own fake-quant eval and 2.4× slower than ConfidenceLUT n=2 eval**
  (35.0 vs 10.96 / 14.30 ms). At 8k it is 3.86 vs 2.84 / 3.60 ms.
- The time is two Inductor-generated int8 shift-add reduction kernels
  (`triton_red_fused__to_copy_bitwise_left_shift_..._sum`, 16.7 + 16.5 ms) from `_pow2.int8_accumulate`:
  `packed[idx].to(int32)` → shift → sum over the 128 cells of a bag.
- It reads 3.76 GB of int8 rows + indices at 32k, so it achieves ≈0.11 TB/s, ~7% of the measured 1.52 TB/s.
- **The report's deploy claim (int8 shift-add ⇒ faster read) is not borne out by this implementation on this
  GPU.** The bytes are 3.6× fewer than n=2 fp32, but the generated kernel is far off the roofline. A hand-written
  kernel would be needed to cash in the byte saving. Re-check on the H100 [HW-dep]; the kernel structure is
  [HW-indep].
- Side finding [HW-dep]: ConfidenceLUT n=2 eval (14.3 ms, plain gather of both cells + blend) is *slower* than its
  own train forward (9.77 ms, `embedding_bag`). The eval path materialises the `[N,G,tph,2,d]` pair.

## Extension / hardware [HW-dep]

- QuantisedConfidenceLUT works on this GPU and torch:
  - the native `p2_scalars` op is registered and enabled;
  - `VALIDATED_ARCHES` includes cc 12.0;
  - it loads in 0.17 s from cache.
- A fresh build of the int8 extension takes ~22 s. This was measured earlier today after clearing a stale build lock
  in `~/.cache/torch_extensions_lutorch_ex`.
- At n=1 the native op is not on the path at all; n=2 eval uses `p2_scalar_kernel`, ~0.95 ms at 32k.

## Must wait for the H100

- All absolute ms, and the ±3% train difference.
- Whether the H100's ~2× bandwidth changes the gather share.
- Whether Inductor's int8 shift-add kernel fares better on Hopper: the deploy-path slowness may be partly codegen
  quality on this arch.
- The n=1 eval fused-gather vs `embedding_bag` gap.
