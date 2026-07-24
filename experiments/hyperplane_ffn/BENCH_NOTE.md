# HyperplaneMultiHeadLUT — front-end cost note

Comparison of the new **learned-hyperplane** index front-end
(`HyperplaneMultiHeadLUT`) against the **fixed anchor-pair** front-end
(`FastMultiHeadLut`) it generalizes. Measured on an **NVIDIA H100 80GB HBM3**,
torch 2.13.0+cu130, via `bench_hyperplane_frontend.py`.

The generalization replaces the cheap two-coordinate gather + subtract
(`x[a] - x[b]`) with a dense affine projection
`a = x @ Wᵀ + b`, a `[B, d_model] × [n_tables·NAP, d_model]` GEMM. That GEMM is
the new dominant FLOP of the front-end.

## Front-end only (index computation, forward), ms/call

| B | d_model | n_tables | NAP | anchor | hyperplane | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 384 | 512 | 6 | 0.223 | 0.329 | 1.5× |
| 4096 | 384 | 3072 | 6 | 1.238 | 1.865 | 1.5× |
| 16384 | 384 | 512 | 6 | 0.833 | 1.249 | 1.5× |
| 4096 | 384 | 1536 | 4 | 0.428 | 0.642 | 1.5× |

In isolation the hyperplane projection costs a steady **~1.5×** the anchor
gather. It is bandwidth-bound (streams `W`, `[n_tables·NAP, d_model]` bf16), not
compute-bound, so the ratio is flat across B and table count.

## Full module, forward + backward (fp32 weights, bf16 autocast), ms/call

| module | B | n_heads | tph | NAP | n_out | mode | fast | hyperplane | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| residual_lut | 4096 | 1 | 512 | 6 | 384 | hard | 5.90 | 5.49 | 0.9× |
| residual_lut | 4096 | 1 | 512 | 6 | 384 | hybrid_smooth | 12.35 | 19.03 | 1.5× |
| qk_lut | 4096 | 6 | 512 | 6 | 128 | hard | 20.62 | 17.55 | 0.9× |
| v_lut | 4096 | 6 | 512 | 4 | 16 | hard | 8.79 | 6.79 | 0.8× |

Takeaways:

- **hard mode**: the projection GEMM is fully hidden under the LUT gather /
  `embedding_bag` reduce and the soft-surrogate backward — the hyperplane
  module is **at parity or slightly faster** (0.8–0.9×) at these shapes. (Note:
  `FastMultiHeadLut`'s anchor-specific `lutorch_cuda` bit-pack kernel only fires
  on the no-grad eval path; the train path both modules time here uses the
  compiled body, so this is an apples-to-apples train comparison.)
- **hybrid_smooth**: the projection is a larger share of the smaller top-2
  forward, so the front-end's ~1.5× shows through to **~1.5×** overall.
- The extra parameters (`w`: `[n_tables, NAP, d_model]`, `b`: `[n_tables, NAP]`)
  are the inherent cost of learned hyperplanes vs stored anchor indices —
  e.g. residual_lut adds `512·6·384 ≈ 1.18M` weights (~4.7 MB fp32).

**Bottom line:** the learned front-end is affordable — it disappears under the
LUT machinery in the hard train/eval path and costs ~1.5× only in the lean
hybrid_smooth forward. The primitive is ready for the A/B experiment (fixed
anchors vs learned hyperplanes) against the 1.20144 bpb anchor; whether learned
hyperplanes *win* on val bpb is the follow-up, not part of issue #64.
