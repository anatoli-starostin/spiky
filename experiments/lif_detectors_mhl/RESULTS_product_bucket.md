# ProductBucketLIFMHL — mixed-radix product-of-detectors sweep

`ProductBucketLIFMHL` (module `spiky.lutorch.product_bucket_lif_mhl`) generalizes the single-detector
`BucketLIFDetectorsMHL`. Each head has **N_det independent M-way LIF bucket detectors** (each with its own
input weights, delay, tau, and trainable strictly-increasing bucket boundaries). The N_det per-detector
bucket digits form a **mixed-radix index into a table of M^N_det joint cells**:

- **HARD forward:** per detector `d`, hard bucket `b_d = searchsorted(t_hard_d, boundaries_d)`; joint index
  `idx = Σ_d b_d · M^(N_det-1-d)` (row-major); gather that cell from the head's `(M^N_det, out)` table.
- **SOFT backward:** each detector has a soft bucket distribution `p_d` (length M). The joint soft
  distribution over the M^N_det grid is the rank-1 tensor product `P = p_0 ⊗ … ⊗ p_{N_det-1}`, evaluated
  **without materializing the outer product** — the `(M,…,M,out)` table is contracted against each `p_d`
  along its axis (N_det sequential einsums), the efficient equivalent of `P·table`. Decoupled
  straight-through: `y_hard` (full table grad → the selected cell) `+ y_soft − y_soft.detach()` with
  `table.detach()` on the soft path (address grad → detectors); forward == hard. Heads sum into out=6.

Hyperplane LUTs are the **M=2** case; the plain bucket model is the **N_det=1** case. `M^N_det` is capped at
4096 cells/head.

Reuses the bucket conventions verbatim: bounded excitatory `w = 2·sigmoid(w_raw)` (hot init), per-detector
delay, `tau = softplus+1.0` (floor 1.0), the O(N) cumsum first-spike membrane, and partition-of-unity soft
bucket membership.

## Sweep (standard distill protocol: frozen int4 Walker2d LUT teacher, `st` train, 6000 steps, batch 256, Adam 3e-3, grad-clip 1.0; eval `hard` on 4096 samples)

| config (M / N_det / heads) | params | R² | RMSE% (of 9.14) | cell-util | dead cells | eff cells/head | detector NMI init→trained |
|---|---|---|---|---|---|---|---|
| M=4 / N_det=3 / heads=8   | 4,008  | 0.375 | 9.39% | 0.55 | 45% | 16/64 | 0.221 → 0.002 |
| M=4 / N_det=3 / heads=32  | 16,032 | 0.503 | 8.38% | 0.45 | 55% | 14/64 | 0.201 → 0.003 |
| M=2 / N_det=6 / heads=8   | 4,848  | 0.429 | 8.98% | 0.91 | 9%  | 40/64 | 0.221 → 0.005 |
| M=2 / N_det=6 / heads=16  | 9,696  | 0.496 | 8.44% | 0.93 | 7%  | 41/64 | 0.217 → 0.003 |
| M=2 / N_det=6 / heads=32  | 19,392 | **0.523** | 8.21% | 0.80 | 20% | 29/64 | 0.218 → 0.006 |

Reference baselines: plain bucket **0.418** @ 4,768 params; rollout LIF net **0.502** @ 4,606 params.

## Key findings

- **(a) Binary radix M=2 deep is the sweet spot.** M=2 detectors learn near-balanced splits (per-detector
  bucket occupancy ≈ 0.98), so the six binary digits behave like six hyperplane bits and the joint cells fill
  **densely (7–20% dead)** — versus the M=4-shallow configs (**45–55% dead**). Same 64 cells, far better
  utilization and higher R² at similar param cost (e.g. M=2/N6/heads=8 = 0.429 vs M=4/N3/heads=8 = 0.375).
- **(b) Training decorrelates the detectors regardless of init.** Pairwise normalized MI between a head's
  detectors starts ≈ 0.2 (the built init shares delay/tau/boundaries across detectors) but drops to ≈ 0.005
  after training. An A/B with an independently-diverse init did *not* raise R² (0.494 vs 0.500) — the
  init-correlation is a transient training removes, not a persistent bug.
- **(c) Accuracy climbs with heads, with diminishing returns:** M=2/N_det=6 gives R² 0.429 → 0.496 → 0.523
  for heads 8 / 16 / 32.
- **(d) M=2 N_det=6 heads=32 (0.523) is the best product config** and edges past the rollout net (0.502) — but
  at ~4× the params (19.4k vs 4.6k). The rollout net remains the most parameter-efficient variant; the product
  model is the highest-accuracy *analytic* (non-rollout) option, with M=2 heads=16 (0.496 @ 9.7k) a good knee.

Reproduce: `PYTHONPATH=../../src python distill_walker2d_product.py --heads 32 --n-det 6 --buckets 2`.
