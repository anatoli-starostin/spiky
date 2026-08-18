# exp_n_0032 — H16/d24/tph128, ORTHOGONAL compress init, tied, 16k (A/B vs exp_n_0031)

IDENTICAL to **exp_n_0031** (H=16, inner d=24, tph=128, nap=6, tied, vanilla backbone, 16k) EXCEPT the
per-head **compress matrices use an ORTHOGONAL-across-heads init** instead of N(0,0.02²). The decisive A/B
on whether orthogonal-compress-init helps loss.

**Init:** after the standard trainer init, the compress weight [n_heads·d, 384] = [384,384] is re-init'd
via `nn.init.orthogonal_(w, gain=0.02·√384)` (config flag `lut_compress_orthogonal=True`). This makes the
16 per-head 24-dim row-subspaces **mutually orthogonal and jointly tile R^384** (verified: pairwise
projection overlap = 0.00000), while **matching the N(0,0.02²) Frobenius/RMS scale** (verified: entry
std 0.0200, ‖W‖_F 7.680 = the Gaussian target) so activation magnitudes stay comparable — a clean isolation
of the SUBSPACE geometry, not scale. (Prior diagnostic: std0.02 gives random-diverse overlap ~0.062;
orthogonal → 0.)

**Params = 36,780,288 (SMOKE-confirmed), identical to exp_n_0031.** FLOPs identical (orthogonal init is
just a different weight init — no matmul change); H·d=384 → ~4× cheaper FFN than dense.

Runs 16k, serial after exp_n_0031. Compare: (a) vs exp_n_0031 (std0.02 twin) = the orthogonal-init A/B;
(b) vs exp_n_0004 (1.21738); (c) vs tied dense 16k (1.19665).
