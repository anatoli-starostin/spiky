// The per-table scalars of the power-of-two read -- THE single definition of the forward integers.
//
// Included by csrc/pow2_int8_read.cu and called, with the same compile flags (--fmad=false: no contraction anywhere), by
//   * p2_scalar_kernel  -- the forward of the spiky_lutorch::p2_scalars custom op (the training forward), and
//   * p2_int8_kernel    -- the inference read (read_fused), inline in the same launch as the int8 accumulation.
// Because both call this one function on the same margins d = z[a] - z[b], training and inference take the same c1, c2,
// q and k' by construction. (The torch implementation in pow2_read.py computes the same quantities and remains the fallback
// when the extension is unavailable; it is not bit-identical to this code -- torch.compile's float32 reduction order and
// libdevice differ -- which is exactly why the CUDA build uses this function for every path it serves.)
//
// For one table with margins d[0..nap-1] (note Sections 2-4):
//   m_p = |d_p|,  c1 = sum_p [d_p > 0] 2^(nap-1-p),  mj = first argmin_p m_p,  mv = m_mj,  c2 = c1 ^ 2^(nap-1-mj)
//   S = sum_p m_p (left fold),  ls = sum_p logsigmoid(beta m_p) (left fold),  logsigmoid(x) = min(0, x) - log1p(exp(-|x|))
//   q = clamp(floor(mv * 2 / (tau ln2) + 1/2), 0, 64)
//   c_q = log2(1 + 2^-q) for q < 8, else 0
//   k' = floor(log2 S + (g + gamma ls) / ln2 - c_q + 1/2);  skip = !(k' >= lo);  kc = clamp(k', lo, hi);  drop = q > Q
//   sh1 = skip ? 15 : kc + 6;  sh2 = (skip || drop) ? 15 : kc + 6 - q
#pragma once

#include <cmath>
#include <cstdint>

namespace p2 {

constexpr uint8_t DISCARD = 15;
constexpr float LN2 = 0.6931471805599453f;

struct TableScalars {
  uint8_t c1, c2, sh, mj;   // addresses, shift byte sh1 | sh2 << 4, argmin anchor pair
  int8_t kc, q;             // clamped k' and q (q <= 64): the STE ratios need them also for skipped / dropped cells
  bool skip, drop;
};

__host__ __device__ inline TableScalars table_scalars(const float* d, int nap, float tau, float g, float beta, float gamma,
                                                      int lo, int hi, int Q) {
  int c1 = 0, mj = 0;
  float S = 0.f, ls = 0.f, mv = 0.f;
  for (int p = 0; p < nap; ++p) {
    const float dp = d[p];
    if (dp > 0.f) c1 |= (1 << (nap - 1 - p));
    const float m = std::fabs(dp);
    S = S + m;
    const float x = beta * m;
    ls = ls + (std::fmin(0.f, x) - std::log1p(std::exp(-std::fabs(x))));
    if (p == 0 || m < mv) { mv = m; mj = p; }
  }
  const float inv = 2.0f / (tau * LN2);
  const float qf = std::fmin(std::fmax(std::floor(mv * inv + 0.5f), 0.f), 64.f);
  const int q = (int)qf;
  const float cq = (q < 8) ? std::log2(1.0f + std::ldexp(1.0f, -q)) : 0.f;
  const float kr = std::floor(std::log2(S) + (g + gamma * ls) / LN2 - cq + 0.5f);
  TableScalars r;
  r.skip = !(kr >= (float)lo);                                      // also true for NaN / -inf (S == 0)
  const int kc = r.skip ? lo : (kr > (float)hi ? hi : (int)kr);
  r.drop = q > Q;
  r.c1 = (uint8_t)c1;
  r.c2 = (uint8_t)(c1 ^ (1 << (nap - 1 - mj)));
  r.mj = (uint8_t)mj;
  r.kc = (int8_t)kc;
  r.q = (int8_t)q;
  const uint8_t sh1 = r.skip ? DISCARD : (uint8_t)(kc + 6);
  const uint8_t sh2 = (r.skip || r.drop) ? DISCARD : (uint8_t)(kc + 6 - q);
  r.sh = sh1 | (sh2 << 4);
  return r;
}

}  // namespace p2
