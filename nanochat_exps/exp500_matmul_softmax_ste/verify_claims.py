"""Numerical verification of the two load-bearing claims behind exp500's deficit.

(a) Hard-sign softmax is an INVERTIBLE reparametrization of the argmax table:
    the orthant->output map is convolution by the Hamming kernel K[s,k]=rho^h(s,k),
    rho=exp(-2/T_sel). Show K is invertible & well-conditioned at the learned T_sel,
    matching the closed form cond = ((1+rho)/(1-rho))^NAP.

(b) Temperature trap: with HARD (integer) ts, the softmax Jacobian dg/dts -> 0 as
    T_sel -> 0 (saturates, backward dies). With SOFT (continuous) ts, boundary tokens
    keep g non-saturated so the Jacobian stays large at small T_sel (gradient flows).
"""
import math
import torch

NAP = 6
K = 1 << NAP
T_SOFT = 0.45          # exp493/exp500 learned ~0.45
LEARNED_TSEL = 0.51    # exp500 learned mean T_sel
torch.manual_seed(0)

# ---- bit matrix [NAP, K], +-1 (row k's bit pattern), MSB-first like the lib ----
codes = torch.arange(K)
bits = ((codes[None, :] >> torch.arange(NAP - 1, -1, -1)[:, None]) & 1)  # [NAP,K] in {0,1}
bit_matrix = (2 * bits - 1).float()                                       # +-1

# ===================== (a) Hamming-kernel conditioning =====================
print("=" * 70)
print("(a) Hamming kernel K[s,k]=rho^h(s,k): invertibility / conditioning")
print("=" * 70)
sk = torch.arange(K)
hdist = (sk[:, None] ^ sk[None, :])                                       # XOR
hdist = torch.tensor([[bin(int(v)).count('1') for v in row] for row in hdist]).float()
print(f"{'T_sel':>7} {'rho':>9} {'cond(K)':>12} {'closed-form':>12} {'min_sv':>10}  invertible?")
for T in (0.10, 0.24, LEARNED_TSEL, 0.72, 1.0, 2.0, 5.0):
    rho = math.exp(-2.0 / T)
    Kmat = rho ** hdist
    sv = torch.linalg.svdvals(Kmat)
    cond = (sv.max() / sv.min()).item()
    closed = ((1 + rho) / (1 - rho)) ** NAP
    print(f"{T:>7.2f} {rho:>9.4f} {cond:>12.3f} {closed:>12.3f} {sv.min():>10.4f}  "
          f"{'YES' if sv.min() > 1e-8 else 'NO'}")
print("-> K invertible at every T_sel>0 (det>0); same function class as argmax.")
print(f"-> at learned T_sel={LEARNED_TSEL}: cond≈{((1+math.exp(-2/LEARNED_TSEL))/(1-math.exp(-2/LEARNED_TSEL)))**NAP:.2f} (mild) -- conditioning is NOT the killer.\n")

# ===================== (b) Temperature trap in the backward =====================
print("=" * 70)
print("(b) Softmax Jacobian ||dg/dts||_F vs T_sel:  HARD (integer ts) vs SOFT")
print("=" * 70)
N = 200_000
sigma = 0.314                                  # gives median|p_soft|~0.32 at T_soft=0.45
d = torch.randn(N, NAP) * sigma
p_soft = d / (T_SOFT + d.abs())
p_hard = torch.where(d > 0, 1.0, -1.0)
ts_hard = p_hard @ bit_matrix                  # [N,K] integer-valued
ts_soft = p_soft @ bit_matrix                  # [N,K] continuous
_fr = (ts_hard - ts_hard.round()).abs().max().item()
print(f"(sanity) median|p_soft|={p_soft.abs().median():.3f}  "
      f"ts_hard integer? max|frac|={_fr:.1e}  "
      f"ts_soft continuous (std={ts_soft.std():.2f})")

def jac_fnorm(ts, T):
    g = torch.softmax(ts / T, dim=-1)          # [N,K]
    S2 = (g * g).sum(-1); S3 = (g * g * g).sum(-1)
    # ||diag(g)-gg^T||_F^2 = S2 - 2 S3 + S2^2 ; J = (1/T)(diag(g)-gg^T)
    fro = torch.sqrt((S2 - 2 * S3 + S2 * S2).clamp_min(0)) / T
    return fro.mean().item()

print(f"\n{'T_sel':>7} {'HARD ||J||_F':>14} {'SOFT ||J||_F':>14} {'soft/hard':>10}")
for T in (0.05, 0.10, 0.20, 0.37, LEARNED_TSEL, 1.0, 2.0):
    h = jac_fnorm(ts_hard, T); s = jac_fnorm(ts_soft, T)
    print(f"{T:>7.2f} {h:>14.4f} {s:>14.4f} {s/max(h,1e-12):>10.2f}")
print("-> HARD: ||J||_F collapses toward 0 as T_sel->0 (one-hot saturation = dead backward).")
print("-> SOFT: ||J||_F stays large / grows as T_sel->0 (boundary tokens keep g non-saturated).")
print("-> the model must keep T_sel moderate to get gradient with hard ts -> forced forward blur.")
