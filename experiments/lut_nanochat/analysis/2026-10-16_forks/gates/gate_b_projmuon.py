"""GATE (b) for projmuonfp8-lrrs (worker GPU): v2 geometry with proj_fp8_muon=True.
- cartridge is FusedConfidenceLUT / IS_FUSED (no reference fallback), forward finite, backward finite;
- the projections really run fp8 (ProjectionMHL fp8_projections), and the fp8 output is numerically sane vs an
  fp32-projection twin with identical weights (decompress randomised: it is zero-init)."""
import torch
from nanochat.lut_ffn import LUTFFNConfig, LUTFeedForward
from spiky.lutorch_ex import FusedConfidenceLUT

torch.manual_seed(0)
D = 1536
m8 = LUTFeedForward(D, LUTFFNConfig(enabled=True, proj_fp8_muon=True), device="cuda")
cart = m8.mhl.cartridge
print("CARTRIDGE_CLASS =", type(cart).__name__, "| IS_FUSED =", isinstance(cart, FusedConfidenceLUT),
      "| read_top_n =", cart.read_top_n, "| nap =", cart.spec.nap)
print("fp8_projections =", getattr(m8.mhl, "fp8_projections", "?"),
      "| compress_fp8_out_dtype =", getattr(m8.mhl, "compress_fp8_out_dtype", "?"))
print("master dtypes: compress W", m8.mhl.compress.weight.dtype, "decompress W", m8.mhl.decompress.weight.dtype)
print("_lut_to_muon tags:", {n: getattr(p, "_lut_to_muon", False) for n, p in m8.mhl.named_parameters() if "compress" in n})
with torch.no_grad():
    m8.mhl.decompress.weight.normal_(0, 0.02)
    m8.mhl.decompress.bias.normal_(0, 0.02)
m32 = LUTFeedForward(D, LUTFFNConfig(enabled=True), device="cuda")
print("state copy:", m32.load_state_dict(m8.state_dict(), strict=False))
m8.eval(); m32.eval()
x = torch.randn(4, 64, D, device="cuda")
with torch.no_grad():
    y8, y32 = m8(x), m32(x)
rel = ((y8 - y32).norm() / y32.norm()).item()
print("FWD_FINITE =", bool(torch.isfinite(y8).all()), "| fp8-vs-fp32-proj rel L2 err = %.3e | max|diff| %.3e (max|y| %.3e)"
      % (rel, (y8 - y32).abs().max(), y32.abs().max()))
m8.train()
xi = x.clone().requires_grad_(True)
yt = m8(xi); (yt * torch.randn_like(yt)).sum().backward()
print("TRAIN fwd/bwd finite =", bool(torch.isfinite(yt).all()), bool(torch.isfinite(xi.grad).all()),
      all(bool(torch.isfinite(p.grad).all()) for p in m8.parameters() if p.grad is not None))
print("GATE_B_PASS =", isinstance(cart, FusedConfidenceLUT) and bool(torch.isfinite(y8).all()) and rel < 0.1)
