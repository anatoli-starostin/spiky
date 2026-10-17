"""GATE (a) for confn2: v2 geometry (h16/tph64/d48, nap8, fp32 projections) at read_top_n=2.
Fused vs reference ConfidenceLUT: forward AND backward. decompress is zero-init in LUTFeedForward, which
would make every upstream cartridge gradient exactly 0 (a vacuous backward check) -> randomise it first,
identically in both modules, via the state_dict copy."""
import torch
from nanochat.lut_ffn import LUTFFNConfig, LUTFeedForward
from spiky.lutorch_ex import FusedConfidenceLUT, ConfidenceLUT

D = 1536
torch.manual_seed(0)

def build(fused):
    cfg = LUTFFNConfig(enabled=True, read_top_n=2, fused=fused)  # everything else = v2 defaults
    assert (cfg.h, cfg.d, cfg.tph, cfg.nap, cfg.read_top_n) == (16, 48, 64, 8, 2)
    assert cfg.proj_bf16 is False and cfg.proj_fp8_muon is False
    return LUTFeedForward(D, cfg, device="cuda")

mf = build(True)
cart = mf.mhl.cartridge
print("CART_CLASS        =", type(cart).__name__)
print("IS_FUSED          =", isinstance(cart, FusedConfidenceLUT))
print("read_top_n        =", cart.read_top_n if hasattr(cart, "read_top_n") else "?")
sp = cart.spec
print("spec              = h_in", sp.h_in, "d_in", sp.d_in, "tph", sp.tph, "nap", sp.nap,
      "anchor_mode", getattr(sp, "anchor_mode", getattr(cart, "anchor_mode", "?")))
print("has log_tau param =", [n for n, _ in cart.named_parameters()])
print("PROJ dtype        =", getattr(mf.mhl, "projection_dtype", None), "| compress W", mf.mhl.compress.weight.dtype)

with torch.no_grad():                                   # non-vacuous backward: decompress != 0
    mf.mhl.decompress.weight.normal_(0, 0.02)
    if mf.mhl.decompress.bias is not None:
        mf.mhl.decompress.bias.normal_(0, 0.02)

mr = build(False)
rcart = mr.mhl.cartridge
print("REF_CLASS         =", type(rcart).__name__, "(reference:", isinstance(rcart, ConfidenceLUT) and not isinstance(rcart, FusedConfidenceLUT), ")")
miss, unexp = mr.load_state_dict(mf.state_dict(), strict=False)
print("state copy missing/unexpected =", list(miss), list(unexp))

mf.eval(); mr.eval()                                    # no table dropout -> identical maths
x = torch.randn(4, 64, D, device="cuda")
g = torch.randn(4, 64, D, device="cuda")

def run(m):
    for p in m.parameters():
        p.grad = None
    xi = x.clone().requires_grad_(True)
    y = m(xi)
    (y * g).sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    return y.detach(), xi.grad.detach(), grads

yf, gxf, gf = run(mf)
yr, gxr, gr = run(mr)
print("FWD_FINITE        =", bool(torch.isfinite(yf).all()), "| ref", bool(torch.isfinite(yr).all()))
print("FWD max|diff|     = %.3e  (max|y| %.3e)" % ((yf - yr).abs().max(), yr.abs().max()))

def cmp(name, a, b):
    d = (a - b).abs().max().item(); s = b.abs().max().item()
    print("  %-44s max|diff| %.3e  max|ref| %.3e  rel %.2e  finite %s" % (name, d, s, d / s if s else float("nan"),
                                                                         bool(torch.isfinite(a).all())))
    return d / s if s else 0.0

print("BACKWARD (fused vs reference):")
worst = cmp("grad x", gxf, gxr)
missing = set(gr) ^ set(gf)
print("  params with grad only on one side:", sorted(missing))
for n in sorted(gr):
    if n in gf:
        worst = max(worst, cmp(n, gf[n], gr[n]))
print("WORST_REL_GRAD    = %.2e" % worst)

# train-mode smoke at n=2 (table dropout on, RNG streams differ by design -> finiteness only)
mf.train()
xi = x.clone().requires_grad_(True)
yt = mf(xi); (yt * g).sum().backward()
print("TRAIN fwd/bwd finite =", bool(torch.isfinite(yt).all()), bool(torch.isfinite(xi.grad).all()),
      all(bool(torch.isfinite(p.grad).all()) for p in mf.parameters() if p.grad is not None))
print("PER_BLOCK params  =", sum(p.numel() for p in mf.parameters()), "-> x24 =", 24 * sum(p.numel() for p in mf.parameters()))
ok = isinstance(cart, FusedConfidenceLUT) and bool(torch.isfinite(yf).all()) and (yf - yr).abs().max().item() < 1e-4 \
     and worst < 1e-3 and not missing
print("GATE_A_PASS       =", ok)
