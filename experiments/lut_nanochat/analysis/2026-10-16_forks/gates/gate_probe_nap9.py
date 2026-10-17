"""GATE 3a+3b probe for nap=9 (512 cells) on the worker.
- Build the nanochat LUT-FFN at h32/tph32/nap9/proj_bf16, fused=True -> FusedConfidenceLUT.
- Assert fused (not reference / not fallback), r=1536, nap=9, proj dtype bf16 w/ fp32 masters, fwd finite.
- Build the same block with fused=False -> reference ConfidenceLUT, copy weights, compare fused vs reference
  in eval (no dropout) -> max abs diff (docstring: bitwise-equal at eval).
- Compute LUT param count per block x24 (authoritative; all blocks identical) and the projection/cartridge split.
Run inside the worker venv with PYTHONPATH=<nanochat root>, CUDA visible.
"""
import torch
from nanochat.lut_ffn import LUTFFNConfig, LUTFeedForward
from spiky.lutorch_ex import FusedConfidenceLUT, ConfidenceLUT

torch.manual_seed(0)
D_MODEL = 1536

def build(fused):
    cfg = LUTFFNConfig(enabled=True, h=32, tph=32, nap=9, proj_bf16=True, fused=fused)
    assert cfg.nap == 9 and cfg.fused is fused and cfg.proj_bf16 is True
    return cfg, LUTFeedForward(D_MODEL, cfg, device="cuda")

cfg, m = build(True)
cart = m.mhl.cartridge
print("CART_CLASS           =", type(cart).__name__)
print("IS_FUSED             =", isinstance(cart, FusedConfidenceLUT))
print("NAP                  =", cart.spec.nap, " -> cells =", 2 ** cart.spec.nap)
pdt = getattr(m.mhl, "projection_dtype", None)
cw = m.mhl.compress.weight.dtype
dw = m.mhl.decompress.weight.dtype
r_out = m.mhl.compress.out_features
print("PROJ_DTYPE           =", pdt)
print("COMPRESS_MASTER_DTYPE=", cw, " DECOMPRESS_MASTER_DTYPE=", dw)
print("R (compress out)     =", r_out, " expected 1536")
print("TABLE_DTYPE          =", cart.weights.dtype)

x = torch.randn(4, 8, D_MODEL, device="cuda")
m.eval()
with torch.no_grad():
    y = m(x)
print("OUT_SHAPE            =", tuple(y.shape))
print("FWD_FINITE           =", bool(torch.isfinite(y.float()).all().item()))

# ---- fused vs reference numeric agreement at nap=9 ----
cfg_r, m_ref = build(False)
rcart = m_ref.mhl.cartridge
print("REF_CART_CLASS       =", type(rcart).__name__, " is_reference_ConfidenceLUT=", isinstance(rcart, ConfidenceLUT) and not isinstance(rcart, FusedConfidenceLUT))
missing, unexpected = m_ref.load_state_dict(m.state_dict(), strict=False)
print("STATE_COPY missing   =", list(missing))
print("STATE_COPY unexpected=", list(unexpected))
m_ref.eval()
with torch.no_grad():
    y_ref = m_ref(x)
diff = (y.float() - y_ref.float()).abs()
print("FUSED_vs_REF_MAXABS  =", float(diff.max().item()))
print("FUSED_vs_REF_MEANABS =", float(diff.mean().item()))
print("REF_FWD_FINITE       =", bool(torch.isfinite(y_ref.float()).all().item()))

# ---- param counts (one block x24, blocks identical) ----
def count(mod):
    return sum(p.numel() for p in mod.parameters())
block_total = count(m)
cart_params = count(cart)
proj_params = count(m.mhl.compress) + count(m.mhl.decompress)
print("PER_BLOCK_TOTAL      =", block_total)
print("  cartridge          =", cart_params)
print("  projections        =", proj_params)
print("LUT_FFN_TOTAL_x24    =", block_total * 24)
print("  (nap=8 fork was 415,309,872; expect ~=", 24 * block_total, ")")

ok = (isinstance(cart, FusedConfidenceLUT) and cart.spec.nap == 9 and pdt == torch.bfloat16
      and cw == torch.float32 and dw == torch.float32 and r_out == 1536
      and bool(torch.isfinite(y.float()).all().item())
      and float(diff.max().item()) < 1e-2)
print("GATE3A_PASS          =", ok)
print("PROBE_DONE")
