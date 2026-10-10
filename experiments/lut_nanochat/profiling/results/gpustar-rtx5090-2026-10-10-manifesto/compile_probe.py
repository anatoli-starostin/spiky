"""Does each Manifesto / Confidence cartridge survive a whole-model torch.compile(model, dynamic=False) (nanochat's
call, base_train.py L337), in fp32 and bf16? Counts graph breaks (torch._dynamo.explain) and runs one compiled train
step (forward + backward). Library = main's lutorch_ex (a041d9d2 content). Small geometry; train mode."""
import sys
import traceback

import torch
import torch.nn as nn

import spiky.lutorch_ex as lx
from spiky.lutorch_ex.cartridges.fused_confidence import FusedConfidenceLUT
from spiky.lutorch_ex.lut_spec import LUTSpec

spec = LUTSpec(h_in=8, h_out=8, tph=16, nap=6, d_in=16, d_out=16, anchor_mode="pairs")
CARTS = {
    "ManifestoHardLUT": lambda: lx.ManifestoHardLUT(spec, seed=0),
    "ManifestoSoftLUT": lambda: lx.ManifestoSoftLUT(spec, seed=0),
    "FusedManifestoHardLUT": lambda: lx.FusedManifestoHardLUT(spec, seed=0),
    "FusedManifestoSoftLUT": lambda: lx.FusedManifestoSoftLUT(spec, seed=0),
    "ConfidenceLUT(n=1)": lambda: lx.ConfidenceLUT(spec, seed=0, read_top_n=1, table_dropout_rate=0.2),
    "FusedConfidenceLUT(n=1)": lambda: FusedConfidenceLUT(spec, seed=0, read_top_n=1, table_dropout_rate=0.2),
    "QuantisedConfidenceLUT(n=2)": lambda: lx.QuantisedConfidenceLUT(spec, seed=0, read_top_n=2),
}


class Block(nn.Module):
    """A stand-in for a nanochat block: residual + ProjectionMHL(cartridge) + a dense layer around it."""

    def __init__(self, cart):
        super().__init__()
        self.inp = nn.Linear(128, 128)
        self.lut = lx.ProjectionMHL(cart, d_model=128)
        self.out = nn.Linear(128, 128)

    def forward(self, x):
        h = self.inp(x)
        return x + self.out(self.lut(h.reshape(-1, 128)).reshape(h.shape))


for dtype in (torch.float32, torch.bfloat16):
    for name, mk in CARTS.items():
        torch._dynamo.reset()
        torch.manual_seed(0)
        model = Block(mk()).cuda().to(dtype).train()
        x = torch.randn(4, 64, 128, device="cuda", dtype=dtype)
        tag = f"{str(dtype).split('.')[-1]:8s} {name:28s}"
        try:
            ex = torch._dynamo.explain(model)(x)
            reasons = sorted({str(b.reason).splitlines()[0][:110] for b in ex.break_reasons})
            cm = torch.compile(model, dynamic=False)
            y = cm(x)
            y.float().sum().backward()
            torch.cuda.synchronize()
            print(f"{tag} OK    graphs={ex.graph_count} breaks={ex.graph_break_count}  "
                  + (" | ".join(reasons[:3]) if reasons else ""))
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
            print(f"{tag} FAIL  {msg}")
        sys.stdout.flush()
