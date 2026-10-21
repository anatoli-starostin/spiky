"""Time one forward+backward of the delta mixers at recall-task sizes (MPS). Usage: python prof_delta.py"""
import sys, time, torch
from models import MIXERS
dev = "mps"
for B, T in [(64, 256), (64, 1024)]:
    for name, kw in [("linear_delta", dict(feat=256)), ("lut_delta", dict(nap=6, feat=256)),
                     ("lut_delta", dict(nap=10, feat=4096)), ("lut_delta", dict(nap=10, feat=4096, chunk=64))]:
        m = MIXERS[name](d=128, H=4, bitflip=2, seed=0, **kw).to(dev)
        x = torch.randn(B, T, 128, device=dev, requires_grad=True)
        for i in range(3):
            if i == 1:
                torch.mps.synchronize(); t = time.time()
            m(x).square().mean().backward()
        torch.mps.synchronize()
        print(f"B={B} T={T} {name} {kw}: {(time.time() - t) / 2:.3f} s per fwd+bwd (one layer)", flush=True)
