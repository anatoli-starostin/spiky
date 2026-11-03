"""[lut_nanochat] CPU unit test of optim.lut_whiten_step: finite, fixed RMS on hit cells, unhit cells stay put,
the update is much better conditioned than the gradient. Run from nanochat/: python tests/test_lut_whiten.py"""
import torch
from nanochat.optim import lut_whiten_step

torch.manual_seed(0)
G, T, K, d = 2, 4, 64, 48
p = torch.randn(G, T, K, d) * 0.01
hit = (torch.rand(G, T, K) > 0.5)
g = (torch.randn(G, T, K, d) + 5 * torch.randn(G, 1, 1, d) * torch.randn(G, T, K, 1)) * hit.unsqueeze(-1)  # lopsided + unhit cells
st = {}
p0 = p.clone()
u = lut_whiten_step(p, g, st, 1.0, 0.9, 0.3, 1e-3, 1e-3)
assert torch.isfinite(p).all()
rows = u.reshape(G, T, K, d)
assert rows[~hit].abs().sum() == 0, "unhit cells must not move"
rms = rows[hit].pow(2).mean().sqrt().item()
assert abs(rms - 0.3) < 0.03, rms


def cond(x):
    ev = torch.linalg.eigvalsh(x.transpose(1, 2) @ x)
    return (ev[:, -1] / ev[:, 0]).mean().item()


cg, cu = cond(g.reshape(G, -1, d)), cond(u.reshape(G, -1, d))
print("cond(grad^T grad) %.1f -> cond(update^T update) %.1f" % (cg, cu))
assert cu < 0.2 * cg
assert torch.allclose(p0 - p, u.reshape(p.shape), atol=1e-6)
for _ in range(3):
    lut_whiten_step(p, g, st, 1.0, 0.9, 0.3, 1e-3, 1e-3)
assert st['step'] == 4 and torch.isfinite(p).all()
print("ok")
