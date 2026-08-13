"""exp_c03 — the LUT policy: HyperplaneMultiHeadLUT as a Walker2d controller (#75).

obs (17) -> [standardise] -> HyperplaneMultiHeadLUT -> [tanh] -> action (6) in [-1,1]

Notes on the two wrappers:

* **Standardisation is not cosmetic.** The LUT's index bits are comparisons *between
  different input coordinates* (anchor pairs) or affine hyperplanes over them, so the
  relative scale of the 17 observation dims changes which rows are addressed. Walker2d
  observations are unbounded and wildly different in scale (z-height ~1.25 vs joint
  velocities up to ±10), so the standardiser is fixed at dataset statistics and saved
  with the model — it is part of the policy.
* **No squashing on the output.** The obvious choice is tanh, but it is actively wrong
  here: the teacher's clipped action sits *exactly* at ±1 about 63% of the time, and
  tanh reaches ±1 only asymptotically — fitting those targets by MSE would demand
  ever-growing pre-activations and the gradient dies. A LUT stores arbitrary reals, so
  the head emits raw values and the environment's own clip(a, -1, 1) does the bounding
  (`eval_lut.py` clips identically). This is the same clip the teacher is subject to.
"""
import os
import torch
import torch.nn as nn

from spiky.lutorch.hyperplane_multi_head_lut import HyperplaneMultiHeadLUT
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut

OBS, ACT = 17, 6


class LUTPolicy(nn.Module):
    def __init__(self, n_anchor_pairs=8, tables_per_head=64, n_heads=1,
                 module="hyperplane", forward_mode="hybrid_smooth",
                 obs_mean=None, obs_std=None, device="cuda", seed=0):
        super().__init__()
        self.cfg = dict(n_anchor_pairs=n_anchor_pairs,
                        tables_per_head=tables_per_head, n_heads=n_heads,
                        module=module, forward_mode=forward_mode, seed=seed)
        kw = dict(input_dim=OBS, n_heads=n_heads, n_outputs=ACT,
                  n_anchor_pairs=n_anchor_pairs, tables_per_head=tables_per_head,
                  forward_mode=forward_mode, weight_dtype=torch.float32,
                  initial_weights_noise=0.01, learnable_temps=True,
                  random_seed=seed, device=device)
        if module == "hyperplane":
            # anchor_pairs init == FastMHL bit-for-bit, then the hyperplanes learn
            self.lut = HyperplaneMultiHeadLUT(hyperplane_init="anchor_pairs",
                                              hyperplane_dtype=torch.float32,
                                              use_bf16=False, **kw)
        else:
            self.lut = FastMultiHeadLut(use_bf16=False, **kw)
        self.register_buffer("obs_mean",
                             torch.zeros(OBS) if obs_mean is None
                             else torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_std",
                             torch.ones(OBS) if obs_std is None
                             else torch.as_tensor(obs_std, dtype=torch.float32))

    def forward(self, obs):
        x = (obs - self.obs_mean) / (self.obs_std + 1e-6)
        y = self.lut(x)                      # [B, n_heads, ACT]
        return y.sum(dim=1)                  # heads summed -> [B, ACT]; env clips

    # ---- accounting -------------------------------------------------------
    def table_params(self):
        return self.lut.weights.numel()

    def index_params(self):
        n = 0
        for name in ("hyperplane_weight", "hyperplane_bias"):
            p = getattr(self.lut, name, None)
            if p is not None:
                n += p.numel()
        return n

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self):
        c = self.cfg
        return (f"{c['module']} NAP={c['n_anchor_pairs']} "
                f"tph={c['tables_per_head']} heads={c['n_heads']} "
                f"rows={2**c['n_anchor_pairs']} "
                f"table={self.table_params():,} idx={self.index_params():,} "
                f"total={self.n_params():,}")


def save(model, path):
    torch.save(dict(cfg=model.cfg, state_dict=model.state_dict(),
                    obs_mean=model.obs_mean.cpu(), obs_std=model.obs_std.cpu()), path)


def load(path, device="cuda"):
    d = torch.load(path, map_location=device, weights_only=False)
    m = LUTPolicy(obs_mean=d["obs_mean"], obs_std=d["obs_std"], device=device,
                  **d["cfg"])
    m.load_state_dict(d["state_dict"])
    return m.to(device).eval()
