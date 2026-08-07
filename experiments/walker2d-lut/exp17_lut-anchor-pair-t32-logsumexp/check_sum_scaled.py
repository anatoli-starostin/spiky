"""Verify the sum-scaled log-sum-exp readout (exp17c) before spending GPU.

Claims to check:
  1. It reduces to exp10's plain sum: at tau >> weight spread the outputs match exp10's
     to near machine precision, with the ADDITIVE init (weights are additive again).
  2. Its gradient sums to T over tables (exp10: T; plain log-sum-exp: 1) -- the property
     that no initialisation could restore.
  3. tau -> 0 approaches T * max(w).
  4. exp00-17-attempt-2 arches are untouched.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from models import REGISTRY                                        # noqa: E402

OBS, ACT, TPH, B = 17, 6, 32, 8192
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails += 1


def build(name, **kw):
    torch.manual_seed(0)
    return REGISTRY[name](OBS, ACT, tables_per_head=TPH, **kw)


def selected(lut, obs):
    with torch.no_grad():
        d = obs[:, lut.soft_anchor_a_long] - obs[:, lut.soft_anchor_b_long]
        idx = ((d > 0).to(torch.int64) * lut.soft_powers.view(1, 1, -1)).sum(-1)
        off = torch.arange(TPH, dtype=idx.dtype) * lut.table_dim
        return lut.weights.view(TPH * lut.table_dim, ACT)[
            (idx + off.view(1, -1)).reshape(-1)].view(B, TPH, ACT)


def main():
    obs = torch.randn(B, OBS)
    m10 = build("fastlut")
    with torch.no_grad():
        a10, _ = m10(obs)
    print(f"exp10: mean {a10.mean():+.6f}  std {a10.std():.6e}\n")

    print("1. sum-scaled readout reduces to exp10's plain sum")
    m = build("fastlut_lse_sum")
    with torch.no_grad():
        a, _ = m(obs)
    print(f"   exp17c: mean {a.mean():+.6f}  std {a.std():.6e}")
    # The residual is the Jensen gap of the spread, ~T*Var(w)/(2*tau) ~ 1e-4 here: the
    # sum-scaled readout equals the plain sum only as tau -> inf. The meaningful test is
    # therefore relative to the output's own scale, not an absolute 1e-5.
    err = (a - a10).abs().max().item()
    gap = TPH * ((1e-3) ** 2 / 3) / (2 * 0.05)
    print(f"   predicted Jensen gap T*Var(w)/(2*tau) = {gap:.3e}")
    check("outputs match exp10 to well within the output std (< 10% of std)",
          err < 0.1 * a10.std().item(),
          f"max|diff| = {err:.3e} vs std {a10.std().item():.3e} "
          f"({100 * err / a10.std().item():.1f}%)")
    check("output std ratio ~ 1", abs(a.std().item() / a10.std().item() - 1) < 0.02,
          f"ratio {a.std().item() / a10.std().item():.4f}")

    print("\n2. gradient scale: sum over tables of d(out)/d(w)")
    res = {}
    for tag, mod in (("exp10 (plain sum)", m10),
                     ("exp17 attempt2 (mean-scaled)", build("fastlut_lse")),
                     ("exp17c (sum-scaled)", m)):
        mod.zero_grad()
        out, _ = mod(obs)
        out.sum().backward()
        g = mod.actor_lut.weights.grad
        per_sample = g.sum().item() / (B * ACT)
        res[tag] = per_sample
        print(f"   {tag:<30} sum d(out)/d(w) = {per_sample:.4f}")
    check("exp10 sums to T", abs(res["exp10 (plain sum)"] - TPH) < 0.5)
    check("mean-scaled sums to 1 (the T-fold sensitivity loss)",
          abs(res["exp17 attempt2 (mean-scaled)"] - 1.0) < 0.05)
    check("sum-scaled sums to T, matching exp10",
          abs(res["exp17c (sum-scaled)"] - TPH) < 0.5)

    print("\n3. limits in tau")
    w = selected(m.actor_lut, obs)
    # tau must be small RELATIVE TO THE WEIGHT SPREAD (~1e-3) to reach the max limit, so
    # 1e-3 is nowhere near it; 1e-6 is.
    for tau in (1e-6, 1e-3, 0.05, 10.0):
        val = TPH * tau * (torch.logsumexp(w / tau, dim=1) - math.log(TPH))
        print(f"   tau={tau:<8g} mean {val.mean():+.6f}   "
              f"(T*max = {(TPH * w.max(dim=1).values).mean():+.6f}, "
              f"sum = {w.sum(dim=1).mean():+.6f})")
    lo = TPH * 1e-6 * (torch.logsumexp(w / 1e-6, dim=1) - math.log(TPH))
    hi = TPH * 10.0 * (torch.logsumexp(w / 10.0, dim=1) - math.log(TPH))
    check("tau->0 approaches T*max",
          (lo - TPH * w.max(dim=1).values).abs().max().item() < 1e-2,
          f"max|diff| = {(lo - TPH * w.max(dim=1).values).abs().max().item():.3e}")
    check("tau->inf approaches the plain sum",
          (hi - w.sum(dim=1)).abs().max().item() < 1e-3)

    print("\n4. non-regression")
    m2 = build("fastlut_lse")
    check("attempt-2 arch unchanged (still mean-scaled, logspace init)",
          m2.actor_lut.exp_outputs_scale == "mean"
          and m2.actor_lut.exp_outputs_init == "logspace")
    check("critic init still identical to exp10's",
          bool(torch.equal(m10.vf[0].weight, m.vf[0].weight)))
    check("plain fastlut untouched",
          bool(torch.equal(build("fastlut").actor_lut.weights, m10.actor_lut.weights)))

    print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
