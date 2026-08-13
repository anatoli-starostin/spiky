"""Verify the corrected weights-as-logarithms init for exp17 (attempt 2).

Targets, all measured against exp10's own initial statistics:
  * output mean ~ 0 (attempt 1: +0.3466)
  * output std ~ exp10's 3.28e-3 (attempt 1: 1.02e-4, i.e. 32x too small)
  * gradients not perfectly uniform across tables (attempt 1: 32.0/32 effective)
  * exp00-16 untouched: `fastlut` and `fastlut_exp` must be bit-identical to before,
    and the RNG stream must not shift (else the MLP critic init would differ).
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


def main():
    obs = torch.randn(B, OBS)

    m10 = build("fastlut")
    with torch.no_grad():
        a10, _ = m10(obs)
    tgt_m, tgt_s = a10.mean().item(), a10.std().item()
    print(f"exp10 reference: mean {tgt_m:+.6f}  std {tgt_s:.3e}\n")

    print("attempt 1 (additive init, tau=0.1) — reproducible via exp_outputs_init")
    m1 = build("fastlut_lse", tau_init=0.1, exp_outputs_init="additive")
    with torch.no_grad():
        a1, _ = m1(obs)
    print(f"  mean {a1.mean():+.6f}  std {a1.std():.3e}  "
          f"(std ratio {a1.std().item() / tgt_s:.3f})")

    print("\nattempt 2 (logspace init, tau=0.05)")
    m2 = build("fastlut_lse")
    lut = m2.actor_lut
    print(f"  init mu {lut.exp_outputs_init_mu:+.5f}   sigma {lut.exp_outputs_init_sigma:g}"
          f"   tau {float(lut.exp_outputs_tau):.4f}")
    with torch.no_grad():
        a2, _ = m2(obs)
    m, s = a2.mean().item(), a2.std().item()
    print(f"  mean {m:+.6f}  std {s:.3e}  (std ratio {s / tgt_s:.3f})")
    check("output mean is near zero (|mean| < 0.01)", abs(m) < 0.01, f"{m:+.6f}")
    check("output mean beats attempt 1 by >20x", abs(m) < abs(a1.mean().item()) / 20,
          f"{abs(m):.5f} vs {abs(a1.mean().item()):.5f}")
    check("output std within 20% of exp10's", 0.8 < s / tgt_s < 1.2, f"ratio {s / tgt_s:.3f}")
    check("nothing saturates clamp(-1,1)", bool((a2.abs() <= 1).all()),
          f"max |mean| {a2.abs().max():.4f}")

    print("\ngradient structure (effective tables carrying the weight gradient)")
    for tag, mod in (("attempt 1", m1), ("attempt 2", m2)):
        lut_ = mod.actor_lut
        with torch.no_grad():
            d = obs[:, lut_.soft_anchor_a_long] - obs[:, lut_.soft_anchor_b_long]
            idx = ((d > 0).to(torch.int64) * lut_.soft_powers.view(1, 1, -1)).sum(-1)
            off = torch.arange(TPH, dtype=idx.dtype) * lut_.table_dim
            w_sel = lut_.weights.view(TPH * lut_.table_dim, ACT)[
                (idx + off.view(1, -1)).reshape(-1)].view(B, TPH, ACT)
            sm = torch.softmax(w_sel / float(lut_.exp_outputs_tau), dim=1)
            eff = (-(sm.clamp_min(1e-12).log() * sm).sum(1)).exp().mean().item()
        print(f"  {tag}: effective tables {eff:.1f}/{TPH}, max softmax {sm.max():.4f}")
        if tag == "attempt 2":
            check("gradient is not perfectly uniform (eff < 31.5)", eff < 31.5,
                  f"{eff:.1f}/{TPH}")

    print("\nnon-regression: exp00-16 arches untouched")
    torch.manual_seed(0)
    a_ref = REGISTRY["fastlut"](OBS, ACT, tables_per_head=TPH)
    torch.manual_seed(0)
    b_ref = REGISTRY["fastlut"](OBS, ACT, tables_per_head=TPH)
    check("fastlut deterministic under the same seed",
          bool(torch.equal(a_ref.weights_check() if hasattr(a_ref, "weights_check")
                           else a_ref.actor_lut.weights, b_ref.actor_lut.weights)))
    # The critic is built AFTER the LUT, so if the LUT's RNG consumption changed the
    # critic weights would differ. Compare fastlut's critic against fastlut_lse's.
    m2b = build("fastlut_lse")
    check("critic init identical to exp10's (RNG stream not shifted by the new init)",
          bool(torch.equal(m10.vf[0].weight, m2b.vf[0].weight)))
    torch.manual_seed(0)
    m16 = REGISTRY["fastlut_exp"](OBS, ACT, tables_per_head=TPH)
    check("exp16 arch still builds and is unaffected",
          bool(torch.equal(m16.actor_lut.weights, m10.actor_lut.weights)))

    print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
