"""Check exp17's initialization against the analysis in models.FastLUTLogSumExpActorCritic.

Predictions to confirm before spending GPU:
  * a constant offset of tau*log(tables_per_head) = 0.1*log(32) = 0.347 on every output;
  * the variable part shrinks by a factor of T=32 vs exp10 (sigma/sqrt(T) vs sigma*sqrt(T));
  * nothing saturates the env's clamp(-1, 1) at init.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from models import REGISTRY                                        # noqa: E402

TPH = 32


def main():
    torch.manual_seed(0)
    m10 = REGISTRY["fastlut"](17, 6, tables_per_head=TPH)
    torch.manual_seed(0)
    m17 = REGISTRY["fastlut_lse"](17, 6, tables_per_head=TPH)

    p10 = sum(p.numel() for p in m10.parameters())
    p17 = sum(p.numel() for p in m17.parameters())
    print(f"params  fastlut {p10:,}   fastlut_lse {p17:,}  (+{p17 - p10} = tau)")
    print(f"extra_log {m17.extra_log()}")

    obs = torch.randn(4096, 17)
    with torch.no_grad():
        a10, _ = m10(obs)
        a17, _ = m17(obs)

    print(f"\nexp10  mean {a10.mean():+.6f}  std {a10.std():.6f}  "
          f"range [{a10.min():+.4f}, {a10.max():+.4f}]")
    print(f"exp17  mean {a17.mean():+.6f}  std {a17.std():.6f}  "
          f"range [{a17.min():+.4f}, {a17.max():+.4f}]")

    pred = 0.1 * math.log(TPH)
    print(f"\npredicted constant offset tau*log(T) = {pred:.4f}   "
          f"measured mean = {a17.mean():+.4f}")
    print(f"predicted std ratio exp10/exp17 = T = {TPH}   "
          f"measured = {a10.std() / a17.std():.1f}")
    print(f"exp17 action means outside clamp(-1,1) at init: "
          f"{(a17.abs() > 1).float().mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
