"""Design the log-space initialization for exp17's log-sum-exp readout.

THE PROBLEM (Anatoli's diagnosis, confirmed numerically below). Under
`out = tau * log( sum_t exp(w_t / tau) )` the table weights sit INSIDE the exponential,
so they are logarithms, not additive contributions. The additive default
`w ~ U(-1e-3, 1e-3)` therefore makes every exp(w/tau) ~ 1, the sum collapses to ~T, and

    out ~ tau * log(T)                     a large near-constant offset
    d out / d w_t = softmax_t ~ 1/T        tiny and nearly uniform

which is exactly the long-warmup signature.

THE FIX, derived rather than guessed:

  * OFFSET. out = 0 requires sum_t exp(w_t/tau) = 1, i.e. each term ~ 1/T, i.e.
        mu = -tau * log(T)
    Centring the weights there makes the readout start at zero. (Equivalently: this turns
    log-SUM-exp into log-MEAN-exp without changing the specified formula.)

  * SPREAD. Log-sum-exp AVERAGES over tables where the plain sum ACCUMULATES:
        additive   std(out) = sigma_a * sqrt(T/3)      (grows as sqrt(T))
        log-sum-exp std(out) ~ std(delta)/sqrt(T)      (shrinks as 1/sqrt(T))
    Matching the two gives sigma_delta = sigma_a * T -- the per-entry spread must be T
    times LARGER, not equal. With sigma_a = 1e-3 and T = 32 that is 0.032.

  * TAU. tau sets how peaked the softmax over tables is. tau >> std(delta) gives a smooth
    mean with uniform ~1/T gradients (Anatoli's complaint); tau ~ std(delta) gives a
    structured softmax and differentiated per-entry gradients. This script measures both
    the output statistics and the gradient non-uniformity across a tau grid so the choice
    is made on numbers.

Usage:  python design_init.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from models import REGISTRY                                        # noqa: E402

OBS, ACT, TPH, NAP, B = 17, 6, 32, 6, 8192
SIGMA_A = 1e-3


def stats(w_sel, tau):
    """Given selected weights [B, T, ACT] and tau, return output + gradient diagnostics."""
    z = w_sel / tau
    out = tau * torch.logsumexp(z, dim=1)                 # [B, ACT]
    sm = torch.softmax(z, dim=1)                          # d out / d w_t
    # non-uniformity: effective number of tables carrying the gradient (perplexity).
    # T means perfectly uniform (no structure); 1 means winner-take-all.
    ent = -(sm.clamp_min(1e-12).log() * sm).sum(dim=1)
    eff = ent.exp().mean().item()
    return out.mean().item(), out.std().item(), sm.max().item(), eff


def main():
    torch.manual_seed(0)
    m10 = REGISTRY["fastlut"](OBS, ACT, tables_per_head=TPH)
    obs = torch.randn(B, OBS)
    with torch.no_grad():
        a10, _ = m10(obs)
    tgt_std = a10.std().item()
    print(f"exp10 reference: out mean {a10.mean():+.6f}  std {tgt_std:.6f}")
    print(f"  (additive: sigma_a*sqrt(T/3) = {SIGMA_A * math.sqrt(TPH / 3):.6f})\n")

    # Emulate a selection: T independent weights per (sample, output).
    g = torch.Generator().manual_seed(1)

    print("CURRENT (attempt 1) init: w ~ U(-1e-3, 1e-3), mu = 0, tau = 0.1")
    w = (torch.rand(B, TPH, ACT, generator=g) - 0.5) * 2 * SIGMA_A
    mo, so, mx, eff = stats(w, 0.1)
    print(f"  out mean {mo:+.6f}   out std {so:.3e}   max softmax {mx:.4f}   "
          f"eff. tables {eff:.1f}/{TPH}")
    print(f"  -> predicted offset tau*log(T) = {0.1 * math.log(TPH):+.4f}; "
          f"out std is {tgt_std / so:.0f}x SMALLER than exp10's\n")

    sigma_d = SIGMA_A * TPH
    print(f"PROPOSED log-space init: w ~ U(mu-{sigma_d:g}, mu+{sigma_d:g}), "
          f"mu = -tau*log(T)")
    print(f"{'tau':>8} {'mu':>10} {'out mean':>11} {'out std':>11} "
          f"{'std/exp10':>10} {'max softmax':>12} {'eff tables':>11}")
    best = None
    for tau in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        mu = -tau * math.log(TPH)
        g2 = torch.Generator().manual_seed(1)
        w = mu + (torch.rand(B, TPH, ACT, generator=g2) - 0.5) * 2 * sigma_d
        mo, so, mx, eff = stats(w, tau)
        print(f"{tau:>8.3f} {mu:>10.4f} {mo:>+11.6f} {so:>11.3e} "
              f"{so / tgt_std:>10.2f} {mx:>12.4f} {eff:>11.1f}")
        score = abs(math.log(so / tgt_std))
        if best is None or score < best[0]:
            best = (score, tau, mu, so, eff)
    print(f"\nclosest match to exp10's output spread: tau = {best[1]}, mu = {best[2]:.4f} "
          f"(std ratio {best[3] / tgt_std:.2f}, eff. tables {best[4]:.1f})")

    print("\nWhat stays structurally different regardless of init:")
    print(f"  d out/d w summed over tables = 1 (log-sum-exp) vs T = {TPH} (additive).")
    print("  So the same weight step moves the action ~T x less. That is a property of")
    print("  averaging vs summing, NOT of the initialisation, and it caps how much of the")
    print("  warmup gap an init fix can close.")


if __name__ == "__main__":
    main()
