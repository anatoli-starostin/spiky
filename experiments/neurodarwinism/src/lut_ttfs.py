"""exp19's LUT readout as a FIXED latency-coded (TTFS) spiking layer.

The central result this file implements and tests: exp19's sum-scaled log-sum-exp readout
is not *approximated* by a first-spike layer — it IS one, exactly, under an
exponential-kernel TTFS neuron. See README.md for the derivation. Briefly:

  a neuron whose PSPs grow as  V(t) = sum_i W_i * exp((t - s_i)/tau_m)  and which fires at
  threshold theta has the closed-form firing time

      t_f = tau_m * ( log(theta) - log( sum_i W_i * exp(-s_i/tau_m) ) )

  which is -tau_m * logsumexp(...) plus a constant. Setting tau_m = tau (the actor's
  learned exp_outputs_tau) makes the bracket exactly the teacher's pre-log quantity S, and
  the action mean is then an EXACT AFFINE function of the output spike time:

      a_o = -tph * (t_f,o - t_cell) + tph*tau*log(theta/tph)

  Two ways to inject the LUT weight w into that sum, both exact and mutually equivalent:
      variant W:  W_c = exp(w_c/tau),  no delay        -> weight-coded
      variant D:  W_c = 1,  delay d_c = D0 - w_c       -> delay-coded  (D0 keeps d >= 0)
  The D0 offset scales the sum by exp(-D0/tau), i.e. shifts every t_f by exactly D0, and is
  absorbed by the decode constant.

Layers (2455 neurons, 12,288 output synapses = exactly one synapse per LUT weight scalar):

  17  input     linear latency code, ONE shared monotone-decreasing map (see below)
 384  race      r_{t,i} = 1[t_a < t_b] and its complement -- "which input spikes first"
2048  cell      one per LUT row (table t, address k); coincidence detector, threshold 6
   6  output    exponential-kernel TTFS neuron, one per action dimension

WHY THE ENCODER MAP MUST BE SHARED ACROSS THE 17 DIMENSIONS. The LUT's address bit is
`x[a] > x[b]`, a comparison BETWEEN two observation dimensions. Under a latency code
t = c - m*x with m > 0 that becomes `t_a < t_b` -- but only if the SAME (c, m) is applied
to both dims. A per-dimension scale or offset silently changes which cell is addressed and
the teacher is no longer reproduced. So the v1 learnable front-end is two scalars, shared.
Generalising the race layer (learnable linear forms on latencies instead of fixed index
pairs) is the v2 path and is deliberately NOT in this file.

Run this file directly for the exactness self-test against the distillation dataset:
    python lut_ttfs.py [--n 20000] [--variant D|W]
"""
import argparse
import math
import os

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NPZ = os.path.join(HERE, "..", "distill_exp19_100k.npz")


class LatencyEncoder(nn.Module):
    """x (normalised obs) -> spike times in [0, T]. Shared, strictly monotone decreasing.

    t_j = c - m * x_j,  m = softplus(m_raw) > 0.  Larger x fires EARLIER, so
    `x_a > x_b  <=>  t_a < t_b` holds for every pair, exactly, for any (c, m>0).

    Both parameters are learnable and they are the ONLY learnable part of the v1 student.
    They cannot change the teacher's function -- every address bit is invariant to them --
    so v1 has an exactly-correct fixed point; what they DO control is where the spikes sit
    inside the [0, T] window, which is what a hardware implementation cares about.
    """

    def __init__(self, x_lo=-7.4, x_hi=8.7, window=1.0):
        super().__init__()
        m0 = window / (x_hi - x_lo)
        self.m_raw = nn.Parameter(torch.tensor(math.log(math.expm1(m0))))
        self.c = nn.Parameter(torch.tensor(m0 * x_hi))
        self.window = float(window)

    @property
    def m(self):
        return nn.functional.softplus(self.m_raw)

    def forward(self, x):                       # x [B, 17] -> t [B, 17]
        return self.c - self.m * x


class LutTTFSReadout(nn.Module):
    """The frozen LUT readout as a spiking layer. Nothing here is learnable.

    variant "W": cell weights enter as synaptic weights   W_c = exp(w_c / tau)
    variant "D": cell weights enter as axonal delays      d_c = D0 - w_c,  W_c = 1
    """

    def __init__(self, weights, anchor_a, anchor_b, tau, variant="D",
                 theta_log_margin=0.5, bit_eps=0.05, t_cell=0.0):
        super().__init__()
        assert variant in ("D", "W")
        self.variant = variant
        T, K, O = weights.shape                          # (32, 64, 6)
        self.n_tables, self.table_dim, self.n_out = T, K, O
        self.tph = T                                     # n_heads == 1
        self.tau = float(tau)
        self.tau_m = float(tau)                          # tau_m = tau makes the map exact
        self.bit_eps = float(bit_eps)
        self.t_cell = float(t_cell)

        self.register_buffer("w", weights.clone())       # (32, 64, 6) the LUT scalars
        self.register_buffer("anchor_a", anchor_a.clone())
        self.register_buffer("anchor_b", anchor_b.clone())
        self.register_buffer("powers",
                             (1 << torch.arange(anchor_a.shape[1] - 1, -1, -1)).long())

        # variant D needs non-negative delays: d = D0 - w with D0 = max(w).
        self.D0 = float(weights.max())
        if variant == "D":
            self.register_buffer("delay", self.D0 - self.w)          # (32, 64, 6) >= 0
            self.register_buffer("syn", torch.ones_like(self.w))
            self._shift = self.D0            # every t_f is shifted by exactly +D0
        else:
            self.register_buffer("delay", torch.zeros_like(self.w))
            self.register_buffer("syn", torch.exp(self.w / self.tau))
            self._shift = 0.0

        # theta must exceed the largest attainable sum so t_f stays positive.
        s_max = float(torch.exp(self.w.max() / self.tau)) * T
        self.theta = math.exp(math.log(s_max) + theta_log_margin)

    # ---- decode: action mean is EXACTLY affine in the output spike time ----------------
    @property
    def decode_slope(self):
        return -float(self.tph)

    @property
    def decode_bias(self):
        # a = -tph*(t_f - t_cell - D0) + tph*tau*log(theta/tph)
        return (self.tph * (self.t_cell + self._shift)
                + self.tph * self.tau * math.log(self.theta / self.tph))

    def decode(self, t_out):
        return self.decode_slope * t_out + self.decode_bias

    # ---- the spiking forward -----------------------------------------------------------
    def gates(self, t_in, hard=True, race=None):
        """Cell activations from input spike times. [B,17] -> [B, 32, 64].

        race: b_{t,i} = 1[t_a < t_b];  cell (t,k) fires iff its 6-bit pattern matches.
        Soft form (for gradients) is a product of sigmoids -- a differentiable
        coincidence detector, which is what a threshold-6 neuron does in the hard limit.

        `race` optionally supplies the 32x6 signed race quantities (see
        LearnableRaceFrontEnd); the default is the fixed anchor-pair difference.
        """
        d = (t_in[:, self.anchor_b] - t_in[:, self.anchor_a]     # [B, 32, 6] >0 == bit 1
             if race is None else race(t_in))
        soft = torch.sigmoid(d / self.bit_eps)
        if hard:
            bits = (d > 0).to(t_in.dtype)
            bits = bits + (soft - soft.detach())                 # straight-through
        else:
            bits = soft
        # pattern match over the 64 addresses, MSB-first
        k = torch.arange(self.table_dim, device=t_in.device)
        kbits = ((k.view(-1, 1) // self.powers.view(1, -1)) % 2).to(t_in.dtype)  # [64, 6]
        b = bits.unsqueeze(2)                                    # [B, 32, 1, 6]
        p = b * kbits + (1 - b) * (1 - kbits)                    # [B, 32, 64, 6]
        # explicit chain instead of p.prod(-1): torch's JIT'd prod-reduction kernel fails
        # to build on this CUDA toolchain (nvrtc cannot find libnvrtc-builtins.so.13.0).
        out = p[..., 0]
        for i in range(1, p.shape[-1]):
            out = out * p[..., i]
        return out                                               # [B, 32, 64]

    def forward(self, t_in, hard=True, return_latency=False, race=None):
        g = self.gates(t_in, hard=hard, race=race)               # [B, 32, 64]
        # each output neuron sums exp((t - t_cell - d_c)/tau_m) * W_c over gated cells
        contrib = self.syn * torch.exp(-self.delay / self.tau_m)  # (32, 64, 6)
        S = torch.einsum("btk,tko->bo", g, contrib)               # [B, 6]
        t_out = self.t_cell + self.tau_m * (math.log(self.theta) - torch.log(S))
        a = self.decode(t_out)
        return (a, t_out) if return_latency else a


class LearnableRaceFrontEnd(nn.Module):
    """v2 front-end: each race neuron takes a LEARNABLE linear form on the 17 latencies.

    The frozen readout's `gates()` only ever needs the 32x6 signed quantities
    `d_{t,i} = t_b - t_a`. Generalising them to `d_{t,i} = sum_j A[t,i,j] * t_j` turns the
    fixed anchor pairs into learnable dendritic weights on the race layer while leaving the
    2048 cell neurons and all 12,288 output synapses frozen.

    A is INITIALISED to the exact one-hot +-1 anchor pattern, so this starts as a bit-exact
    copy of the teacher and can only move away from it under a training signal. That gives
    the harness a known-correct fixed point to validate against before any learning.
    """

    def __init__(self, readout, n_in=17):
        super().__init__()
        T, NAP = readout.anchor_a.shape
        A = torch.zeros(T, NAP, n_in, dtype=readout.w.dtype)
        t_idx = torch.arange(T).view(-1, 1).expand(T, NAP)
        i_idx = torch.arange(NAP).view(1, -1).expand(T, NAP)
        A[t_idx, i_idx, readout.anchor_b] = 1.0        # d = t_b - t_a
        A[t_idx, i_idx, readout.anchor_a] = -1.0
        self.A = nn.Parameter(A)

    def forward(self, t_in):                            # [B, 17] -> [B, 32, 6]
        return torch.einsum("bj,tij->bti", t_in, self.A)


class SpikingLutStudent(nn.Module):
    """Encoder (learnable, 2 scalars) + frozen spiking readout."""

    def __init__(self, readout, encoder=None, learnable_races=False):
        super().__init__()
        self.enc = encoder if encoder is not None else LatencyEncoder()
        self.readout = readout
        self.race = LearnableRaceFrontEnd(readout) if learnable_races else None
        for p in self.readout.parameters():
            p.requires_grad_(False)

    def forward(self, x, hard=True, return_latency=False):
        return self.readout(self.enc(x), hard=hard, return_latency=return_latency,
                            race=self.race)


def build_from_npz(path=DEFAULT_NPZ, variant="D", device="cpu", dtype=torch.float64):
    import numpy as np
    Z = np.load(path)
    ro = LutTTFSReadout(
        torch.tensor(Z["weights"], dtype=dtype),
        torch.tensor(Z["anchor_a"], dtype=torch.long),
        torch.tensor(Z["anchor_b"], dtype=torch.long),
        float(Z["tau"]), variant=variant).to(device)
    return ro, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--variant", default="both", choices=["D", "W", "both"])
    a = ap.parse_args()
    import numpy as np

    variants = ["D", "W"] if a.variant == "both" else [a.variant]
    for v in variants:
        ro, Z = build_from_npz(a.npz, variant=v)
        st = SpikingLutStudent(ro)
        x = torch.tensor(Z["x_norm"][: a.n], dtype=torch.float64)
        y = torch.tensor(Z["y_action_mean_f64"][: a.n])
        with torch.no_grad():
            ahat, t_out = st(x, hard=True, return_latency=True)
        err = (ahat - y).abs()
        print(f"\nvariant {v}   theta={ro.theta:.4g}  tau_m={ro.tau_m:.6f}  "
              f"D0={ro.D0:.4f}  decode: a = {ro.decode_slope:.1f}*t + {ro.decode_bias:.6f}")
        print(f"  spike times t_out   min {t_out.min():.6f}  max {t_out.max():.6f}")
        print(f"  |a_spiking - a_LUT| max {err.max():.3e}  mean {err.mean():.3e}"
              f"   (action scale std {y.std():.3f})")
        if v == "D":
            print(f"  delays d = D0 - w   min {ro.delay.min():.6f}  max {ro.delay.max():.6f}"
                  f"   (all >= 0: {bool((ro.delay >= 0).all())})")
        else:
            print(f"  synaptic W = exp(w/tau)  min {ro.syn.min():.6f}  max {ro.syn.max():.4f}")

    # how "first-to-spike" is the log-sum-exp in practice?
    ro, Z = build_from_npz(a.npz, variant="D")
    x = torch.tensor(Z["x_norm"][:20000], dtype=torch.float64)
    st = SpikingLutStudent(ro)
    with torch.no_grad():
        g = ro.gates(st.enc(x))
        contrib = ro.syn * torch.exp(-ro.delay / ro.tau_m)
        per_table = torch.einsum("btk,tko->bto", g, contrib)      # [B, 32, 6]
        share = per_table / per_table.sum(1, keepdim=True)
        eff = 1.0 / (share ** 2).sum(1)                           # participation ratio
        top1 = share.max(dim=1).values
    print(f"\nsoft-min sharpness over the {ro.tph} tables (20k samples):")
    print(f"  effective tables contributing  mean {eff.mean():.2f}  "
          f"min {eff.min():.2f}  max {eff.max():.2f}   (1 = pure first-to-spike, "
          f"{ro.tph} = plain sum)")
    print(f"  share of the earliest synapse  mean {top1.mean():.3f}  max {top1.max():.3f}")


if __name__ == "__main__":
    main()
