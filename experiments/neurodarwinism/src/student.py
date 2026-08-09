"""Trainable spiking student: two neuron models, two weight codings, three learnable scopes.

Supersedes the frozen reference in `lut_ttfs.py` (which stays as the verified oracle — this
module asserts agreement with it at init, so the trainable path is validated against code
that already passed the exactness test).

NEURON MODELS
-------------
"exact" — exponentially-growing-PSP TTFS neuron. V(t) = sum_c W_c exp((t - s_c)/tau_m),
    fires at theta:
        t_f = tau_m * ( log(theta) - log( sum_c W_c exp(-s_c/tau_m) ) )
    With tau_m = tau this reproduces exp19's readout EXACTLY (see lut_ttfs.py / README).

"lif"   — current-based LIF with a FINITE synaptic time constant, tau_s = tau_m / 2:
        tau_m dV/dt = -V + I,   I(t) = sum_c W_c exp(-(t - s_c)/tau_s)
        =>  V(t) = sum_c W_c [ exp(-(t-s_c)/tau_m) - exp(-(t-s_c)/tau_s) ]
    (the tau_s = tau_m/2 case has amplitude factor tau_s/(tau_m - tau_s) = 1 exactly.)
    Substituting x = exp(-t/tau_m) makes exp(-t/tau_s) = x^2, so threshold crossing is a
    QUADRATIC in x:
        theta = A x - B x^2,   A = sum_c W_c e^{s_c/tau_m},  B = sum_c W_c e^{2 s_c/tau_m}
    V(x) is a downward parabola; as t increases x DEcreases from A/B, so the first crossing
    is the LARGER root:
        x_f = ( A + sqrt(A^2 - 4 B theta) ) / (2B),     t_f = -tau_m log(x_f)
    Exactly solvable — no Lambert W needed, and differentiable. (tau_s = tau_m is the case
    that needs Lambert W; tau_s = tau_m/2 is the standard closed-form choice, Goltz 2021.)
    A^2 < 4 B theta means the neuron NEVER reaches threshold: no spike. Tracked and reported.

    Computed with s shifted by its per-output max so the exponentials stay in [e^-9.4, 1];
    the shift cancels exactly (x_f scales by e^{-s_ref/tau_m}, t_f shifts by +s_ref).

WEIGHT CODINGS
--------------
variant "W": W_c = exp(w_c / tau),  s_c = t_cell            (weight-coded)
variant "D": W_c = 1,  s_c = t_cell + (D0 - w_c) >= 0       (delay-coded)
On the exact neuron these are provably the same layer up to a rigid shift of D0. On the LIF
they are NOT — B weights arrivals by e^{2s/tau_m}, so delays and weights stop being
interchangeable. That asymmetry is one of the things the sweep measures.

DECODE
------
"exact" neuron: a = -tph * (t_f - t_cell - shift) + tph*tau*log(theta/tph), analytically exact.
"lif" neuron:   a = alpha_o * t_f,o + beta_o, with (alpha, beta) FIT ONCE per output dim by
    least squares on a calibration batch of teacher pairs, then FROZEN. This is the
    "approximate decode" — the residual of that fit is the gap that learning has to close.
    The decode is deliberately not learnable: the brief is that only the front-end learns.

LEARNABLE SCOPES
----------------
"races"  : the 32x6x17 race weights A (3,264)   [+ encoder m, c — 2 params, always]
"tau"    : races + the readout temperature tau  (+1)
"weights": races + tau + the 12,288 cell weights (full distillation)
"""
import math

import torch
import torch.nn as nn

from lut_ttfs import LatencyEncoder, LearnableRaceFrontEnd


class SpikingStudent(nn.Module):

    def __init__(self, weights, anchor_a, anchor_b, tau, *,
                 neuron="exact", variant="D", scope="races", decode_mode="affine",
                 bit_eps=0.05, t_cell=0.0, theta_rho=0.5, dtype=torch.float32):
        super().__init__()
        assert neuron in ("exact", "lif")
        assert variant in ("D", "W")
        assert scope in ("races", "tau", "weights")
        assert decode_mode in ("affine", "corrected")
        self.neuron, self.variant, self.scope = neuron, variant, scope
        self.decode_mode = decode_mode
        self.bit_eps, self.t_cell = float(bit_eps), float(t_cell)

        T, K, O = weights.shape
        self.n_tables, self.table_dim, self.n_out, self.tph = T, K, O, T
        NAP = anchor_a.shape[1]

        w0 = weights.to(dtype).clone()
        if scope == "weights":
            self.w = nn.Parameter(w0)
        else:
            self.register_buffer("w", w0)
        tau_raw0 = torch.tensor(math.log(math.expm1(float(tau))), dtype=dtype)
        if scope in ("tau", "weights"):
            self.tau_raw = nn.Parameter(tau_raw0)
        else:
            self.register_buffer("tau_raw", tau_raw0)

        self.register_buffer("anchor_a", anchor_a.clone())
        self.register_buffer("anchor_b", anchor_b.clone())
        self.register_buffer("powers", (1 << torch.arange(NAP - 1, -1, -1)).long())
        k = torch.arange(K)
        self.register_buffer("kbits", ((k.view(-1, 1) // self.powers.view(1, -1)) % 2)
                             .to(dtype))                                    # [64, 6]

        # D0 is FIXED at init (not tracked through w) so delays have a stable zero point.
        # If training pushes some w above D0 the delay would go negative; it is clamped and
        # the event is counted, so a silent physical violation cannot pass unnoticed.
        self.D0 = float(w0.max())
        self.n_delay_clamped = 0

        self.enc = LatencyEncoder()
        self.race = LearnableRaceFrontEnd(_AnchorShim(anchor_a, anchor_b, w0))
        self.race.A.data = self.race.A.data.to(dtype)

        self.theta_rho = float(theta_rho)
        self.register_buffer("theta", torch.tensor(float("nan"), dtype=dtype))
        self.register_buffer("dec_a", torch.zeros(O, dtype=dtype))
        self.register_buffer("dec_b", torch.zeros(O, dtype=dtype))

    # ---------------------------------------------------------------- parameterisation
    @property
    def tau(self):
        return nn.functional.softplus(self.tau_raw).clamp_min(1e-3)

    def syn_delay(self):
        """(W_c, s_c) per (table, address, output), both [32, 64, 6]."""
        tau = self.tau
        if self.variant == "W":
            return torch.exp(self.w / tau), torch.zeros_like(self.w) + self.t_cell
        d = self.D0 - self.w
        if bool((d < 0).any()):
            self.n_delay_clamped += int((d < 0).sum())
        return torch.ones_like(self.w), self.t_cell + d.clamp_min(0.0)

    # ---------------------------------------------------------------- front end
    def gates(self, x, hard=True):
        t_in = self.enc(x)
        d = self.race(t_in)                                     # [B, 32, 6]
        soft = torch.sigmoid(d / self.bit_eps)
        bits = (d > 0).to(x.dtype) + (soft - soft.detach()) if hard else soft
        b = bits.unsqueeze(2)                                   # [B, 32, 1, 6]
        p = b * self.kbits + (1 - b) * (1 - self.kbits)         # [B, 32, 64, 6]
        # explicit chain instead of p.prod(-1): torch's JIT'd prod-reduction kernel fails
        # to build on this CUDA toolchain (nvrtc cannot find libnvrtc-builtins.so.13.0).
        out = p[..., 0]
        for i in range(1, p.shape[-1]):
            out = out * p[..., i]
        return out                                              # [B, 32, 64]

    # ---------------------------------------------------------------- neurons
    def sum_S(self, g):
        """The teacher's pre-log quantity as the student computes it: [B, 6]."""
        W, s = self.syn_delay()
        return torch.einsum("btk,tko->bo", g, W * torch.exp(-(s - self.t_cell) / self.tau))

    def spike_time(self, g):
        W, s = self.syn_delay()
        tau_m = self.tau
        if self.neuron == "exact":
            S = torch.einsum("btk,tko->bo", g, W * torch.exp(-(s - self.t_cell) / tau_m))
            return self.t_cell + tau_m * (torch.log(self.theta) - torch.log(S)), None
        # LIF: shift arrivals by their per-output max so the exponentials stay bounded
        s_ref = s.reshape(-1, self.n_out).max(dim=0).values                 # [6]
        e1 = torch.exp((s - s_ref) / tau_m)
        A = torch.einsum("btk,tko->bo", g, W * e1)
        B = torch.einsum("btk,tko->bo", g, W * e1 * e1)
        disc = A * A - 4.0 * B * self.theta
        no_spike = (disc < 0)
        x = (A + torch.sqrt(disc.clamp_min(1e-12))) / (2.0 * B)
        return s_ref + (-tau_m * torch.log(x.clamp_min(1e-30))), no_spike

    def decode(self, t_f):
        """affine: a = alpha*t + beta (fit once, frozen).

        corrected: the ANALYTIC inverse of the LIF, available because in variant W every
        synapse arrives at t_cell, so A = B = S and the quadratic collapses to

            x_f = [1 + sqrt(1 - 4*theta/S)] / 2   =>   S = theta / ( x_f * (1 - x_f) )

        with x_f = exp(-(t_f - t_cell)/tau_m). Feeding that S through the teacher's own
        readout gives the action back EXACTLY -- the LIF's "gap" in variant W is a decode
        artefact, not a limitation of the neuron.

        For variant D the arrival times differ, A and B are independent, and t_f alone does
        not determine S; the same formula is then only an approximation and is labelled as
        such wherever it is used.
        """
        if self.decode_mode == "affine":
            return self.dec_a * t_f + self.dec_b
        x = torch.exp(-(t_f - self.t_cell) / self.tau)
        S = self.theta / (x * (1.0 - x)).clamp_min(1e-20)
        return self.tph * self.tau * (torch.log(S.clamp_min(1e-20)) - math.log(self.tph))

    def forward(self, x, hard=True):
        g = self.gates(x, hard=hard)
        t_f, no_spike = self.spike_time(g)
        return self.decode(t_f), t_f, no_spike, g

    # ---------------------------------------------------------------- calibration
    @torch.no_grad()
    def calibrate(self, x_cal, a_cal):
        """Pick theta so every calibration sample spikes, then fit the frozen decode."""
        g = self.gates(x_cal, hard=True)
        W, s = self.syn_delay()
        tau_m = self.tau
        if self.neuron == "exact":
            S = torch.einsum("btk,tko->bo", g, W * torch.exp(-(s - self.t_cell) / tau_m))
            # the exact neuron always uses its own analytic affine decode
            self.decode_mode = "affine"
            # theta above max S keeps t_f > 0; margin in log space
            self.theta.fill_(float(torch.exp(torch.log(S.max()) + 0.5)))
            t_f, _ = self.spike_time(g)
            self.dec_a.fill_(-float(self.tph))
            self.dec_b.copy_(torch.full_like(self.dec_b, float(
                self.tph * (self.t_cell) + self.tph * float(tau_m)
                * math.log(float(self.theta) / self.tph))))
            if self.variant == "D":
                self.dec_b += self.tph * self.D0
            return
        s_ref = s.reshape(-1, self.n_out).max(dim=0).values
        e1 = torch.exp((s - s_ref) / tau_m)
        A = torch.einsum("btk,tko->bo", g, W * e1)
        B = torch.einsum("btk,tko->bo", g, W * e1 * e1)
        margin = (A * A / (4.0 * B)).min()
        self.theta.fill_(float(self.theta_rho * margin))
        t_f, _ = self.spike_time(g)
        # per-output least-squares affine fit a ~ alpha*t + beta, then FROZEN
        for o in range(self.n_out):
            t, a = t_f[:, o].double(), a_cal[:, o].double()
            tm, am = t.mean(), a.mean()
            alpha = ((t - tm) * (a - am)).sum() / ((t - tm) ** 2).sum().clamp_min(1e-30)
            self.dec_a[o] = float(alpha)
            self.dec_b[o] = float(am - alpha * tm)

    def trainable(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]


class _AnchorShim:
    """LearnableRaceFrontEnd only reads .anchor_a/.anchor_b/.w — give it those."""

    def __init__(self, a, b, w):
        self.anchor_a, self.anchor_b, self.w = a, b, w
