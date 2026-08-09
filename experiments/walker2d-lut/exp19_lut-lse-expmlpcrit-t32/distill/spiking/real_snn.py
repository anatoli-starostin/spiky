"""A REAL, fully-simulated spiking network approximating the exp19 LUT teacher.

No functional shortcuts. Every layer is discrete-time membrane integration with spikes as
events; there is no `t_a < t_b` comparison and no product-of-bits anywhere. What was
analytic in `lut_ttfs.py` / `student.py` is here an actual circuit:

  INPUT  17 neurons, one spike each at  t_j = c - m*x_j  in [0, T_in].
  RACE   384 order-detecting LIF neurons = 2 per (table, anchor-pair): r+ fires if a
         arrives first, r- fires if b arrives first. Each has exactly TWO real synapses --
         an excitatory one from its "winner" input and an inhibitory VETO from the other --
         with real weights and real (learnable, continuously-interpolated) delays. Order is
         detected by membrane dynamics: the excitatory PSP climbs toward threshold and the
         veto arrives in time to cancel it, or does not.
  CELL   2048 coincidence-detector LIF neurons, one per LUT row, each with 6 synapses from
         the race neurons matching its 6-bit address. Long membrane time constant so drive
         LATCHES, plus the validated fixed-period square-wave inhibitory gate.
  OUTPUT 6 LIF TTFS neurons, tau_s = tau_m/2, one synapse from every cell, weight-coded
         (variant W) and initialised to exp(w/tau). Action is decoded from spike TIMING.

DISCRETE NEURON (current-based LIF, the standard Zenke form):
    I[n+1] = alpha_s * I[n] + sum_j W_j * S_j[n]
    V[n+1] = alpha_m * V[n] + I[n]                       (no reset -- single-spike TTFS)
    S[n]   = H(V[n] - theta),  emitted at most ONCE per neuron per sample
with alpha = exp(-dt/tau). Gradients use a fast-sigmoid surrogate for H.

WHY SINGLE-SPIKE: this is a TTFS code. Every neuron is latched off after its first spike,
which is what makes "spike time" a well-defined readout and keeps BPTT memory bounded.

TIME RESOLUTION IS THE BINDING CONSTRAINT, and it is worth stating up front. The action is
affine in the output spike time with slope tph/k (k = the output time-constant scale), and
spike times are quantised to dt, so quantisation alone floors the error at roughly

    err_norm  ~  (tph / k) * dt / sqrt(12) / action_std

Increasing k or shrinking dt both fix it and both cost simulation steps. That trade-off is
measured rather than assumed -- see RESULTS_real_snn.md.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "..", "distill_exp19_100k.npz")
OUT = os.path.join(HERE, "results")


# ------------------------------------------------------------------ surrogate spike
class SpikeFn(torch.autograd.Function):
    """H(v) forward, fast-sigmoid surrogate backward: dS/dv = 1/(1 + slope*|v|)^2."""

    @staticmethod
    def forward(ctx, v, slope):
        ctx.save_for_backward(v)
        ctx.slope = slope
        return (v > 0).to(v.dtype)

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        return g / (1.0 + ctx.slope * v.abs()) ** 2, None


def spike(v, slope=25.0):
    return SpikeFn.apply(v, slope)


def psp_peak(alpha_s, alpha_m, n_max=4000):
    """Peak of the unit-weight single-spike PSP, so weights can be normalised to it."""
    n = torch.arange(1, n_max, dtype=torch.float64)
    if abs(alpha_s - alpha_m) < 1e-9:
        v = n * alpha_m ** (n - 1)
    else:
        v = (torch.tensor(alpha_s, dtype=torch.float64) ** n
             - torch.tensor(alpha_m, dtype=torch.float64) ** n) / (alpha_s - alpha_m)
    return float(v.max())


# ------------------------------------------------------------------ the network
class RealSNN(nn.Module):

    def __init__(self, weights, anchor_a, anchor_b, tau, *, dt=1 / 64, T_in=1.0,
                 n_steps=160, k_out=4.0,
                 tau_s_race=0.02, tau_m_race=0.04, veto=6.0, race_lat=0.03,
                 tau_s_veto=10.0,
                 tau_s_cell=0.05, tau_m_cell=15.0, cell_margin=0.45,
                 gate_open=None, gate_G=50.0, slope=25.0, theta_frac=0.8, decode="corrected",
                 train_race=True, train_cell=False, train_out=False):
        super().__init__()
        T, K, O = weights.shape                       # (32, 64, 6)
        NAP = anchor_a.shape[1]
        self.nT, self.K, self.O, self.NAP = T, K, O, NAP
        self.tph = T
        self.dt, self.T_in, self.n_steps = dt, T_in, n_steps
        self.slope = slope
        self.theta_frac = float(theta_frac)
        self.decode_mode = decode
        self.tau = float(tau)

        # ---- input latency code (the constants the analytic work used) --------------
        self.register_buffer("enc_c", torch.tensor(0.5403726697))
        self.register_buffer("enc_m", torch.tensor(0.0621117949))

        # ---- RACE wiring: 2 neurons per (table, pair). r+ index = j, r- index = j + 192
        self.nR = 2 * T * NAP                                          # 384
        a_flat, b_flat = anchor_a.reshape(-1), anchor_b.reshape(-1)    # (192,)
        self.register_buffer("exc_src", torch.cat([a_flat, b_flat]))   # winner input
        self.register_buffer("inh_src", torch.cat([b_flat, a_flat]))   # veto input
        self.a_s_r, self.a_m_r = math.exp(-dt / tau_s_race), math.exp(-dt / tau_m_race)
        # The veto needs its OWN, LONG synaptic time constant. With a single fast synaptic
        # current the inhibition is a transient PSP: if the winner spikes much earlier than
        # the loser, its veto has decayed away by the time the loser's excitation arrives,
        # and the wrong-order neuron fires anyway. Measured: that failure mode alone capped
        # order accuracy at 0.83 and produced errors at gaps up to 0.19 T. A slow
        # (effectively latching) inhibitory synapse is what makes the detector work at all.
        self.a_i_r = math.exp(-dt / tau_s_veto)
        pk_r = psp_peak(self.a_s_r, self.a_m_r)
        # Excitatory weight normalised so ONE input spike peaks at exactly 2*theta, i.e.
        # the neuron is guaranteed to fire from a single winner spike; the veto has to be
        # strong AND early enough to stop it. theta_race = 1 by convention.
        self.w_exc = nn.Parameter(torch.full((self.nR,), 2.0 / pk_r), requires_grad=train_race)
        self.w_inh = nn.Parameter(torch.full((self.nR,), veto * 2.0 / pk_r),
                                  requires_grad=train_race)
        # Real axonal delays, in TIME units, learnable and continuously interpolated.
        # The veto is deliberately FASTER than the excitation (delay 0 vs race_lat): that
        # asymmetry is what makes the detector prefer the earlier input.
        self.d_exc = nn.Parameter(torch.full((self.nR,), float(race_lat)),
                                  requires_grad=train_race)
        self.d_inh = nn.Parameter(torch.zeros(self.nR), requires_grad=train_race)
        self.theta_race = 1.0

        # ---- CELL wiring: 2048 rows x 6 synapses, gathered from the 384 race neurons ----
        kk = torch.arange(K)
        kbits = ((kk.view(-1, 1) // (1 << torch.arange(NAP - 1, -1, -1)).view(1, -1)) % 2)
        M = torch.zeros(T * K, self.nR)
        for t in range(T):
            for i in range(NAP):
                jp, jm = t * NAP + i, T * NAP + t * NAP + i        # r+ , r-
                rows = torch.arange(K) + t * K
                M[rows, jp] = kbits[:, i].float()                  # bit 1 -> a first -> r+
                M[rows, jm] = 1.0 - kbits[:, i].float()            # bit 0 -> b first -> r-
        self.register_buffer("cell_M", M)                          # (2048, 384), 6 ones/row
        self.a_s_c, self.a_m_c = math.exp(-dt / tau_s_cell), math.exp(-dt / tau_m_cell)
        pk_c = psp_peak(self.a_s_c, self.a_m_c)
        self.w_cell = nn.Parameter(torch.tensor(1.0 / pk_c), requires_grad=train_cell)
        # threshold strictly between 5 and 6 unit PSPs -> ALL SIX races required
        self.theta_cell = 5.0 + float(cell_margin)
        # validated square-wave gate: closed until `gate_open`, then released
        self.gate_open = float(T_in + 0.05 if gate_open is None else gate_open)
        self.gate_G = float(gate_G)

        # ---- OUTPUT: 6 TTFS LIF, tau_s = tau_m/2, weight-coded from the teacher --------
        self.tau_m_o = k_out * self.tau
        self.a_m_o, self.a_s_o = math.exp(-dt / self.tau_m_o), math.exp(-dt / (self.tau_m_o / 2))
        pk_o = psp_peak(self.a_s_o, self.a_m_o)
        Wo = torch.exp(weights / self.tau).reshape(T * K, O)       # (2048, 6) analytic init
        self.register_buffer("out_scale", torch.tensor(1.0 / pk_o))
        self.w_out = nn.Parameter(Wo.clone(), requires_grad=train_out)
        self.theta_out = nn.Parameter(torch.ones(O), requires_grad=False)   # set by calibrate
        # affine decode from the OUTPUT SPIKE TIME (12 params, fitted then learnable)
        self.dec_a = nn.Parameter(torch.full((O,), -float(T) / k_out))
        self.dec_b = nn.Parameter(torch.zeros(O))
        # "corrected": invert the LIF analytically instead of fitting a straight line.
        # For synchronous arrivals A == B == S, so x_f = exp(-(t-t_ref)/tau_m) satisfies
        # S = theta / (x_f (1 - x_f)); feeding that S through the teacher readout is exact
        # in continuous time. Here it is only APPROXIMATE (discrete steps, a spread of cell
        # firing times, dropped cells), so t_ref and the final affine are left learnable.
        self.t_ref = nn.Parameter(torch.zeros(O))
        # "mlp": a small learnable per-output decode of the spike time. The analytic forms
        # above assume the CONTINUOUS LIF; the simulated one is discrete, its 32 inputs are
        # not perfectly simultaneous, and some are missing, so its true t -> a transfer
        # function is smooth but not either closed form. 16 hidden units per output dim
        # (6 x 49 = 294 params) absorb that mismatch without letting the decode see anything
        # except the output spike time.
        H = 16
        self.mlp_w1 = nn.Parameter(torch.randn(O, H) * 0.5)
        self.mlp_b1 = nn.Parameter(torch.linspace(-2, 2, H).repeat(O, 1))
        self.mlp_w2 = nn.Parameter(torch.randn(O, H) * (1.0 / H))
        self.mlp_b2 = nn.Parameter(torch.zeros(O))

    # -------------------------------------------------------------- input raster
    def input_current(self, x):
        """Per-race-synapse arrival raster [B, nR, N] for exc and inh, with real delays.

        One input spike per dimension at t_j = c - m*x_j. A synapse with delay d receives
        it at t_j + d, split linearly across the two adjacent time bins so the delay is a
        genuine continuous parameter with a gradient, not a rounded index.
        """
        B = x.shape[0]
        t_in = self.enc_c - self.enc_m * x                         # [B, 17]
        N = self.n_steps
        out = []
        for src, delay, w in ((self.exc_src, self.d_exc, self.w_exc),
                              (self.inh_src, self.d_inh, self.w_inh)):
            arr = (t_in[:, src] + delay.view(1, -1)) / self.dt     # [B, nR] in bins
            lo = arr.floor()
            frac = arr - lo
            lo = lo.long().clamp(0, N - 1)
            hi = (lo + 1).clamp(0, N - 1)
            R = torch.zeros(B, self.nR, N, device=x.device, dtype=x.dtype)
            amp = w.view(1, -1)
            R.scatter_add_(2, lo.unsqueeze(-1), (amp * (1 - frac)).unsqueeze(-1))
            R.scatter_add_(2, hi.unsqueeze(-1), (amp * frac).unsqueeze(-1))
            out.append(R)
        return out[0], out[1]                                      # exc, inh  [B, nR, N]

    # -------------------------------------------------------------- the simulation
    def forward(self, x, record=False):
        B = x.shape[0]
        dev, dtp = x.device, x.dtype
        inj_e, inj_i = self.input_current(x)                       # [B, nR, N] each
        gate_n = int(self.gate_open / self.dt)

        Ir = torch.zeros(B, self.nR, device=dev, dtype=dtp)
        Ii = torch.zeros_like(Ir)
        Vr = torch.zeros_like(Ir)
        fired_r = torch.zeros_like(Ir)
        Ic = torch.zeros(B, self.nT * self.K, device=dev, dtype=dtp)
        Vc = torch.zeros_like(Ic)
        fired_c = torch.zeros_like(Ic)
        Io = torch.zeros(B, self.O, device=dev, dtype=dtp)
        Vo = torch.zeros_like(Io)
        fired_o = torch.zeros_like(Io)
        t_out = torch.zeros(B, self.O, device=dev, dtype=dtp)
        n_cell_spk = torch.zeros(B, device=dev, dtype=dtp)
        rec = []

        for n in range(self.n_steps):
            # ---- RACE: fast excitation, slow (latching) veto -------------------------
            Vr = self.a_m_r * Vr + Ir - Ii
            Ir = self.a_s_r * Ir + inj_e[:, :, n]
            Ii = self.a_i_r * Ii + inj_i[:, :, n]
            s_r = spike(Vr - self.theta_race, self.slope) * (1.0 - fired_r)
            fired_r = torch.clamp(fired_r + s_r, max=1.0)

            # ---- CELL (coincidence + square-wave inhibitory gate) --------------------
            Vc = self.a_m_c * Vc + Ic
            Ic = self.a_s_c * Ic + self.w_cell * (s_r @ self.cell_M.t())
            gate = 0.0 if n >= gate_n else self.gate_G
            s_c = spike(Vc - self.theta_cell - gate, self.slope) * (1.0 - fired_c)
            fired_c = torch.clamp(fired_c + s_c, max=1.0)
            n_cell_spk = n_cell_spk + s_c.sum(-1)

            # ---- OUTPUT TTFS ---------------------------------------------------------
            Vo = self.a_m_o * Vo + Io
            Io = self.a_s_o * Io + self.out_scale * (s_c @ self.w_out)
            s_o = spike(Vo - self.theta_out.view(1, -1), self.slope) * (1.0 - fired_o)
            t_out = t_out + s_o * (n * self.dt)
            fired_o = torch.clamp(fired_o + s_o, max=1.0)
            if record:
                rec.append((float(s_r.sum()), float(s_c.sum()), float(s_o.sum())))

        # neurons that never fired are pinned to the end of the window
        t_out = t_out + (1.0 - fired_o) * (self.n_steps * self.dt)
        a = self.decode(t_out)
        info = dict(t_out=t_out, fired_o=fired_o, cell_spikes=n_cell_spk,
                    race_spikes=fired_r.sum(-1), rec=rec)
        return a, info

    def decode(self, t_out):
        if self.decode_mode == "mlp":
            z = (t_out - self.t_ref).unsqueeze(-1) * self.mlp_w1 + self.mlp_b1
            h = torch.tanh(z)
            return (h * self.mlp_w2).sum(-1) + self.mlp_b2
        if self.decode_mode == "affine":
            return self.dec_a * t_out + self.dec_b
        xf = torch.exp(-(t_out - self.t_ref) / self.tau_m_o).clamp(1e-6, 1.0 - 1e-6)
        S = self.theta_out.view(1, -1) / (xf * (1.0 - xf))
        return self.dec_a * torch.log(S) + self.dec_b

    # -------------------------------------------------------------- calibration
    @torch.no_grad()
    def calibrate(self, x, a_ref):
        """Pick output thresholds so every neuron spikes mid-window, then fit the decode."""
        # Run with an unreachable threshold to see the membrane scale, then place theta
        # below the WEAKEST sample's peak so every sample actually emits a spike. Using the
        # batch maximum here (the obvious thing) leaves ~30% of samples silent, and a silent
        # output neuron is pinned to the end of the window -- a large, one-sided error.
        self.theta_out.fill_(1e9)                       # never fires -> record peak V
        peaks = self._peak_vo(x)                        # [B, O] per-sample peaks
        self.theta_out.copy_(self.theta_frac * peaks.quantile(0.005, dim=0))
        _, info = self.forward(x)
        t = info["t_out"]
        if self.decode_mode == "mlp":
            # centre and scale the time axis, then least-squares the linear head
            self.t_ref.copy_(t.mean(0))
            sd = t.std(0).clamp_min(1e-6)
            self.mlp_w1.copy_((torch.randn_like(self.mlp_w1) * 0.5 + 1.0) / sd.unsqueeze(-1))
            zz = torch.tanh((t - self.t_ref).unsqueeze(-1) * self.mlp_w1 + self.mlp_b1)
            for o in range(self.O):
                Hm = zz[:, o, :].double()
                Hm = torch.cat([Hm, torch.ones_like(Hm[:, :1])], 1)
                # RIDGE, not plain least squares: t_out is quantised to dt so the tanh
                # features are near-collinear and an unregularised solve overflows.
                G = Hm.t() @ Hm + 1e-3 * torch.eye(Hm.shape[1], dtype=Hm.dtype,
                                                   device=Hm.device)
                sol = torch.linalg.solve(G, Hm.t() @ a_ref[:, o].double().unsqueeze(-1))
                self.mlp_w2[o] = sol[:-1, 0].float()
                self.mlp_b2[o] = float(sol[-1, 0])
            return dict(mode="mlp", t_range=[float(t.min()), float(t.max())],
                        fired_frac=float(info["fired_o"].mean()),
                        cell_spikes=float(info["cell_spikes"].mean()),
                        theta_out=self.theta_out.tolist())
        if self.decode_mode != "affine":
            self.t_ref.copy_(t.min(0).values - 1e-3)
            xf = torch.exp(-(t - self.t_ref) / self.tau_m_o).clamp(1e-6, 1.0 - 1e-6)
            t = torch.log(self.theta_out.view(1, -1) / (xf * (1.0 - xf)))
        for o in range(self.O):
            to, ao = t[:, o].double(), a_ref[:, o].double()
            tm, am = to.mean(), ao.mean()
            var = ((to - tm) ** 2).sum()
            al = ((to - tm) * (ao - am)).sum() / var.clamp_min(1e-30)
            self.dec_a[o] = float(al)
            self.dec_b[o] = float(am - al * tm)
        return dict(theta_out=self.theta_out.tolist(),
                    fired_frac=float(info["fired_o"].mean()),
                    t_range=[float(t.min()), float(t.max())],
                    cell_spikes=float(info["cell_spikes"].mean()))

    @torch.no_grad()
    def _peak_vo(self, x):
        B = x.shape[0]
        dev, dtp = x.device, x.dtype
        inj_e, inj_i = self.input_current(x)
        gate_n = int(self.gate_open / self.dt)
        Ir = torch.zeros(B, self.nR, device=dev, dtype=dtp); Vr = torch.zeros_like(Ir)
        Ii = torch.zeros_like(Ir); fr = torch.zeros_like(Ir)
        Ic = torch.zeros(B, self.nT * self.K, device=dev, dtype=dtp); Vc = torch.zeros_like(Ic)
        fc = torch.zeros_like(Ic)
        Io = torch.zeros(B, self.O, device=dev, dtype=dtp); Vo = torch.zeros_like(Io)
        pk = torch.full((B, self.O), -1e30, device=dev, dtype=dtp)
        for n in range(self.n_steps):
            Vr = self.a_m_r * Vr + Ir - Ii
            Ir = self.a_s_r * Ir + inj_e[:, :, n]
            Ii = self.a_i_r * Ii + inj_i[:, :, n]
            s_r = (Vr > self.theta_race).to(dtp) * (1.0 - fr)
            fr = torch.clamp(fr + s_r, max=1.0)
            Vc = self.a_m_c * Vc + Ic
            Ic = self.a_s_c * Ic + self.w_cell * (s_r @ self.cell_M.t())
            gate = 0.0 if n >= gate_n else self.gate_G
            s_c = (Vc > self.theta_cell + gate).to(dtp) * (1.0 - fc)
            fc = torch.clamp(fc + s_c, max=1.0)
            Vo = self.a_m_o * Vo + Io
            Io = self.a_s_o * Io + self.out_scale * (s_c @ self.w_out)
            pk = torch.maximum(pk, Vo)
        return pk
