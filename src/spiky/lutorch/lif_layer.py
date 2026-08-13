"""LIFLayer — a minimal, stackable single-layer LIF spiking transform (times -> times).

A pared-down sibling of LIFMultiHeadLUT: it keeps ONLY the LIF timing/membrane machinery and
emits the differentiable first-spike TIME of each output neuron. There is NO head concept, NO
bucketing, NO table, NO mixed-radix gather, NO readout — the output IS the spike times, so the
layer stacks: layer_{k+1}.in_dim == layer_k.out_dim, feeding times straight in as the next
layer's inputs (the universal-timing arrival conversion applies again in each layer).

Mechanism (D neurons, each with N synapses; the SAME machinery as universal-timing LIFMultiHeadLUT):
  • Universal timing: arrival a_ij = clamp(baseline_C - gain_j * x_j + delay_ij, 0, t_window), with a
    LEARNABLE per-input-channel gain (init t_window*3/32 = the old fixed alpha=3) and a single
    trainable scalar baseline_C (init 0.5*t_window). Delays are FREE/signed; positivity is enforced
    on the FINAL arrival, not on the delay.
  • Bounded-excitatory weights w = w_max*sigmoid(w_raw) (hot init), per-neuron tau = softplus(tau_raw)+1.0.
  • Closed-form O(N) cumsum LIF membrane V = exp(-a/tau) * cumsum(w * exp(a/tau)) over sorted arrivals.
  • Differentiable SOFT first-spike time via the crossing temperature T_cross (per neuron): with
    crossing prob c_k = sigmoid((V_k - theta)/T_cross) at the k-th arrival and survival S_k = prod(1-c),
    the first-spike distribution is p_k = c_k * S_{k-1} and the expected time is
        t_spike = sum_k p_k * a_k  +  S_last * t_window.
    A neuron that never reaches threshold (S_last ~ 1) outputs t_window (the latest possible time) —
    the natural "no-spike" value; every output is guaranteed to lie in [0, t_window].

forward(x: [B, in_dim]) -> [B, out_dim]. The output transform is per-instance (output_transform):
 - "rescale" (DEFAULT, inter-layer): an INPUT-INDEPENDENT learnable per-channel affine
   out = rescale_scale*t_spike + rescale_bias, init to standardize the raw spike times (out0 = (t-mean)/std).
   Same affine for every input (fixed learnable params, NOT per-sample statistics) — it conditions the
   timing code for the next layer WITHOUT the sample-dependent distortion a per-sample LayerNorm causes.
 - "log" (final readout ONLY): out = log(t_spike + log_eps). Apply at the LAST layer, before the action decode.
 - "linear": the raw first-spike time (tests / manual use).
Fully differentiable (no straight-through / no train/eval branch — the soft time IS the output).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LIFLayer"]


class LIFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, *, w_max=2.0, t_window=32.0, delay_init_std=0.0,
                 freeze_temperature=False, output_transform="rescale", log_eps=None, gain_init=None,
                 rescale_init_mean=12.0, rescale_init_std=1.0, device=None):
        # output_transform: how the raw first-spike time t_spike becomes the layer output.
        #   "rescale" (DEFAULT, the inter-layer conditioner): an INPUT-INDEPENDENT learnable per-channel
        #       affine  out = rescale_scale * t_spike + rescale_bias, initialized to STANDARDIZE the raw
        #       spike-time range (scale = 1/rescale_init_std, bias = -rescale_init_mean/rescale_init_std,
        #       i.e. out0 = (t - mean)/std). Same affine for EVERY input (fixed learnable params, NOT
        #       per-sample statistics) -> it conditions the timing code for the next layer WITHOUT the
        #       sample-dependent distortion a per-sample LayerNorm would introduce.
        #   "log" (final readout only): out = log(t_spike + log_eps). Apply ONLY at the last layer, before
        #       the action decode.
        #   "linear": raw first-spike time (for tests / manual use).
        # log_eps: epsilon floor so an immediate spike (t->0) doesn't map to -inf under "log". Default
        #   1e-3*t_window (= 0.032 at t_window=32); floors log at log(eps) ~ -3.44 for t->0.
        # rescale_init_mean / rescale_init_std: the raw spike-time init stats used to standardize the
        #   "rescale" output (measured ~12 / ~1 for a layer fed ~unit-variance input). Learnable, so per-
        #   layer differences are absorbed during training; the init just keeps stacked arrivals in range.
        # gain_init: initial (learnable) per-input-channel gain. Default t_window*3/32 (unit-variance calib).
        super().__init__()
        if output_transform not in ("rescale", "log", "linear"):
            raise ValueError(f"output_transform must be 'rescale', 'log' or 'linear', got {output_transform!r}")
        N, D = int(in_dim), int(out_dim)
        self.in_dim = N
        self.out_dim = D
        self.w_max = float(w_max)
        self.t_window = float(t_window)
        self.output_transform = str(output_transform)
        self.log_eps = float(1e-3 * self.t_window if log_eps is None else log_eps)
        dev = device or torch.device("cpu")
        gain0 = self.t_window * 3.0 / 32.0 if gain_init is None else float(gain_init)
        # per-neuron per-synapse delays: FREE/signed. std==0 => exact zeros (neutral start, no RNG draw).
        if float(delay_init_std) > 0.0:
            init = float(delay_init_std) * torch.randn(D, N, device=dev)
        else:
            init = torch.zeros(D, N, device=dev)
        self.delay = nn.Parameter(init)                                       # (D, N)
        self.w_raw = nn.Parameter(-2.2 + 0.5 * torch.randn(D, N, device=dev))  # (D, N) bounded-excitatory hot init
        self.tau_raw = nn.Parameter(torch.ones(D, device=dev))                # (D,) per-neuron tau
        # universal timing: learnable per-input-channel gain + a single trainable scalar baseline C.
        self.gain = nn.Parameter(torch.full((N,), gain0, device=dev))         # (N,) per-input-channel
        self.baseline_C = nn.Parameter(torch.tensor(0.5 * self.t_window, device=dev))        # scalar
        # per-neuron soft crossing temperature (init exp(0)=1.0). Trainable unless frozen.
        trainable_T = not bool(freeze_temperature)
        self.log_T_cross = nn.Parameter(torch.zeros(D, device=dev), requires_grad=trainable_T)  # (D,)
        self.register_buffer("theta_mem", torch.tensor(1.0, device=dev))
        # "rescale" output transform: an INPUT-INDEPENDENT per-channel affine (learnable scale+bias, the
        # SAME for every input — not per-sample stats). Init standardizes the raw spike times:
        #   out0 = (t_spike - rescale_init_mean) / rescale_init_std. Only allocated for output_transform
        # == "rescale"; the log/linear paths carry no extra params.
        if self.output_transform == "rescale":
            std0 = float(rescale_init_std)
            self.rescale_scale = nn.Parameter(torch.full((D,), 1.0 / std0, device=dev))              # (D,)
            self.rescale_bias = nn.Parameter(torch.full((D,), -float(rescale_init_mean) / std0, device=dev))

    @property
    def w(self):
        return self.w_max * torch.sigmoid(self.w_raw)          # (D, N)

    @property
    def tau(self):
        return F.softplus(self.tau_raw) + 1.0                  # (D,)

    @property
    def T_cross(self):
        return torch.exp(self.log_T_cross)                     # (D,)

    @torch.compile
    def forward(self, x):
        """x: [B, in_dim] -> [B, out_dim] first-spike times in [0, t_window]."""
        B, N, D = x.shape[0], self.in_dim, self.out_dim
        # universal-timing arrival: (B, D, N)
        a = torch.clamp(self.baseline_C
                        - self.gain.view(1, 1, N) * x.view(B, 1, N)
                        + self.delay.unsqueeze(0), 0.0, self.t_window)
        a_srt, idx = torch.sort(a, dim=-1)                                    # (B, D, N)
        w_srt = self.w.unsqueeze(0).expand(B, -1, -1).gather(-1, idx)         # (B, D, N)
        tv = self.tau.view(1, D, 1)
        V = torch.exp(-a_srt / tv) * torch.cumsum(w_srt * torch.exp(a_srt / tv), dim=-1)   # O(N) cumsum membrane
        # differentiable soft first-spike time
        Tc = self.T_cross.view(1, D, 1)
        c = torch.sigmoid((V - self.theta_mem) / Tc)                          # per-arrival crossing prob
        surv = torch.cumprod(1.0 - c, dim=-1)
        surv_prev = torch.cat([torch.ones_like(surv[..., :1]), surv[..., :-1]], dim=-1)
        p = c * surv_prev                                                     # first-spike-at-k prob
        t_spike = (p * a_srt).sum(-1) + surv[..., -1] * self.t_window         # (B, D); no-spike -> t_window
        if self.output_transform == "rescale":
            return self.rescale_scale * t_spike + self.rescale_bias           # input-independent affine (default)
        if self.output_transform == "log":
            return torch.log(t_spike + self.log_eps)                          # final-readout log-time
        return t_spike                                                        # raw first-spike time

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
