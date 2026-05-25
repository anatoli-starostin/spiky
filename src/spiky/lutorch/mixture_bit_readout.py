"""Mixture-of-tables BitReadout — Mixture-of-Softmaxes with popcount components.

Each of T "tables" is a NAP=B (=15) popcount expert: it projects the residual to
a B-bit sign code `p_t`, every token has a learned B-bit code `b_{t,v}`, and the
table's match score `p_t·b_{t,v}` (a popcount) defines a Hamming-normal softmax
over the vocab:
    P_t(v) = softmax_v( scale_t · p_t(x)·b_{t,v} )
A context gate mixes the T experts:
    P(v) = Σ_t π_t(x) · P_t(v)          π(x) = softmax(x·W_gate)
and the head returns log P(v) (already normalized).

Why a MIXTURE and not a sum: summing the tables' match scores (or multiplying the
experts) collapses algebraically to a single flat dot — one unimodal Hamming-normal,
rank-bounded. AVERAGING the per-table softmaxes is multimodal and breaks the
softmax rank bottleneck (this is Mixture-of-Softmaxes, Yang et al.). That's the
only combination where "many NAP=15 tables" buys expressiveness a flat readout can't.

Each component is matmul-free (popcount); the gate is a tiny D×T linear. Forward
returns normalized log-probs so `F.cross_entropy(logP, target)` = `-logP[target]`
(the extra softmax is a no-op since logP is normalized) and eval works unchanged.
Trade-off vs the flat BitReadout: the per-table softmax normalizer needs all V, so
this is not sub-linear at inference (but it is multimodal + richer).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MixtureBitReadout(nn.Module):
    def __init__(self, input_dim, vocab_size, n_tables, bits_per_table=15,
                 sign_temp=0.5, logit_scale_init=None, device=None, seed=0):
        super().__init__()
        self.D = input_dim
        self.V = vocab_size
        self.T = n_tables
        self.B = bits_per_table
        self.sign_temp = sign_temp

        dev = device or torch.device("cpu")
        g = torch.Generator(device=dev).manual_seed(int(seed))
        # T tables x B anchor pairs over the residual coords.
        a = torch.randint(0, input_dim, (n_tables, bits_per_table), generator=g, device=dev)
        b = torch.randint(0, input_dim, (n_tables, bits_per_table), generator=g, device=dev)
        clash = a == b
        b[clash] = (b[clash] + 1) % input_dim
        self.register_buffer("anchor_a", a)
        self.register_buffer("anchor_b", b)
        # per-table per-token code latent -> sign = ±1 code.
        self.latent = nn.Parameter(torch.randn(n_tables, vocab_size, bits_per_table, generator=g, device=dev))
        # per-table logit scale. 1/sqrt(B) gives ~std-1 logits => each expert is
        # near-uniform and the average of T of them is flatter still (weak gradient,
        # the exp547 crawl). A larger init makes each expert peaked enough to give a
        # real sharpening signal.
        si = (1.0 / math.sqrt(bits_per_table)) if logit_scale_init is None else logit_scale_init
        self.scale = nn.Parameter(torch.full((n_tables,), float(si), device=dev))
        # context gate over the T experts.
        self.gate = nn.Parameter(torch.randn(input_dim, n_tables, generator=g, device=dev) * (1.0 / math.sqrt(input_dim)))

    def _table_logcomp(self, x, log_pi_t, t):
        """log π_t + log P_t(·) for one expert -> [N, V]. Checkpointed."""
        d = x[:, self.anchor_a[t]] - x[:, self.anchor_b[t]]      # [N, B]
        p_soft = d / (self.sign_temp + d.abs())
        p = p_soft + (torch.sign(d) - p_soft).detach()           # STE sign [N, B]
        z = self.latent[t]                                       # [V, B]
        bcode = z + (torch.sign(z) - z).detach()                 # STE sign [V, B]
        match = (p @ bcode.t()) * self.scale[t]                  # [N, V] popcount
        logPt = F.log_softmax(match, dim=-1)                     # [N, V]
        return log_pi_t.unsqueeze(-1) + logPt                    # [N, V]

    def forward(self, x):
        """x: [N, D] -> normalized mixture log-probs logP [N, V]."""
        log_pi = F.log_softmax(x @ self.gate, dim=-1)            # [N, T]
        logP = None
        for t in range(self.T):
            # per-table checkpoint: recompute the [N,V] expert in backward so only
            # one expert's activation lives at a time (else T retained ~ 40GB).
            comp = checkpoint(self._table_logcomp, x, log_pi[:, t], t,
                              use_reentrant=False)               # [N, V]
            logP = comp if logP is None else torch.logaddexp(logP, comp)
        return logP                                              # [N, V] (Σ_v exp = 1)

    def extra_repr(self):
        return (f"D={self.D}, V={self.V}, T={self.T} experts x {self.B} bits | "
                f"Mixture-of-Softmaxes (popcount), deploy=codes[{self.T}x{self.V}x{self.B}b]+gate")
