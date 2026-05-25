"""BitReadout head — the unembedder as a popcount.

Insight (user, exp544 follow-up): a TinyMHLut table's match score for row r is
ts[r] = Σ_i p_i(x)·bitmatrix[i,r], a binary dot of x's anchor signs against the
row's ±1 pattern. If every vocab token owns a ±1 code b_v, then

    logit(v) = p(x) · b_v          p_i = sign(x[a_i] - x[b_i])  ∈ {±1}^P
                                    b_v ∈ {±1}^P  (learned token code)

is a full DOT PRODUCT over a P-bit code (rank P) — same richness class as a Linear
head — but binary, so at inference it's an XNOR-popcount (matmul-free), and the
softmax over codes is Hamming-smooth (a product of per-bit Bernoullis).

Contrast with ScoreLUT (exp542-544): that read m SCALARS (rank-m, too weak, capped
~+0.2 vs Linear). This reads a full P-dim binary dot, so it should match Linear's
expressiveness while staying matmul-free.

Forward is the exact hard popcount (sign·sign) via STE, so the logged bpb is the
deployable model: p uses a soft-sign backward (gradient to the residual x); the
token codes b use a straight-through sign (gradient to a per-token latent). At
deploy keep only the P-bit codes per token + the anchor pairs; logits = popcount.
"""
import math
import torch
import torch.nn as nn


class BitReadoutHead(nn.Module):
    def __init__(self, input_dim, vocab_size, n_bits, sign_temp=0.5,
                 device=None, seed=0):
        super().__init__()
        self.D = input_dim
        self.V = vocab_size
        self.P = n_bits
        self.sign_temp = sign_temp
        self.hard = False  # eval toggle; forward is hard-popcount either way (STE)

        dev = device or torch.device("cpu")
        g = torch.Generator(device=dev).manual_seed(int(seed))
        # P fixed anchor pairs over the residual coords (ranking comparisons).
        a = torch.randint(0, input_dim, (n_bits,), generator=g, device=dev)
        b = torch.randint(0, input_dim, (n_bits,), generator=g, device=dev)
        clash = a == b                                  # avoid a==b (d would be 0)
        b[clash] = (b[clash] + 1) % input_dim
        self.register_buffer("anchor_a", a)
        self.register_buffer("anchor_b", b)
        # per-token code latent -> sign = ±1 code.
        self.latent = nn.Parameter(torch.randn(vocab_size, n_bits, generator=g, device=dev))
        # learnable logit temperature: popcount is a sum of P (=n_bits) ±1 terms,
        # so its std is ~sqrt(P) (~32 at P=1024) -> the softmax saturates and CE
        # explodes unless the logits are scaled to O(1). init at 1/sqrt(P).
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(n_bits)))

    def forward(self, x):
        """x: [N, D] residual -> logits [N, V]. Forward is the exact popcount."""
        d = x[:, self.anchor_a] - x[:, self.anchor_b]            # [N, P]
        # x's bit code: hard sign forward, soft-sign backward (gradient to x).
        p_soft = d / (self.sign_temp + d.abs())
        p = p_soft + (torch.sign(d) - p_soft).detach()          # STE: fwd=sign(d)
        # token codes: hard sign forward, straight-through to latent.
        b = self.latent + (torch.sign(self.latent) - self.latent).detach()  # fwd=sign(latent)
        logits = (p @ b.t()) * self.logit_scale                  # [N, V]  scaled popcount
        return logits

    def extra_repr(self):
        return (f"D={self.D}, V={self.V}, P={self.P} bits | rank={self.P}, "
                f"deploy=codes[{self.V}x{self.P}b]+anchors[2x{self.P}], popcount")
