"""ScoreLUT head — produce logits DIRECTLY from the backbone LUTs, no D-readout
matmul.

The backbone's residual LUTs emit a score table S [N, m, k] (their n_outputs is
set to m*k, accumulated across layers). Each vocab token reads m scores from it —
one slot per subspace, picked by a learned code — and weights+sums them:

    logit(v) = Σ_j  w_v[j] · S[j, code_v[j]]

This is the VocabLUT readout, but S comes straight from the LUTs instead of
x_resid·C, so there is no intermediate D=384 dot-product / matmul on the V side —
deploy is a gather (m lookups + adds per token).

Codes are learned: each token has a query latent z_v[j] (d_z dims) matched against
m static slot-key codebooks K_j [k, d_z]; the slot is chosen by soft attention
(forward soft -> directed gradient) and hardened to argmax at deploy. `hard=True`
uses the exact argmax-gather (the deployable head). `_code_usage` reports the
fraction of slots used (collapse monitor).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreLUTHead(nn.Module):
    def __init__(self, vocab_size, m, k, d_z=8, assign_temp=1.0,
                 assign_init_std=1.0, weight_init_noise=0.5, device=None, seed=0):
        super().__init__()
        self.V = vocab_size
        self.m = m
        self.k = k
        self.d_z = d_z
        self.assign_temp = assign_temp
        self.hard = False
        self._code_usage = None

        dev = device or torch.device("cpu")
        g = torch.Generator(device=dev).manual_seed(int(seed))
        # Query latents + slot keys. init std is large enough that the assignment
        # softmax is DISCRIMINATIVE from step 1 (different tokens read different
        # slots) -- a tiny init leaves softmax ~uniform => all tokens read the
        # subspace average => logits identical for all tokens => symmetry only
        # breaks via glacial gradients (the 3.15 plateau).
        self.latent = nn.Parameter(torch.randn(vocab_size, m, d_z, generator=g, device=dev) * assign_init_std)
        self.keys = nn.Parameter(torch.randn(m, k, d_z, generator=g, device=dev) * assign_init_std)
        # per-token, per-subspace readout weights, init ~1/m with noise so tokens
        # also differ in their readout from the start.
        w = torch.full((vocab_size, m), 1.0 / m, device=dev)
        w = w + torch.randn(vocab_size, m, generator=g, device=dev) * (weight_init_noise / m)
        self.weights = nn.Parameter(w)

    def _assign_logits(self):
        # [V, m, k]: token query . slot key, per subspace.
        return torch.einsum('vmd,mkd->vmk', self.latent, self.keys) / self.assign_temp

    def forward(self, S):
        """S: [N, m, k] score table from the backbone -> logits [N, V]."""
        al = self._assign_logits()                                  # [V, m, k]
        if self.hard:
            code = al.argmax(dim=-1)                                # [V, m]
            sel = F.one_hot(code, self.k).to(S.dtype)               # [V, m, k]
            usage_codes = code
        else:
            sel = F.softmax(al, dim=-1)                             # [V, m, k] soft
            usage_codes = al.argmax(dim=-1)
        A = sel * self.weights.unsqueeze(-1)                        # [V, m, k]
        logits = torch.einsum('nmk,vmk->nv', S, A)                 # [N, V]
        with torch.no_grad():
            self._code_usage = sum(int(usage_codes[:, j].unique().numel())
                                   for j in range(self.m)) / (self.m * self.k)
        return logits

    def extra_repr(self):
        return (f"V={self.V}, m={self.m}, k={self.k}, d_z={self.d_z} | "
                f"score_space={self.m*self.k}, deploy=codes[{self.V}x{self.m}]+w[{self.V}x{self.m}]")
