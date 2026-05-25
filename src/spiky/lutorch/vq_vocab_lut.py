"""VQ VocabLUT — a product-quantized (VQ) unembedder, the *transposed* LUT.

Duality with the backbone TinyMultiHeadLut (both are sum-pooled gathers,
`embedding_bag(table, indices, sum)`):
  - backbone LUT: the INDEX is computed from x (anchor bits), the TABLE is static
    learned weights.
  - VocabLUT head: the TABLE is computed from x (`x_j · C_j`), the INDEX is the
    token's learned code. Tokens are "output prototypes" addressing the codebooks.

The V×D logit-weight matrix is parameterized as the VQ of a learnable per-token
latent into m sub-codebooks:
    logit(v) = x · concat_j C_j[code_v[j]],   code_v[j] = argmin_k ||latent_v[j] - C_j[k]||
Trained VQ-VAE style: forward uses the quantized weights via STE (grad -> latent),
codebooks learned via the codebook loss, latent pulled to codes via commitment.
At deploy only `codes [V, m]` (b bits each) + `codebooks [m, 2^b, D/m]` are kept,
and the V-side is matmul-free (one tiny `2^b·D` precompute + V·m lookups).
Training materializes the dense quantized weight and does a normal matmul (cheap
to implement; the table trick is an inference detail giving identical logits).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVocabLUT(nn.Module):
    def __init__(self, input_dim, vocab_size, m, b, commit_beta=0.25,
                 device=None, seed=0):
        super().__init__()
        if input_dim % m != 0:
            raise ValueError(f"input_dim {input_dim} must be divisible by m {m}")
        self.D = input_dim
        self.V = vocab_size
        self.m = m
        self.b = b
        self.k = 1 << b
        self.sub = input_dim // m
        self.commit_beta = commit_beta

        dev = device or torch.device("cpu")
        g = torch.Generator(device=dev).manual_seed(int(seed))

        # latent: full-capacity per-token weight, init like nn.Linear(D, V) weight.
        bound = 1.0 / math.sqrt(input_dim)
        self.latent = nn.Parameter(
            (torch.rand(vocab_size, input_dim, generator=g, device=dev) * 2 - 1) * bound
        )
        # codebooks init from random latent subvectors (ensures correct scale + coverage).
        cb = torch.empty(m, self.k, self.sub, device=dev)
        with torch.no_grad():
            for j in range(m):
                idx = torch.randint(0, vocab_size, (self.k,), generator=g, device=dev)
                cb[j] = self.latent[idx, j * self.sub:(j + 1) * self.sub]
        self.codebooks = nn.Parameter(cb)

        self._vq_loss = None        # set each forward; add to the task loss
        self._code_usage = None     # fraction of codebook entries used (monitoring)

    @torch.no_grad()
    def _assign(self):
        """codes [m, V]: nearest codebook entry per token per subspace (no grad)."""
        codes = torch.empty(self.m, self.V, dtype=torch.long, device=self.latent.device)
        for j in range(self.m):
            lat_j = self.latent[:, j * self.sub:(j + 1) * self.sub]
            codes[j] = torch.cdist(lat_j, self.codebooks[j]).argmin(dim=-1)
        return codes

    def quantized_weight(self):
        """Differentiable (wrt codebooks) quantized weight [V, D] + codes [m, V]."""
        codes = self._assign()
        eq = self.codebooks.gather(1, codes.unsqueeze(-1).expand(-1, -1, self.sub))  # [m,V,sub]
        e_q = eq.permute(1, 0, 2).reshape(self.V, self.D)                            # [V,D]
        return e_q, codes

    def forward(self, x):
        """x: [N, D] -> logits [N, V]. Stashes `_vq_loss` for the caller to add."""
        e_q, codes = self.quantized_weight()
        lat = self.latent
        # STE: forward uses quantized weight, gradient flows to the latent.
        e_q_ste = lat + (e_q - lat).detach()
        logits = x @ e_q_ste.t()

        commit = F.mse_loss(lat, e_q.detach())          # grad -> latent
        codebook = F.mse_loss(e_q, lat.detach())        # grad -> codebooks
        self._vq_loss = codebook + self.commit_beta * commit
        with torch.no_grad():
            used = sum(int(codes[j].unique().numel()) for j in range(self.m))
            self._code_usage = used / (self.m * self.k)
        return logits

    def extra_repr(self):
        return (f"V={self.V}, D={self.D}, m={self.m}, b={self.b} (k={self.k}), "
                f"sub={self.sub}, deploy_params=codes[{self.V}x{self.m}x{self.b}b]+"
                f"codebooks[{self.m}x{self.k}x{self.sub}]")
