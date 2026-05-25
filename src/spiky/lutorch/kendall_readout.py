"""KendallReadout — unembedder as an (approximate) Kendall-tau popcount, tied to
the embeddings.

The residual LUTs now map to E (=64): the residual stream is a *predicted token
embedding* ê, in the same space as tok_emb. The logit for token v is the rank
agreement (Kendall tau) between ê and the stored embedding emb_v:

    logit(v) = τ(ê, emb_v) ∝ Σ_p sign(ê_{i_p}-ê_{j_p}) · sign(emb_{v,i_p}-emb_{v,j_p})
             = ⟨ s_ê , s_{emb_v} ⟩            (a popcount over partial-order codes)

over K = E·log₂E sampled coordinate pairs (a sorting-network's worth of
comparisons captures the rank order; full Kendall would be E² pairs). The token
codes are NOT learned separately — they are derived from the embeddings (tied), so
the embedder *is* the unembedder. Matmul-free at deploy (popcount of 1-bit codes);
~32× less head bandwidth than a Linear head; the residual LUTs also shrink (E vs D).

STE: both codes are hard sign() in the forward with soft-sign backward, so
gradients flow to ê (the residual LUTs) and to emb (co-adapts).
"""
import math
import torch
import torch.nn as nn


def _ste_sign(d, temp):
    soft = d / (temp + d.abs())
    return soft + (torch.sign(d) - soft).detach()   # fwd = sign(d), bwd = d(soft)


class KendallReadout(nn.Module):
    def __init__(self, embed_dim, n_pairs, sign_temp=0.5, hard_sign=True,
                 full_pairs=False, learnable_sign_temp=False, device=None, seed=0):
        super().__init__()
        self.E = embed_dim
        # hard_sign=True  -> ±1 codes, popcount, matmul-free deploy (STE backward).
        # hard_sign=False -> soft rank-similarity (continuous), real low-rank matmul.
        self.hard_sign = hard_sign
        dev = device or torch.device("cpu")
        # sign temperature (log-space). Backward-only when hard_sign (shapes the STE
        # gradient); learnable -> the loss self-tunes how steeply gradients flow
        # through the rank comparisons. No effect on the deployed popcount.
        _lst = torch.tensor(math.log(sign_temp), device=dev)
        if learnable_sign_temp:
            self.log_sign_temp = nn.Parameter(_lst)
        else:
            self.register_buffer("log_sign_temp", _lst)
        g = torch.Generator(device=dev).manual_seed(int(seed))
        if full_pairs:
            # ALL i<j pairs -> exact Kendall tau (E(E-1)/2 comparisons).
            i, j = torch.triu_indices(embed_dim, embed_dim, offset=1, device=dev)
        else:
            i = torch.randint(0, embed_dim, (n_pairs,), generator=g, device=dev)
            j = torch.randint(0, embed_dim, (n_pairs,), generator=g, device=dev)
            clash = i == j
            j[clash] = (j[clash] + 1) % embed_dim
        self.K = int(i.numel())
        self.register_buffer("pair_i", i)
        self.register_buffer("pair_j", j)
        # popcount over K bits has std ~sqrt(K); scale to O(1) logits.
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(self.K)))

    def _code(self, z):
        T = self.log_sign_temp.exp()
        d = z[:, self.pair_i] - z[:, self.pair_j]                                   # [*, K]
        if self.hard_sign:
            return _ste_sign(d, T)                                                  # ±1 (STE)
        return d / (T + d.abs())                                                    # soft rank in (-1,1)

    def forward(self, x, emb):
        """x: predicted embedding [N, E]; emb: vocab embeddings [V, E] -> [N, V]."""
        s_e = self._code(x)                                                          # [N, K]
        s_v = self._code(emb)                                                        # [V, K]
        return (s_e @ s_v.t()) * self.logit_scale                                   # [N, V]

    def extra_repr(self):
        return (f"E={self.E}, K={self.K} pairs (E·log2E), TIED to embeddings | "
                f"logit = approx-Kendall popcount, matmul-free at deploy")
