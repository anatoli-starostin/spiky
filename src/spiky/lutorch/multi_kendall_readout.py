"""MultiKendallReadout — many partial-order views, nonlinearly aggregated.

Generalizes the single-set Kendall head: instead of one set of pairs, use T
"weightless LUT" tables, each a set of NAP coordinate pairs (a partial-order
extractor). Apply the SAME tables to both the predicted embedding ê and every
vocab embedding emb_v; each table t yields a per-table Kendall τ_t (popcount of
the two partial-order codes over its NAP pairs). Then a small SHARED MLP
aggregates the T per-table τ's into the logit:

    τ_t(ê, emb_v) = Σ_{p∈t} sign(ê_i-ê_j)·sign(emb_{v,i}-emb_{v,j})   (popcount, per table)
    logit(v)      = logit_scale · MLP( τ_1(v)/NAP, …, τ_T(v)/NAP )

Why the MLP and not a sum: summing the τ's collapses to one flat popcount over all
T·NAP pairs (the tables vanish — that's the existing single-set head). A nonlinear
aggregator makes the tables matter (can express "agree on most views", "view 3
dominates", etc.) — the lever the flat/sum head can't reach. Trained from all
tables (no gate/responsibility), so it avoids the mixture cold-start.

Tied to the embeddings (codes derived from emb_v, no separate output params). Hard
popcount per table (STE: soft-sign backward). V is chunked + checkpointed because
the per-table τ tensor is [N, V, T].
"""
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MultiKendallReadout(nn.Module):
    def __init__(self, embed_dim, n_tables, nap, mlp_hidden=16, sign_temp=0.5,
                 v_chunk=8192, device=None, seed=0):
        super().__init__()
        self.E = embed_dim
        self.T = n_tables
        self.NAP = nap
        self.sign_temp = sign_temp
        self.v_chunk = v_chunk
        dev = device or torch.device("cpu")
        g = torch.Generator(device=dev).manual_seed(int(seed))
        i = torch.randint(0, embed_dim, (n_tables, nap), generator=g, device=dev)
        j = torch.randint(0, embed_dim, (n_tables, nap), generator=g, device=dev)
        clash = i == j
        j[clash] = (j[clash] + 1) % embed_dim
        self.register_buffer("pair_i", i)                       # [T, NAP]
        self.register_buffer("pair_j", j)
        self.mlp = nn.Sequential(
            nn.Linear(n_tables, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, 1))
        # the MLP shrinks the small τ/NAP input, so the raw output std is tiny;
        # scale up to get ~std-1 initial logits (learnable, self-corrects).
        self.logit_scale = nn.Parameter(torch.tensor(float(nap)))

    def _codes(self, z):
        d = z[:, self.pair_i] - z[:, self.pair_j]               # [M, T, NAP]
        soft = d / (self.sign_temp + d.abs())
        return soft + (torch.sign(d) - soft).detach()           # ±1 STE

    def _chunk_logit(self, s_e, s_v_chunk):
        tau = torch.einsum("ntp,vtp->nvt", s_e, s_v_chunk) / self.NAP   # [N, chunk, T] in [-1,1]
        return self.mlp(tau).squeeze(-1) * self.logit_scale            # [N, chunk]

    def forward(self, x, emb):
        """x: [N, E] predicted embedding; emb: [V, E] vocab embeddings -> [N, V]."""
        s_e = self._codes(x)                                    # [N, T, NAP]
        s_v = self._codes(emb)                                  # [V, T, NAP]
        if self.v_chunk and self.v_chunk < emb.shape[0]:
            # memory-bounded path: chunk V + recompute in backward (slower).
            outs = [checkpoint(self._chunk_logit, s_e, s_v[c0:c0 + self.v_chunk],
                               use_reentrant=False)
                    for c0 in range(0, emb.shape[0], self.v_chunk)]
            return torch.cat(outs, dim=1)
        return self._chunk_logit(s_e, s_v)                     # one-shot (faster, more memory)

    def extra_repr(self):
        return (f"E={self.E}, T={self.T} tables x NAP={self.NAP} pairs, MLP-aggregated | "
                f"tied to embeddings, hard popcount per table")
