"""Multi-Map 3D Unembedder (MDN head) — drop-in replacement for a dense Linear(384->V) unembedder.

Per spec /tmp/mdn-head-experiment-spec.md. Given hidden h (d_model), a single linear P predicts, for
each of M mixture components: a mixture logit alpha_m and, per map n in [N], a 3D mean mu and a 6-number
lower-triangular Cholesky factor L (full-covariance 3D Gaussian). Token v has learned coords X[v,n] in
each 3D map. Score:
    log phi_m(v) = sum_n [ sum_i log L_ii,mn  -  0.5 * (x_v^n - mu_mn)^T (L L^T)_mn (x_v^n - mu_mn) ]
    logit_v      = logsumexp_m [ log pi_m + log phi_m(v) ] + b_v          (soft intersection = product of densities)
    P(v|h)       = softmax_v(logit_v)                                     (discriminative categorical head)

Interface matches an unembedder: forward(h) -> logits [.., V]. Also `.decorrelation()` and `.param_count()`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# lower-tri 3x3 packing of the 6 raw numbers: (0,0),(1,0),(1,1),(2,0),(2,1),(2,2)
_DIAG_IDX = [0, 2, 5]     # raw positions that are diagonal entries
_TRI_ROWS = [0, 1, 1, 2, 2, 2]
_TRI_COLS = [0, 0, 1, 0, 1, 2]


class MDNHead(nn.Module):
    def __init__(self, d_model, vocab_size, n_maps=11, n_mix=8, block=3,
                 gamma_dec=1e-2, diag_clamp=4.0, x_init=None, b_init=None, device=None):
        super().__init__()
        assert block == 3, "spec is 3D maps (block=3); other block sizes are a separate ablation"
        self.d = d_model; self.V = vocab_size; self.N = n_maps; self.M = n_mix
        self.B = block; self.gamma_dec = gamma_dec; self.diag_clamp = diag_clamp
        per_comp = 1 + n_maps * (3 + 6)                     # alpha + per-map (mu3 + L6) = 1 + 9N
        self.out_dim = n_mix * per_comp
        self.P = nn.Linear(d_model, self.out_dim, device=device)   # h -> all params
        # token coordinate table X: [V, N, 3]  (embedding-like)
        if x_init is not None:
            self.X = nn.Parameter(torch.as_tensor(x_init, dtype=torch.float32, device=device).reshape(vocab_size, n_maps, 3))
        else:
            self.X = nn.Parameter(0.02 * torch.randn(vocab_size, n_maps, 3, device=device))
        # per-token bias
        b0 = torch.zeros(vocab_size, device=device) if b_init is None else torch.as_tensor(b_init, dtype=torch.float32, device=device)
        self.b = nn.Parameter(b0)
        # init P small so early logits ~ b (unigram); zero mu/L-offdiag, unit precision (L diag raw=0)
        nn.init.normal_(self.P.weight, std=1e-3)
        nn.init.zeros_(self.P.bias)

    # ---- parameter groups helper: X (embedding-like) + b -> no weight decay; P.weight -> wd ----
    def param_groups(self):
        return dict(decay=[self.P.weight], no_decay=[self.X, self.b, self.P.bias])

    @staticmethod
    def param_count(d_model, vocab_size, n_maps=11, n_mix=8):
        per_comp = 1 + n_maps * 9
        P = d_model * (n_mix * per_comp) + (n_mix * per_comp)   # weight + bias
        X = vocab_size * n_maps * 3
        b = vocab_size
        return dict(total=P + X + b, X=X, P=P, b=b)

    def _unpack(self, h):
        """h [B,d] -> logpi [B,M], mu [B,M,N,3], Ldiag_raw [B,M,N,3] (clamped), Lam [B,M,N,3,3]."""
        Bn = h.shape[0]
        o = self.P(h).view(Bn, self.M, 1 + self.N * 9)
        alpha = o[:, :, 0]                                   # [B,M]
        logpi = F.log_softmax(alpha, dim=1)                  # [B,M]
        rest = o[:, :, 1:].view(Bn, self.M, self.N, 9)
        mu = rest[..., :3]                                   # [B,M,N,3]
        raw6 = rest[..., 3:]                                 # [B,M,N,6]
        # build L [B,M,N,3,3]
        L = h.new_zeros(Bn, self.M, self.N, 3, 3)
        diag_raw = torch.clamp(raw6[..., _DIAG_IDX], -self.diag_clamp, self.diag_clamp)   # [B,M,N,3]
        for k, (r, c) in enumerate(zip(_TRI_ROWS, _TRI_COLS)):
            if k in (0, 2, 5):
                L[..., r, c] = torch.exp(torch.clamp(raw6[..., k], -self.diag_clamp, self.diag_clamp))
            else:
                L[..., r, c] = raw6[..., k]
        logdet_half = diag_raw.sum(-1)                       # [B,M,N]  (= sum_i log L_ii)
        return logpi, mu, logdet_half, L                     # return Cholesky L directly

    def _logits_rows(self, h):
        """h [b,d] -> logits [b,V]. Loops only over the N maps, vectorizing the M components into
        one batched einsum per map (16-32x fewer kernel launches than the m,n double loop)."""
        logpi, mu, logdet_half, L = self._unpack(h)          # mu [b,M,N,3], L [b,M,N,3,3]
        b = h.shape[0]; bm = b * self.M
        Lam = L @ L.transpose(-1, -2)                        # [b,M,N,3,3] precision
        # q_mn(v) = x^T Λ x - 2 x^T Λμ + μ^T Λμ  (expanded Mahalanobis -> only [V,small]x[small,bM] matmuls)
        logphi = logdet_half.sum(2).unsqueeze(-1)            # [b,M,1]
        for n in range(self.N):
            xn = self.X[:, n, :]                             # [V,3]
            Ln = Lam[:, :, n]                                # [b,M,3,3]
            mn = mu[:, :, n, :]                              # [b,M,3]
            xquad = torch.stack([xn[:, 0] ** 2, xn[:, 1] ** 2, xn[:, 2] ** 2,
                                 2 * xn[:, 0] * xn[:, 1], 2 * xn[:, 0] * xn[:, 2],
                                 2 * xn[:, 1] * xn[:, 2]], dim=1)         # [V,6]
            lam6 = torch.stack([Ln[..., 0, 0], Ln[..., 1, 1], Ln[..., 2, 2],
                                Ln[..., 0, 1], Ln[..., 0, 2], Ln[..., 1, 2]], dim=-1)   # [b,M,6]
            t1 = torch.matmul(xquad, lam6.reshape(bm, 6).t()).reshape(self.V, b, self.M).permute(1, 2, 0)  # [b,M,V]
            Lammu = torch.matmul(Ln, mn.unsqueeze(-1)).squeeze(-1)       # [b,M,3]
            t2 = torch.matmul(xn, Lammu.reshape(bm, 3).t()).reshape(self.V, b, self.M).permute(1, 2, 0)    # [b,M,V]
            t3 = (mn * Lammu).sum(-1).unsqueeze(-1)                      # [b,M,1]
            logphi = logphi - 0.5 * (t1 - 2.0 * t2 + t3)
        return torch.logsumexp(logphi + logpi.unsqueeze(-1), dim=1) + self.b.unsqueeze(0)  # [b,V]

    def forward(self, h, chunk=1024):
        """h [.., d] -> logits [.., V]. When training, gradient-checkpoints over batch chunks so
        the 88 per-(m,n) [b,V] activations are recomputed in backward instead of all held (else OOM)."""
        shp = h.shape[:-1]
        h = h.reshape(-1, self.d)
        if chunk and h.shape[0] > chunk:
            train_cp = self.training and torch.is_grad_enabled()
            outs = []
            for i in range(0, h.shape[0], chunk):
                hc = h[i:i + chunk]
                oc = cp.checkpoint(self._logits_rows, hc, use_reentrant=False) if train_cp else self._logits_rows(hc)
                outs.append(oc)
            logits = torch.cat(outs, 0)
        else:
            logits = self._logits_rows(h)
        return logits.reshape(*shp, self.V)

    def decorrelation(self):
        """Between-block (between-map) decorrelation penalty on standardized X columns.
        Penalize corr between the 3N columns but NOT within a 3-block (leave full-cov free)."""
        Xc = self.X.reshape(self.V, self.N * 3)
        Xz = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
        C = (Xz.T @ Xz) / self.V                             # [3N,3N] correlation
        d = self.N * 3
        mask = torch.ones(d, d, device=C.device)
        for n in range(self.N):                              # zero the N diagonal 3x3 blocks
            mask[3 * n:3 * n + 3, 3 * n:3 * n + 3] = 0.0
        off = C * mask
        return self.gamma_dec * (off.pow(2).sum()) / (d * d - d)
