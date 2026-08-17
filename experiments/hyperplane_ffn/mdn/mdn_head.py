"""Multi-Map B-D Unembedder (MDN head) — drop-in replacement for a dense Linear(d->V) unembedder,
generalized to arbitrary per-map block dim B (was fixed at 3). Also a plain low-rank-linear control head.

MDN: per map n, token v has coords X[v,n] in R^B; the head predicts (from h) a mean mu and a lower-tri
Cholesky L per (component m, map n). Score:
  log phi_m(v) = sum_n [ sum_i log L_ii  - 0.5 (x_v-mu)^T (L L^T) (x_v-mu) ]
  logit_v = logsumexp_m [ log pi_m + log phi_m(v) ] + b_v          (soft intersection, discriminative)
Quadratic evaluated via the expanded form x^T Λ x - 2 x^T Λμ + μ^T Λμ using [V, B^2]x[B^2, bM] matmuls
(works for any B), gradient-checkpointed over batch chunks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def _tri_idx(B):
    """lower-tri (row,col) pairs row-major, and the packed-array indices of the diagonal entries."""
    rows, cols, diag = [], [], []
    k = 0
    for i in range(B):
        for j in range(i + 1):
            rows.append(i); cols.append(j)
            if i == j:
                diag.append(k)
            k += 1
    return rows, cols, diag


class MDNHead(nn.Module):
    def __init__(self, d_model, vocab_size, n_maps=11, n_mix=8, block=3,
                 gamma_dec=1e-2, diag_clamp=4.0, x_init=None, b_init=None, device=None):
        super().__init__()
        self.d = d_model; self.V = vocab_size; self.N = n_maps; self.M = n_mix
        self.B = block; self.gamma_dec = gamma_dec; self.diag_clamp = diag_clamp
        self._rows, self._cols, self._diag = _tri_idx(block)
        self.ltri = block * (block + 1) // 2                 # Cholesky params per (m,n)
        per_comp = 1 + n_maps * (block + self.ltri)          # alpha + per-map (mu + L)
        self.out_dim = n_mix * per_comp
        self.P = nn.Linear(d_model, self.out_dim, device=device)
        if x_init is not None:
            self.X = nn.Parameter(torch.as_tensor(x_init, dtype=torch.float32, device=device).reshape(vocab_size, n_maps, block))
        else:
            self.X = nn.Parameter(0.02 * torch.randn(vocab_size, n_maps, block, device=device))
        b0 = torch.zeros(vocab_size, device=device) if b_init is None else torch.as_tensor(b_init, dtype=torch.float32, device=device)
        self.b = nn.Parameter(b0)
        nn.init.normal_(self.P.weight, std=1e-3)
        nn.init.zeros_(self.P.bias)

    def param_groups(self):
        return dict(decay=[self.P.weight], no_decay=[self.X, self.b, self.P.bias])

    @staticmethod
    def param_count(d_model, vocab_size, n_maps=11, n_mix=8, block=3):
        ltri = block * (block + 1) // 2
        per_comp = 1 + n_maps * (block + ltri)
        P = d_model * (n_mix * per_comp) + (n_mix * per_comp)
        X = vocab_size * n_maps * block
        return dict(total=P + X + vocab_size, X=X, P=P, b=vocab_size)

    def _unpack(self, h):
        Bn = h.shape[0]; B = self.B
        o = self.P(h).view(Bn, self.M, 1 + self.N * (B + self.ltri))
        logpi = F.log_softmax(o[:, :, 0], dim=1)             # [b,M]
        rest = o[:, :, 1:].view(Bn, self.M, self.N, B + self.ltri)
        mu = rest[..., :B]                                   # [b,M,N,B]
        raw = rest[..., B:]                                  # [b,M,N,ltri]
        L = h.new_zeros(Bn, self.M, self.N, B, B)
        for k, (r, c) in enumerate(zip(self._rows, self._cols)):
            if k in self._diag:
                L[..., r, c] = torch.exp(torch.clamp(raw[..., k], -self.diag_clamp, self.diag_clamp))
            else:
                L[..., r, c] = raw[..., k]
        logdet_half = torch.clamp(raw[..., self._diag], -self.diag_clamp, self.diag_clamp).sum(-1)  # [b,M,N]
        return logpi, mu, logdet_half, L

    def _logits_rows(self, h):
        B = self.B; b = h.shape[0]; bm = b * self.M
        logpi, mu, logdet_half, L = self._unpack(h)
        Lam = L @ L.transpose(-1, -2)                        # [b,M,N,B,B] precision
        logphi = logdet_half.sum(2).unsqueeze(-1)            # [b,M,1]
        for n in range(self.N):
            xn = self.X[:, n, :]                             # [V,B]
            Ln = Lam[:, :, n]                                # [b,M,B,B]
            mn = mu[:, :, n, :]                              # [b,M,B]
            # x^T Λ x  via outer-product features [V,B*B] · Λ_flat[b,M,B*B]
            xouter = (xn.unsqueeze(2) * xn.unsqueeze(1)).reshape(self.V, B * B)   # [V,B*B]
            t1 = torch.matmul(Ln.reshape(bm, B * B), xouter.t()).reshape(b, self.M, self.V)  # [b,M,V]
            Lammu = torch.matmul(Ln, mn.unsqueeze(-1)).squeeze(-1)                # [b,M,B]
            t2 = torch.matmul(Lammu, xn.t()).reshape(b, self.M, self.V)          # [b,M,V]
            t3 = (mn * Lammu).sum(-1).unsqueeze(-1)                              # [b,M,1]
            logphi = logphi - 0.5 * (t1 - 2.0 * t2 + t3)
        return torch.logsumexp(logphi + logpi.unsqueeze(-1), dim=1) + self.b.unsqueeze(0)

    def forward(self, h, chunk=1024):
        shp = h.shape[:-1]; h = h.reshape(-1, self.d)
        if chunk and h.shape[0] > chunk:
            train_cp = self.training and torch.is_grad_enabled()
            outs = []
            for i in range(0, h.shape[0], chunk):
                hc = h[i:i + chunk]
                outs.append(cp.checkpoint(self._logits_rows, hc, use_reentrant=False) if train_cp else self._logits_rows(hc))
            logits = torch.cat(outs, 0)
        else:
            logits = self._logits_rows(h)
        return logits.reshape(*shp, self.V)

    def decorrelation(self):
        Xc = self.X.reshape(self.V, self.N * self.B)
        Xz = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
        C = (Xz.T @ Xz) / self.V
        d = self.N * self.B
        mask = torch.ones(d, d, device=C.device)
        for n in range(self.N):
            mask[self.B * n:self.B * n + self.B, self.B * n:self.B * n + self.B] = 0.0
        off = C * mask
        return self.gamma_dec * off.pow(2).sum() / max(1, d * d - d)


class LowRankLinearHead(nn.Module):
    """Decisive control: plain low-rank unembedder logits = (h @ Vd) @ Uv^T + b, W ≈ Uv Vd^T rank r.
    Params = d*r + V*r + V. Match r so V*r ≈ MDN head's V*B*N (i.e. r ≈ B*N)."""
    def __init__(self, d_model, vocab_size, rank, b_init=None, device=None):
        super().__init__()
        self.d = d_model; self.V = vocab_size; self.r = rank
        self.Vd = nn.Linear(d_model, rank, bias=False, device=device)     # down-project h
        self.Uv = nn.Parameter(0.02 * torch.randn(vocab_size, rank, device=device))
        b0 = torch.zeros(vocab_size, device=device) if b_init is None else torch.as_tensor(b_init, dtype=torch.float32, device=device)
        self.b = nn.Parameter(b0)
        nn.init.normal_(self.Vd.weight, std=0.02)

    def param_groups(self):
        return dict(decay=[self.Vd.weight, self.Uv], no_decay=[self.b])

    @staticmethod
    def param_count(d_model, vocab_size, rank):
        return dict(total=d_model * rank + vocab_size * rank + vocab_size,
                    Uv=vocab_size * rank, Vd=d_model * rank, b=vocab_size)

    def decorrelation(self):
        return torch.zeros((), device=self.Uv.device)

    def forward(self, h):
        z = self.Vd(h)                                        # [..,r]
        return torch.matmul(z, self.Uv.t()) + self.b          # [..,V]
