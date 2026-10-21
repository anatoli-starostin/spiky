"""Tiny 2-layer models for MQAR that differ only in the sequence mixer.

Mixers:
  softmax   - causal softmax attention (upper bound)
  linear    - linear attention, phi = elu+1, feature dim F  (state per head = F x d_head)
  lut_hard  - drawer memory: tph tables x 2**nap cells, F = tph * 2**nap (same state size).
              Each token writes v into the cell addressed by its key's anchor-pair sign bits and
              reads the cells addressed by its query's bits; read = sum over tables / count.
  lut_nc    - drawer memory + spiky's neighbour cell and confidence score on read and write.

The drawer memory is linear attention with a sparse binary feature map (one-hot cell per table),
so the comparison with `linear` at equal F is a comparison at equal recurrent state size.
Address gradients use spiky's recipe: hard forward, soft full-K surrogate backward.
"""
import math

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F


def kernel_attention(phi_q, phi_k, v, eps=1e-6):
    """Causal normalized kernel attention. phi_*: [B,H,T,F], v: [B,H,T,dh]."""
    T = v.shape[2]
    w = phi_q @ phi_k.transpose(-1, -2)
    mask = torch.ones(T, T, device=v.device, dtype=torch.bool).tril()
    w = w.masked_fill(~mask, 0.0)
    return (w @ v) / (w.sum(-1, keepdim=True) + eps)


def rope(x, base=10000.0):
    """Rotary position embedding on the last dim of x: [B,H,T,dh] (standard half-split form)."""
    T, dh = x.shape[-2], x.shape[-1]
    inv = base ** (-torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., : dh // 2], x[..., dh // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class SoftmaxMixer(nn.Module):
    def __init__(self, d, H, use_rope=False, **_):
        super().__init__()
        self.H, self.use_rope = H, use_rope
        self.qkv = nn.Linear(d, 3 * d, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, d // self.H).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            q, k = rope(q), rope(k)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)


class LinearMixer(nn.Module):
    def __init__(self, d, H, feat=256, out_gate=False, **_):
        super().__init__()
        self.H, self.feat, self.out_gate = H, feat, out_gate
        self.qk = nn.Linear(d, 2 * H * feat, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        if out_gate:  # same RMS-norm + gate as the drawer mixer, for a fair comparison
            self.gate = nn.Linear(d, d, bias=True)
            self.norm = nn.RMSNorm(d // H)

    def forward(self, x):
        B, T, d = x.shape
        q, k = self.qk(x).view(B, T, 2, self.H, self.feat).permute(2, 0, 3, 1, 4)
        v = self.v(x).view(B, T, self.H, d // self.H).transpose(1, 2)
        out = kernel_attention(F.elu(q) + 1, F.elu(k) + 1, v)
        if self.out_gate:
            out = self.norm(out) * torch.sigmoid(self.gate(x)).view(B, T, self.H, d // self.H).transpose(1, 2)
        return out.transpose(1, 2).reshape(B, T, d)


class LutMemMixer(nn.Module):
    def __init__(self, d, H, feat=256, nap=6, dk=32, nbr_conf=False, tie_qk=False, use_nbr=None, use_conf=None,
                 seed=0, write="add", read_power=1, topk=0, bitflip=0, init_mem=False, out_gate=False, learned_addr=0, signed=False, keycheck=False, **_):
        super().__init__()
        # signed: Count-Sketch writes -- per table, each token also gets a +-1 sign from one extra fixed anchor pair;
        #         it writes sign*value and reads sign*cell, so colliding unrelated items cancel instead of piling up
        # keycheck: each cell also stores the sum of its writers' address vectors; a token weights each table's read by
        #           softmax over tables of (its address . the cell's mean address) -- crowded/mismatched cells count less
        self.signed, self.keycheck = signed, keycheck
        if keycheck:
            self.log_kc = nn.Parameter(torch.tensor(-0.5 * math.log(dk)))
        # write: "add" = cells accumulate (sum/count); "decay" = learned per-head exponential forgetting;
        #        "ema" = per-cell exponential moving average with a learned per-token rate beta (beta->1 = overwrite)
        # read_power: weight = (number of tables where q and k share a cell) ** read_power (2 ~ soft intersection)
        self.write, self.read_power = write, read_power
        self.hardness = 1.0  # soft-to-hard annealing: 0 = soft full-K cell distribution forward, 1 = hard cells
        self.topk = topk     # >0: bounded soft read/write -- the k most likely cells per table, weighted by P
        # bitflip=b: cheap bounded soft read -- the 2**b cells reached by flipping any subset of the b least-certain
        # bits, weighted exp(-sum of flipped margins / tau) (= their probability ratio); no scoring of all cells
        self.bitflip = bitflip
        # init_mem: learned initial cell contents + learned prior counts (SDM-style parametric memory): every read also
        #           sees trained per-cell values, like a lutorch_ex table, on top of what the sequence wrote
        # out_gate: RMS-normalise the read and gate it element-wise from the token (SDM-style read processing)
        self.init_mem, self.out_gate = init_mem, out_gate
        # learned_addr=k: SDM-style learned addressing control -- each cell of each table owns a learned key vector;
        # a token's cells are its top-k cells by score z . key, weighted by softmax over those k scores (no anchors)
        self.learned_addr = learned_addr
        if learned_addr:
            self.cell_keys = nn.Parameter(torch.randn(H, feat // (2 ** nap), 2 ** nap, dk) * 0.5)
        dh = d // H
        if init_mem:
            self.mem0 = nn.Parameter(torch.zeros(H, (feat // (2 ** nap)) * (2 ** nap), dh))
            self.cnt0 = nn.Parameter(torch.full((H, (feat // (2 ** nap)) * (2 ** nap)), -2.0))  # softplus -> ~0.13
        if out_gate:
            self.gate = nn.Linear(d, d, bias=True)
            self.norm = nn.RMSNorm(dh)
        if write == "decay":
            self.decay_logit = nn.Parameter(torch.full((H,), 4.0))  # sigmoid(4) ~ 0.982 per step
        if write == "ema":
            self.beta = nn.Linear(d, H)
        self.tie_qk = tie_qk  # Reformer-style shared Q/K projection: same token -> same address
        K = 2 ** nap
        assert feat % K == 0, "feat must be tph * 2**nap"
        self.H, self.nap, self.K, self.tph, self.dk = H, nap, K, feat // K, dk
        self.use_nbr = nbr_conf if use_nbr is None else use_nbr
        self.use_conf = nbr_conf if use_conf is None else use_conf
        self.qk = nn.Linear(d, (1 if tie_qk else 2) * H * dk, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        # fixed random anchor pairs, shared by query and key (same hash on both sides)
        g = torch.Generator().manual_seed(seed)
        pairs = torch.combinations(torch.arange(dk), 2)
        idx = torch.stack([torch.stack([torch.randperm(len(pairs), generator=g)[:nap]
                                        for _ in range(self.tph)]) for _ in range(H)])  # [H,tph,nap]
        self.register_buffer("anc_a", pairs[idx, 0].reshape(H, -1))
        self.register_buffer("anc_b", pairs[idx, 1].reshape(H, -1))
        if signed:  # one extra fixed anchor pair per table for the sign
            sidx = torch.randint(len(pairs), (H, self.tph), generator=g)
            self.register_buffer("sgn_a", pairs[sidx, 0])
            self.register_buffer("sgn_b", pairs[sidx, 1])
        order = torch.arange(nap - 1, -1, -1)
        self.register_buffer("pow2", 2 ** order)
        cells = torch.arange(K)
        self.register_buffer("cell_bits", ((cells[:, None] >> order[None, :]) & 1).float())  # [K,nap]
        self.log_tau = nn.Parameter(torch.zeros(()))  # surrogate temperature
        # spiky ConfidenceLUT n=2 scalars, stored as logs; init beta=2, gamma=1, tau=0.5
        if self.use_conf:
            self.log_beta = nn.Parameter(torch.tensor(math.log(2.0)))
            self.log_gamma = nn.Parameter(torch.tensor(0.0))
        if self.use_nbr:
            self.log_rtau = nn.Parameter(torch.tensor(math.log(0.5)))

    def features(self, z):
        """z: [B,H,T,dk] -> phi [B,H,T,tph*K]."""
        B, H, T, _ = z.shape
        z = z / (z.abs().mean(-1, keepdim=True) + 1e-6)  # spiky's pre-LUT MeanAbsNorm
        if self.learned_addr:
            scores = torch.einsum("bhtd,hckd->bhtck", z, self.cell_keys) / self.log_tau.exp()   # [B,H,T,tph,K]
            vals, idx = scores.topk(self.learned_addr, dim=-1)
            w = torch.softmax(vals, -1)
            phi = torch.zeros_like(scores).scatter(-1, idx, w)
            return phi.reshape(B, H, T, self.tph * self.K)
        ia = self.anc_a[None, :, None, :].expand(B, H, T, -1)
        ib = self.anc_b[None, :, None, :].expand(B, H, T, -1)
        u = (z.gather(-1, ia) - z.gather(-1, ib)).view(B, H, T, self.tph, self.nap)
        c = ((u > 0).long() * self.pow2).sum(-1)  # [B,H,T,tph]
        hard = F.one_hot(c, self.K).float()
        m = u.abs()
        if self.use_nbr:
            ustar, jstar = m.min(-1)
            c_nb = c ^ self.pow2[jstar]
            vco = torch.sigmoid(-2 * ustar / self.log_rtau.exp()).unsqueeze(-1)  # <= 0.5
            hard = (1 - vco) * hard + vco * F.one_hot(c_nb, self.K).float()
        if self.use_conf:
            s = torch.exp(self.log_gamma.exp() * F.logsigmoid(self.log_beta.exp() * m).sum(-1)).unsqueeze(-1)
            hard = s * hard
        # soft full-K surrogate (gradient only): p(cell) = prod_j sigmoid(+-u_j / tau)
        tau = self.log_tau.exp()
        lp1, lp0 = F.logsigmoid(u / tau), F.logsigmoid(-u / tau)
        P = (lp1 @ self.cell_bits.T + lp0 @ (1 - self.cell_bits).T).exp()
        if self.bitflip:
            tau = self.log_tau.exp()
            m = u.abs()
            mb, jb = (-m).topk(self.bitflip, dim=-1)                   # b least-certain bits per table
            mb = -mb
            subs = torch.arange(2 ** self.bitflip, device=u.device)
            sel = ((subs[:, None] >> torch.arange(self.bitflip, device=u.device)[None, :]) & 1)  # [2^b, b]
            flipmask = (sel[None, None, None, None] * self.pow2[jb][..., None, :]).sum(-1)     # [B,H,T,tph,2^b]
            cells = c[..., None] ^ flipmask
            w = torch.exp(-(sel.float()[None, None, None, None] * mb[..., None, :]).sum(-1) / tau)
            w = w / w.sum(-1, keepdim=True)
            phi = torch.zeros(B, H, T, self.tph, self.K, device=u.device).scatter_add(-1, cells, w)
            return phi.reshape(B, H, T, self.tph * self.K)
        if self.topk:
            vals, idx = P.topk(self.topk, dim=-1)                      # k most likely cells per table
            w = vals / (vals.sum(-1, keepdim=True) + 1e-9)             # renormalised soft weights, differentiable
            phi = torch.zeros_like(P).scatter(-1, idx, w)
            return phi.reshape(B, H, T, self.tph * self.K)
        a = self.hardness
        phi = hard + (P - P.detach()) if a >= 1.0 else a * hard + (1 - a) * P + a * (P - P.detach())
        return phi.reshape(B, H, T, self.tph * self.K)

    def addresses(self, x):
        """Hard cell index per table for each token (tied): [B,H,T,tph]. Used by diagnostics."""
        B, T, _ = x.shape
        z = self.qk(x).view(B, T, self.H, self.dk).transpose(1, 2)
        z = z / (z.abs().mean(-1, keepdim=True) + 1e-6)
        ia = self.anc_a[None, :, None, :].expand(B, self.H, T, -1)
        ib = self.anc_b[None, :, None, :].expand(B, self.H, T, -1)
        u = (z.gather(-1, ia) - z.gather(-1, ib)).view(B, self.H, T, self.tph, self.nap)
        return ((u > 0).long() * self.pow2).sum(-1), u

    def table_read(self, z, v, eps=1e-6):
        """Per-table drawer read (dense O(T^2) training form) for signed writes and/or key-checked table weighting."""
        B, H, T, _ = z.shape
        phi = self.features(z).view(B, H, T, self.tph, self.K)
        mask = torch.ones(T, T, dtype=torch.bool, device=z.device).tril()
        M = torch.einsum("bhjtk,bhitk->bhtji", phi, phi) * mask                 # [B,H,tph,T(read),T(write)]
        den = M.sum(-1, keepdim=True)                                            # cell counts seen by each reader
        W = M
        if self.signed:
            zn = z / (z.abs().mean(-1, keepdim=True) + 1e-6)
            ia = self.sgn_a[None, :, None, :].expand(B, H, T, -1)
            ib = self.sgn_b[None, :, None, :].expand(B, H, T, -1)
            us = zn.gather(-1, ia) - zn.gather(-1, ib)                           # [B,H,T,tph]
            soft = torch.tanh(us / self.log_tau.exp())
            s = (torch.sign(us) + soft - soft.detach()).permute(0, 1, 3, 2)      # hard +-1, tanh surrogate gradient
            W = M * s[..., :, None] * s[..., None, :]
        num = W @ v[:, :, None]                                                  # [B,H,tph,T,dh]
        if not self.keycheck:
            return num.sum(2) / (den.sum(2) + eps)
        zn = z / (z.abs().mean(-1, keepdim=True) + 1e-6)
        kbar = (M @ zn[:, :, None]) / (den + eps)                                # mean address stored in the cell
        score = (kbar * zn[:, :, None]).sum(-1) * self.log_kc.exp()             # [B,H,tph,T]
        score = score.masked_fill(den.squeeze(-1) == 0, -1e4)
        a = torch.softmax(score, dim=2)
        return (a[..., None] * num / (den + eps)).sum(2)

    def forward(self, x):
        B, T, d = x.shape
        v = self.v(x).view(B, T, self.H, d // self.H).transpose(1, 2)
        if self.tie_qk and (self.signed or self.keycheck):
            assert self.write == "add" and self.read_power == 1 and not self.init_mem
            out = self.table_read(self.qk(x).view(B, T, self.H, self.dk).transpose(1, 2), v)
            if self.out_gate:
                out = self.norm(out) * torch.sigmoid(self.gate(x)).view(B, T, self.H, d // self.H).transpose(1, 2)
        elif self.tie_qk and (self.init_mem or self.out_gate) and self.write == "add" and self.read_power == 1:
            phi = self.features(self.qk(x).view(B, T, self.H, self.dk).transpose(1, 2))
            w = (phi @ phi.transpose(-1, -2)) * torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            num, den = w @ v, w.sum(-1, keepdim=True)
            if self.init_mem:
                num = num + phi @ self.mem0[None]
                den = den + (phi * F.softplus(self.cnt0)[None, :, None, :]).sum(-1, keepdim=True)
            out = num / (den + 1e-6)
            if self.out_gate:
                out = self.norm(out) * torch.sigmoid(self.gate(x)).view(B, T, self.H, d // self.H).transpose(1, 2)
        elif self.tie_qk and (self.write != "add" or self.read_power != 1):
            phi = self.features(self.qk(x).view(B, T, self.H, self.dk).transpose(1, 2))
            out = self.kernel_general(phi, v, x)
        elif self.tie_qk:
            phi = self.features(self.qk(x).view(B, T, self.H, self.dk).transpose(1, 2))
            out = kernel_attention(phi, phi, v)
        else:
            zq, zk = self.qk(x).view(B, T, 2, self.H, self.dk).permute(2, 0, 3, 1, 4)
            out = kernel_attention(self.features(zq), self.features(zk), v)
        return out.transpose(1, 2).reshape(B, T, d)


def _causal(T, device):
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def kernel_general(self, phi, v, x, eps=1e-6):
    """Tied drawer read with decay / ema writes and/or powered match counts. Dense O(T^2) training form.

    phi: [B,H,T,tph*K] (hard forward + surrogate gradient). Per-table match M_t[j,i] = <phi_t(j), phi_t(i)>.
    """
    B, H, T, _ = phi.shape
    mask = _causal(T, phi.device)
    if self.write == "ema":
        pt = phi.view(B, H, T, self.tph, self.K)
        M = torch.einsum("bhjtk,bhitk->bhtji", pt, pt)                       # [B,H,tph,T,T]
        Mh = M.detach().clamp(max=1.0) * mask                                  # hard same-cell indicator
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2)                     # [B,H,T] write rate per token
        logkeep = torch.log1p(-beta.clamp(max=1 - 1e-4))                       # log(1-beta_k)
        # C[j,i] = sum_{i<k<=j} Mh[j,k] * log(1-beta_k): later writes to the query's cell shrink older ones
        A = Mh * logkeep[:, :, None, None, :]                                  # [B,H,tph,T(j),T(k)]
        C = A.flip(-1).cumsum(-1).flip(-1) - A          # suffix sum over k > i (k <= j via Mh's causal mask)
        w = (M * mask) * beta[:, :, None, None, :] * torch.exp(C)             # [B,H,tph,T,T]
        w = w.sum(2)
    else:
        w = phi @ phi.transpose(-1, -2)                                        # shared-cell counts
        if self.read_power != 1:
            w = w.clamp(min=0) ** self.read_power
        if self.write == "decay":
            g = torch.sigmoid(self.decay_logit)[None, :, None, None]           # [1,H,1,1]
            idx = torch.arange(T, device=phi.device)
            dist = (idx[:, None] - idx[None, :]).clamp(min=0).float()
            w = w * g ** dist
        w = w * mask
    return (w @ v) / (w.sum(-1, keepdim=True) + eps)


LutMemMixer.kernel_general = kernel_general


class LocalSoftmaxMixer(nn.Module):
    """Causal softmax attention restricted to the last `window` tokens."""
    def __init__(self, d, H, window=16, **_):
        super().__init__()
        self.H, self.window = H, window
        self.qkv = nn.Linear(d, 3 * d, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, d // self.H).permute(2, 0, 3, 1, 4)
        idx = torch.arange(T, device=x.device)
        allowed = (idx[:, None] >= idx[None, :]) & (idx[:, None] - idx[None, :] < self.window)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return o.transpose(1, 2).reshape(B, T, d)


class HybridMixer(nn.Module):
    """Half the width: local softmax window (short range); other half: tied drawer memory (long range)."""
    def __init__(self, d, H, window=16, nbr=False, **kw):
        super().__init__()
        self.local = LocalSoftmaxMixer(d // 2, H // 2, window=window)
        self.mem = LutMemMixer(d // 2, H // 2, tie_qk=True, use_nbr=nbr, use_conf=False, **kw)

    def forward(self, x):
        h = x.shape[-1] // 2
        return torch.cat([self.local(x[..., :h]), self.mem(x[..., h:])], -1)


def delta_chunk(KK, QK, KS, QS, V, beta):
    """One chunk of the delta rule S_t = S_{t-1} + beta_t k_t (v_t - S_{t-1}^T k_t)^T, o_t = S_t^T q_t (WY form).
    KK, QK: [..., c, c] key-key / query-key products; KS, QS: [..., c, dh] reads of the incoming state.
    Returns outputs [..., c, dh] and U [..., c, dh] such that S_out = S_in + K^T U."""
    A = (beta[..., None] * KK).tril(-1)
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    U = torch.linalg.solve_triangular(I + A, beta[..., None] * (V - KS), upper=False, unitriangular=True)
    return QS + QK.tril() @ U, U


class DeltaDrawerMixer(LutMemMixer):
    """Delta drawers = DeltaNet whose key is sparse: the tied LUT address with flip-b weights, scaled to unit norm
    (each token touches tph * 2**b cells). Cells store values; a token corrects its cells towards its own value:
    r = k.M; M += beta_t k (v_t - r). With beta = 1 the token reads back exactly v_t. Output = read after the write.
    Training uses the chunked WY form with sparse gathers/scatters; memory size (tph * 2**nap cells) costs no
    extra compute per token."""
    def __init__(self, d, H, bitflip=2, out_gate=True, chunk=16, **kw):
        super().__init__(d, H, tie_qk=True, bitflip=bitflip, out_gate=out_gate, **kw)
        assert bitflip > 0, "delta drawers use the flip-b read (weights carry the address gradient)"
        self.beta = nn.Linear(d, H)  # write strength per token and head
        self.chunk = chunk

    def sparse_addr(self, z):
        """z: [B,H,T,dk] -> global cell ids [B,H,T,n] and unit-norm key weights [B,H,T,n]."""
        B, H, T, _ = z.shape
        z = z / (z.abs().mean(-1, keepdim=True) + 1e-6)
        ia = self.anc_a[None, :, None, :].expand(B, H, T, -1)
        ib = self.anc_b[None, :, None, :].expand(B, H, T, -1)
        u = (z.gather(-1, ia) - z.gather(-1, ib)).view(B, H, T, self.tph, self.nap)
        # coarse-to-fine bit curriculum (anneal_frac > 0 drives hardness 0 -> 1): only the first nb bits of each
        # table address are used (the rest read as 0), nb growing from 4 to nap -- big drawers first, which match by
        # chance and so give a learning signal, then finer ones. Memory is per sequence, so the layout may change.
        nb = min(self.nap, 4 + round(self.hardness * (self.nap - 4))) if self.nap > 4 else self.nap
        active = (torch.arange(self.nap, device=u.device) < nb)
        c = ((u > 0).long() * self.pow2 * active).sum(-1)
        tau = self.log_tau.exp()
        mb, jb = (-u.abs().masked_fill(~active, float("inf"))).topk(self.bitflip, dim=-1)
        mb = -mb
        subs = torch.arange(2 ** self.bitflip, device=u.device)
        sel = ((subs[:, None] >> torch.arange(self.bitflip, device=u.device)[None, :]) & 1)
        flipmask = (sel[None, None, None, None] * self.pow2[jb][..., None, :]).sum(-1)
        cells = c[..., None] ^ flipmask                                          # [B,H,T,tph,2^b]
        w = torch.exp(-(sel.float()[None, None, None, None] * mb[..., None, :]).sum(-1) / tau)
        w = w / w.sum(-1, keepdim=True)                                          # per table: sums to 1
        cells = cells + (torch.arange(self.tph, device=u.device) * self.K)[:, None]
        w = w.reshape(B, H, T, -1)
        return cells.reshape(B, H, T, -1), w / w.norm(dim=-1, keepdim=True)

    def forward(self, x, sequential=False):
        B, T, d = x.shape
        dh, m = d // self.H, 2 ** self.bitflip
        v = self.v(x).view(B, T, self.H, dh).transpose(1, 2)                     # [B,H,T,dh]
        idx, w = self.sparse_addr(self.qk(x).view(B, T, self.H, self.dk).transpose(1, 2))
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2)                        # [B,H,T]
        M = v.new_zeros(B, self.H, self.tph * self.K, dh)
        n = idx.shape[-1]
        outs = []
        step = 1 if sequential else self.chunk
        for c0 in range(0, T, step):
            ic, wc = idx[:, :, c0:c0 + step], w[:, :, c0:c0 + step]
            c = ic.shape[2]
            flat = ic.reshape(B, self.H, c * n, 1).expand(-1, -1, -1, dh)
            KS = (wc[..., None] * M.gather(2, flat).view(B, self.H, c, n, dh)).sum(3)
            it, wt = ic.view(B, self.H, c, self.tph, m), wc.view(B, self.H, c, self.tph, m)
            eq = it[:, :, :, None, :, :, None] == it[:, :, None, :, :, None, :]    # same cell, per table
            KK = (eq * wt[:, :, :, None, :, :, None] * wt[:, :, None, :, :, None, :]).sum((-1, -2, -3))
            o, U = delta_chunk(KK, KK, KS, KS, v[:, :, c0:c0 + c], beta[:, :, c0:c0 + c])
            M = M.scatter_add(2, flat, (wc[..., None] * U[:, :, :, None, :]).reshape(B, self.H, c * n, dh))
            outs.append(o)
        out = torch.cat(outs, 2)
        if self.out_gate:
            out = self.norm(out) * torch.sigmoid(self.gate(x)).view(B, T, self.H, dh).transpose(1, 2)
        return out.transpose(1, 2).reshape(B, T, d)


class DeltaLinearMixer(nn.Module):
    """Fair dense baseline (DeltaNet): S <- S + beta_t k_t (v_t - S^T k_t)^T, o_t = S_t^T q_t, with L2-normalised
    elu+1 features of size feat (state feat x dh per head, as linear attention) and the same RMS-norm + gate.
    Chunked WY form."""
    def __init__(self, d, H, feat=256, out_gate=True, chunk=32, **_):
        super().__init__()
        self.H, self.feat, self.out_gate, self.chunk = H, feat, out_gate, chunk
        self.qk = nn.Linear(d, 2 * H * feat, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.beta = nn.Linear(d, H)
        if out_gate:
            self.gate = nn.Linear(d, d, bias=True)
            self.norm = nn.RMSNorm(d // H)

    def forward(self, x, sequential=False):
        B, T, d = x.shape
        dh = d // self.H
        q, k = self.qk(x).view(B, T, 2, self.H, self.feat).permute(2, 0, 3, 1, 4)
        q, k = F.normalize(F.elu(q) + 1, dim=-1), F.normalize(F.elu(k) + 1, dim=-1)
        v = self.v(x).view(B, T, self.H, dh).transpose(1, 2)
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2)
        S = v.new_zeros(B, self.H, self.feat, dh)
        outs = []
        step = 1 if sequential else self.chunk
        for c0 in range(0, T, step):
            qc, kc = q[:, :, c0:c0 + step], k[:, :, c0:c0 + step]
            o, U = delta_chunk(kc @ kc.transpose(-1, -2), qc @ kc.transpose(-1, -2), kc @ S, qc @ S,
                               v[:, :, c0:c0 + step], beta[:, :, c0:c0 + step])
            S = S + kc.transpose(-1, -2) @ U
            outs.append(o)
        out = torch.cat(outs, 2)
        if self.out_gate:
            out = self.norm(out) * torch.sigmoid(self.gate(x)).view(B, T, self.H, dh).transpose(1, 2)
        return out.transpose(1, 2).reshape(B, T, d)


MIXERS = {
    "softmax": SoftmaxMixer,
    "linear": LinearMixer,
    "lut_hard": lambda **kw: LutMemMixer(nbr_conf=False, **kw),
    "lut_nc": lambda **kw: LutMemMixer(nbr_conf=True, **kw),
    "lut_hard_tied": lambda **kw: LutMemMixer(nbr_conf=False, tie_qk=True, **kw),
    "lut_nc_tied": lambda **kw: LutMemMixer(nbr_conf=True, tie_qk=True, **kw),
    "lut_nbr_tied": lambda **kw: LutMemMixer(use_nbr=True, use_conf=False, tie_qk=True, **kw),
    "lut_conf_tied": lambda **kw: LutMemMixer(use_nbr=False, use_conf=True, tie_qk=True, **kw),
    "softmax_rope": lambda **kw: SoftmaxMixer(use_rope=True, **kw),
    "local": LocalSoftmaxMixer,
    "hybrid": HybridMixer,
    "lut_delta": DeltaDrawerMixer,
    "linear_delta": DeltaLinearMixer,
}


class Block(nn.Module):
    def __init__(self, d, H, mixer, layer, shift_k=1, anchor_seed=0, **kw):
        super().__init__()
        self.shift_k = shift_k
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        # token shift over the shift_k previous tokens: identical for all mixers
        self.shift = nn.Linear((shift_k + 1) * d, d, bias=False)
        self.mixer = MIXERS[mixer](d=d, H=H, seed=1000 * anchor_seed + layer, **kw)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.ln1(x)
        past = [F.pad(h[:, :-j], (0, 0, j, 0)) for j in range(1, self.shift_k + 1)]
        h = self.shift(torch.cat([h] + past, -1))
        x = x + self.mixer(h)
        return x + self.mlp(self.ln2(x))


def layer_mixer(mixer, i, layers):
    """'local_then_<m>': first layer is a local softmax window, the remaining layers use <m> at full width."""
    if mixer.startswith("local_then_"):
        return "local" if i == 0 else mixer[len("local_then_"):]
    return mixer


class Model(nn.Module):
    def __init__(self, vocab, d=128, H=4, layers=2, mixer="softmax", **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, H, layer_mixer(mixer, i, layers), i, **kw) for i in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def set_hardness(model, a):
    """Set the soft-to-hard annealing coefficient of every drawer mixer (1 = hard, the deployable form)."""
    for m in model.modules():
        if isinstance(m, LutMemMixer):
            m.hardness = float(a)
