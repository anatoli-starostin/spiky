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
    def __init__(self, d, H, feat=256, **_):
        super().__init__()
        self.H, self.feat = H, feat
        self.qk = nn.Linear(d, 2 * H * feat, bias=False)
        self.v = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        q, k = self.qk(x).view(B, T, 2, self.H, self.feat).permute(2, 0, 3, 1, 4)
        v = self.v(x).view(B, T, self.H, d // self.H).transpose(1, 2)
        out = kernel_attention(F.elu(q) + 1, F.elu(k) + 1, v)
        return out.transpose(1, 2).reshape(B, T, d)


class LutMemMixer(nn.Module):
    def __init__(self, d, H, feat=256, nap=6, dk=32, nbr_conf=False, tie_qk=False, use_nbr=None, use_conf=None,
                 seed=0, write="add", read_power=1, topk=0, bitflip=0, **_):
        super().__init__()
        # write: "add" = cells accumulate (sum/count); "decay" = learned per-head exponential forgetting;
        #        "ema" = per-cell exponential moving average with a learned per-token rate beta (beta->1 = overwrite)
        # read_power: weight = (number of tables where q and k share a cell) ** read_power (2 ~ soft intersection)
        self.write, self.read_power = write, read_power
        self.hardness = 1.0  # soft-to-hard annealing: 0 = soft full-K cell distribution forward, 1 = hard cells
        self.topk = topk     # >0: bounded soft read/write -- the k most likely cells per table, weighted by P
        # bitflip=b: cheap bounded soft read -- the 2**b cells reached by flipping any subset of the b least-certain
        # bits, weighted exp(-sum of flipped margins / tau) (= their probability ratio); no scoring of all cells
        self.bitflip = bitflip
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

    def forward(self, x):
        B, T, d = x.shape
        v = self.v(x).view(B, T, self.H, d // self.H).transpose(1, 2)
        if self.tie_qk and (self.write != "add" or self.read_power != 1):
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
