"""
Ground-truth reference implementation of the spike_QK transformer from spike_QK.ipynb.

Encapsulates forward/backward logic for testing and comparison with the lutorch stack.
Not an nn.Module; plain class with forward(x) and backward(learning_rate).

Usage:
    model = GT_spike_QK_Transformer(device=..., **kwargs)
    logits = model.forward(x)   # x: [B, context_size] long
    # Caller computes NLL gradient and writes it into model.output, then:
    model.backward(learning_rate)
"""

from __future__ import annotations

import math
from typing import List

import torch


# --- Helpers (mirror notebook) ---

def _U(u: torch.Tensor) -> torch.Tensor:
    return 0.5 / (1 + torch.abs(u))


def _Up(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sign(x) / (1 + torch.abs(x)) ** 2


def _allocate_PE_buckets(context_size: int, num_buckets: int, device: torch.device) -> torch.Tensor:
    PE_buckets = torch.zeros(context_size, dtype=torch.long, device=device)
    if num_buckets <= 1:
        return PE_buckets
    B_half = num_buckets // 2
    for pos in range(context_size):
        if pos < B_half:
            PE_buckets[pos] = pos
        else:
            log_term = math.log(pos / B_half)
            log_max_dist = math.log(context_size / B_half)
            scale = (num_buckets - B_half) / log_max_dist
            log_bucket = B_half + int(scale * log_term)
            PE_buckets[pos] = min(log_bucket, num_buckets - 1)
    return PE_buckets


def _allocate_RPE_matrix(buckets: torch.Tensor, device: torch.device) -> torch.Tensor:
    context_size = buckets.shape[0]
    indices = torch.arange(context_size, device=device)
    diff = indices.unsqueeze(1) - indices.unsqueeze(0)
    diff = diff.clamp(min=0)
    return buckets[diff]


def _get_balanced_indices(total_needed: int, embedding_dim: int, device: torch.device) -> torch.Tensor:
    num_full_perms = math.ceil(total_needed / embedding_dim)
    indices_list = []
    for _ in range(num_full_perms):
        indices_list.append(torch.randperm(embedding_dim, device=device))
    flat_indices = torch.cat(indices_list)[:total_needed]
    return flat_indices


# --- Internal LUT/cache classes (take device and dims) ---


class _Anchors:
    def __init__(self, n_c: int, embedding_dim: int, device: torch.device):
        self.a = torch.randint(0, embedding_dim, (n_c,), device=device, dtype=torch.long)
        self.b = torch.randint(0, embedding_dim, (n_c,), device=device, dtype=torch.long)
        mask = self.a == self.b
        while mask.any():
            self.b[mask] = torch.randint(0, embedding_dim, (mask.sum().item(),), device=device, dtype=torch.long)
            mask = self.a == self.b


class _MultiHeadLUT:
    def __init__(
        self,
        n_t: int,
        n_c: int,
        size: int,
        y_dim: int,
        num_heads: int,
        embedding_dim: int,
        device: torch.device,
        init_zeros: bool = False,
    ):
        self.n_t = n_t
        self.n_c = n_c
        self.size = size
        self.y_dim = y_dim
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.device = device

        total_anchors_needed = num_heads * n_t * n_c
        flat_a = _get_balanced_indices(total_anchors_needed, embedding_dim, device)
        flat_b = _get_balanced_indices(total_anchors_needed, embedding_dim, device)
        self.anchors_a = flat_a.view(num_heads, n_t, n_c)
        self.anchors_b = flat_b.view(num_heads, n_t, n_c)
        self.flat_a = self.anchors_a.view(-1)
        self.flat_b = self.anchors_b.view(-1)

        mask = self.anchors_a == self.anchors_b
        while mask.any():
            self.anchors_b[mask] = torch.randint(
                0, embedding_dim, (mask.sum().item(),), device=device, dtype=torch.long
            )
            mask = self.anchors_a == self.anchors_b

        if init_zeros:
            self.S = torch.zeros((n_t, size, num_heads, y_dim), device=device)
        else:
            self.S = torch.randn((n_t, size, num_heads, y_dim), device=device) * 0.001

        self.nt_range = torch.arange(self.n_t, device=device)
        self.powers = (1 << torch.arange(n_c, device=device, dtype=torch.long)).view(1, 1, -1)


class _SingleLUT:
    def __init__(
        self,
        n_t: int,
        n_c: int,
        size: int,
        y_dim: int,
        embedding_dim: int,
        device: torch.device,
    ):
        self.n_t = n_t
        self.n_c = n_c
        self.size = size
        self.y_dim = y_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.anchors = [_Anchors(n_c, embedding_dim, device) for _ in range(n_t)]
        self.S = torch.randn((n_t, size, y_dim), device=device) * 0.001
        self.A_stacked = torch.stack([n.a for n in self.anchors])
        self.B_stacked = torch.stack([n.b for n in self.anchors])
        self.nt_range = torch.arange(self.n_t, device=device)
        self.powers = 1 << torch.arange(n_c, device=device, dtype=torch.long)


class _LUTCache:
    def __init__(self, n_t: int, device: torch.device, num_heads: int = 1, force_3d: bool = False):
        self.n_t = n_t
        self.num_heads = num_heads
        self.force_3d = force_3d
        self.device = device
        self.j = None
        self.r_min = None
        self.u_min = None
        self.mean_z = None

    def resize(self, num_tokens: int):
        if self.num_heads > 1 or self.force_3d:
            self.j = torch.zeros(
                (num_tokens, self.num_heads, self.n_t), dtype=torch.long, device=self.device
            )
            self.r_min = torch.zeros(
                (num_tokens, self.num_heads, self.n_t), dtype=torch.long, device=self.device
            )
            self.u_min = torch.zeros((num_tokens, self.num_heads, self.n_t), device=self.device)
        else:
            self.j = torch.zeros((num_tokens, self.n_t), dtype=torch.long, device=self.device)
            self.r_min = torch.zeros((num_tokens, self.n_t), dtype=torch.long, device=self.device)
            self.u_min = torch.zeros((num_tokens, self.n_t), device=self.device)
        self.mean_z = torch.zeros((num_tokens,), device=self.device)


class _JointAttentionCache:
    def __init__(self, num_heads: int, n_t: int, device: torch.device):
        self.num_heads = num_heads
        self.n_t = n_t
        self.device = device
        self.j_joint = None
        self.r_min_joint = None
        self.u_min_joint = None

    def resize(self, num_pairs: int):
        self.j_joint = torch.zeros(
            (num_pairs, self.num_heads, self.n_t), dtype=torch.long, device=self.device
        )
        self.r_min_joint = torch.zeros(
            (num_pairs, self.num_heads, self.n_t), dtype=torch.long, device=self.device
        )
        self.u_min_joint = torch.zeros((num_pairs, self.num_heads, self.n_t), device=self.device)


# --- Kernel-style functions (operate on m's state) ---


def _cache_index_single(lut: _SingleLUT, cache: _LUTCache, x: torch.Tensor) -> None:
    u = x[:, lut.A_stacked] - x[:, lut.B_stacked]
    bits = (u > 0).long()
    cache.j[:] = (bits * lut.powers).sum(dim=2)
    abs_u = torch.abs(u)
    cache.r_min[:] = abs_u.argmin(dim=2)
    cache.u_min[:] = u.gather(2, cache.r_min.unsqueeze(2)).squeeze(2)


def _cache_index_multi(lut: _MultiHeadLUT, cache: _LUTCache, x: torch.Tensor) -> None:
    B = x.shape[0]
    val_a = torch.index_select(x, 1, lut.flat_a).view(B, lut.num_heads, lut.n_t, lut.n_c)
    val_b = torch.index_select(x, 1, lut.flat_b).view(B, lut.num_heads, lut.n_t, lut.n_c)
    u = val_a - val_b
    j = torch.where(u > 0, lut.powers, 0).sum(dim=-1)
    abs_u = u.abs()
    r_min = abs_u.argmin(dim=-1)
    u_min = u.gather(-1, r_min.unsqueeze(-1)).squeeze(-1)
    cache.j[:] = j.view_as(cache.j)
    cache.r_min[:] = r_min.view_as(cache.r_min)
    cache.u_min[:] = u_min.view_as(cache.u_min)


def _cache_index_diff(
    lut: _MultiHeadLUT,
    joint_cache: _JointAttentionCache,
    z: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    q_combine: float,
    k_combine: float,
    qk_combine: float,
) -> None:
    z_Q = z[rows]
    z_K = z[cols]
    diff = q_combine * z_Q + k_combine * z_K + qk_combine * z_Q * z_K
    num_pairs = diff.shape[0]
    val_a = torch.index_select(diff, 1, lut.flat_a).view(
        num_pairs, lut.num_heads, lut.n_t, lut.n_c
    )
    val_b = torch.index_select(diff, 1, lut.flat_b).view(
        num_pairs, lut.num_heads, lut.n_t, lut.n_c
    )
    u = val_a - val_b
    j = torch.where(u > 0, lut.powers, 0).sum(dim=-1)
    abs_u = u.abs()
    r_min = abs_u.argmin(dim=-1)
    u_min = u.gather(-1, r_min.unsqueeze(-1)).squeeze(-1)
    joint_cache.j_joint[:] = j
    joint_cache.r_min_joint[:] = r_min
    joint_cache.u_min_joint[:] = u_min


def _grad_backward_single(
    x_gradient: torch.Tensor,
    incoming_grad: torch.Tensor,
    S_j: torch.Tensor,
    S_jbar: torch.Tensor,
    u_min: torch.Tensor,
    r_min: torch.Tensor,
    lut: _SingleLUT,
) -> None:
    gi = torch.einsum("pni,pi->pn", S_jbar - S_j, incoming_grad)
    v = gi * _Up(u_min)
    a_idx = lut.A_stacked[lut.nt_range, r_min]
    b_idx = lut.B_stacked[lut.nt_range, r_min]
    x_gradient.scatter_add_(1, a_idx, v)
    x_gradient.scatter_add_(1, b_idx, -v)


def _grad_backward_diff(
    x_gradient: torch.Tensor,
    incoming_grad: torch.Tensor,
    S_j: torch.Tensor,
    S_jbar: torch.Tensor,
    u_min: torch.Tensor,
    r_min: torch.Tensor,
    lut: _MultiHeadLUT,
    rows: torch.Tensor,
    cols: torch.Tensor,
    z: torch.Tensor,
    q_combine: float,
    k_combine: float,
    qk_combine: float,
    device: torch.device,
) -> None:
    diff = S_jbar - S_j
    if incoming_grad.dim() == 3:
        incoming_grad = incoming_grad.squeeze(-1)
    gi = diff * incoming_grad.unsqueeze(-1)
    v = gi * _Up(u_min)
    N, H, Nt = v.shape
    v_flat = v.reshape(-1)
    h_idx = torch.arange(H, device=device).view(1, -1, 1).expand(N, H, Nt)
    nt_idx = torch.arange(Nt, device=device).view(1, 1, -1).expand(N, H, Nt)
    a_indices = lut.anchors_a[h_idx, nt_idx, r_min]
    b_indices = lut.anchors_b[h_idx, nt_idx, r_min]
    a_indices_flat = a_indices.reshape(-1)
    b_indices_flat = b_indices.reshape(-1)
    rows_expanded = rows.view(N, 1, 1).expand(N, H, Nt).reshape(-1)
    cols_expanded = cols.view(N, 1, 1).expand(N, H, Nt).reshape(-1)
    K_at_a = z[cols_expanded, a_indices_flat]
    K_at_b = z[cols_expanded, b_indices_flat]
    Q_at_a = z[rows_expanded, a_indices_flat]
    Q_at_b = z[rows_expanded, b_indices_flat]
    v_Q_a = v_flat * (q_combine + qk_combine * K_at_a)
    v_Q_b = v_flat * (q_combine + qk_combine * K_at_b)
    x_gradient.index_put_((rows_expanded, a_indices_flat), v_Q_a, accumulate=True)
    x_gradient.index_put_((rows_expanded, b_indices_flat), -v_Q_b, accumulate=True)
    v_K_a = v_flat * (k_combine + qk_combine * Q_at_a)
    v_K_b = v_flat * (k_combine + qk_combine * Q_at_b)
    x_gradient.index_put_((cols_expanded, a_indices_flat), v_K_a, accumulate=True)
    x_gradient.index_put_((cols_expanded, b_indices_flat), -v_K_b, accumulate=True)


def _grad_backward_multi(
    x_gradient: torch.Tensor,
    incoming_grad: torch.Tensor,
    S_j: torch.Tensor,
    S_jbar: torch.Tensor,
    u_min: torch.Tensor,
    r_min: torch.Tensor,
    lut: _MultiHeadLUT,
    target_indices: torch.Tensor,
    device: torch.device,
) -> None:
    diff = S_jbar - S_j
    if diff.dim() == 3:
        gi = diff * incoming_grad.squeeze(-1).unsqueeze(-1)
    else:
        gi = torch.einsum("nhkd,nhd->nhk", diff, incoming_grad)
    v = gi * _Up(u_min)
    N, H, Nt = v.shape
    v_flat = v.reshape(-1)
    h_idx = torch.arange(H, device=device).view(1, -1, 1).expand(N, H, Nt)
    nt_idx = torch.arange(Nt, device=device).view(1, 1, -1).expand(N, H, Nt)
    a_indices = lut.anchors_a[h_idx, nt_idx, r_min]
    b_indices = lut.anchors_b[h_idx, nt_idx, r_min]
    a_indices_flat = a_indices.reshape(-1)
    b_indices_flat = b_indices.reshape(-1)
    token_indices_flat = target_indices.view(N, 1, 1).expand(N, H, Nt).reshape(-1)
    x_gradient.index_put_((token_indices_flat, a_indices_flat), v_flat, accumulate=True)
    x_gradient.index_put_((token_indices_flat, b_indices_flat), -v_flat, accumulate=True)


def _LUT_forward_single(
    lut: _SingleLUT,
    cache: _LUTCache,
    y: torch.Tensor,
    smooth_forward: bool,
) -> None:
    j = cache.j
    S_j = lut.S[lut.nt_range, j]
    if smooth_forward:
        r_min = cache.r_min
        u_min = cache.u_min
        jbar = j ^ (1 << r_min)
        S_jbar = lut.S[lut.nt_range, jbar]
        U_weight = _U(u_min).unsqueeze(-1)
        output = S_j + U_weight * (S_jbar - S_j)
    else:
        output = S_j
    y.add_(output.sum(dim=1))


def _LUT_backward_single(
    lut: _SingleLUT,
    cache: _LUTCache,
    x_gradient: torch.Tensor,
    y_gradient: torch.Tensor,
    learning_rate: float,
    smooth_forward: bool,
    device: torch.device,
) -> None:
    j = cache.j
    r_min = cache.r_min
    u_min = cache.u_min
    jbar = j ^ (1 << r_min)
    S_j = lut.S[lut.nt_range, j]
    S_jbar = lut.S[lut.nt_range, jbar]
    _grad_backward_single(x_gradient, y_gradient, S_j, S_jbar, u_min, r_min, lut)
    U_weight = _U(u_min)
    weight_j = 1.0 - U_weight
    weight_jbar = U_weight
    flat_indices_j = (lut.nt_range.unsqueeze(0) * lut.size + j).view(-1)
    flat_indices_jbar = (lut.nt_range.unsqueeze(0) * lut.size + jbar).view(-1)
    raw_grads = y_gradient.unsqueeze(1).expand(-1, lut.n_t, -1).reshape(-1, lut.y_dim)
    weight_j_flat = weight_j.view(-1, 1).expand(-1, lut.y_dim)
    weight_jbar_flat = weight_jbar.view(-1, 1).expand(-1, lut.y_dim)
    grads_j = raw_grads * weight_j_flat
    grads_jbar = raw_grads * weight_jbar_flat
    update_j = grads_j * -learning_rate
    update_jbar = grads_jbar * -learning_rate
    lut.S.view(-1, lut.y_dim).index_add_(0, flat_indices_j, update_j)
    lut.S.view(-1, lut.y_dim).index_add_(0, flat_indices_jbar, update_jbar)


def _unembed_forward(
    z: torch.Tensor,
    W_embed: torch.Tensor,
    output: torch.Tensor,
    unembed_temperature: float,
) -> None:
    z_norm = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
    logits = z_norm @ W_embed.T / unembed_temperature
    output[:] = logits


def _unembed_backward(
    z: torch.Tensor,
    W_embed: torch.Tensor,
    grad_output: torch.Tensor,
    x_gradient: torch.Tensor,
    learning_rate: float,
    unembed_temperature: float,
    unembed_lr_scale: float,
    W_embed_M: torch.Tensor,
    W_embed_V: torch.Tensor,
    W_embed_step: torch.Tensor,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
) -> None:
    z_norms = z.norm(dim=-1, keepdim=True) + 1e-6
    z_norm = z / z_norms
    grad_W = (grad_output.T @ z_norm) / unembed_temperature
    grad_z_norm = (grad_output @ W_embed) / unembed_temperature
    dot_product = (grad_z_norm * z_norm).sum(dim=-1, keepdim=True)
    grad_z = (grad_z_norm - z_norm * dot_product) / z_norms
    x_gradient.add_(grad_z)
    W_embed_step.add_(1)
    step = W_embed_step.float()
    W_embed_M.mul_(adam_beta1).add_(grad_W, alpha=1 - adam_beta1)
    W_embed_V.mul_(adam_beta2).addcmul_(grad_W, grad_W, value=1 - adam_beta2)
    bias_correction1 = 1 - torch.pow(adam_beta1, step)
    bias_correction2 = 1 - torch.pow(adam_beta2, step)
    m_hat = W_embed_M / bias_correction1
    v_hat = W_embed_V / bias_correction2
    update = m_hat / (v_hat.sqrt() + adam_epsilon)
    W_embed.sub_(update * learning_rate * unembed_lr_scale)


def _attention_forward_dense(
    m: "GT_spike_QK_Transformer",
    lut_a: _MultiHeadLUT,
    lut_v: _MultiHeadLUT,
    cache_a: _LUTCache,
    cache_v: _LUTCache,
    joint_cache: _JointAttentionCache,
    y_output: torch.Tensor,
    training: bool,
    layer_idx: int,
) -> None:
    B = m.batch_size
    C = m.context_size
    H = lut_a.num_heads
    Nt_a = lut_a.n_t
    Nt_v = lut_v.n_t
    device = m.device
    rows = m.batched_rows_Q
    cols = m.batched_cols_K
    rpe = m.batched_RPE_A
    pos_buckets = m.positional_buckets_a

    j_joint = joint_cache.j_joint
    r_min = joint_cache.r_min_joint
    u_min = joint_cache.u_min_joint

    rpe_expanded = rpe.view(-1, 1, 1)
    j_pair = j_joint * pos_buckets + rpe_expanded

    nt_idx = torch.arange(Nt_a, device=device).view(1, 1, -1)
    h_idx = torch.arange(H, device=device).view(1, -1, 1)

    S_j_A = lut_a.S[nt_idx, j_pair, h_idx].squeeze(-1)

    if m.smooth_forward:
        j_bar = j_joint ^ (1 << r_min)
        j_bar_pair = j_bar * pos_buckets + rpe_expanded
        S_jbar_A = lut_a.S[nt_idx, j_bar_pair, h_idx].squeeze(-1)
        U_weight = _U(u_min)
        smooth_scores = S_j_A + U_weight * (S_jbar_A - S_j_A)
        raw_scores = smooth_scores.sum(dim=-1)
    else:
        raw_scores = S_j_A.sum(dim=-1)

    scaled_scores = raw_scores / m.attention_temperature
    row_max = torch.zeros((m.total_tokens, H), device=device).fill_(-float("inf"))
    rows_expanded = rows.unsqueeze(1).expand(-1, H)
    row_max.scatter_reduce_(0, rows_expanded, scaled_scores, reduce="amax", include_self=False)
    exps = torch.exp(scaled_scores - row_max[rows])
    row_sum = torch.zeros((m.total_tokens, H), device=device)
    row_sum.index_add_(0, rows, exps)
    probs = exps / (row_sum[rows] + 1e-10)

    j_tokens_V = cache_v.j.view(B, C, lut_v.num_heads, Nt_v)
    nt_idx_v = torch.arange(Nt_v, device=device).view(1, 1, 1, -1)
    h_idx_v = torch.arange(lut_v.num_heads, device=device).view(1, 1, -1, 1)
    S_j_V = lut_v.S[nt_idx_v, j_tokens_V, h_idx_v]

    if m.smooth_forward:
        r_min_tokens_V = cache_v.r_min.view(B, C, lut_v.num_heads, Nt_v)
        u_min_tokens_V = cache_v.u_min.view(B, C, lut_v.num_heads, Nt_v)
        jbar_tokens_V = j_tokens_V ^ (1 << r_min_tokens_V)
        S_jbar_V = lut_v.S[nt_idx_v, jbar_tokens_V, h_idx_v]
        U_V = _U(u_min_tokens_V).unsqueeze(-1)
        v_vals_lut = (S_j_V + U_V * (S_jbar_V - S_j_V)).sum(dim=-2)
    else:
        v_vals_lut = S_j_V.sum(dim=-2)

    v_vals = v_vals_lut.view(m.total_tokens, H, -1)
    weighted_v = probs.unsqueeze(-1) * v_vals[cols]
    out = torch.zeros((m.total_tokens, H, m.head_dim), device=device)
    out.index_add_(0, rows, weighted_v)
    y_output[:] = out.view(m.total_tokens, -1)


def _attention_backward_batched(
    m: "GT_spike_QK_Transformer",
    lut_a: _MultiHeadLUT,
    lut_v: _MultiHeadLUT,
    cache_a: _LUTCache,
    cache_v: _LUTCache,
    joint_cache: _JointAttentionCache,
    x_gradient: torch.Tensor,
    y_gradient: torch.Tensor,
    learning_rate: float,
) -> None:
    device = m.device
    rows, cols = m.batched_rows_Q, m.batched_cols_K
    pos_buckets = m.positional_buckets_a

    j_joint = joint_cache.j_joint
    r_min_joint = joint_cache.r_min_joint
    u_min_joint = joint_cache.u_min_joint

    rpe_expanded = m.batched_RPE_A.view(-1, 1, 1)
    j_pair = j_joint * pos_buckets + rpe_expanded

    nt_idx_a = torch.arange(lut_a.n_t, device=device).view(1, 1, -1)
    nt_idx_v = torch.arange(lut_v.n_t, device=device).view(1, 1, -1)
    h_idx = torch.arange(lut_a.num_heads, device=device).view(1, -1, 1)

    S_j_A = lut_a.S[nt_idx_a, j_pair, h_idx].squeeze(-1)
    j_bar = j_joint ^ (1 << r_min_joint)
    j_bar_pair = j_bar * pos_buckets + rpe_expanded
    S_jbar_A = lut_a.S[nt_idx_a, j_bar_pair, h_idx].squeeze(-1)

    U_joint = _U(u_min_joint)
    smooth_A = S_j_A + U_joint * (S_jbar_A - S_j_A)
    raw_scores = smooth_A.sum(dim=-1)
    scaled_scores = raw_scores / m.attention_temperature

    row_max = torch.zeros((m.total_tokens, lut_a.num_heads), device=device).fill_(-float("inf"))
    rows_expanded = rows.unsqueeze(1).expand(-1, lut_a.num_heads)
    row_max.scatter_reduce_(0, rows_expanded, scaled_scores, reduce="amax", include_self=False)
    exps = torch.exp(scaled_scores - row_max[rows])
    row_sum = torch.zeros((m.total_tokens, lut_a.num_heads), device=device)
    row_sum.index_add_(0, rows, exps)
    probs = exps / (row_sum[rows] + 1e-10)

    grad_out = y_gradient[rows].view(-1, lut_a.num_heads, m.head_dim)
    B, C = m.batch_size, m.context_size
    Nt_v = lut_v.n_t

    j_tokens_V = cache_v.j.view(B, C, lut_v.num_heads, Nt_v)
    r_min_tokens_V = cache_v.r_min.view(B, C, lut_v.num_heads, Nt_v)
    u_min_tokens_V = cache_v.u_min.view(B, C, lut_v.num_heads, Nt_v)
    j_V = j_tokens_V.view(m.total_tokens, lut_v.num_heads, Nt_v)
    r_min_V = r_min_tokens_V.view(m.total_tokens, lut_v.num_heads, Nt_v)
    u_min_V = u_min_tokens_V.view(m.total_tokens, lut_v.num_heads, Nt_v)
    j_V_cols = j_V[cols]
    r_min_V_cols = r_min_V[cols]
    u_min_V_cols = u_min_V[cols]
    jbar_V_cols = j_V_cols ^ (1 << r_min_V_cols)

    S_j_V = lut_v.S[nt_idx_v, j_V_cols, h_idx]
    S_jbar_V = lut_v.S[nt_idx_v, jbar_V_cols, h_idx]
    U_V = _U(u_min_V_cols).unsqueeze(-1)
    smooth_V = S_j_V + U_V * (S_jbar_V - S_j_V)
    V_sum = smooth_V.sum(dim=2)
    grad_V_sum = grad_out * probs.unsqueeze(-1)
    grad_w = (grad_out * V_sum).sum(dim=-1)
    weighted_grad_w = probs * grad_w
    row_dot_sum = torch.zeros((m.total_tokens, lut_a.num_heads), device=device)
    row_dot_sum.index_add_(0, rows, weighted_grad_w)
    grad_a = probs * (grad_w - row_dot_sum[rows])
    grad_a = grad_a / m.attention_temperature

    _grad_backward_multi(
        x_gradient, grad_V_sum, S_j_V, S_jbar_V,
        u_min_V_cols, r_min_V_cols, lut_v, cols, device
    )

    U_V_weight = _U(u_min_V_cols)
    weight_j_V = 1.0 - U_V_weight
    weight_jbar_V = U_V_weight
    idx_V_j_flat = (
        nt_idx_v * (lut_v.size * lut_v.num_heads) + j_V_cols * lut_v.num_heads + h_idx
    ).view(-1)
    idx_V_jbar_flat = (
        nt_idx_v * (lut_v.size * lut_v.num_heads) + jbar_V_cols * lut_v.num_heads + h_idx
    ).view(-1)
    grad_V_param = grad_V_sum.unsqueeze(2).expand(-1, -1, lut_v.n_t, -1).reshape(-1, lut_v.y_dim)
    weight_j_V_flat = weight_j_V.view(-1, 1).expand(-1, lut_v.y_dim)
    weight_jbar_V_flat = weight_jbar_V.view(-1, 1).expand(-1, lut_v.y_dim)
    grad_V_j = grad_V_param * weight_j_V_flat
    grad_V_jbar = grad_V_param * weight_jbar_V_flat
    update_v_j = grad_V_j * -learning_rate
    update_v_jbar = grad_V_jbar * -learning_rate
    lut_v.S.view(-1, lut_v.y_dim).index_add_(0, idx_V_j_flat, update_v_j)
    lut_v.S.view(-1, lut_v.y_dim).index_add_(0, idx_V_jbar_flat, update_v_jbar)

    grad_a_expanded = grad_a.unsqueeze(-1)
    _grad_backward_diff(
        x_gradient, grad_a_expanded, S_j_A, S_jbar_A,
        u_min_joint, r_min_joint, lut_a, rows, cols, m.z,
        m.q_combine_mode, m.k_combine_mode, m.qk_combine_mode, device,
    )

    weight_j_A = 1.0 - U_joint
    weight_jbar_A = U_joint
    idx_A_j_flat = (
        nt_idx_a * (lut_a.size * lut_a.num_heads) + j_pair * lut_a.num_heads + h_idx
    ).view(-1)
    idx_A_jbar_flat = (
        nt_idx_a * (lut_a.size * lut_a.num_heads) + j_bar_pair * lut_a.num_heads + h_idx
    ).view(-1)
    grad_A_param = grad_a.unsqueeze(2).expand(-1, -1, lut_a.n_t).reshape(-1, 1)
    weight_j_A_flat = weight_j_A.view(-1, 1)
    weight_jbar_A_flat = weight_jbar_A.view(-1, 1)
    grad_A_j = grad_A_param * weight_j_A_flat
    grad_A_jbar = grad_A_param * weight_jbar_A_flat
    update_a_j = grad_A_j * -learning_rate
    update_a_jbar = grad_A_jbar * -learning_rate
    lut_a.S.view(-1, 1).index_add_(0, idx_A_j_flat, update_a_j)
    lut_a.S.view(-1, 1).index_add_(0, idx_A_jbar_flat, update_a_jbar)


# --- Main class ---


class GT_spike_QK_Transformer:
    """
    Ground-truth spike_QK transformer: forward/backward only.
    All hyperparameters are set in the constructor.
    """

    def __init__(
        self,
        device: torch.device,
        # Sequence / vocab
        context_size: int = 32,
        vocab_size: int = 257,
        # Model dimensions
        embedding_dim: int = 32,
        num_layers: int = 6,
        num_heads: int = 4,
        # Positional
        positional_buckets_a: int = 8,
        # Attention LUT
        attention_a_n_t: int = 32,
        attention_a_n_c: int = 14,
        attention_v_n_t: int = 16,
        attention_v_n_c: int = 14,
        attention_temperature: float = 0.25,
        # Q-K combination
        q_combine_mode: float = 1.0,
        k_combine_mode: float = -2.0,
        qk_combine_mode: float = 0.0,
        # FFN
        include_ffn: bool = True,
        ffn_n_t: int = 16,
        ffn_n_c: int = 14,
        # Unembed
        unembed_temperature: float = 0.1,
        unembed_lr_scale: float = 1.0,
        # Forward
        smooth_forward: bool = False,
        noise_jitter_scale: float = 0.0,
        # Adam (unembed)
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        # LR drops (optional)
        learning_rate_drop: float = 0.9,
        learning_rate_drop_times: List[int] | None = None,
    ):
        if learning_rate_drop_times is None:
            learning_rate_drop_times = []

        self.device = device
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.positional_buckets_a = positional_buckets_a
        self.attention_a_n_t = attention_a_n_t
        self.attention_a_n_c = attention_a_n_c
        self.attention_v_n_t = attention_v_n_t
        self.attention_v_n_c = attention_v_n_c
        self.attention_temperature = attention_temperature
        self.q_combine_mode = q_combine_mode
        self.k_combine_mode = k_combine_mode
        self.qk_combine_mode = qk_combine_mode
        self.include_ffn = include_ffn
        self.ffn_n_t = ffn_n_t
        self.ffn_n_c = ffn_n_c
        self.unembed_temperature = unembed_temperature
        self.unembed_lr_scale = unembed_lr_scale
        self.smooth_forward = smooth_forward
        self.noise_jitter_scale = noise_jitter_scale
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.learning_rate_drop = learning_rate_drop
        self.learning_rate_drop_times = learning_rate_drop_times

        assert embedding_dim % num_heads == 0
        self.head_dim = embedding_dim // num_heads

        self.PE_buckets_A = _allocate_PE_buckets(
            context_size, positional_buckets_a, device
        )
        self.RPE_matrix_A = _allocate_RPE_matrix(self.PE_buckets_A, device)
        self.tril_indices = torch.tril_indices(
            context_size, context_size, offset=0, device=device
        )

        self.FFN: List[_SingleLUT] = []
        self.FFN_cache: List[_LUTCache] = []
        self.V: List[_MultiHeadLUT] = []
        self.A: List[_MultiHeadLUT] = []
        self.V_cache: List[_LUTCache] = []
        self.A_cache: List[_LUTCache] = []
        self.A_joint_cache: List[_JointAttentionCache] = []

        for _ in range(num_layers):
            self.FFN.append(
                _SingleLUT(
                    ffn_n_t, ffn_n_c, 1 << ffn_n_c, embedding_dim, embedding_dim, device
                )
            )
            self.FFN_cache.append(_LUTCache(ffn_n_t, device, num_heads=1))

            self.V.append(
                _MultiHeadLUT(
                    attention_v_n_t,
                    attention_v_n_c,
                    1 << attention_v_n_c,
                    self.head_dim,
                    num_heads,
                    embedding_dim,
                    device,
                )
            )
            self.A.append(
                _MultiHeadLUT(
                    attention_a_n_t,
                    attention_a_n_c,
                    (1 << attention_a_n_c) * positional_buckets_a,
                    1,
                    num_heads,
                    embedding_dim,
                    device,
                    init_zeros=False,
                )
            )
            self.V_cache.append(
                _LUTCache(attention_v_n_t, device, num_heads=num_heads, force_3d=True)
            )
            self.A_cache.append(
                _LUTCache(attention_a_n_t, device, num_heads=num_heads, force_3d=True)
            )
            self.A_joint_cache.append(
                _JointAttentionCache(num_heads, attention_a_n_t, device)
            )

        self.W_embed = torch.randn(vocab_size, embedding_dim, device=device) * 0.1
        self.W_embed_M = torch.zeros_like(self.W_embed)
        self.W_embed_V = torch.zeros_like(self.W_embed)
        self.W_embed_step = torch.zeros(1, dtype=torch.long, device=device)

        self.layer_lr_multipliers = torch.ones(num_layers, device=device)
        self.lr_drop_next_idx = 0

        self.batch_size = 0
        self.total_tokens = 0
        self.input_tokens: torch.Tensor | None = None
        self.output_tokens: torch.Tensor | None = None
        self.z: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.batched_rows_Q: torch.Tensor | None = None
        self.batched_cols_K: torch.Tensor | None = None
        self.batched_RPE_A: torch.Tensor | None = None

    def set_batch_size(self, new_batch_size: int) -> None:
        if self.batch_size == new_batch_size:
            return
        self.batch_size = new_batch_size
        self.total_tokens = new_batch_size * self.context_size

        rows_local = self.tril_indices[0]
        cols_local = self.tril_indices[1]
        offsets = torch.arange(new_batch_size, device=self.device) * self.context_size

        self.batched_rows_Q = (
            rows_local.unsqueeze(0) + offsets.unsqueeze(1)
        ).view(-1)
        self.batched_cols_K = (
            cols_local.unsqueeze(0) + offsets.unsqueeze(1)
        ).view(-1)
        self.batched_RPE_A = self.RPE_matrix_A[rows_local, cols_local].repeat(
            new_batch_size
        )

        self.input_tokens = torch.zeros(
            self.total_tokens, dtype=torch.long, device=self.device
        )
        self.output_tokens = torch.zeros(
            self.total_tokens, dtype=torch.long, device=self.device
        )
        self.z = torch.zeros(
            self.total_tokens, self.embedding_dim, device=self.device
        )
        self.output = torch.zeros(
            self.total_tokens, self.vocab_size, device=self.device
        )

        num_pairs = len(self.batched_rows_Q)
        for l in range(self.num_layers):
            self.FFN_cache[l].resize(self.total_tokens)
            self.V_cache[l].resize(self.total_tokens)
            self.A_cache[l].resize(self.total_tokens)
            self.A_joint_cache[l].resize(num_pairs)

    def _add_noise(self, z: torch.Tensor, training: bool) -> torch.Tensor:
        if training and self.noise_jitter_scale > 0:
            return z + (torch.randn_like(z, device=self.device) * self.noise_jitter_scale)
        return z

    def _model_forward(self, training: bool = False) -> None:
        for l in range(self.num_layers):
            _cache_index_multi(self.V[l], self.V_cache[l], self._add_noise(self.z, training))
            _cache_index_diff(
                self.A[l],
                self.A_joint_cache[l],
                self.z,
                self.batched_rows_Q,
                self.batched_cols_K,
                self.q_combine_mode,
                self.k_combine_mode,
                self.qk_combine_mode,
            )
            attention_delta = torch.zeros_like(self.z)
            _attention_forward_dense(
                self,
                self.A[l],
                self.V[l],
                self.A_cache[l],
                self.V_cache[l],
                self.A_joint_cache[l],
                attention_delta,
                training=training,
                layer_idx=l,
            )
            self.z.add_(attention_delta)

            if self.include_ffn:
                _cache_index_single(
                    self.FFN[l],
                    self.FFN_cache[l],
                    self._add_noise(self.z, training),
                )
                _LUT_forward_single(
                    self.FFN[l],
                    self.FFN_cache[l],
                    self.z,
                    self.smooth_forward,
                )

        self.output.zero_()
        _unembed_forward(
            self.z,
            self.W_embed,
            self.output,
            self.unembed_temperature,
        )

    def _model_backward(self, learning_rate: float) -> None:
        y_grad = torch.zeros(
            self.total_tokens, self.embedding_dim, device=self.device
        )
        x_grad = torch.zeros(
            self.total_tokens, self.embedding_dim, device=self.device
        )

        _unembed_backward(
            self.z,
            self.W_embed,
            self.output,
            x_grad,
            learning_rate,
            self.unembed_temperature,
            self.unembed_lr_scale,
            self.W_embed_M,
            self.W_embed_V,
            self.W_embed_step,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
        )

        for l in range(self.num_layers - 1, -1, -1):
            layer_lr = learning_rate * self.layer_lr_multipliers[l]
            if self.include_ffn:
                y_grad.copy_(x_grad)
                _LUT_backward_single(
                    self.FFN[l],
                    self.FFN_cache[l],
                    x_grad,
                    y_grad,
                    layer_lr,
                    self.smooth_forward,
                    self.device,
                )
            y_grad.copy_(x_grad)
            _attention_backward_batched(
                self,
                self.A[l],
                self.V[l],
                self.A_cache[l],
                self.V_cache[l],
                self.A_joint_cache[l],
                x_grad,
                y_grad,
                layer_lr,
            )

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Run one forward pass. x: [batch_size, context_size] long tensor of token ids.
        Returns logits [batch_size * context_size, vocab_size].
        After computing the NLL gradient (e.g. probs - one_hot), write it into
        self.output and call backward(learning_rate).
        """
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, C = x.shape
        assert C == self.context_size
        self.set_batch_size(B)
        self.input_tokens[:] = x.reshape(-1)
        self.z[:] = self.W_embed[self.input_tokens]
        self._model_forward(training=training)
        return self.output

    def backward(self, learning_rate: float) -> None:
        """
        Run one backward pass and update parameters.
        Expects self.output to already hold the logits gradient (e.g. probs - one_hot
        with invalid positions zeroed). Same convention as spike_QK.ipynb model_backward.
        """
        self._model_backward(learning_rate)
