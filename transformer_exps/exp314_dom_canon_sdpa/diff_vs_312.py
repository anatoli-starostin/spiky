"""Compare exp312 and exp314 forward paths stage by stage.

Build both q/k/v/out pipelines from the same BitLUTs and shared LN params,
feed the same input, and print per-stage tensor differences. Any non-trivial
divergence points to an architectural mismatch.
"""
import sys, os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/starost/spiky')

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.ranking_tools import (
    RankAttention, DominanceCanonicalize, DominanceToVector,
)

DEVICE = 'cuda:0'
cfg = json.load(open('/home/starost/spiky/transformer_exps/exp312_bf16_soft/config.json'))
E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
D_QK_P = d_qk * (d_qk - 1) // 2
D_V_P = d_v * (d_v - 1) // 2

torch.manual_seed(0)

# --- Shared modules (one instance per role; exp312 and exp314 paths share these) ---
def _make_lut(n_outputs, n_heads, input_nap, output_nap, tph, n_inputs, seed_off):
    return BitPermutationLUT(
        n_inputs=n_inputs, n_outputs=n_outputs, n_heads=n_heads,
        input_nap=input_nap, output_nap=output_nap, tph=tph,
        random_seed=cfg['random_seed'] + seed_off,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg['bit_lut_latent_dtype'],
        device=DEVICE,
    )

q_perm = _make_lut(d_qk, H, cfg['qk_input_nap'], cfg['qk_output_nap'], cfg['qk_tph'], E, 0)
k_perm = _make_lut(d_qk, H, cfg['qk_input_nap'], cfg['qk_output_nap'], cfg['qk_tph'], E, 100)
v_perm = _make_lut(d_v, H, cfg['v_input_nap'], cfg['v_output_nap'], cfg['v_tph'], E, 200)
out_proj = _make_lut(E, 1, cfg['out_input_nap'], cfg['out_output_nap'], cfg['out_tph'], H*d_v, 400)

# Shared LN params (same instance fed to both paths)
q_norm = nn.LayerNorm(d_qk).to(DEVICE)
k_norm = nn.LayerNorm(d_qk).to(DEVICE)
out_norm = nn.LayerNorm(E).to(DEVICE)
attn_scale = torch.tensor(0.25, device=DEVICE)  # frozen for fair compare

# exp312's RankAttention (no internal attn_scale — we apply it manually to match exp314 semantics)
rank_attn = RankAttention(
    d_qk=d_qk, d_v=D_V_P,
    smooth_mode=False,
    temperature=0.1,
    sdpa_temperature=1.0,
    sdpa_forward_temperature=1.0,
    learnable_attn_scale_init=None,
).to(DEVICE)

# exp314's DominanceCanonicalize — with LN weights copied from q_norm/k_norm
q_canon = DominanceCanonicalize(d_qk, smooth_mode=False, temperature=0.1).to(DEVICE)
k_canon = DominanceCanonicalize(d_qk, smooth_mode=False, temperature=0.1).to(DEVICE)
q_canon.d2v.ln.weight.data.copy_(q_norm.weight.data)
q_canon.d2v.ln.bias.data.copy_(q_norm.bias.data)
k_canon.d2v.ln.weight.data.copy_(k_norm.weight.data)
k_canon.d2v.ln.bias.data.copy_(k_norm.bias.data)

# exp314's attn_to_vec (normalise=False) and out_to_vec with LN copied from out_norm
attn_to_vec = DominanceToVector(d_v, normalise=False).to(DEVICE)
out_to_vec = DominanceToVector(E, normalise=True).to(DEVICE)
out_to_vec.ln.weight.data.copy_(out_norm.weight.data)
out_to_vec.ln.bias.data.copy_(out_norm.bias.data)

# --- Input ---
B, T = 2, 16
x = torch.randn(B, T, E, device=DEVICE)
pos_emb = torch.randn(T, E, device=DEVICE) * 0.1
xp = (x + pos_emb.unsqueeze(0)).reshape(B*T, E)
x_flat = x.reshape(B*T, E)

def diff(a, b, label):
    d_max = (a - b).abs().max().item()
    d_mean = (a - b).abs().mean().item()
    print(f"  {label:30s}  max={d_max:.3e}  mean={d_mean:.3e}  shape={list(a.shape)}")
    return d_max

print("=" * 70)
print("STAGE 1: Q/K bit-LUT outputs (same BitLUT, same input → must match)")
q_dom_312 = q_perm(xp)
k_dom_312 = k_perm(xp)
q_dom_314 = q_perm(xp)
k_dom_314 = k_perm(xp)
diff(q_dom_312, q_dom_314, "q_dom")
diff(k_dom_312, k_dom_314, "k_dom")

print("\nSTAGE 2: Borda projection → d_qk vector")
borda_m = q_perm.dom_borda_m  # exp312 uses q_perm.dom_borda_m for both q and k
q_312 = torch.einsum('bhp,kp->bhk', q_dom_312, borda_m)
k_312 = torch.einsum('bhp,kp->bhk', k_dom_312, borda_m)
# exp314: same einsum inside DominanceToVector
canon_borda_m = q_canon.d2v.borda_m
q_borda_314 = torch.einsum('bhp,kp->bhk', q_dom_314, canon_borda_m)
k_borda_314 = torch.einsum('bhp,kp->bhk', k_dom_314, canon_borda_m)
diff(q_312, q_borda_314, "q Borda (pre-LN)")
diff(k_312, k_borda_314, "k Borda (pre-LN)")
diff(borda_m, canon_borda_m, "borda_m buffers")

print("\nSTAGE 3: LayerNorm on d_qk")
# exp312 reshapes, permutes, then applies LN at (B, H, T, d_qk)
q_312_r = q_312.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
k_312_r = k_312.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
q_312_ln = q_norm(q_312_r)
k_312_ln = k_norm(k_312_r)
# exp314 applies LN at (B*T, H, d_qk) inside DominanceCanonicalize
q_314_ln_pre = q_canon.d2v.ln(q_borda_314)  # (B*T, H, d_qk)
k_314_ln_pre = k_canon.d2v.ln(k_borda_314)
# reshape to match 312
q_314_ln = q_314_ln_pre.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
k_314_ln = k_314_ln_pre.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
diff(q_312_ln, q_314_ln, "q after LN")
diff(k_312_ln, k_314_ln, "k after LN")

print("\nSTAGE 4: ste_rank_projection / VectorToDominance (expected: algebraically same)")
# exp312 uses rank_attn.ste_rank_projection → uses rank_attn.pairs (randomly permuted!)
rq_312 = rank_attn.ste_rank_projection(q_312_ln)  # (B, H, T, M=276)
rk_312 = rank_attn.ste_rank_projection(k_312_ln)
# exp314 uses VectorToDominance inside canon → canonical triu pairs order
# We must reshape 314's LN'd vector to the same layout and apply v2d
# To compare per-element, project 312's q_ln through 314's v2d as well (same pairs as canon)
rq_314 = q_canon.v2d(q_314_ln.permute(0, 2, 1, 3).reshape(B*T, H, d_qk)).reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
rk_314 = k_canon.v2d(k_314_ln.permute(0, 2, 1, 3).reshape(B*T, H, d_qk)).reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)

# First: check pairs equality
rank_attn_pairs = rank_attn.pairs
canon_pairs = q_canon.v2d.pairs
print(f"  rank_attn.pairs[:, :5]: {rank_attn_pairs[:, :5].cpu().numpy()}")
print(f"  canon_pairs[:, :5]:     {canon_pairs[:, :5].cpu().numpy()}")
same_pairs = torch.equal(rank_attn_pairs, canon_pairs)
print(f"  pairs tensors equal: {same_pairs}")

# Element-wise compare (will differ if pairs are reordered)
diff(rq_312, rq_314, "rq (element-wise)")
# But as sets of values per (b, h, t), they should match as multisets
rq_312_sorted = rq_312.sort(dim=-1).values
rq_314_sorted = rq_314.sort(dim=-1).values
diff(rq_312_sorted, rq_314_sorted, "rq sorted (same multiset?)")

print("\nSTAGE 5: SDPA — compute full attention output, compare")
# V dominance (same BitLUT, same input)
v_dom = v_perm(x_flat).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)  # (B, H, T, P_v)

# exp312 attention: rq scaled by attn_scale, default SDPA scale 1/sqrt(M)
rq_312_scaled = rq_312 * attn_scale
attn_dom_312 = F.scaled_dot_product_attention(rq_312_scaled, rk_312, v_dom, is_causal=True)
# exp314 attention: same math but on rq_314 (canonical pair order)
rq_314_scaled = rq_314 * attn_scale
attn_dom_314 = F.scaled_dot_product_attention(rq_314_scaled, rk_314, v_dom, is_causal=True)

diff(attn_dom_312, attn_dom_314, "attn_dom (SDPA output)")

print("\nSTAGE 6: Borda projection after SDPA → d_v")
v_borda_m = v_perm.dom_borda_m
attn_312 = torch.einsum('bhtp,kp->bhtk', attn_dom_312, v_borda_m)
attn_314 = attn_to_vec(attn_dom_314)  # normalise=False → same einsum
diff(attn_312, attn_314, "attn (d_v vector)")

print("\nSTAGE 7: out_proj + Borda + LN → [B, T, E]")
out_in_312 = attn_312.permute(0, 2, 1, 3).reshape(B*T, H*d_v)
out_in_314 = attn_314.permute(0, 2, 1, 3).reshape(B*T, H*d_v)
diff(out_in_312, out_in_314, "out_in (pre-out_proj)")
out_dom_312 = out_proj(out_in_312)
out_dom_314 = out_proj(out_in_314)
diff(out_dom_312, out_dom_314, "out_dom (post-out_proj)")

out_borda_m = out_proj.dom_borda_m
out_312_noln = torch.einsum('bhp,kp->bhk', out_dom_312, out_borda_m).squeeze(1).reshape(B, T, E)
out_312 = out_norm(out_312_noln)
out_314 = out_to_vec(out_dom_314).squeeze(1).reshape(B, T, E)
diff(out_312, out_314, "final out (post-LN)")

print("\nSTAGE 8: Per-sample rank_attn.pairs reorder check")
# If STAGE 4 diff is large due to pair permutation, confirm by reshuffling
# rq_312's pair dim into canonical order.
# rank_attn.pairs is (2, M) in random order; canonical_pairs is (2, M) in triu order.
# Find a permutation π such that rank_attn.pairs[:, π] == canonical_pairs[:, :].
ra = rank_attn_pairs.cpu()
cp = canon_pairs.cpu()
# Build lookup: canonical index p' corresponds to which rank_attn index p?
ra_keys = ra[0] * d_qk + ra[1]
cp_keys = cp[0] * d_qk + cp[1]
# perm[p'] = p such that ra[p] == cp[p']
perm = torch.tensor([int((ra_keys == cp_keys[p_]).nonzero(as_tuple=True)[0]) for p_ in range(cp_keys.numel())])
print(f"  perm[:10] = {perm[:10].cpu().numpy()}  (rank_attn pair index for each canonical pair)")
rq_312_reordered = rq_312[..., perm.to(rq_312.device)]
diff(rq_312_reordered, rq_314, "rq reorderd vs 314")
