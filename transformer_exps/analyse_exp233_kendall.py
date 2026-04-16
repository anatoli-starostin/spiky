"""
Kendall tau analysis on exp233 (residual architecture).
Compares out_proj input (attention output) vs post-residual stream state (x + norm1(proj)).
This measures what the residual stream actually changes in rank space.
"""
import sys, os, json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer_exps.common import make_sampler, BOS_ID, CONTEXT_SIZE
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy

DEVICE = 'cuda:0'
EXP_DIR = 'transformer_exps/exp233_mixed_nap_qk5'

with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
NAP_QK = cfg['nap_qk']
NAP = cfg['nap']
TPH = cfg['tph']
N_LAYERS = cfg['num_layers']

torch.manual_seed(cfg['random_seed'])


def _make_lut(n_heads, n_outputs, nap, seed_offset):
    return MultiHeadLut(
        input_dim=E, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=TPH,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
        initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, recompute_in_backward=True,
    )


# Per-layer captures
captured = {i: {'attn_out': None, 'post_resid': None} for i in range(N_LAYERS)}


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.q_lut = _make_lut(H, d_qk, NAP_QK, layer_idx)
        self.k_lut = _make_lut(H, d_qk, NAP_QK, 100+layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v_lut = _make_lut(H, d_v, NAP, 200+layer_idx)
        self.out_proj = _make_lut(1, E, NAP, 400+layer_idx)
        self.norm1 = nn.LayerNorm(E)
        self.ffn = _make_lut(1, E, NAP, 600+layer_idx)
        self.norm2 = nn.LayerNorm(E)

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = x + pos_emb.unsqueeze(0)
        xp_flat = xp.reshape(B*T, _E)
        x_flat = x.reshape(B*T, _E)
        q = self.q_lut(xp_flat).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_lut(xp_flat).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_lut(x_flat).reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_flat = attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)  # out_proj input
        proj = self.out_proj(attn_flat).squeeze(1).reshape(B, T, _E)
        x_after_resid = x + self.norm1(proj)  # post-residual state

        # Capture: out_proj input (attn_flat) vs post-residual (x_after_resid flat)
        captured[self.layer_idx]['attn_out'] = attn_flat.detach().clone()
        captured[self.layer_idx]['post_resid'] = x_after_resid.reshape(B*T, _E).detach().clone()

        x = x_after_resid
        ffn_out = self.ffn(x.reshape(B*T, _E)).squeeze(1).reshape(B, T, _E)
        x = x + self.norm2(ffn_out)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(cfg['vocab_size'], E)
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.randn(CONTEXT_SIZE, E) * 0.1) for _ in range(N_LAYERS)])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.unembedder = nn.Sequential(
            nn.Linear(E, 128), nn.GELU(), nn.Linear(128, cfg['vocab_size'], bias=False),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
        return self.unembedder(x)


model = Model().to(DEVICE)
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print('Model loaded.')


def kendall_tau_batch(x, y):
    """x, y: [N, D]. Both must have same D."""
    N, D = x.shape
    i, j = torch.triu_indices(D, D, offset=1, device=x.device)
    dx = x[:, i] - x[:, j]
    dy = y[:, i] - y[:, j]
    agree = torch.sign(dx) * torch.sign(dy)
    tau = agree.sum(dim=1) / agree.shape[1]
    return tau


sampler = make_sampler(DEVICE, random_seed=1)
N_BATCHES = 10
BATCH_SIZE = 128

# Accumulate taus across batches
all_layer_taus = {i: [] for i in range(N_LAYERS)}

print(f'Running {N_BATCHES} batches of {BATCH_SIZE}...')
with torch.no_grad():
    for bi in range(N_BATCHES):
        x = sampler.sample_training_batch(BATCH_SIZE).long()
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        model(inp)
        for i in range(N_LAYERS):
            attn_out = captured[i]['attn_out']      # [B*T, H*d_v=32]
            post = captured[i]['post_resid']        # [B*T, E=32]
            tau = kendall_tau_batch(attn_out, post)
            all_layer_taus[i].append(tau)

print()
print('=' * 60)
print(f'exp233: τ(out_proj input, post-residual stream)')
print('=' * 60)
print(f'{"Layer":>6} | {"Mean τ":>8} {"Std τ":>8} {"Min τ":>8} {"Max τ":>8}')
print('-' * 60)
all_taus = []
for i in range(N_LAYERS):
    tau = torch.cat(all_layer_taus[i])
    all_taus.append(tau)
    print(f'{i:>6} | {tau.mean().item():>8.4f} {tau.std().item():>8.4f} '
          f'{tau.min().item():>8.4f} {tau.max().item():>8.4f}')
print('-' * 60)
total = torch.cat(all_taus)
print(f'{"All":>6} | {total.mean().item():>8.4f} {total.std().item():>8.4f} '
      f'{total.min().item():>8.4f} {total.max().item():>8.4f}')
