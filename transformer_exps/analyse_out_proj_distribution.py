"""
Load exp270 / exp271 checkpoint and measure the distribution of out_proj output
values (before LayerNorm). The question: is it a uniform ladder (integer-like
Borda counts), or does it have continuous "closeness" structure?

For each layer's out_proj, we capture the output tensor on real data and compute:
  - per-sample sorted values (min, percentiles, max)
  - per-sample gap distribution (how are adjacent sorted values spaced?)
  - globally: is the sorted-value histogram peaked on a uniform ladder?
"""
import sys, os, json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer_exps.common import make_sampler, BOS_ID, CONTEXT_SIZE
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.permutational_lut import PermutationalLut

EXP_DIR = 'transformer_exps/exp270_perm_ste_full'
DEVICE = 'cuda:0'

with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
NAP_QK = cfg['nap_qk']
NAP_V = cfg['nap_v']
TPH = cfg['tph']
TPH_V = cfg.get('tph_v', TPH)
INAP_OUT = cfg['input_nap_out']
ONAP_OUT = cfg['output_nap_out']
TPH_OUT = cfg['tph_out']
N_LAYERS = cfg['num_layers']
SOFT_MODE = cfg.get('soft_mode', 'rational')
TEMP = cfg.get('temperature', 0.1)

torch.manual_seed(0)


def _make_lut(n_heads, n_outputs, nap, seed_offset, tph=None):
    if tph is None:
        tph = TPH
    return MultiHeadLut(
        input_dim=E, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
        initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, recompute_in_backward=True,
    )


def _make_perm_outproj(seed_offset):
    return PermutationalLut(
        n_inputs=E, n_outputs=E,
        input_nap=INAP_OUT, output_nap=ONAP_OUT,
        n_heads=1, tph=TPH_OUT,
        pair_mode=cfg.get('pair_mode', 'scrambled'),
        soft_mode=SOFT_MODE,
        temperature=TEMP,
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        recompute_in_backward=True,
        initial_weights_noise=0.001,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_lut = _make_lut(H, d_qk, NAP_QK, layer_idx)
        self.k_lut = _make_lut(H, d_qk, NAP_QK, 100 + layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v_lut = _make_lut(H, d_v, NAP_V, 200 + layer_idx, tph=TPH_V)
        self.out_proj = _make_perm_outproj(400 + layer_idx)
        self.out_norm = nn.LayerNorm(E)

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B*T, _E)
        x_flat = x.reshape(B*T, _E)
        q = self.q_lut(xp).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_lut(xp).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_lut(x_flat).reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)).squeeze(1).reshape(B, T, _E)
        x = self.out_norm(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(cfg['vocab_size'], E)
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.randn(CONTEXT_SIZE, E) * 0.1) for _ in range(N_LAYERS)])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        concat_dim = E * N_LAYERS
        self.unembedder = nn.Sequential(
            nn.Linear(concat_dim, concat_dim * 4),
            nn.ReLU(),
            nn.Linear(concat_dim * 4, cfg['vocab_size']),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        outs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            outs.append(x)
        concat = torch.cat(outs, dim=-1)
        return self.unembedder(concat)


model = Model().to(DEVICE)
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print(f'Loaded checkpoint from {EXP_DIR}')

# Hook out_proj outputs (pre-LayerNorm) for every layer
captured = {i: [] for i in range(N_LAYERS)}

def make_hook(idx):
    def hook(module, inp, out):
        captured[idx].append(out.detach().squeeze(1).cpu())
    return hook

hooks = []
for i, layer in enumerate(model.layers):
    h = layer.out_proj.register_forward_hook(make_hook(i))
    hooks.append(h)

# Run a few batches
sampler = make_sampler(DEVICE, random_seed=1)
N_BATCHES = 5
BATCH_SIZE = 64

print(f'\nRunning {N_BATCHES} batches of {BATCH_SIZE}...\n')
with torch.no_grad():
    for _ in range(N_BATCHES):
        x = sampler.sample_training_batch(BATCH_SIZE).long()
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        model(inp)

for h in hooks:
    h.remove()

# Stats per layer
print(f'{"layer":>5} | {"n_samples":>9} {"mean":>9} {"std":>9} | '
      f'{"sorted 0":>10} {"sorted mid":>10} {"sorted -1":>10} | '
      f'{"gap p50":>9} {"gap p99":>9} | {"unique%":>8}')
print('-' * 120)

for i in range(N_LAYERS):
    # Each captured entry is [B*T, E]. Concatenate.
    all_out = torch.cat(captured[i], dim=0)  # [N_samples, E]
    N = all_out.shape[0]

    # Sort each row ascending
    sorted_vals, _ = all_out.sort(dim=-1)

    # Sample statistics
    mean_vals = sorted_vals.mean(dim=0)  # [E]
    std_vals = sorted_vals.std(dim=0)
    # Differences between adjacent sorted values (per sample)
    gaps = sorted_vals[:, 1:] - sorted_vals[:, :-1]  # [N, E-1]
    gap_flat = gaps.flatten()

    # Fraction of "unique" values: count distinct values per row, average
    # If this is near 1 (E distinct values), values are well-separated;
    # if much less, many sorted values are equal (integer ladder)
    uniq_frac = 0.0
    for row in all_out[:256]:  # sample 256 rows for speed
        uniq_frac += row.unique().numel() / E
    uniq_frac /= min(256, N)

    print(f'{i:>5} | {N:>9} {all_out.mean().item():>9.4f} {all_out.std().item():>9.4f} | '
          f'{mean_vals[0].item():>10.3f} {mean_vals[E//2].item():>10.3f} {mean_vals[-1].item():>10.3f} | '
          f'{gap_flat.quantile(0.5).item():>9.4f} {gap_flat.quantile(0.99).item():>9.4f} | '
          f'{uniq_frac*100:>7.2f}%')

# Also dump a distribution of adjacent gaps for a specific layer
print('\n=== Layer 0 gap distribution (quantiles) ===')
all_out = torch.cat(captured[0], dim=0)
sorted_vals, _ = all_out.sort(dim=-1)
gaps = (sorted_vals[:, 1:] - sorted_vals[:, :-1]).flatten()
for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f'  p{int(q*100):>2} = {gaps.quantile(q).item():.4f}')
print(f'  mean = {gaps.mean().item():.4f}')
print(f'  cv   = {(gaps.std() / gaps.mean()).item():.3f}')
print(f'  # near-zero gaps (<0.01×mean): {((gaps < 0.01 * gaps.mean()).float().mean().item())*100:.2f}%')

print('\n=== Layer 5 gap distribution (quantiles) ===')
all_out = torch.cat(captured[5], dim=0)
sorted_vals, _ = all_out.sort(dim=-1)
gaps = (sorted_vals[:, 1:] - sorted_vals[:, :-1]).flatten()
for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f'  p{int(q*100):>2} = {gaps.quantile(q).item():.4f}')
print(f'  mean = {gaps.mean().item():.4f}')
print(f'  cv   = {(gaps.std() / gaps.mean()).item():.3f}')
print(f'  # near-zero gaps (<0.01×mean): {((gaps < 0.01 * gaps.mean()).float().mean().item())*100:.2f}%')
