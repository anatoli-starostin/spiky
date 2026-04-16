"""
Measure Kendall tau between out_proj inputs and outputs in exp252.
Tells us how much rank structure each out_proj LUT preserves vs shuffles.
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
EXP_DIR = 'transformer_exps/exp252_concat_mlp_unemb'

with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
NAP_QK = cfg['nap_qk']
NAP = cfg['nap']
NAP_OUT = cfg.get('nap_out', NAP)
TPH = cfg['tph']
TPH_OUT = cfg.get('tph_out', TPH)
N_LAYERS = cfg['num_layers']

torch.manual_seed(cfg['random_seed'])


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


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_lut = _make_lut(H, d_qk, NAP_QK, layer_idx)
        self.k_lut = _make_lut(H, d_qk, NAP_QK, 100+layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v_lut = _make_lut(H, d_v, NAP, 200+layer_idx)
        self.out_proj = _make_lut(1, E, NAP_OUT, 400+layer_idx, tph=TPH_OUT)
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
        layer_outputs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            layer_outputs.append(x)
        return self.unembedder(torch.cat(layer_outputs, dim=-1))


model = Model().to(DEVICE)
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print('Model loaded.')


def kendall_tau_batch(x, y):
    """
    Compute Kendall tau for each row.
    x, y: [N, D]
    Returns: [N] tensor of tau values in [-1, 1]
    """
    N, D = x.shape
    # All pairs of columns
    i, j = torch.triu_indices(D, D, offset=1, device=x.device)
    dx = x[:, i] - x[:, j]
    dy = y[:, i] - y[:, j]
    # Concordant: same sign; discordant: opposite sign
    agree = torch.sign(dx) * torch.sign(dy)  # +1 concordant, -1 discordant, 0 tie
    # Kendall tau = (concordant - discordant) / n_pairs
    tau = agree.sum(dim=1) / agree.shape[1]
    return tau


# Hooks to capture out_proj inputs and outputs
captured = {i: {'in': [], 'out': []} for i in range(N_LAYERS)}

def make_hook(layer_idx):
    def hook(module, inp, out):
        x_in = inp[0].detach()   # [B*T, H*d_v=32]
        x_out = out.detach().squeeze(1)  # [B*T, E=32]
        captured[layer_idx]['in'].append(x_in)
        captured[layer_idx]['out'].append(x_out)
    return hook

hooks = []
for i, layer in enumerate(model.layers):
    h = layer.out_proj.register_forward_hook(make_hook(i))
    hooks.append(h)

# Run data through model
sampler = make_sampler(DEVICE, random_seed=1)
N_BATCHES = 10
BATCH_SIZE = 128

print(f'Running {N_BATCHES} batches of {BATCH_SIZE}...')
with torch.no_grad():
    for i in range(N_BATCHES):
        x = sampler.sample_training_batch(BATCH_SIZE).long()
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        model(inp)

for h in hooks:
    h.remove()

# Compute Kendall tau per layer
print()
print('=' * 60)
print(f'{"Layer":>6} | {"Mean τ":>8} {"Std τ":>8} {"Min τ":>8} {"Max τ":>8}')
print('=' * 60)

all_taus = []
for layer_idx in range(N_LAYERS):
    x_in = torch.cat(captured[layer_idx]['in'], dim=0)
    x_out = torch.cat(captured[layer_idx]['out'], dim=0)
    tau = kendall_tau_batch(x_in, x_out)
    all_taus.append(tau)
    print(f'{layer_idx:>6} | {tau.mean().item():>8.4f} {tau.std().item():>8.4f} '
          f'{tau.min().item():>8.4f} {tau.max().item():>8.4f}')

print('=' * 60)
total_tau = torch.cat(all_taus)
print(f'{"Overall":>6} | {total_tau.mean().item():>8.4f} {total_tau.std().item():>8.4f} '
      f'{total_tau.min().item():>8.4f} {total_tau.max().item():>8.4f}')

# Interpretation
print()
print('Interpretation:')
print('  τ = +1: output preserves input ranking exactly')
print('  τ =  0: output ranking uncorrelated with input')
print('  τ = -1: output ranking is reversed input')
