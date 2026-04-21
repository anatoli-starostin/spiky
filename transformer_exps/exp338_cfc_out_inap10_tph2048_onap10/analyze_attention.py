"""Analyze exp338's attention patterns per layer.

Loads checkpoint, forwards training batches, captures q/k at every
layer (via forward hooks on q_perm/k_perm), recomputes softmax attention
over the canonicalized dominance, and reports:

  - mean attention row averaged over samples & heads & positions
  - locality metric: fraction of attention within distance k, for k in
    {1, 2, 4, 8, 16, 32, 64, 128}
  - saves a PNG with one subplot per layer (attention vs. positional
    distance) to exp338_cfc_out_inap10_tph2048_onap10/attention_stats.png
"""
import os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spiky.util.text_snippet_sampler import TextSnippetSampler
from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.ranking_tools import DominanceCanonicalize, DominanceToVector

EXP = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP, 'config.json')) as f: cfg = json.load(f)

DEVICE = 'cuda:0'
CONTEXT_SIZE = cfg['context_size']
VOCAB_SIZE = cfg['vocab_size']
BOS_ID = 256
DATA_PATH = os.path.normpath(os.path.join(EXP, '..', '..', 'workbooks', 'fineweb_texts.txt'))
torch.manual_seed(cfg['random_seed'])

E, H, d_qk, d_v = cfg['embedding_dim'], cfg['n_heads'], cfg['d_qk'], cfg['d_v']
N_LAYERS = cfg['num_layers']
D_QK_P, D_V_P = d_qk*(d_qk-1)//2, d_v*(d_v-1)//2
PS = [list(range(h*d_v, (h+1)*d_v)) for h in range(H)] if cfg.get('out_partition_by_head', False) else None


def _make_qk(so): return BitPermutationLUT(n_inputs=E, n_outputs=d_qk, n_heads=H, input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'], tph=cfg['qk_tph'], random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'], latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'), device=DEVICE)
def _make_v(so): return BitPermutationLUT(n_inputs=E, n_outputs=d_v, n_heads=H, input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'], tph=cfg['v_tph'], random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'], latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'), device=DEVICE)
def _make_out(so): return BitPermutationLUT(n_inputs=H*d_v, n_outputs=E, n_heads=1, input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'], tph=cfg['out_tph'], random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'], latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'), device=DEVICE, partition_sets=PS)


class LUTBlock(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.q_perm = _make_qk(i); self.k_perm = _make_qk(100+i)
        self.v_perm = _make_v(200+i); self.out_proj = _make_out(400+i)
        ct = cfg.get('canon_temperature', 0.1)
        self.q_canon = DominanceCanonicalize(d_qk, temperature=ct); self.k_canon = DominanceCanonicalize(d_qk, temperature=ct)
        self.attn_to_vec = DominanceToVector(d_v, normalise=False); self.out_to_vec = DominanceToVector(E)
        self.attn_scale = nn.Parameter(torch.tensor(float(cfg.get('learnable_attn_scale_init', 0.25))))

    def forward(self, x, pe):
        B, T, _E = x.shape
        xp = (x + pe.unsqueeze(0)).reshape(B*T, _E); xf = x.reshape(B*T, _E)
        q_dom = self.q_canon(self.q_perm(xp)); k_dom = self.k_canon(self.k_perm(xp))
        q = q_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3) * self.attn_scale  # [B,H,T,P]
        k = k_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        v_dom = self.v_perm(xf).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)
        # Compute attention weights manually (SDPA doesn't expose them).
        scores = torch.einsum('bhip,bhjp->bhij', q, k) / math.sqrt(q.shape[-1])  # [B,H,T,T]
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)   # [B, H, T, T]
        self._last_attn = attn.detach()
        attn_dom = torch.einsum('bhij,bhjp->bhip', attn, v_dom)
        a = self.attn_to_vec(attn_dom)
        oi = a.permute(0, 2, 1, 3).reshape(B*T, H*d_v)
        od = self.out_proj(oi)
        return self.out_to_vec(od).squeeze(1).reshape(B, T, _E)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.zeros(CONTEXT_SIZE, E)) for _ in range(N_LAYERS)])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        cd = E*N_LAYERS
        self.unembedder = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd, cd*4), nn.ReLU(), nn.Linear(cd*4, VOCAB_SIZE))

    def forward(self, t):
        x = self.token_embedder(t)
        for layer, pe in zip(self.layers, self.pos_embs):
            x = layer(x, pe)
        return x


model = Model().to(DEVICE)
sd = torch.load(os.path.join(EXP, 'checkpoint.pt'), map_location=DEVICE, weights_only=False)
model.load_state_dict(sd)
model.eval()

sampler = TextSnippetSampler(DATA_PATH, CONTEXT_SIZE, 10_000, DEVICE, random_seed=7)

# Accumulate mean attention pattern per layer: attn[b,h,i,j] -> reduce over b,h,i to get f(j-i) = f(dist).
T = CONTEXT_SIZE
dist_acc = torch.zeros(N_LAYERS, T, device=DEVICE)   # [layers, dist=0..T-1]
count_acc = torch.zeros(N_LAYERS, T, device=DEVICE)

N_BATCHES = 20
with torch.no_grad():
    for b in range(N_BATCHES):
        x = sampler.sample_training_batch(cfg['batch_size']).long()
        inp = torch.empty_like(x); inp[:, 0] = BOS_ID; inp[:, 1:] = x[:, :-1]
        _ = model(inp)
        for l, layer in enumerate(model.layers):
            attn = layer._last_attn  # [B, H, T, T]
            B = attn.shape[0]
            # For each (b, h, i, j) where j <= i, distance = i - j.
            # Sum attention into dist bins.
            for d in range(T):
                # rows i where j = i - d exists
                rows_i = torch.arange(d, T, device=DEVICE)
                cols_j = rows_i - d
                vals = attn[:, :, rows_i, cols_j]        # [B, H, T-d]
                dist_acc[l, d] += vals.sum().item()
                count_acc[l, d] += B * attn.shape[1] * (T - d)
        print(f'batch {b+1}/{N_BATCHES}')

# Per-layer mean attention per distance.
mean_attn = (dist_acc / count_acc.clamp(min=1)).cpu().numpy()  # [L, T]

# Locality metric: fraction of attention mass within distance k,
# averaged over rows (positions).
ks = [1, 2, 4, 8, 16, 32, 64, 128]
cum_mass = {}
with torch.no_grad():
    for b in range(5):  # small re-pass
        x = sampler.sample_training_batch(cfg['batch_size']).long()
        inp = torch.empty_like(x); inp[:, 0] = BOS_ID; inp[:, 1:] = x[:, :-1]
        _ = model(inp)
        for l, layer in enumerate(model.layers):
            attn = layer._last_attn  # [B, H, T, T]
            B, Hh, T0, _ = attn.shape
            for k in ks:
                # For each row i, sum attn[i, max(0,i-k+1) : i+1]
                tot = 0.0; cnt = 0
                for i in range(T0):
                    lo = max(0, i - k + 1)
                    tot += attn[:, :, i, lo:i+1].sum().item()
                    cnt += B * Hh
                cum_mass.setdefault((l, k), [0.0, 0])
                cum_mass[(l, k)][0] += tot
                cum_mass[(l, k)][1] += cnt
        print(f'  locality batch {b+1}/5')

print('\n=== Locality (fraction of attention mass within distance k) ===')
print(f"{'layer':>6} " + ' '.join(f'{k:>7}' for k in ks))
for l in range(N_LAYERS):
    vals = [cum_mass[(l, k)][0] / cum_mass[(l, k)][1] for k in ks]
    print(f'{l:>6} ' + ' '.join(f'{v:>7.4f}' for v in vals))

# Plot: one subplot per layer — mean attention vs distance (log-x).
fig, axes = plt.subplots(1, N_LAYERS, figsize=(3*N_LAYERS, 3), sharey=True)
for l in range(N_LAYERS):
    ax = axes[l]
    ax.plot(range(T), mean_attn[l])
    ax.set_xscale('symlog', linthresh=1)
    ax.set_title(f'layer {l}')
    ax.set_xlabel('distance'); ax.grid(True, alpha=0.3)
axes[0].set_ylabel('mean attention')
plt.tight_layout()
out_png = os.path.join(EXP, 'attention_stats.png')
plt.savefig(out_png, dpi=110); plt.close()
print(f'\nsaved {out_png}')
