"""Collect (out_proj input, out_proj output) pairs from exp338 for each layer.

Loads the checkpoint, forwards 100 batches of training data, and saves
per-layer .pt files containing:
  {'inputs':  [N, 64]  float32  (= H*d_v concat fed into out_proj)
   'outputs': [N, P]   float32  (= dominance out of out_proj; P = 496 = C(32,2))}
"""
import os, sys, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.util.text_snippet_sampler import TextSnippetSampler
from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.ranking_tools import DominanceCanonicalize, DominanceToVector

SRC_EXP = '/home/starost/spiky/transformer_exps/exp338_cfc_out_inap10_tph2048_onap10'
OUT_DIR = '/home/starost/spiky/transformer_exps/distill_exp338/data'
N_BATCHES = 100  # 100 * 8 * 128 = 102400 samples per layer

with open(os.path.join(SRC_EXP, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
CONTEXT_SIZE = cfg['context_size']
VOCAB_SIZE = cfg['vocab_size']
BOS_ID = 256
DATA_PATH = os.path.normpath(
    os.path.join(SRC_EXP, '..', '..', 'workbooks', 'fineweb_texts.txt')
)

torch.manual_seed(cfg['random_seed'])

E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
N_LAYERS = cfg['num_layers']
D_QK_P = d_qk * (d_qk - 1) // 2
D_V_P = d_v * (d_v - 1) // 2

_OUT_PARTITION_SETS = (
    [list(range(h * d_v, (h + 1) * d_v)) for h in range(H)]
    if cfg.get('out_partition_by_head', False) else None
)


def _make_qk(so):
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_qk, n_heads=H,
        input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
        tph=cfg['qk_tph'], random_seed=cfg['random_seed']+so,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'), device=DEVICE,
    )
def _make_v(so):
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_v, n_heads=H,
        input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'],
        tph=cfg['v_tph'], random_seed=cfg['random_seed']+so,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'), device=DEVICE,
    )
def _make_out(so):
    return BitPermutationLUT(
        n_inputs=H*d_v, n_outputs=E, n_heads=1,
        input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'],
        tph=cfg['out_tph'], random_seed=cfg['random_seed']+so,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'), device=DEVICE,
        partition_sets=_OUT_PARTITION_SETS,
    )


class LUTBlock(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.q_perm = _make_qk(i); self.k_perm = _make_qk(100+i)
        self.v_perm = _make_v(200+i); self.out_proj = _make_out(400+i)
        ct = cfg.get('canon_temperature', 0.1)
        self.q_canon = DominanceCanonicalize(d_qk, temperature=ct)
        self.k_canon = DominanceCanonicalize(d_qk, temperature=ct)
        self.attn_to_vec = DominanceToVector(d_v, normalise=False)
        self.out_to_vec = DominanceToVector(E)
        self.attn_scale = nn.Parameter(torch.tensor(float(cfg.get('learnable_attn_scale_init', 0.25))))

    def forward(self, x, pe):
        B, T, _E = x.shape
        xp = (x + pe.unsqueeze(0)).reshape(B*T, _E)
        xf = x.reshape(B*T, _E)
        q = self.q_canon(self.q_perm(xp)).reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        k = self.k_canon(self.k_perm(xp)).reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        v = self.v_perm(xf).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)
        ad = F.scaled_dot_product_attention(q*self.attn_scale, k, v, is_causal=True)
        a = self.attn_to_vec(ad)
        oi = a.permute(0, 2, 1, 3).reshape(B*T, H*d_v)
        od = self.out_proj(oi)
        return self.out_to_vec(od).squeeze(1).reshape(B, T, _E)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.randn(CONTEXT_SIZE, E)*0.1) for _ in range(N_LAYERS)])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        cd = E*N_LAYERS
        self.unembedder = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd, cd*4), nn.ReLU(), nn.Linear(cd*4, VOCAB_SIZE))

    def forward(self, t):
        x = self.token_embedder(t)
        outs = []
        for layer, pe in zip(self.layers, self.pos_embs):
            x = layer(x, pe); outs.append(x)
        return self.unembedder(torch.cat(outs, dim=-1))


model = Model().to(DEVICE)
sd = torch.load(os.path.join(SRC_EXP, 'checkpoint.pt'), map_location=DEVICE, weights_only=False)
model.load_state_dict(sd)
model.eval()
print(f'loaded {SRC_EXP}/checkpoint.pt — {sum(p.numel() for p in model.parameters()):,} adam params')

# Hooks: capture out_proj input + output per layer.
inputs = {i: [] for i in range(N_LAYERS)}
outputs = {i: [] for i in range(N_LAYERS)}

def make_hook(layer_idx):
    def hook(module, args, output):
        x = args[0].detach()              # [B*T, 64]
        y = output.detach().squeeze(1)    # [B*T, 1, P] -> [B*T, P]
        inputs[layer_idx].append(x.cpu())
        outputs[layer_idx].append(y.cpu())
    return hook

handles = []
for i, layer in enumerate(model.layers):
    handles.append(layer.out_proj.register_forward_hook(make_hook(i)))

# Use a distinct sampler seed so we don't train the distillation target on
# exp338's own training positions (not strictly necessary but cleaner).
sampler = TextSnippetSampler(DATA_PATH, CONTEXT_SIZE, 10_000, DEVICE, random_seed=42)

print(f'collecting {N_BATCHES} batches × {cfg["batch_size"]} × {CONTEXT_SIZE} = '
      f'{N_BATCHES * cfg["batch_size"] * CONTEXT_SIZE:,} samples per layer')

with torch.no_grad():
    for bi in range(N_BATCHES):
        x = sampler.sample_training_batch(cfg['batch_size']).long()
        inp = torch.empty_like(x); inp[:, 0] = BOS_ID; inp[:, 1:] = x[:, :-1]
        _ = model(inp)
        if (bi + 1) % 10 == 0:
            print(f'  batch {bi+1}/{N_BATCHES}')

for h in handles: h.remove()

os.makedirs(OUT_DIR, exist_ok=True)
for i in range(N_LAYERS):
    X = torch.cat(inputs[i], dim=0)
    Y = torch.cat(outputs[i], dim=0)
    path = os.path.join(OUT_DIR, f'layer_{i}.pt')
    torch.save({'inputs': X, 'outputs': Y}, path)
    print(f'saved layer {i}: inputs {tuple(X.shape)} outputs {tuple(Y.shape)} -> {path}')

print('done.')
