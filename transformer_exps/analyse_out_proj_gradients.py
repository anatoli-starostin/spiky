"""
Load the exp270 checkpoint (PermLut STE at step 100K) and measure gradient
statistics on the inner LUT weights of each layer's out_proj.

We run 50 training steps and report:
  - |grad| percentiles (how much are we asking the weights to move?)
  - |grad| / |weight| ratio (signal-to-weight ratio per layer)
  - fraction of "effective" grads (|grad| > eps × max)
  - temporal coefficient of variation (grad stability across minibatches)
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
print(f'Loaded checkpoint from {EXP_DIR}')

sampler = make_sampler(DEVICE, random_seed=1)
N_STEPS = 50
BATCH_SIZE = 128

# Accumulate per-step gradient stats for each layer's out_proj inner weights
per_step_grad_norms = {i: [] for i in range(N_LAYERS)}   # L2 norm of grad
per_step_weight_norms = {i: [] for i in range(N_LAYERS)}  # L2 norm of weight
all_grads = {i: [] for i in range(N_LAYERS)}  # concatenated flat tensors

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f'\nRunning {N_STEPS} forward/backward steps with batch_size={BATCH_SIZE}...\n')
for step in range(N_STEPS):
    x = sampler.sample_training_batch(BATCH_SIZE).long()
    inp = torch.empty_like(x)
    inp[:, 0] = BOS_ID
    inp[:, 1:] = x[:, :-1]
    logits = model(inp)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))
    optimizer.zero_grad()
    loss.backward()

    for i, layer in enumerate(model.layers):
        w = layer.out_proj.inner.projection.weights
        g = w.grad
        if g is None:
            continue
        per_step_grad_norms[i].append(g.norm().item())
        per_step_weight_norms[i].append(w.norm().item())
        if step == N_STEPS - 1:  # save full flat grad only on last step
            all_grads[i] = g.detach().flatten().cpu()

    # Don't actually step — we just want gradient stats, not to walk the model
    # (keeps gradients stable across iterations for CV measurement)

print(f'{"layer":>5} | {"mean |g|":>11} {"cv(|g|)":>10} | '
      f'{"mean |w|":>11} | {"|g|/|w|":>12} | {"p50 |g_i|":>12} {"p99 |g_i|":>12} {"dead%":>8}')
print('-' * 100)

for i in range(N_LAYERS):
    gn = torch.tensor(per_step_grad_norms[i])
    wn = torch.tensor(per_step_weight_norms[i])
    mean_gn = gn.mean().item()
    cv_gn = (gn.std() / gn.mean()).item()
    mean_wn = wn.mean().item()
    ratio = mean_gn / mean_wn

    # Per-element statistics from the last step's full grad tensor
    g_flat = all_grads[i].abs()
    p50 = g_flat.quantile(0.5).item()
    p99 = g_flat.quantile(0.99).item()
    # Fraction of near-zero elements (below 1e-3 × p99)
    dead_thr = p99 * 1e-3 if p99 > 0 else 0
    dead_pct = (g_flat < dead_thr).float().mean().item() * 100

    print(f'{i:>5} | {mean_gn:>11.4e} {cv_gn:>10.3f} | {mean_wn:>11.4e} | '
          f'{ratio:>12.4e} | {p50:>12.4e} {p99:>12.4e} {dead_pct:>7.2f}%')

print()
print('Legend:')
print('  mean |g|   : mean L2 norm of grad tensor across steps')
print('  cv(|g|)    : coefficient of variation (std/mean) of |g| — minibatch noise')
print('  mean |w|   : mean L2 norm of weight tensor')
print('  |g|/|w|    : signal-to-weight ratio (small → model cannot move much per step)')
print('  p50/p99 |g_i|: per-element grad magnitude percentiles')
print('  dead%      : fraction of grad elements < 1e-3 × p99')
