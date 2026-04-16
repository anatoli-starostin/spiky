"""
exp256_perm_lut — exp252 architecture, but out_proj is a PermutationalLut (scrambled mode).
Q/K/V remain standard MultiHeadLut. Concat layer outputs + MLP unembedder.
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import (
    make_sampler, evaluate_model, count_params, save_summary,
    BOS_ID, CONTEXT_SIZE,
)
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.permutational_lut import PermutationalLut

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
SEQ_LEN = CONTEXT_SIZE
torch.manual_seed(cfg['random_seed'])

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
SOFT_MODE = cfg.get('soft_mode', 'rational')
TEMP = cfg.get('temperature', 0.5)
TEMP_END = cfg.get('temperature_anneal_end', None)
N_LAYERS = cfg['num_layers']


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


# ---- Dominance-V helpers ----
#
# Convert a real-valued V vector of shape [..., d_v] into a centred signed
# dominance vector of shape [..., P] where P = d_v*(d_v-1)/2. Each entry is
# soft_vote(V[i] - V[j]) for all i<j pairs. Uses STE (hard forward, rational bwd).
#
# After SDPA mixes these dominance vectors linearly, we project back to a
# d_v-dimensional rank vector via a Borda-style {-1,0,+1} matrix: each
# dimension k sums +d_{kj} for j>k and -d_{jk} for j<k.

_DV_P = d_v * (d_v - 1) // 2
# Pair index tensors (i<j enumerated in row-major order)
_DV_PAIR_I, _DV_PAIR_J = torch.triu_indices(d_v, d_v, offset=1).tolist()
# Borda projection matrix [d_v, P]: M[k, p] = +1 if i_p==k, -1 if j_p==k, 0 otherwise
_DV_BORDA_M = torch.zeros(d_v, _DV_P)
for p, (i, j) in enumerate(zip(_DV_PAIR_I, _DV_PAIR_J)):
    _DV_BORDA_M[i, p] = 1.0
    _DV_BORDA_M[j, p] = -1.0


def ranks_to_dominance(v, temperature):
    """Convert [..., d_v] real vector to [..., P] signed dominance vector.

    Uses STE: hard forward (sign * 0.5) with rational backward surrogate.
    """
    # v: [..., d_v]
    i_idx = torch.tensor(_DV_PAIR_I, device=v.device)
    j_idx = torch.tensor(_DV_PAIR_J, device=v.device)
    d_raw = v.index_select(-1, i_idx) - v.index_select(-1, j_idx)  # [..., P]
    hard = torch.where(d_raw > 0, 0.5, -0.5).to(v.dtype)
    soft = 0.5 * d_raw / (temperature + d_raw.abs())
    return hard.detach() + (soft - soft.detach())


def dominance_to_ranks(d_vec, borda_m):
    """Aggregate [..., P] signed dominance vector into [..., d_v] rank vector.

    borda_m: [d_v, P] with +1/-1 entries — same {a,b,pair} structure.
    Output = d_vec @ borda_m.T  →  [..., d_v]
    """
    return torch.einsum('...p,kp->...k', d_vec, borda_m)


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
        # Register Borda projection as a per-layer buffer (shared shape, not weights)
        self.register_buffer('borda_m', _DV_BORDA_M.clone())

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B*T, _E)
        x_flat = x.reshape(B*T, _E)

        q = self.q_lut(xp).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_lut(xp).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # V: [B, T, H, d_v] — the real-valued ranking vector from v_lut
        v_ranks = self.v_lut(x_flat).reshape(B, T, H, d_v)
        # Convert ranks → centred signed dominance [B, T, H, P]
        temp = self.out_proj.temperature  # reuse the PermLut temperature for consistency
        v_dom = ranks_to_dominance(v_ranks, temp)
        # SDPA expects [B, H, T, P]
        v_dom = v_dom.permute(0, 2, 1, 3)
        # Q/K are rank vectors (real), dot product gives attention weights as usual
        attn_dom = F.scaled_dot_product_attention(q, k, v_dom, is_causal=True)  # [B, H, T, P]
        # Aggregate back to ranks
        attn_ranks = dominance_to_ranks(attn_dom, self.borda_m)  # [B, H, T, d_v]

        x = self.out_proj(attn_ranks.permute(0, 2, 1, 3).reshape(B*T, H*d_v)).squeeze(1).reshape(B, T, _E)
        x = self.out_norm(x)
        return x


class ConcatLayersTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(cfg['vocab_size'], E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.randn(SEQ_LEN, E) * 0.1) for _ in range(N_LAYERS)])
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


def get_lr_scale(step):
    n_steps = cfg['n_steps']
    warmup = int(cfg.get('lr_warmup_fraction', 0.1) * n_steps)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


sampler = make_sampler(DEVICE, random_seed=1)
model = ConcatLayersTransformer().to(DEVICE)

PERM_LR_MULT = cfg.get('perm_lr_mult', 1.0)
dense_params, perm_params, lut_params = [], [], []
for n, p in model.named_parameters():
    if 'unembedder' in n:
        dense_params.append(p)
    elif 'out_proj' in n:
        # PermutationalLut params (inner MultiHeadLut weights)
        perm_params.append(p)
    else:
        lut_params.append(p)
optimizer = torch.optim.Adam([
    {'params': dense_params, 'lr': cfg['lr'] * 3},
    {'params': perm_params, 'lr': cfg['lr'] * PERM_LR_MULT},
    {'params': lut_params, 'lr': cfg['lr']},
])
print(f'param groups: dense={sum(p.numel() for p in dense_params):,} '
      f'perm={sum(p.numel() for p in perm_params):,} '
      f'lut={sum(p.numel() for p in lut_params):,}')
print(f'lr: dense={cfg["lr"]*3:.4f} perm={cfg["lr"]*PERM_LR_MULT:.4f} lut={cfg["lr"]:.4f}')
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

total_params, trainable_params = count_params(model)
print(f'Parameters: {total_params:,}')
print(f'Q/K nap={NAP_QK} tph={TPH}, V nap={NAP_V} tph={TPH_V}')
print(f'OutProj PermLut: input_nap={INAP_OUT} output_nap={ONAP_OUT} tph={TPH_OUT} soft_mode={SOFT_MODE} temp={TEMP}')

csv_path = os.path.join(EXP_DIR, 'metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'val_loss'])

train_losses, val_losses, val_steps = [], [], []
best_val = float('inf')
ema = None
t0 = time.time()

def set_perm_temperature(model, temp):
    for layer in model.layers:
        if hasattr(layer, 'out_proj') and hasattr(layer.out_proj, 'temperature'):
            layer.out_proj.temperature = temp


model.train()
for step in range(cfg['n_steps']):
    if TEMP_END is not None:
        progress = step / max(cfg['n_steps'] - 1, 1)
        cur_temp = TEMP + (TEMP_END - TEMP) * progress
        set_perm_temperature(model, cur_temp)
    x = sampler.sample_training_batch(cfg['batch_size']).long()
    inp = torch.empty_like(x)
    inp[:, 0] = BOS_ID
    inp[:, 1:] = x[:, :-1]
    logits = model(inp)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    lv = loss.item()
    ema = lv if ema is None else 0.99*ema + 0.01*lv

    if step % 100 == 0:
        cur_t = model.layers[0].out_proj.temperature
        print(f'step {step:6d} | loss={ema:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | temp={cur_t:.3f}')

    if step % cfg['test_every'] == 0:
        val = evaluate_model(model, sampler, cfg['test_batch_size'])
        if val < best_val:
            best_val = val
        print(f'[VAL] step {step}: {val:.4f}')
        train_losses.append(ema)
        val_losses.append(val)
        val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{val:.6f}'])
        csv_f.flush()

csv_f.close()
elapsed = time.time() - t0

plt.figure(figsize=(8, 4))
plt.plot(val_steps, train_losses, label='train')
plt.plot(val_steps, val_losses, label='val')
plt.xlabel('steps'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=100)
plt.close()

summary = {
    'exp_name': cfg['exp_name'],
    'best_val_loss': best_val,
    'final_val_loss': val_losses[-1] if val_losses else None,
    'total_params': total_params,
    'trainable_params': trainable_params,
    'training_time_hours': round(elapsed / 3600, 3),
}
save_summary(EXP_DIR, summary)
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
