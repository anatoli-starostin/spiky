"""
Resume exp270 checkpoint with temperature=1.0 and lr=1e-4 constant.
Run for 10K steps, log val loss every 1000.
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer_exps.common import (
    make_sampler, evaluate_model, count_params, save_summary,
    BOS_ID, CONTEXT_SIZE,
)
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.permutational_lut import PermutationalLut

SRC_EXP = 'transformer_exps/exp270_perm_ste_full'
OUT_DIR = 'transformer_exps/exp272_resume_temp1'
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(SRC_EXP, 'config.json')) as f:
    cfg = json.load(f)

# Override hyperparameters for the resume
cfg['exp_name'] = 'exp272_resume_temp1'
cfg['description'] = 'Resume exp270 checkpoint with temperature=1.0, lr=1e-4 constant, 10K steps.'
cfg['temperature'] = 1.0
cfg['lr'] = 1e-4
cfg['n_steps'] = 10000
cfg['lr_schedule'] = 'constant'
cfg['lr_warmup_fraction'] = 0.0

with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump(cfg, f, indent=2)

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
TEMP = cfg['temperature']
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
        outs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            outs.append(x)
        concat = torch.cat(outs, dim=-1)
        return self.unembedder(concat)


sampler = make_sampler(DEVICE, random_seed=1)
model = Model().to(DEVICE)
print(f'Loading checkpoint from {SRC_EXP}...')
ckpt = torch.load(os.path.join(SRC_EXP, 'checkpoint.pt'), map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
print(f'Loaded. Temperature set to {TEMP}, lr={cfg["lr"]:.2e} constant.')

# Verify the temperature override actually took effect in all PermLuts
for i, layer in enumerate(model.layers):
    assert layer.out_proj.temperature == TEMP, f'layer {i} temperature mismatch'
print(f'All {N_LAYERS} PermLut layers confirmed at temperature={TEMP}')

dense_params = [p for n, p in model.named_parameters() if 'unembedder' in n]
lut_params = [p for n, p in model.named_parameters() if 'unembedder' not in n]
optimizer = torch.optim.Adam([
    {'params': dense_params, 'lr': cfg['lr'] * 3},
    {'params': lut_params, 'lr': cfg['lr']},
])

total_params, _ = count_params(model)
print(f'Parameters: {total_params:,}')

csv_path = os.path.join(OUT_DIR, 'metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'val_loss'])

val_init = evaluate_model(model, sampler, cfg['test_batch_size'])
print(f'[VAL] step 0 (pre-resume): {val_init:.4f}')
csv_w.writerow([0, '', f'{val_init:.6f}'])
csv_f.flush()

best_val = val_init
ema = None
t0 = time.time()

model.train()
for step in range(1, cfg['n_steps'] + 1):
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

    lv = loss.item()
    ema = lv if ema is None else 0.99 * ema + 0.01 * lv

    if step % 100 == 0:
        print(f'step {step:6d} | loss={ema:.4f}')

    if step % 1000 == 0:
        val = evaluate_model(model, sampler, cfg['test_batch_size'])
        if val < best_val:
            best_val = val
        print(f'[VAL] step {step}: {val:.4f}')
        csv_w.writerow([step, f'{ema:.6f}', f'{val:.6f}'])
        csv_f.flush()

csv_f.close()
elapsed = time.time() - t0

summary = {
    'exp_name': cfg['exp_name'],
    'best_val_loss': best_val,
    'initial_val_loss': val_init,
    'total_params': total_params,
    'training_time_hours': round(elapsed / 3600, 3),
    'note': 'Resumed exp270 with temperature=1.0, lr=1e-4 constant',
}
save_summary(OUT_DIR, summary)
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
