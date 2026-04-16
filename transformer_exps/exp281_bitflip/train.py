"""
exp281_bitflip — exp280 arch but PermLut weights trained via BitFlipOptimizer (±1 binary).
STE backward for gradient flow. Adam for dense/embedding params. ctx=32, 25K bs=64.
"""
import sys, os, json, math, time, csv
os.environ['SPIKY_PERMLUT_NO_CUSTOM_CUDA'] = '1'
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
from spiky.lutorch.permutational_lut import PermutationalLut
from spiky.lutorch.ranking_tools import RankAttention
from spiky.lutorch.bit_flip_optimizer import BitFlipOptimizer

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
SEQ_LEN = cfg['context_size']
torch.manual_seed(cfg['random_seed'])

E = cfg['embedding_dim']
POS_DIM = cfg['pos_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
D_V_P = d_v * (d_v - 1) // 2  # 28 for d_v=8
N_LAYERS = cfg['num_layers']
SOFT_MODE = cfg['soft_mode']
TEMP = cfg['temperature']
RANK_ATTN_TEMP = cfg['rank_attn_temperature']

PERM_KWARGS = dict(
    pair_mode='scrambled',
    soft_mode=SOFT_MODE,
    temperature=TEMP,
    device=DEVICE,
    recompute_in_backward=True,
    initial_weights_noise=0.001,
)


def _make_qk_perm(seed_offset):
    return PermutationalLut(
        n_inputs=E + POS_DIM, n_outputs=d_qk, n_heads=H,
        input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
        tph=cfg['qk_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        **PERM_KWARGS,
    )


def _make_v_perm(seed_offset):
    return PermutationalLut(
        n_inputs=E, n_outputs=d_v, n_heads=H,
        input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'],
        tph=cfg['v_tph'],
        return_dominance=True,
        random_seed=cfg['random_seed'] + seed_offset,
        **PERM_KWARGS,
    )


def _make_out_perm(seed_offset):
    return PermutationalLut(
        n_inputs=H * d_v, n_outputs=E, n_heads=1,
        input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'],
        tph=cfg['out_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        **PERM_KWARGS,
    )


class FullPermBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_perm = _make_qk_perm(layer_idx)
        self.k_perm = _make_qk_perm(100 + layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v_perm = _make_v_perm(200 + layer_idx)
        self.rank_attn = RankAttention(
            d_qk=d_qk, d_v=D_V_P,
            smooth_mode=False,
            temperature=RANK_ATTN_TEMP,
            sdpa_temperature=1.0,
            sdpa_forward_temperature=1.0,
        )
        self.out_proj = _make_out_perm(400 + layer_idx)
        self.out_norm = nn.LayerNorm(E)
        # Borda matrix from V PermLut for converting dominance back to ranks
        self.register_buffer('borda_m', self.v_perm.dom_borda_m.clone())

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        # Concat positional embeddings for Q/K
        xp = torch.cat([x, pos_emb.unsqueeze(0).expand(B, -1, -1)], dim=-1)  # [B, T, E+POS_DIM]

        q = self.q_perm(xp.reshape(B * T, -1))  # [B*T, H, d_qk]
        k = self.k_perm(xp.reshape(B * T, -1))
        q = self.q_norm(q.reshape(B, T, H, d_qk).permute(0, 2, 1, 3))
        k = self.k_norm(k.reshape(B, T, H, d_qk).permute(0, 2, 1, 3))

        # V: dominance output directly from PermLut
        v_dom = self.v_perm(x.reshape(B * T, _E))  # [B*T, H, P]
        v_dom = v_dom.reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)  # [B, H, T, P]

        # RankAttention: Q/K in pair-dominance, V already dominance
        attn_dom = self.rank_attn(q, k, v_dom, is_causal=True)  # [B, H, T, P]
        # Borda-aggregate back to rank vectors
        attn = torch.einsum('bhtp,kp->bhtk', attn_dom, self.borda_m)  # [B, H, T, d_v]

        out = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v))
        out = self.out_norm(out.squeeze(1).reshape(B, T, E))
        return out


class FullPermTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(cfg['vocab_size'], E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.pos_embs = nn.ParameterList(
            [nn.Parameter(torch.randn(SEQ_LEN, POS_DIM) * 0.1) for _ in range(N_LAYERS)]
        )
        self.layers = nn.ModuleList([FullPermBlock(i) for i in range(N_LAYERS)])
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
        return self.unembedder(torch.cat(outs, dim=-1))


def get_lr_scale(step):
    n_steps = cfg['n_steps']
    warmup = int(cfg.get('lr_warmup_fraction', 0.1) * n_steps)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


sampler = make_sampler(DEVICE, random_seed=1)
model = FullPermTransformer().to(DEVICE)

# Separate dense params (Adam) from PermLut weights (BitFlip)
dense_params = []
perm_modules = []
for n, p in model.named_parameters():
    if 'inner.projection.weights' not in n:
        dense_params.append(p)
for layer in model.layers:
    perm_modules.extend([layer.q_perm, layer.k_perm, layer.v_perm, layer.out_proj])

optimizer = torch.optim.Adam(dense_params, lr=cfg['lr'])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

bit_opt = BitFlipOptimizer(
    perm_modules,
    lr=0.01,
    lr_schedule_fn=get_lr_scale,
)

total_params, trainable_params = count_params(model)
n_perm = sum(m.inner.projection.weights.numel() for m in perm_modules)
n_dense = sum(p.numel() for p in dense_params)
print(f'Parameters: {total_params:,} (dense={n_dense:,}, perm_binary={n_perm:,})')
print(f'Context size: {SEQ_LEN}, batch size: {cfg["batch_size"]}')
print(f'Embedding: {E}, pos_dim: {POS_DIM} (concat)')
print(f'Q/K PermLut: in_nap={cfg["qk_input_nap"]} out_nap={cfg["qk_output_nap"]} tph={cfg["qk_tph"]} d_qk={d_qk}')
print(f'V PermLut: in_nap={cfg["v_input_nap"]} out_nap={cfg["v_output_nap"]} tph={cfg["v_tph"]} d_v={d_v} (dominance P={D_V_P})')
print(f'Out PermLut: in_nap={cfg["out_input_nap"]} out_nap={cfg["out_output_nap"]} tph={cfg["out_tph"]}')
print(f'BitFlip: n_samples={cfg.get("bitflip_n_samples", 32)}, flip_ratio={cfg.get("bitflip_flip_ratio", 0.05)}')
print(f'RankAttn temp={RANK_ATTN_TEMP}, soft_mode={SOFT_MODE}, temp={TEMP}')

csv_path = os.path.join(EXP_DIR, 'metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'val_loss'])

train_losses, val_losses, val_steps = [], [], []
best_val = float('inf')
ema = None
t0 = time.time()

model.train()
for step in range(cfg['n_steps']):
    x = sampler.sample_training_batch(cfg['batch_size']).long()
    inp = torch.empty_like(x)
    inp[:, 0] = BOS_ID
    inp[:, 1:] = x[:, :-1]
    logits = model(inp)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, V), x.reshape(B * T))
    optimizer.zero_grad()
    bit_opt.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    bit_opt.step()

    lv = loss.item()
    ema = lv if ema is None else 0.99 * ema + 0.01 * lv

    if step % 100 == 0:
        print(f'step {step:6d} | loss={ema:.4f} | lr={scheduler.get_last_lr()[0]:.2e}')

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
