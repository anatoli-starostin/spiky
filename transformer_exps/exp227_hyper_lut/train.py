"""
exp227_hyper_lut — HyperLUT transformer.
n_pairs=256, hidden_dim=1024, rational soft_mode, temp=0.1.
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
from spiky.lutorch.hyper_lut import HyperLUT, HyperLUTBackbone

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
VOCAB_SIZE = cfg['vocab_size']
SEQ_LEN = CONTEXT_SIZE
torch.manual_seed(cfg['random_seed'])

E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
N_PAIRS = cfg['n_pairs']
HID = cfg['hidden_dim']
TEMP = cfg['temperature']
SOFT = cfg['soft_mode']
SEED = cfg['random_seed']


def _make_hyper(n_heads, n_outputs, seed_offset):
    h = HyperLUT(
        input_dim=E, n_heads=n_heads, n_outputs=n_outputs,
        n_pairs=N_PAIRS, hidden_dim=HID,
        temperature=TEMP, soft_mode=SOFT,
        random_seed=SEED+seed_offset, device=DEVICE,
        use_layer_norm=True,
    )
    with torch.no_grad():
        nn.init.normal_(h.backbone.fc1_weight, std=0.001)
        nn.init.normal_(h.fc2.weight, std=0.001)
    return h


class HyperBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q = _make_hyper(H, d_qk, layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k = _make_hyper(H, d_qk, 100+layer_idx)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v = _make_hyper(H, d_v, 200+layer_idx)
        self.v_norm = nn.LayerNorm(d_v)
        self.out_proj = _make_hyper(1, E, 400+layer_idx)
        self.out_norm = nn.LayerNorm(E)
        self.ffn = _make_hyper(1, E, 600+layer_idx)
        self.ffn_norm = nn.LayerNorm(E)

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B*T, _E)
        x_flat = x.reshape(B*T, _E)

        q = self.q_norm(self.q(xp)).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_norm(self.k(xp)).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        v = self.v_norm(self.v(x_flat)).reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        proj = self.out_norm(self.out_proj(attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)).squeeze(1)).reshape(B, T, _E)
        x = x + proj

        ffn_out = self.ffn_norm(self.ffn(x.reshape(B*T, _E)).squeeze(1)).reshape(B, T, _E)
        x = x + ffn_out
        return x


class UniformHyperLUTTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        n_layers = cfg['num_layers']
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.randn(SEQ_LEN, E) * 0.1) for _ in range(n_layers)])
        self.layers = nn.ModuleList([HyperBlock(i) for i in range(n_layers)])
        self.unembedder = nn.Sequential(
            nn.Linear(E, 128), nn.GELU(), nn.Linear(128, VOCAB_SIZE, bias=False),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
        return self.unembedder(x)


def get_lr_scale(step):
    n_steps = cfg['n_steps']
    warmup = int(cfg.get('lr_warmup_fraction', 0.1) * n_steps)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


sampler = make_sampler(DEVICE, random_seed=1)
model = UniformHyperLUTTransformer().to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
    {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

total_params, trainable_params = count_params(model)
print(f'Parameters: {total_params:,}')
print(f'HyperLUT: n_pairs={N_PAIRS}, hidden_dim={HID}, temp={TEMP}, soft_mode={SOFT}')

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
    loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    lv = loss.item()
    ema = lv if ema is None else 0.99*ema + 0.01*lv

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
