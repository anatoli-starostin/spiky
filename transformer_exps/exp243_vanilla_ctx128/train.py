"""
exp243_vanilla_ctx128 — Vanilla causal transformer with context_size=128.
Same architecture as exp001 but longer context.
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from spiky.util.text_snippet_sampler import TextSnippetSampler

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
CONTEXT_SIZE = cfg['context_size']
VOCAB_SIZE = cfg['vocab_size']
BOS_ID = 256
TESTING_LENGTH = 10_000
DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'workbooks', 'fineweb_texts.txt')
)

torch.manual_seed(cfg['random_seed'])
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

sampler = TextSnippetSampler(DATA_PATH, CONTEXT_SIZE, TESTING_LENGTH, DEVICE, random_seed=cfg['random_seed'])


class SDPATransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.permute(0,2,1,3).reshape(B, T, C)
        attn = self.out_proj(attn)
        x = x + self.norm1(self.dropout(attn))
        x = x + self.norm2(self.dropout(self.ffn(x)))
        return x


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, max_len, dropout, ff_mult):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.token_emb.weight.data.uniform_(-1, 1)
        self.register_buffer('pos_emb', torch.empty(max_len, d_model).uniform_(-1, 1))
        self.layers = nn.ModuleList([
            SDPATransformerLayer(d_model, n_heads, d_model * ff_mult, dropout)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.token_emb(x) + self.pos_emb[:x.shape[1]]
        for layer in self.layers:
            h = layer(h)
        return self.out(h)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def evaluate_model(model, sampler, batch_size):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in sampler.testing_batches_iterator(batch_size):
            inp = torch.empty(batch.shape[0], batch.shape[1], dtype=torch.long, device=batch.device)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1].long()
            logits = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), batch.long().reshape(B*T))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def get_lr_scale(step):
    n_steps = cfg['n_steps']
    warmup = int(cfg.get('lr_warmup_fraction', 0.1) * n_steps)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


model = CharTransformer(
    vocab_size=cfg['vocab_size'],
    d_model=cfg['d_model'],
    n_heads=cfg['n_heads'],
    num_layers=cfg['num_layers'],
    max_len=CONTEXT_SIZE,
    dropout=cfg['dropout'],
    ff_mult=cfg['dim_feedforward_multiplier'],
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

total_params, trainable_params = count_params(model)
print(f'Parameters: {total_params:,}')
print(f'Vanilla transformer: d_model={cfg["d_model"]}, n_heads={cfg["n_heads"]}, layers={cfg["num_layers"]}, context={CONTEXT_SIZE}')

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
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
