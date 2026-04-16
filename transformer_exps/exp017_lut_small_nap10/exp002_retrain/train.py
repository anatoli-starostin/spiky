"""
Retrain vanilla baseline (exp002 config) with checkpoint save at the end.
d_model=256, n_heads=4, 6 layers, ff_mult=4, bs=128, lr=0.001, warmup+cosine, 100K steps.
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from transformer_exps.common import make_sampler, BOS_ID, CONTEXT_SIZE

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda:0'
SEQ_LEN = CONTEXT_SIZE
torch.manual_seed(1)

D_MODEL = 256
N_HEADS = 4
NUM_LAYERS = 6
FF_MULT = 4
N_STEPS = 100000
BATCH_SIZE = 128
LR = 0.001
WARMUP_FRACTION = 0.1
TEST_EVERY = 1000


class SDPATransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.permute(0,2,1,3).reshape(B, T, C)
        x = x + self.norm1(self.out_proj(attn))
        x = x + self.norm2(self.ffn(x))
        return x


class CharTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(257, D_MODEL)
        self.token_emb.weight.data.uniform_(-1, 1)
        self.register_buffer('pos_emb', torch.empty(SEQ_LEN, D_MODEL).uniform_(-1, 1))
        self.layers = nn.ModuleList([
            SDPATransformerLayer(D_MODEL, N_HEADS, D_MODEL * FF_MULT)
            for _ in range(NUM_LAYERS)
        ])
        self.out = nn.Linear(D_MODEL, 257)

    def forward(self, x):
        h = self.token_emb(x) + self.pos_emb[:x.shape[1]]
        for layer in self.layers:
            h = layer(h)
        return self.out(h)


def evaluate(model, sampler):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in sampler.testing_batches_iterator(BATCH_SIZE):
            inp = torch.empty(batch.shape[0], batch.shape[1], dtype=torch.long, device=DEVICE)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1].long()
            logits = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), batch.long().reshape(B*T))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def get_lr_scale(step):
    warmup = int(WARMUP_FRACTION * N_STEPS)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(N_STEPS - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


sampler = make_sampler(DEVICE, random_seed=1)
model = CharTransformer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

ema = None
best_val = float('inf')
train_losses, val_losses, val_steps = [], [], []
t0 = time.time()

model.train()
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
    optimizer.step()
    scheduler.step()

    lv = loss.item()
    ema = lv if ema is None else 0.99*ema + 0.01*lv

    if step % 100 == 0:
        print(f"step {step:6d} | loss={ema:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    if step % TEST_EVERY == 0:
        val = evaluate(model, sampler)
        if val < best_val:
            best_val = val
        print(f"[VAL] step {step}: {val:.4f}")
        train_losses.append(ema)
        val_losses.append(val)
        val_steps.append(step)

elapsed = time.time() - t0

torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print("Checkpoint saved.")

plt.figure(figsize=(8, 4))
plt.plot(val_steps, train_losses, label='train')
plt.plot(val_steps, val_losses, label='val')
plt.xlabel('steps'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=100)
plt.close()

summary = {
    'best_val_loss': best_val,
    'final_val_loss': val_losses[-1] if val_losses else None,
    'total_params': total_params,
    'training_time_hours': round(elapsed / 3600, 3),
}
print(json.dumps(summary, indent=2))
