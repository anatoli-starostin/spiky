"""
GRU baseline: character-level GRU language model on same dataset.
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from transformer_exps.common import make_sampler, evaluate_model, count_params, save_summary, BOS_ID, CONTEXT_SIZE

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda:0'
SEQ_LEN = CONTEXT_SIZE
torch.manual_seed(42)

VOCAB_SIZE = 257
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 2
N_STEPS = 50000
BATCH_SIZE = 256
LR = 0.001


class GRUCell(nn.Module):
    """Single GRU cell: z = sigmoid(W_z x + U_z h), r = sigmoid(W_r x + U_r h), h' = tanh(W_h x + U_h (r*h))."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h):
        # x: [B, input_dim], h: [B, hidden_dim]
        xh = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.W_z(xh))
        r = torch.sigmoid(self.W_r(xh))
        xrh = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.W_h(xrh))
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class GRULayer(nn.Module):
    """Unrolled GRU over sequence length."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cell = GRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [B, T, input_dim]
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        outputs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outputs.append(h)
        return torch.stack(outputs, dim=1)  # [B, T, hidden_dim]


class CharGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.layers = nn.ModuleList([
            GRULayer(EMBEDDING_DIM if i == 0 else HIDDEN_DIM, HIDDEN_DIM)
            for i in range(NUM_LAYERS)
        ])
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.out(h)


sampler = make_sampler(DEVICE, random_seed=1)
model = CharGRU().to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"GRU Parameters: {total_params:,}")
print(f"Config: embed={EMBEDDING_DIM}, hidden={HIDDEN_DIM}, layers={NUM_LAYERS}")


def get_lr_scale(step):
    warmup = int(0.1 * N_STEPS)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(N_STEPS - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

train_losses, val_losses, val_steps = [], [], []
best_val = float('inf')
ema = None
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

    if step % 1000 == 0:
        val = evaluate_model(model, sampler, BATCH_SIZE)
        if val < best_val:
            best_val = val
        print(f"[VAL] step {step}: {val:.4f}")
        train_losses.append(ema)
        val_losses.append(val)
        val_steps.append(step)

elapsed = time.time() - t0

plt.figure(figsize=(8, 4))
plt.plot(val_steps, train_losses, label='train')
plt.plot(val_steps, val_losses, label='val')
plt.xlabel('steps'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=100)
plt.close()

summary = {
    'exp_name': 'exp_gru_baseline',
    'best_val_loss': best_val,
    'final_val_loss': val_losses[-1] if val_losses else None,
    'total_params': total_params,
    'training_time_hours': round(elapsed / 3600, 3),
}
save_summary(EXP_DIR, summary)
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
