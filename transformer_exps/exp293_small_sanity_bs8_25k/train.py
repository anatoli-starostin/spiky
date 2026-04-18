"""
exp290_sdpa_dom_bs8_25k — exp288 fork. Q, K PermLut also use return_dominance=True.
Attention = F.scaled_dot_product_attention over 120-D dominance (not RankAttention).
No q_norm/k_norm. /output_nap done inside PermLut (library). /(d_v-1) done after Borda.
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
from spiky.lutorch.permutational_lut import PermutationalLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
CONTEXT_SIZE = cfg['context_size']
VOCAB_SIZE = cfg['vocab_size']
BOS_ID = 256
RAW_VOCAB_SIZE = 256
TESTING_LENGTH = 10_000
DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'workbooks', 'fineweb_texts.txt')
)

torch.manual_seed(cfg['random_seed'])

E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
N_LAYERS = cfg['num_layers']
SOFT_MODE = cfg['soft_mode']
TEMP = cfg['temperature']
D_QK_P = d_qk * (d_qk - 1) // 2
D_V_P = d_v * (d_v - 1) // 2

SCRAMBLED_POLICY = AnchorSamplingPolicy(cfg.get('scrambled_policy', 'full_coverage'))
INPUT_ANCHOR_POLICY = AnchorSamplingPolicy(cfg.get('input_anchor_policy', 'full_coverage'))

PERM_KWARGS = dict(
    pair_mode='scrambled',
    soft_mode=SOFT_MODE,
    temperature=TEMP,
    scrambled_policy=SCRAMBLED_POLICY,
    input_anchor_policy=INPUT_ANCHOR_POLICY,
    device=DEVICE,
    recompute_in_backward=True,
    initial_weights_noise=0.001,
)


def _make_qk_perm(seed_offset):
    return PermutationalLut(
        n_inputs=E, n_outputs=d_qk, n_heads=H,
        input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
        tph=cfg['qk_tph'],
        return_dominance=True,
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


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_perm = _make_qk_perm(layer_idx)
        self.k_perm = _make_qk_perm(100 + layer_idx)
        self.v_perm = _make_v_perm(200 + layer_idx)
        self.out_proj = _make_out_perm(400 + layer_idx)
        self.out_norm = nn.LayerNorm(E)
        self.register_buffer('borda_m', self.v_perm.dom_borda_m.clone())

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, _E)
        x_flat = x.reshape(B * T, _E)

        q = self.q_perm(xp).reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)  # [B, H, T, P_qk]
        k = self.k_perm(xp).reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)  # [B, H, T, P_qk]
        v_dom = self.v_perm(x_flat).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)  # [B, H, T, P_v]

        attn_dom = F.scaled_dot_product_attention(q, k, v_dom, is_causal=True)  # [B, H, T, P_v]
        attn = torch.einsum('bhtp,kp->bhtk', attn_dom, self.borda_m)  # [B, H, T, d_v]; borda_m pre-scaled by 1/sqrt(d_v-1)

        x = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)).squeeze(1).reshape(B, T, _E)
        x = self.out_norm(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.pos_embs = nn.ParameterList(
            [nn.Parameter(torch.randn(CONTEXT_SIZE, E) * 0.1) for _ in range(N_LAYERS)]
        )
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        concat_dim = E * N_LAYERS
        self.unembedder = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, concat_dim * 4),
            nn.ReLU(),
            nn.Linear(concat_dim * 4, VOCAB_SIZE),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        outs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            outs.append(x)
        concat = torch.cat(outs, dim=-1)
        return self.unembedder(concat)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def generate_text(model, prefix, length, device):
    ctx = list(prefix.encode('utf-8'))
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            trunc = ctx[-(CONTEXT_SIZE - 1):]
            x = torch.zeros([1, CONTEXT_SIZE], dtype=torch.long, device=device)
            x[0, 0] = BOS_ID
            x[0, 1:1+len(trunc)] = torch.tensor(trunc, dtype=torch.long, device=device)
            pos = len(trunc)
            logits = model(x)
            probs = torch.softmax(logits[0, pos, :RAW_VOCAB_SIZE], dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ctx.append(next_id)
    return bytes(c for c in ctx if 0 <= c < 256).decode('utf-8', errors='replace')


def evaluate_model(model, sampler, batch_size):
    model.eval()
    losses = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in sampler.testing_batches_iterator(batch_size):
            inp = torch.empty(batch.shape[0], batch.shape[1], dtype=torch.long, device=batch.device)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1].long()
            logits = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), batch.long().reshape(B*T))
            losses.append(loss.item())
    gen = generate_text(model, 'Once upon a time ', length=100, device=device)
    print(f'[GEN]: {gen}')
    model.train()
    return sum(losses) / len(losses)


def get_lr_scale(step):
    n_steps = cfg['n_steps']
    warmup = int(cfg.get('lr_warmup_fraction', 0.1) * n_steps)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


sampler = TextSnippetSampler(DATA_PATH, CONTEXT_SIZE, TESTING_LENGTH, DEVICE, random_seed=1)
model = Model().to(DEVICE)
model = torch.compile(model, dynamic=False)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

total_params, _ = count_params(model)
print(f'Parameters: {total_params:,}')
print(f'Context size: {CONTEXT_SIZE}, batch size: {cfg["batch_size"]}')
print(f'soft_mode={SOFT_MODE}, temp={TEMP}')
print(f'Q/K PermLut (dominance): in_nap={cfg["qk_input_nap"]} out_nap={cfg["qk_output_nap"]} tph={cfg["qk_tph"]} d_qk={d_qk} P_qk={D_QK_P}')
print(f'V PermLut (dominance): in_nap={cfg["v_input_nap"]} out_nap={cfg["v_output_nap"]} tph={cfg["v_tph"]} d_v={d_v} P_v={D_V_P}')
print(f'Out PermLut: in_nap={cfg["out_input_nap"]} out_nap={cfg["out_output_nap"]} tph={cfg["out_tph"]}')
print('SDPA over dominance; Borda / (d_v-1)')

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
    'training_time_hours': round(elapsed / 3600, 3),
}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
