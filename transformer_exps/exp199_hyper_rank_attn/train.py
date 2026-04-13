"""
exp199_hyper_rank_attn — HyperLUT-based Q/K/V/OutProj with SDPA attention.
Q/K take cat(x, pos_emb), V takes x, standard SDPA, no FFN.
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
from spiky.lutorch.hyper_lut import HyperLUT

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
SEQ_LEN = CONTEXT_SIZE
torch.manual_seed(cfg['random_seed'])

E = cfg['embedding_dim']
P = cfg['positional_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
temp = cfg.get('temperature', 0.1)
soft_mode = cfg.get('soft_mode', 'sigmoid')


class HyperBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_hyper = HyperLUT(
            input_dim=E+P, n_heads=H, n_outputs=d_qk,
            n_pairs=cfg['qk_n_pairs'], hidden_dim=cfg['qk_hidden'],
            temperature=temp, soft_mode=soft_mode,
            random_seed=cfg['random_seed']+layer_idx, device=DEVICE,
        )
        self.k_hyper = HyperLUT(
            input_dim=E+P, n_heads=H, n_outputs=d_qk,
            n_pairs=cfg['qk_n_pairs'], hidden_dim=cfg['qk_hidden'],
            temperature=temp, soft_mode=soft_mode,
            random_seed=cfg['random_seed']+100+layer_idx, device=DEVICE,
        )
        self.v_hyper = HyperLUT(
            input_dim=E, n_heads=H, n_outputs=d_v,
            n_pairs=cfg['v_n_pairs'], hidden_dim=cfg['v_hidden'],
            temperature=temp, soft_mode=soft_mode,
            random_seed=cfg['random_seed']+200+layer_idx, device=DEVICE,
        )
        self.out_proj = HyperLUT(
            input_dim=H*d_v, n_heads=1, n_outputs=E,
            n_pairs=cfg['outproj_n_pairs'], hidden_dim=cfg['outproj_hidden'],
            temperature=temp, soft_mode=soft_mode,
            random_seed=cfg['random_seed']+400+layer_idx, device=DEVICE,
        )
        self.norm = nn.LayerNorm(E)

    def forward(self, x, pos_emb):
        B, T, _E = x.shape

        xp = torch.cat([x, pos_emb.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        xp_flat = xp.reshape(B*T, E+P)
        x_flat = x.reshape(B*T, _E)

        q = self.q_hyper(xp_flat).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_hyper(xp_flat).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        v = self.v_hyper(x_flat).reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)

        proj = self.out_proj(attn).squeeze(1).reshape(B, T, _E)

        return x + self.norm(proj)


class HyperRankAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(cfg['vocab_size'], E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.pos_emb = nn.Parameter(torch.randn(SEQ_LEN, P) * 0.1)
        self.layers = nn.ModuleList([HyperBlock(i) for i in range(cfg['num_layers'])])
        self.unembedder = nn.Linear(E, cfg['vocab_size'], bias=False)

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer in self.layers:
            x = layer(x, self.pos_emb)
        return self.unembedder(x)


def get_lr_scale(step):
    n_steps = cfg['n_steps']
    warmup = int(cfg.get('lr_warmup_fraction', 0.1) * n_steps)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


sampler = make_sampler(DEVICE, random_seed=1)
model = HyperRankAttn().to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
    {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

total_params, trainable_params = count_params(model)
print(f'Parameters: {total_params:,} total, {trainable_params:,} trainable')
print(f'Architecture: HyperLUT Q/K(pairs={cfg["qk_n_pairs"]},h={cfg["qk_hidden"]}) + V(pairs={cfg["v_n_pairs"]},h={cfg["v_hidden"]}) + SDPA + OutProj(pairs={cfg["outproj_n_pairs"]},h={cfg["outproj_hidden"]}), {soft_mode}')

csv_path = os.path.join(EXP_DIR, 'metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'val_loss'])

train_losses, val_losses, steps_log = [], [], []
best_val_loss = float('inf')
best_step = 0
ema = None
alpha = 0.01
start_time = time.time()

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
    ema = lv if ema is None else (1-alpha)*ema + alpha*lv

    if step % 100 == 0:
        lr = scheduler.get_last_lr()[0]
        print(f'step {step:6d} | train_loss={ema:.4f} | lr={lr:.2e}')

    if step % cfg['test_every'] == 0:
        val_loss = evaluate_model(model, sampler, cfg['test_batch_size'])
        print(f'[VAL] step {step}: val_loss={val_loss:.4f}')
        train_losses.append(ema)
        val_losses.append(val_loss)
        steps_log.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{val_loss:.6f}'])
        csv_f.flush()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = step

csv_f.close()
elapsed = time.time() - start_time

plt.figure(figsize=(8, 4))
plt.plot(steps_log, train_losses, label='train')
plt.plot(steps_log, val_losses, label='val')
plt.xlabel('steps'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=100)
plt.close()

summary = {
    'exp_name': cfg['exp_name'],
    'best_val_loss': best_val_loss,
    'final_val_loss': val_losses[-1] if val_losses else None,
    'best_step': best_step,
    'total_steps': cfg['n_steps'],
    'total_params': total_params,
    'trainable_params': trainable_params,
    'training_time_hours': round(elapsed / 3600, 3),
}
save_summary(EXP_DIR, summary)
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('Checkpoint saved.')
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
