"""
exp197_rank_attn_v2 — RankAttention revisited with modern config.
Q/K LUTs take cat(x, pos_emb), V LUT takes x, SDPA attention, out_proj LUT, FFN LUT.
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


def _make_lut(input_dim, n_heads, n_outputs, nap, tph, seed_offset):
    return MultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
        smooth_mode=cfg['smooth_mode'], n_alternatives=cfg['n_alternatives'],
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
        initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
        recompute_in_backward=True,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_lut = _make_lut(E + P, H, d_qk, cfg['qk_nap'], cfg['qk_tph'], layer_idx)
        self.k_lut = _make_lut(E + P, H, d_qk, cfg['qk_nap'], cfg['qk_tph'], 100 + layer_idx)
        self.v_lut = _make_lut(E, H, d_v, cfg['v_nap'], cfg['v_tph'], 200 + layer_idx)
        self.out_proj = _make_lut(H * d_v, 1, E, cfg['outproj_nap'], cfg['outproj_tph'], 400 + layer_idx)
        self.norm1 = nn.LayerNorm(E)
        self.ffn = _make_lut(E, 1, E, cfg['ffn_nap'], cfg['ffn_tph'], 600 + layer_idx)
        self.norm2 = nn.LayerNorm(E)

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = torch.cat([x, pos_emb.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        xp_flat = xp.reshape(B * T, E + P)

        q = self.q_lut(xp_flat).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_lut(xp_flat).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        v = self.v_lut(x.reshape(B * T, _E)).reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        proj = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)).squeeze(1).reshape(B, T, _E)
        x = x + self.norm1(proj)

        ffn_out = self.ffn(x.reshape(B * T, _E)).squeeze(1).reshape(B, T, _E)
        x = x + self.norm2(ffn_out)
        return x


class LUTRankAttnV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(cfg['vocab_size'], E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        n_layers = cfg['num_layers']
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.randn(SEQ_LEN, P) * 0.1) for _ in range(n_layers)])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(n_layers)])
        self.unembedder = nn.Sequential(
            nn.Linear(E, 256), nn.GELU(), nn.Linear(256, cfg['vocab_size'], bias=False),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
        return self.unembedder(x)


# ── LR schedule ────────────────────────────────────────────────────────────────

def get_lr_scale(step):
    n_steps = cfg['n_steps']
    warmup = int(cfg.get('lr_warmup_fraction', 0.1) * n_steps)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


# ── Run ────────────────────────────────────────────────────────────────────────

sampler = make_sampler(DEVICE, random_seed=1)
model = LUTRankAttnV2().to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
    {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

total_params, trainable_params = count_params(model)
print(f'Parameters: {total_params:,} total, {trainable_params:,} trainable')
print(f'Architecture: Q/K LUT(cat(x,pos), nap={cfg["qk_nap"]}, tph={cfg["qk_tph"]}) + V LUT + SDPA + OutProj LUT, no FFN')

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
    loss = F.cross_entropy(logits.reshape(B * T, V), x.reshape(B * T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    lv = loss.item()
    ema = lv if ema is None else (1 - alpha) * ema + alpha * lv

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
