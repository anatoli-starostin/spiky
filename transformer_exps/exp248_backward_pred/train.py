"""
exp233_mixed_nap_qk5 — Q/K use nap=5, V/OutProj/FFN use nap=12.
tph=128, QK LayerNorm.
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
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
NAP_QK = cfg['nap_qk']
NAP = cfg['nap']
TPH = cfg['tph']


def _make_lut(n_heads, n_outputs, nap, seed_offset):
    return MultiHeadLut(
        input_dim=E, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=TPH,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
        initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, recompute_in_backward=True,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_lut = _make_lut(H, d_qk, NAP_QK, layer_idx)
        self.k_lut = _make_lut(H, d_qk, NAP_QK, 100+layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v_lut = _make_lut(H, d_v, NAP, 200+layer_idx)
        self.out_proj = _make_lut(1, E, NAP, 400+layer_idx)
        self.ffn = _make_lut(1, E, NAP, 600+layer_idx)
        # Backward predictor: predict input from output (same capacity as out_proj)
        self.backward_pred = _make_lut(1, E, NAP, 700+layer_idx)
        self.pred_loss = None  # stored per forward for external access

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        x_in = x  # save input for backward prediction
        xp = (x + pos_emb.unsqueeze(0)).reshape(B*T, _E)
        x_flat = x.reshape(B*T, _E)

        q = self.q_lut(xp).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_lut(xp).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_lut(x_flat).reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)).squeeze(1).reshape(B, T, _E)

        x = self.ffn(x.reshape(B*T, _E)).squeeze(1).reshape(B, T, _E)

        # Backward prediction: predict input rankings from output
        if self.training:
            x_pred = self.backward_pred(x.reshape(B*T, _E)).squeeze(1)  # [B*T, E]
            x_target = x_in.detach().reshape(B*T, _E)
            # Ranking loss: for all pairs (i,j), prediction should preserve ordering
            # Use precomputed pair indices for efficiency
            if not hasattr(self, '_pair_i'):
                idx = torch.triu_indices(_E, _E, offset=1, device=x.device)
                self._pair_i = idx[0]
                self._pair_j = idx[1]
            pi, pj = self._pair_i, self._pair_j
            # Target: sign of difference for each pair
            d_target = x_target[:, pi] - x_target[:, pj]  # [B*T, n_pairs]
            target_sign = (d_target > 0).float()  # 1 if x[i] > x[j], 0 otherwise
            # Prediction: soft difference
            d_pred = x_pred[:, pi] - x_pred[:, pj]  # [B*T, n_pairs]
            # Binary cross-entropy: does prediction agree on ordering?
            self.pred_loss = F.binary_cross_entropy_with_logits(d_pred, target_sign)

        return x


class UniformLUTTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(cfg['vocab_size'], E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        n_layers = cfg['num_layers']
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.randn(SEQ_LEN, E) * 0.1) for _ in range(n_layers)])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(n_layers)])
        self.unembedder = nn.Sequential(
            nn.Linear(E, 128), nn.GELU(), nn.Linear(128, cfg['vocab_size'], bias=False),
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
model = UniformLUTTransformer().to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
    {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

total_params, trainable_params = count_params(model)
print(f'Parameters: {total_params:,}')
print(f'Mixed NAP: Q/K nap={NAP_QK}, V/OutProj/FFN nap={NAP}, tph={TPH}')

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
    ce_loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))
    # Add backward prediction losses from all layers
    pred_weight = cfg.get('backward_pred_weight', 0.1)
    pred_losses = [layer.pred_loss for layer in model.layers if layer.pred_loss is not None]
    pred_loss = sum(pred_losses) / len(pred_losses) if pred_losses else 0.0
    loss = ce_loss + pred_weight * pred_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    lv = ce_loss.item()
    plv = pred_loss.item() if isinstance(pred_loss, torch.Tensor) else 0.0
    ema = lv if ema is None else 0.99*ema + 0.01*lv

    if step % 100 == 0:
        print(f'step {step:6d} | ce={ema:.4f} pred={plv:.4f} | lr={scheduler.get_last_lr()[0]:.2e}')

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
