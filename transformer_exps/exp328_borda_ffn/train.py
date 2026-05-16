"""exp314_dom_canon_sdpa -- fork of exp312 with DominanceCanonicalize+SDPA.

Architectural changes vs exp312:
  - RankAttention is removed. Q/K dominance is cleaned with
    DominanceCanonicalize (Borda + LN + rank-projection back to ±1) and fed
    directly to F.scaled_dot_product_attention as pairwise rank features.
    The `learnable_attn_scale_init` scalar is preserved and folded into the
    SDPA scale parameter (= attn_scale / sqrt(P_qk)).
  - Manual Borda einsums + nn.LayerNorm(attn), nn.LayerNorm(E) are replaced
    with DominanceToVector modules after attention (d_v) and after out_proj
    (E) — both include Borda + LN in one module.
  - q_norm/k_norm/out_norm are dropped (their roles subsumed by the new
    modules).
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
from spiky.lutorch.ranking_tools import (
    BordaFFN, DominanceCanonicalize, DominanceToVector,
)

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
D_QK_P = d_qk * (d_qk - 1) // 2
D_V_P = d_v * (d_v - 1) // 2


D_OUT_P = E * (E - 1) // 2


class LUTBlock(nn.Module):
    """No-LUT baseline mirroring exp314. BordaFFN n_outputs = P (pair
    count) — its output is "entangled Borda" in pair-dim. q/k/out_proj
    outputs go through DominanceCanonicalize (P→P via Borda bottleneck);
    v skips canonicalize. attn_to_vec / out_to_vec project dominance →
    Borda item-dim (matches exp314)."""

    def __init__(self, layer_idx):
        super().__init__()
        ffn_mult = cfg.get('ffn_mult', 4)
        # K-WTA sparsity on FFN hidden activations (fraction of hidden_dim).
        # If cfg['sparsity_fraction'] > 0, keep only top sparsity_fraction
        # of the `ffn_mult * n_heads * n_outputs` activations per sample.
        sfrac = cfg.get('sparsity_fraction', 0.0)
        def _sk(n_out, n_head):
            if sfrac <= 0.0 or sfrac >= 1.0:
                return None
            hidden = ffn_mult * n_head * n_out
            return max(1, int(round(hidden * sfrac)))

        # BordaFFN always operates in Borda (= pair) dim: n_outputs = P.
        self.q_proj   = BordaFFN(n_inputs=E,       n_outputs=D_QK_P,  n_heads=H, ffn_mult=ffn_mult, sparsity_k=_sk(D_QK_P, H))
        self.k_proj   = BordaFFN(n_inputs=E,       n_outputs=D_QK_P,  n_heads=H, ffn_mult=ffn_mult, sparsity_k=_sk(D_QK_P, H))
        self.v_proj   = BordaFFN(n_inputs=E,       n_outputs=D_V_P,   n_heads=H, ffn_mult=ffn_mult, sparsity_k=_sk(D_V_P, H))
        self.out_proj = BordaFFN(n_inputs=H * d_v, n_outputs=D_OUT_P, n_heads=1, ffn_mult=ffn_mult, sparsity_k=_sk(D_OUT_P, 1))

        canon_t = cfg.get('canon_temperature', 0.1)
        # P → P via Borda-bottleneck (D2V → V2D STE). Stays in pair/Borda dim.
        self.q_canon   = DominanceCanonicalize(d_qk, temperature=canon_t)
        self.k_canon   = DominanceCanonicalize(d_qk, temperature=canon_t)
        self.out_canon = DominanceCanonicalize(E,    temperature=canon_t)

        # Dominance (P) → Borda item-dim.
        self.attn_to_vec = DominanceToVector(d_v, normalise=False)   # no LN (matches exp314)
        self.out_to_vec  = DominanceToVector(E)                      # Borda + LN → E

        self.attn_scale = nn.Parameter(torch.tensor(
            float(cfg.get('learnable_attn_scale_init', 0.25))
        ))

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, _E)

        q_dom = self.q_canon(self.q_proj(xp))                             # [B*T, H, P_qk]
        k_dom = self.k_canon(self.k_proj(xp))
        v_dom = self.v_proj(xp)                                           # [B*T, H, P_v] (no canon)

        q = q_dom.view(B, T, H, D_QK_P).permute(0, 2, 1, 3)               # [B, H, T, P_qk]
        k = k_dom.view(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        v = v_dom.view(B, T, H, D_V_P ).permute(0, 2, 1, 3)               # [B, H, T, P_v]

        attn_dom = F.scaled_dot_product_attention(
            q * self.attn_scale, k, v, is_causal=True,
        )                                                                 # [B, H, T, P_v]
        attn = self.attn_to_vec(attn_dom)                                 # [B, H, T, d_v]

        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_dom = self.out_canon(self.out_proj(out_in))                   # [B*T, 1, P_out]
        out = self.out_to_vec(out_dom).squeeze(1).reshape(B, T, _E)       # [B, T, E]
        return out


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


# --- Build model + split params ---
sampler = TextSnippetSampler(DATA_PATH, CONTEXT_SIZE, TESTING_LENGTH, DEVICE, random_seed=1)
model = Model().to(DEVICE)

adam_params = list(model.parameters())
print(f'Adam-managed parameters: {sum(p.numel() for p in adam_params):,}')

adam_opt = torch.optim.AdamW(
    adam_params, lr=cfg['adam_lr'],
    weight_decay=cfg.get('adam_weight_decay', 0.0),
)
adam_scheduler = torch.optim.lr_scheduler.LambdaLR(adam_opt, get_lr_scale)

print(f'BordaFFN baseline (no LUTs): q/k n_outputs={d_qk}, v n_outputs={d_v}, out n_outputs={E}, H={H}')

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

    adam_opt.zero_grad()
    loss.backward()
    adam_opt.step()
    adam_scheduler.step()

    lv = loss.item()
    ema = lv if ema is None else 0.99*ema + 0.01*lv

    if step % 100 == 0:
        print(f'step {step:6d} | loss={ema:.4f} | lr={adam_scheduler.get_last_lr()[0]:.2e}')

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

total_params = sum(p.numel() for p in model.parameters())
summary = {
    'exp_name': cfg['exp_name'],
    'best_val_loss': best_val,
    'final_val_loss': val_losses[-1] if val_losses else None,
    'adam_params': total_params,
    'n_bit_luts': 0,
    'training_time_hours': round(elapsed / 3600, 3),
}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
