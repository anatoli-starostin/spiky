"""
exp284_domv_qkvbig — exp276 fork with Q/K/V nap=6 tph=256 (was nap=5 tph=128).
out_proj unchanged. ctx=128, 100K bs=32.
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
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import RankAttention

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
NAP_QK = cfg['nap_qk']
NAP_V = cfg['nap_v']
TPH = cfg['tph']
NAP_OUT = cfg['nap_out']
TPH_OUT = cfg['tph_out']
N_LAYERS = cfg['num_layers']
RANK_ATTN_TEMP = cfg.get('rank_attn_temperature', 0.1)
DOM_V_TEMP = cfg.get('dom_v_temperature', 0.1)

# ---- Dominance-V helpers (mirror exp274) ----
_DV_P = d_v * (d_v - 1) // 2
_DV_PAIR_I, _DV_PAIR_J = torch.triu_indices(d_v, d_v, offset=1).tolist()
_DV_BORDA_M = torch.zeros(d_v, _DV_P)
for _p, (_i, _j) in enumerate(zip(_DV_PAIR_I, _DV_PAIR_J)):
    _DV_BORDA_M[_i, _p] = 1.0
    _DV_BORDA_M[_j, _p] = -1.0


def ranks_to_dominance(v, temperature):
    """[..., d_v] -> [..., P] centred signed dominance (STE: hard fwd, rational bwd)."""
    i_idx = torch.tensor(_DV_PAIR_I, device=v.device)
    j_idx = torch.tensor(_DV_PAIR_J, device=v.device)
    d_raw = v.index_select(-1, i_idx) - v.index_select(-1, j_idx)
    hard = torch.where(d_raw > 0, 0.5, -0.5).to(v.dtype)
    soft = 0.5 * d_raw / (temperature + d_raw.abs())
    return hard.detach() + (soft - soft.detach())


def dominance_to_ranks(d_vec, borda_m):
    """[..., P] -> [..., d_v] via Borda projection."""
    return torch.einsum('...p,kp->...k', d_vec, borda_m)


def _make_lut(n_heads, n_outputs, nap, seed_offset, tph=None):
    if tph is None:
        tph = TPH
    return MultiHeadLut(
        input_dim=E, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
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
        self.k_lut = _make_lut(H, d_qk, NAP_QK, 100 + layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v_lut = _make_lut(H, d_v, NAP_V, 200 + layer_idx)
        # RankAttention projects Q/K into pair-dominance space internally.
        # We also pass V already converted to dominance (shape [B,H,T,P=28]).
        self.rank_attn = RankAttention(
            d_qk=d_qk, d_v=_DV_P,
            smooth_mode=False,
            temperature=RANK_ATTN_TEMP,
            sdpa_temperature=1.0,
            sdpa_forward_temperature=1.0,
        )
        # Regular MultiHeadLut out_proj, matching exp255's settings
        self.out_proj = _make_lut(1, E, NAP_OUT, 400 + layer_idx, tph=TPH_OUT)
        self.out_norm = nn.LayerNorm(E)
        self.register_buffer('borda_m', _DV_BORDA_M.clone())

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B*T, _E)
        x_flat = x.reshape(B*T, _E)

        q = self.q_lut(xp).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_lut(xp).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_lut(x_flat).reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        # Convert V rank vectors into centred signed dominance pair vectors
        v_dom = ranks_to_dominance(v, DOM_V_TEMP)  # [B, H, T, P]

        # RankAttention: Q/K → pair-dominance features → SDPA over dominance V
        attn_dom = self.rank_attn(q, k, v_dom, is_causal=True)  # [B, H, T, P]
        # Borda-aggregate the mixed dominance back into rank vectors
        attn = dominance_to_ranks(attn_dom, self.borda_m)  # [B, H, T, d_v]

        x = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)).squeeze(1).reshape(B, T, _E)
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

dense_params = [p for n, p in model.named_parameters() if 'unembedder' in n]
lut_params = [p for n, p in model.named_parameters() if 'unembedder' not in n]
optimizer = torch.optim.Adam([
    {'params': dense_params, 'lr': cfg['lr'] * 3},
    {'params': lut_params, 'lr': cfg['lr']},
])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

total_params, _ = count_params(model)
print(f'Parameters: {total_params:,}')
print(f'Context size: {CONTEXT_SIZE}, batch size: {cfg["batch_size"]}')
print(f'RankAttention temperature: {RANK_ATTN_TEMP}')
print(f'Q/K nap={NAP_QK} tph={TPH}, V nap={NAP_V} tph={TPH}')
print(f'Out MultiHeadLut: nap={NAP_OUT} tph={TPH_OUT}')

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
