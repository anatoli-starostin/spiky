"""exp299_bit_permlut -- fork of exp297 with BitPermutationLUT + BitPermutationLUTOptimizer.

Each layer's q_perm, k_perm, v_perm, out_proj is a BitPermutationLUT (1-bit
weights, CUDA-only, CANONICAL_DISTINCT). They're trained by
BitPermutationLUTOptimizer (fp8 latent + fp8 moments Adam, per-table scaled).
Embeddings, LayerNorms, RankAttention's learnable scale and unembedder use
regular torch.optim.Adam.

BitPermutationLUT emits dominance only; q/k/out_proj need a Borda projection
back to their nominal dim (d_qk / E).
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
from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.bit_permutation_lut_optimizer import BitPermutationLUTOptimizer
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
N_LAYERS = cfg['num_layers']
D_QK_P = d_qk * (d_qk - 1) // 2
D_V_P = d_v * (d_v - 1) // 2


def _make_qk(seed_offset):
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_qk, n_heads=H,
        input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
        tph=cfg['qk_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'),
        device=DEVICE,
    )


def _make_v(seed_offset):
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_v, n_heads=H,
        input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'],
        tph=cfg['v_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'),
        device=DEVICE,
    )


def _make_out(seed_offset):
    return BitPermutationLUT(
        n_inputs=H * d_v, n_outputs=E, n_heads=1,
        input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'],
        tph=cfg['out_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'),
        device=DEVICE,
    )


def _make_ffn(seed_offset):
    # FFN BitPermLUT after out_proj. Input/output both E; capacity mirrors
    # out_proj (input_nap, output_nap, tph identical).
    return BitPermutationLUT(
        n_inputs=E, n_outputs=E, n_heads=1,
        input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'],
        tph=cfg['out_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'),
        device=DEVICE,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_perm = _make_qk(layer_idx)
        self.k_perm = _make_qk(100 + layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v_perm = _make_v(200 + layer_idx)
        self.rank_attn = RankAttention(
            d_qk=d_qk, d_v=D_V_P,
            smooth_mode=False,
            temperature=cfg.get('rank_attn_temperature', 0.1),
            sdpa_temperature=1.0,
            sdpa_forward_temperature=1.0,
            learnable_attn_scale_init=cfg.get('learnable_attn_scale_init', None),
        )
        self.out_proj = _make_out(400 + layer_idx)
        self.out_norm = nn.LayerNorm(E)
        # FFN BitPermLUT (same-capacity E->E) after out_proj + LN.
        self.ffn = _make_ffn(500 + layer_idx)
        self.ffn_norm = nn.LayerNorm(E)
        # Pre-scaled Borda matrices (1/sqrt(n_outputs-1)): dom -> rank.
        self.register_buffer('q_borda_m', self.q_perm.dom_borda_m.clone())
        self.register_buffer('v_borda_m', self.v_perm.dom_borda_m.clone())
        self.register_buffer('out_borda_m', self.out_proj.dom_borda_m.clone())
        self.register_buffer('ffn_borda_m', self.ffn.dom_borda_m.clone())

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, _E)
        x_flat = x.reshape(B * T, _E)

        # Q / K dominance -> Borda -> d_qk ranks.
        q_dom = self.q_perm(xp)                                       # [B*T, H, D_QK_P]
        k_dom = self.k_perm(xp)
        q = torch.einsum('bhp,kp->bhk', q_dom, self.q_borda_m)         # [B*T, H, d_qk]
        k = torch.einsum('bhp,kp->bhk', k_dom, self.q_borda_m)
        q = q.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)               # [B, H, T, d_qk]
        k = k.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # V stays in dominance space.
        v_dom = self.v_perm(x_flat).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)  # [B, H, T, P_v]

        attn_dom = self.rank_attn(q, k, v_dom, is_causal=True)         # [B, H, T, P_v]
        attn = torch.einsum('bhtp,kp->bhtk', attn_dom, self.v_borda_m)  # [B, H, T, d_v]

        # Out: H*d_v -> E via dominance + Borda.
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_dom = self.out_proj(out_in)                                 # [B*T, 1, P_out]
        out = torch.einsum('bhp,kp->bhk', out_dom, self.out_borda_m).squeeze(1).reshape(B, T, _E)
        out = self.out_norm(out)

        # FFN: E -> E via BitPermLUT + Borda + LN.
        ffn_in = out.reshape(B * T, _E)
        ffn_dom = self.ffn(ffn_in)                                      # [B*T, 1, P_ffn]
        ffn_out = torch.einsum('bhp,kp->bhk', ffn_dom, self.ffn_borda_m).squeeze(1).reshape(B, T, _E)
        ffn_out = self.ffn_norm(ffn_out)
        return ffn_out


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

bit_luts = []
for layer in model.layers:
    bit_luts += [layer.q_perm, layer.k_perm, layer.v_perm, layer.out_proj, layer.ffn]

# Regular Adam params = everything except those under bit_luts' control.
# bit_luts have zero Parameters (bit_weights is a buffer), so model.parameters()
# naturally excludes them.
adam_params = list(model.parameters())
print(f'bit LUTs: {len(bit_luts)} modules')
print(f'Adam-managed parameters: {sum(p.numel() for p in adam_params):,}')

adam_opt = torch.optim.Adam(adam_params, lr=cfg['adam_lr'])
adam_scheduler = torch.optim.lr_scheduler.LambdaLR(adam_opt, get_lr_scale)
bit_opt = BitPermutationLUTOptimizer(
    bit_luts,
    lr=cfg['bit_lut_lr'],
    beta1=cfg.get('bit_lut_beta1', 0.9),
    beta2=cfg.get('bit_lut_beta2', 0.999),
    lr_schedule_fn=get_lr_scale,
)

print(f'Q/K BitPermLUT: in_nap={cfg["qk_input_nap"]} out_nap={cfg["qk_output_nap"]} tph={cfg["qk_tph"]} d_qk={d_qk} P_qk={D_QK_P}')
print(f'V BitPermLUT:   in_nap={cfg["v_input_nap"]} out_nap={cfg["v_output_nap"]} tph={cfg["v_tph"]} d_v={d_v} P_v={D_V_P}')
print(f'Out BitPermLUT: in_nap={cfg["out_input_nap"]} out_nap={cfg["out_output_nap"]} tph={cfg["out_tph"]}')
print('Borda (pre-scaled) projects dominance -> d_qk / d_v / E')

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
    bit_opt.zero_grad()
    loss.backward()
    adam_opt.step()
    adam_scheduler.step()
    bit_opt.step()

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
    'n_bit_luts': len(bit_luts),
    'training_time_hours': round(elapsed / 3600, 3),
}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
bit_opt.close()
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
