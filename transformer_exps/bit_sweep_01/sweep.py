"""Short 5K-step DraftBPLUTO sweep on the exp299 architecture.

No warmup, constant lr, no decay. Purpose: early-reject bad hyperparameter
configs. Reports final val_loss after 5K steps for each config.
"""
import sys, os, json, math, time, csv, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from spiky.util.text_snippet_sampler import TextSnippetSampler
from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.draft_bpluto import DraftBPLUTO
from spiky.lutorch.ranking_tools import RankAttention

DEVICE = 'cuda:0'
CONTEXT_SIZE = 128
VOCAB_SIZE = 257
BOS_ID = 256
RAW_VOCAB_SIZE = 256
TESTING_LENGTH = 10_000
DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'workbooks', 'fineweb_texts.txt')
)

# exp299 architecture.
E, H = 32, 4
d_qk, d_v = 24, 16
N_LAYERS = 6
D_QK_P = d_qk * (d_qk - 1) // 2
D_V_P = d_v * (d_v - 1) // 2
QK_INAP, QK_ONAP, QK_TPH = 5, 24, 192
V_INAP, V_ONAP, V_TPH = 5, 16, 128
OUT_INAP, OUT_ONAP, OUT_TPH = 10, 32, 1024
RANDOM_SEED = 42


def _make_qk(seed_offset, init_std, soft):
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_qk, n_heads=H,
        input_nap=QK_INAP, output_nap=QK_ONAP, tph=QK_TPH,
        random_seed=RANDOM_SEED + seed_offset,
        initial_weights_noise=init_std,
        soft_backward=soft,
        device=DEVICE,
    )


def _make_v(seed_offset, init_std, soft):
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_v, n_heads=H,
        input_nap=V_INAP, output_nap=V_ONAP, tph=V_TPH,
        random_seed=RANDOM_SEED + seed_offset,
        initial_weights_noise=init_std,
        soft_backward=soft,
        device=DEVICE,
    )


def _make_out(seed_offset, init_std, soft):
    return BitPermutationLUT(
        n_inputs=H * d_v, n_outputs=E, n_heads=1,
        input_nap=OUT_INAP, output_nap=OUT_ONAP, tph=OUT_TPH,
        random_seed=RANDOM_SEED + seed_offset,
        initial_weights_noise=init_std,
        soft_backward=soft,
        device=DEVICE,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx, init_std, soft):
        super().__init__()
        self.q_perm = _make_qk(layer_idx, init_std, soft)
        self.k_perm = _make_qk(100 + layer_idx, init_std, soft)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v_perm = _make_v(200 + layer_idx, init_std, soft)
        self.rank_attn = RankAttention(
            d_qk=d_qk, d_v=D_V_P,
            smooth_mode=False, temperature=0.1,
            sdpa_temperature=1.0, sdpa_forward_temperature=1.0,
            learnable_attn_scale_init=0.25,
        )
        self.out_proj = _make_out(400 + layer_idx, init_std, soft)
        self.out_norm = nn.LayerNorm(E)
        self.register_buffer('q_borda_m', self.q_perm.dom_borda_m.clone())
        self.register_buffer('v_borda_m', self.v_perm.dom_borda_m.clone())
        self.register_buffer('out_borda_m', self.out_proj.dom_borda_m.clone())

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, _E)
        x_flat = x.reshape(B * T, _E)
        q_dom = self.q_perm(xp)
        k_dom = self.k_perm(xp)
        q = torch.einsum('bhp,kp->bhk', q_dom, self.q_borda_m).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = torch.einsum('bhp,kp->bhk', k_dom, self.q_borda_m).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q = self.q_norm(q); k = self.k_norm(k)
        v_dom = self.v_perm(x_flat).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)
        attn_dom = self.rank_attn(q, k, v_dom, is_causal=True)
        attn = torch.einsum('bhtp,kp->bhtk', attn_dom, self.v_borda_m)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_dom = self.out_proj(out_in)
        out = torch.einsum('bhp,kp->bhk', out_dom, self.out_borda_m).squeeze(1).reshape(B, T, _E)
        return self.out_norm(out)


class Model(nn.Module):
    def __init__(self, init_std, soft):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.pos_embs = nn.ParameterList(
            [nn.Parameter(torch.randn(CONTEXT_SIZE, E) * 0.1) for _ in range(N_LAYERS)]
        )
        self.layers = nn.ModuleList([LUTBlock(i, init_std, soft) for i in range(N_LAYERS)])
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


def evaluate(model, sampler, batch_size):
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
            loss = F.cross_entropy(logits.reshape(B * T, V), batch.long().reshape(B * T))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def run_one(cfg):
    torch.manual_seed(RANDOM_SEED)
    model = Model(init_std=cfg['init_std'], soft=cfg['soft']).to(DEVICE)
    bit_luts = []
    for layer in model.layers:
        bit_luts += [layer.q_perm, layer.k_perm, layer.v_perm, layer.out_proj]
    adam_opt = torch.optim.Adam(list(model.parameters()), lr=cfg['adam_lr'], weight_decay=cfg.get('adam_wd', 0.0))
    bit_opt = DraftBPLUTO(
        bit_luts,
        lr=cfg['bit_lr'],
        beta1=cfg.get('beta1', 0.9),
        beta2=cfg.get('beta2', 0.999),
        weight_decay=cfg.get('bit_wd', 0.0),
        lr_schedule_fn=None,       # constant lr
    )
    sampler = TextSnippetSampler(DATA_PATH, CONTEXT_SIZE, TESTING_LENGTH, DEVICE, random_seed=1)
    ema = None
    t0 = time.time()
    model.train()
    BS = 8
    for step in range(cfg['n_steps']):
        x = sampler.sample_training_batch(BS).long()
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        logits = model(inp)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), x.reshape(B * T))
        adam_opt.zero_grad(); bit_opt.zero_grad()
        loss.backward()
        adam_opt.step(); bit_opt.step()
        lv = loss.item()
        ema = lv if ema is None else 0.99 * ema + 0.01 * lv
    val = evaluate(model, sampler, 256)
    bit_opt.close()
    return {'ema_train_loss': ema, 'val_loss': val, 'wall_s': time.time() - t0}


# --- sweep definitions ---
BASE = dict(
    n_steps=5000,
    adam_lr=1e-3,
    bit_lr=1e-3,
    init_std=0.001,
    bit_wd=0.0,
    adam_wd=0.0,
    soft=False,
    beta1=0.9, beta2=0.999,
)

def override(**kw):
    c = dict(BASE); c.update(kw); return c

SWEEP = [
    ('baseline (exp307 @5K, no sched)',      override()),
    ('bit_lr=3e-3',                          override(bit_lr=3e-3)),
    ('bit_lr=3e-4',                          override(bit_lr=3e-4)),
    ('init_std=0.01',                        override(init_std=0.01)),
    ('init_std=0.01 + bit_lr=3e-3',          override(init_std=0.01, bit_lr=3e-3)),
    ('bit_wd=0.01',                          override(bit_wd=0.01)),
    ('beta2=0.99',                           override(beta2=0.99)),
    ('soft_backward=True',                   override(soft=True)),
]

out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_results.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['name', 'bit_lr', 'init_std', 'bit_wd', 'beta2', 'soft', 'val_loss', 'ema_train_loss', 'wall_s'])

for name, cfg in SWEEP:
    print(f'\n=== {name} ===', flush=True)
    print(f'cfg: {cfg}', flush=True)
    r = run_one(cfg)
    print(f'  val_loss={r["val_loss"]:.4f}, ema_train={r["ema_train_loss"]:.4f}, wall={r["wall_s"]:.1f}s', flush=True)
    with open(out_csv, 'a', newline='') as f:
        csv.writer(f).writerow([
            name, cfg['bit_lr'], cfg['init_std'], cfg['bit_wd'], cfg['beta2'],
            cfg['soft'], f'{r["val_loss"]:.4f}', f'{r["ema_train_loss"]:.4f}', f'{r["wall_s"]:.1f}',
        ])

print(f'\nResults CSV: {out_csv}')
