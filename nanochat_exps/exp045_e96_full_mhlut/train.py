"""exp045_e96_full_mhlut — fp32 MultiHeadLut transformer (no BitPermutationLUT).

E=96, H=6, d_qk=32, d_v=16. Joint Q/K MultiHeadLut split into per-head q/k via
VectorToDominance(32). V and out_proj also MultiHeadLut, output treated as
dominance directly. Dominance residual stream from exp044 (raw sum, learnable
alpha gate per layer). Final D2V + MLP unembedder.
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.multi_bit_permutation_lut import MultiBitPermutationLUT
from spiky.lutorch.multi_bit_permutation_lut_optimizer import MultiBitPermutationLUTOptimizer
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import DominanceToVector, VectorToDominance

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E           = cfg['embedding_dim']
H           = cfg['n_heads']
d_qk        = cfg['d_qk']
d_v         = cfg['d_v']
N_LAYERS    = cfg['num_layers']
D_QK_P      = d_qk * (d_qk - 1) // 2
D_V_P       = d_v * (d_v - 1) // 2
P_STREAM    = E * (E - 1) // 2
DEVICE_BS   = cfg['device_batch_size']
TOTAL_BS    = cfg['total_batch_size']
N_STEPS     = cfg['n_steps']
EVAL_EVERY  = cfg['eval_every']
EVAL_STEPS  = cfg['eval_steps']
WARMUP_FRAC = cfg['lr_warmup_fraction']

_POS_EMB_CFG = cfg.get('pos_emb_dim', 0)
_POS_EMB_ACTIVE = isinstance(_POS_EMB_CFG, int) and _POS_EMB_CFG > 0
def _pos_emb_dim(layer_idx):
    return _POS_EMB_CFG if _POS_EMB_ACTIVE else E


# --- Tokenizer + dataloader ---------------------------------------------------
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
BOS_ID = tokenizer.get_bos_token_id()
print(f'Vocab size: {VOCAB_SIZE}, BOS id: {BOS_ID}')

train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='train', device=DEVICE
)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE
)
token_bytes = get_token_bytes(device=DEVICE)


# --- LUT factories ------------------------------------------------------------
_MHLUT_KWARGS = dict(
    smooth_mode=True,
    n_alternatives=1,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    recompute_in_backward=True,
)

def _make_qk_joint(layer_idx, seed_offset):
    """Joint Q/K MultiHeadLut: input=E (sum-mode), n_heads=H, n_outputs=2*d_qk."""
    n_inputs = E + (_pos_emb_dim(layer_idx) if _POS_EMB_ACTIVE else 0)
    return MultiHeadLut(
        input_dim=n_inputs,
        n_heads=H,
        n_outputs=2 * d_qk,
        n_anchor_pairs=cfg['qk_input_nap'],
        tables_per_head=cfg['qk_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **_MHLUT_KWARGS,
    )

def _make_v(layer_idx, seed_offset):
    """V MultiBitPermutationLUT K=4: in=E, n_heads=H, d_head=d_v, P_v=120 (CLT-scaled dominance)."""
    return MultiBitPermutationLUT(
        n_inputs=E,
        n_outputs=d_v,
        n_heads=H,
        input_nap=cfg['v_input_nap'],
        output_nap=cfg['v_output_nap'],
        tph=cfg['v_tph'],
        bit_width=cfg['mb_bit_width'],
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['mb_lut_init_std'],
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        device=DEVICE,
    )

_OUT_TPH_PER_LAYER = cfg['out_tph_per_layer']
def _make_out(layer_idx, seed_offset):
    """out_proj MultiBitPermutationLUT K=4: in=H*d_v, n_heads=1, d_head=E, P_E=4560 (CLT-scaled dominance)."""
    tph = _OUT_TPH_PER_LAYER[layer_idx]
    return MultiBitPermutationLUT(
        n_inputs=H * d_v,
        n_outputs=E,
        n_heads=1,
        input_nap=cfg['out_input_nap'],
        output_nap=cfg['out_output_nap'],
        tph=tph,
        bit_width=cfg['mb_bit_width'],
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['mb_lut_init_std'],
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        device=DEVICE,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qk_joint = _make_qk_joint(layer_idx, layer_idx)
        self.v_lut    = _make_v(layer_idx, 200 + layer_idx)
        self.out_proj = _make_out(layer_idx, 400 + layer_idx)

        canon_t = cfg.get('canon_temperature', 0.1)
        self.qk_v2d = VectorToDominance(d_qk, smooth_mode=False, temperature=canon_t)

        self.stream_to_x = DominanceToVector(E, normalise=True)
        self.attn_to_vec = DominanceToVector(d_v, normalise=False)

        self.block_out_alpha = nn.Parameter(torch.tensor(1.0))

        self.pos_dim = _pos_emb_dim(layer_idx)
        if _POS_EMB_ACTIVE:
            self.qk_input_ln = nn.LayerNorm(E + self.pos_dim)

        self.attn_scale = nn.Parameter(torch.tensor(
            float(cfg.get('learnable_attn_scale_init', 0.25))
        ))

    def forward(self, stream_raw, pos_emb):
        B, T, _ = stream_raw.shape
        x = self.stream_to_x(stream_raw)                                # [B, T, E]

        if _POS_EMB_ACTIVE:
            pos = pos_emb.unsqueeze(0).expand(B, -1, -1)
            xp = torch.cat([x, pos], dim=-1)
            xp = self.qk_input_ln(xp).reshape(B * T, E + self.pos_dim)
        else:
            xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, E)
        x_flat = x.reshape(B * T, E)

        qk_out = self.qk_joint(xp)                                      # [B*T, H, 2*d_qk]
        q_vec = qk_out[..., :d_qk]
        k_vec = qk_out[..., d_qk:]
        q_dom = self.qk_v2d(q_vec)                                      # [B*T, H, P_qk]
        k_dom = self.qk_v2d(k_vec)
        q = q_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        k = k_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)

        v_dom = self.v_lut(x_flat).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)

        attn_dom = F.scaled_dot_product_attention(
            q * self.attn_scale, k, v_dom,
            is_causal=True,
        )
        attn = self.attn_to_vec(attn_dom)

        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        block_out_dom = self.out_proj(out_in).squeeze(1).reshape(B, T, P_STREAM)

        return stream_raw + self.block_out_alpha * block_out_dom


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        _pos_dim_fn = _pos_emb_dim if _POS_EMB_ACTIVE else (lambda i: E)
        _pos_init_scale = cfg.get('pos_emb_init_scale', 0.1)
        self.pos_embs = nn.ParameterList([
            nn.Parameter(torch.randn(CONTEXT_SIZE, _pos_dim_fn(i)) * _pos_init_scale)
            for i in range(N_LAYERS)
        ])
        self.token_to_dom = VectorToDominance(E, smooth_mode=False, temperature=cfg.get('canon_temperature', 0.1))
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.final_d2v = DominanceToVector(E, normalise=True)
        self.unembedder = nn.Sequential(
            nn.Linear(E, 4 * E),
            nn.ReLU(),
            nn.Linear(4 * E, VOCAB_SIZE),
        )

    def get_device(self):
        return self.token_embedder.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        x = self.token_embedder(tokens)                       # [B, T, E]
        stream_raw = self.token_to_dom(x)                     # [B, T, P_STREAM]
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            stream_raw = layer(stream_raw, pos_emb)
        h = self.final_d2v(stream_raw)                        # [B, T, E]
        logits = self.unembedder(h)                           # [B, T, V]
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1,
            )
        return logits


# --- Build + optimiser --------------------------------------------------------
model = Model().to(DEVICE)

# Collect MultiBit LUTs (V + out_proj per layer) — managed by MultiBitPermutationLUTOptimizer.
mb_luts = []
for layer in model.layers:
    mb_luts += [layer.v_lut, layer.out_proj]

adam_params = list(model.parameters())
adam_param_count = sum(p.numel() for p in adam_params)
print(f'P_STREAM = {P_STREAM} (E={E})')
print(f'MultiBit LUTs: {len(mb_luts)} (V+out_proj × {N_LAYERS} layers)')
print(f'Adam-managed parameters: {adam_param_count:,}')

def get_lr_scale(step):
    n = N_STEPS
    w = int(WARMUP_FRAC * n)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

decay_params   = [p for p in adam_params if p.ndim >= 2]
nodecay_params = [p for p in adam_params if p.ndim < 2]
adam_groups = [
    dict(params=decay_params,   lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=cfg.get('weight_decay', 0.0)),
    dict(params=nodecay_params, lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=0.0),
]
adam_opt = torch.optim.AdamW(adam_groups)
for g in adam_opt.param_groups:
    g['initial_lr'] = g['lr']

mb_opt = MultiBitPermutationLUTOptimizer(
    mb_luts,
    lr=cfg['mb_lut_lr'],
    beta1=cfg.get('mb_lut_beta1', 0.9),
    beta2=cfg.get('mb_lut_beta2', 0.95),
    lr_schedule_fn=get_lr_scale,
)

print(f'Q/K joint MHLut:   in_nap={cfg["qk_input_nap"]} tph={cfg["qk_tph"]} d_qk={d_qk} P_qk={D_QK_P}')
print(f'V MultiBitPermLUT (K={cfg["mb_bit_width"]}): in_nap={cfg["v_input_nap"]} out_nap={cfg["v_output_nap"]} tph={cfg["v_tph"]} P_v={D_V_P}')
print(f'out_proj MultiBitPermLUT (K={cfg["mb_bit_width"]}): in_nap={cfg["out_input_nap"]} out_nap={cfg["out_output_nap"]} tph={_OUT_TPH_PER_LAYER} P_E={P_STREAM}')

# --- Training loop ------------------------------------------------------------
tokens_per_step = DEVICE_BS * CONTEXT_SIZE
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'Tokens/micro-batch: {tokens_per_step:,} | grad_accum: {grad_accum} | effective batch: {grad_accum * tokens_per_step:,} tokens')

csv_path = os.path.join(EXP_DIR, 'metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'val_bpb'])

train_losses_logged, val_bpbs, val_steps = [], [], []
ema = None
best_bpb = float('inf')
t0 = time.time()

model.train()
for step in range(1, N_STEPS + 1):
    lr_scale = get_lr_scale(step)
    for g in adam_opt.param_groups:
        g['lr'] = g['initial_lr'] * lr_scale

    adam_opt.zero_grad()
    mb_opt.zero_grad()
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, targets=y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum

    adam_opt.step()
    mb_opt.step()

    ema = accum_loss if ema is None else 0.99 * ema + 0.01 * accum_loss

    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | loss={ema:.4f} | lr={lr_scale * cfg["adam_lr"]:.2e}')

    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval()
        val_loader = val_loader_factory()
        bpb = evaluate_bpb(model, val_loader, EVAL_STEPS, token_bytes)
        if bpb < best_bpb:
            best_bpb = bpb
        print(f'[VAL] step {step}: bpb={bpb:.4f}')
        train_losses_logged.append(ema)
        val_bpbs.append(bpb)
        val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}'])
        csv_f.flush()
        model.train()

csv_f.close()
elapsed = time.time() - t0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, train_losses_logged, label='train (ema)')
ax1.set(xlabel='step', ylabel='cross-entropy loss', title='Training Loss')
ax1.legend(); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, 'o-', color='tab:orange', label='val bpb')
ax2.set(xlabel='step', ylabel='bits per byte', title='Validation BPB')
ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120)
plt.close()

summary = {
    'exp_name': cfg['exp_name'],
    'best_val_bpb': best_bpb,
    'final_val_bpb': val_bpbs[-1] if val_bpbs else None,
    'adam_params': adam_param_count,
    'training_time_hours': round(elapsed / 3600, 3),
}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

mb_opt.close()
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
