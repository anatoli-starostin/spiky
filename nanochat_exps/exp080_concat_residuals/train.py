"""nanochat_exps/exp080_concat_residuals — concat-residual LUT chain with bipartitioned pair sampling.

Architecture:
  - E=16 token embedding, O=16 out_proj output dim. Each LUTBlock appends its
    out_proj output to its input (DenseNet-style growth). The residual stream
    grows: L0 input=16, L1 input=32, ..., L5 input=96, final concat=112.
  - QK and V LUT inputs at L>=1 are bipartitioned via partition_sets:
        zoneA = [0..15]  (original token-embedding slot)
        zoneB = [16..end] (concatenated prior block outputs)
    Pairs are sampled WITHIN each zone (no cross-zone pairs).
  - out_proj LUT input is bipartitioned by head: H zones of d_v elements each
    (no cross-head pairs).
  - Inter-block residuals are real-valued (LayerNorm only on out_proj output).
  - Before the unembedder, the final 112-dim concat is canonicalized via
    V2D -> D2V -> Linear(112, vocab).

Per-layer dims:
  L0: in=16  (no partition)            out_proj=16 -> next_in=32
  L1: in=32  (16 | 16)                 out_proj=16 -> next_in=48
  L2: in=48  (16 | 32)                 out_proj=16 -> next_in=64
  L3: in=64  (16 | 48)                 out_proj=16 -> next_in=80
  L4: in=80  (16 | 64)                 out_proj=16 -> next_in=96
  L5: in=96  (16 | 80)                 out_proj=16 -> unembedder_in=112

How to launch:

    PYTHONPATH=/home/starost/nanochat \\
        /home/starost/spiky/.venv/bin/python \\
        -u nanochat_exps/exp080_concat_residuals/train.py \\
        > nanochat_exps/exp080_concat_residuals/stdout.log 2>&1 &
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

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import VectorToDominance, DominanceToVector

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E           = cfg['embedding_dim']
O           = cfg['out_proj_output_dim']
H           = cfg['n_heads']
d_qk        = cfg['d_qk']
d_v         = cfg['d_v']
N_LAYERS    = cfg['num_layers']
D_QK_P      = d_qk * (d_qk - 1) // 2
DEVICE_BS   = cfg['device_batch_size']
TOTAL_BS    = cfg['total_batch_size']
N_STEPS     = cfg['n_steps']
EVAL_EVERY  = cfg['eval_every']
EVAL_STEPS  = cfg['eval_steps']
WARMUP_FRAC = cfg['lr_warmup_fraction']

# Per-layer growing input dims: L0=E, L1=E+O, L2=E+2*O, ...
LAYER_INPUT_DIMS = [E + i * O for i in range(N_LAYERS)]
UNEMBEDDER_DIM = E + N_LAYERS * O  # token_emb + N_LAYERS * out_proj


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


# --- Partition helpers --------------------------------------------------------
def _qkv_partitions(input_dim):
    """Bipartition over the residual stream: [0..O-1] | [O..input_dim-1].
    L0 has input_dim==E==O, so a single zone — return None (no partition)."""
    if input_dim <= O:
        return None
    return [list(range(O)), list(range(O, input_dim))]

def _out_proj_partitions():
    """H zones over the SDPA-output (H*d_v) for out_proj: one zone per head."""
    return [list(range(h * d_v, (h + 1) * d_v)) for h in range(H)]


# --- LUT block helpers --------------------------------------------------------
_TINY_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
)


class LUTBlock(nn.Module):
    def __init__(self, layer_idx, input_dim):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_dim = input_dim

        qkv_part = _qkv_partitions(input_dim)
        out_part = _out_proj_partitions()

        self.qk_joint = TinyMultiHeadLut(
            input_dim=input_dim,
            n_heads=H,
            n_outputs=2 * d_qk,
            n_anchor_pairs=cfg['qk_input_nap'],
            tables_per_head=cfg['qk_tph'],
            partition_sets=qkv_part,
            random_seed=cfg['random_seed'] + layer_idx,
            device=DEVICE,
            **_TINY_KWARGS,
        )
        self.v_lut = TinyMultiHeadLut(
            input_dim=input_dim,
            n_heads=H,
            n_outputs=d_v,
            n_anchor_pairs=cfg['v_input_nap'],
            tables_per_head=cfg['v_tph'],
            partition_sets=qkv_part,
            random_seed=cfg['random_seed'] + 200 + layer_idx,
            device=DEVICE,
            **_TINY_KWARGS,
        )
        self.out_proj = TinyMultiHeadLut(
            input_dim=H * d_v,
            n_heads=1,
            n_outputs=O,
            n_anchor_pairs=cfg['out_input_nap'],
            tables_per_head=cfg['out_tph'],
            partition_sets=out_part,
            random_seed=cfg['random_seed'] + 400 + layer_idx,
            device=DEVICE,
            **_TINY_KWARGS,
        )

        canon_t = cfg.get('canon_temperature', 0.1)
        self.qk_v2d = VectorToDominance(d_qk, smooth_mode=False, temperature=canon_t)
        self.out_ln = nn.LayerNorm(O)

        self.attn_scale = nn.Parameter(torch.tensor(
            float(cfg.get('learnable_attn_scale_init', 0.25))
        ))

    def forward(self, x_resid, pos_emb):
        # x_resid: [B, T, input_dim] (real-valued residual stream)
        B, T, D = x_resid.shape
        x_in = x_resid + pos_emb.unsqueeze(0)               # [B, T, D]
        x_flat = x_in.reshape(B * T, D)

        qk_out = self.qk_joint(x_flat)                      # [B*T, H, 2*d_qk]
        q_vec = qk_out[..., :d_qk]
        k_vec = qk_out[..., d_qk:]
        q_dom = self.qk_v2d(q_vec)
        k_dom = self.qk_v2d(k_vec)
        q = q_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        k = k_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)

        v_vec = self.v_lut(x_flat)                          # [B*T, H, d_v]
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3) # [B, H, T, d_v]

        attn = F.scaled_dot_product_attention(
            q * self.attn_scale, k, v, is_causal=True,
        )                                                   # [B, H, T, d_v]

        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_real = self.out_proj(out_in).squeeze(1)         # [B*T, O]
        out_real = self.out_ln(out_real).reshape(B, T, O)   # [B, T, O]

        # Concat residual: append this block's output to its input.
        return torch.cat([x_resid, out_real], dim=-1)       # [B, T, D + O]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)

        pos_init = cfg.get('pos_emb_init_scale', 0.1)
        self.pos_embs = nn.ParameterList([
            nn.Parameter(torch.randn(CONTEXT_SIZE, dim) * pos_init)
            for dim in LAYER_INPUT_DIMS
        ])

        self.layers = nn.ModuleList([
            LUTBlock(i, LAYER_INPUT_DIMS[i]) for i in range(N_LAYERS)
        ])

        canon_t = cfg.get('canon_temperature', 0.1)
        self.final_v2d = VectorToDominance(UNEMBEDDER_DIM, smooth_mode=False, temperature=canon_t)
        self.final_d2v = DominanceToVector(UNEMBEDDER_DIM, normalise=True)
        self.unembedder = nn.Linear(UNEMBEDDER_DIM, VOCAB_SIZE)

    def get_device(self):
        return self.token_embedder.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        x = self.token_embedder(tokens)                     # [B, T, E]
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)                           # appends out_proj output
        # x is now [B, T, UNEMBEDDER_DIM]
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        dom = self.final_v2d(flat)                          # [B*T, P=D*(D-1)/2] in ±1
        rank = self.final_d2v(dom)                          # [B*T, D] Borda + LN
        logits = self.unembedder(rank).reshape(B, T, VOCAB_SIZE)
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1,
            )
        return logits


# --- Build + optimiser --------------------------------------------------------
model = Model().to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f'Total params (all fp32): {n_params:,}')
print(f'Per-layer input dims: {LAYER_INPUT_DIMS}')
print(f'Unembedder input dim:  {UNEMBEDDER_DIM}')

def get_lr_scale(step):
    n = N_STEPS
    w = int(WARMUP_FRAC * n)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

decay_params   = [p for p in model.parameters() if p.ndim >= 2]
nodecay_params = [p for p in model.parameters() if p.ndim < 2]
adam_groups = [
    dict(params=decay_params,   lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=cfg.get('weight_decay', 0.0)),
    dict(params=nodecay_params, lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=0.0),
]
optimizer = torch.optim.AdamW(adam_groups)
for g in optimizer.param_groups:
    g['initial_lr'] = g['lr']

print(f'Q/K Joint MHLut: in_nap={cfg["qk_input_nap"]} tph={cfg["qk_tph"]} d_qk={d_qk} (n_outputs=2*d_qk={2*d_qk})')
print(f'V MHLut:         in_nap={cfg["v_input_nap"]} tph={cfg["v_tph"]} d_v={d_v} (n_outputs=d_v)')
print(f'out_proj MHLut:  in_nap={cfg["out_input_nap"]} tph={cfg["out_tph"]} (n_outputs=O={O}) + LayerNorm | H={H} partition zones')

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
    for g in optimizer.param_groups:
        g['lr'] = g['initial_lr'] * lr_scale

    optimizer.zero_grad()
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, targets=y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum

    optimizer.step()

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
    'n_params': n_params,
    'training_time_hours': round(elapsed / 3600, 3),
    'layer_input_dims': LAYER_INPUT_DIMS,
    'unembedder_dim': UNEMBEDDER_DIM,
}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
