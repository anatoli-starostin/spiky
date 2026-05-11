    """nanochat_exps/exp252_tinymhlut_soft — fork of exp251 with TinyMHLut(soft).

Single change vs exp251: replace SoftMultiHeadLUT(hard=True, learnable_temps=True)
with TinyMultiHeadLut(backward_mode='soft', learnable_temps=True). Gradients are
identical (fp32 exact for dL/dweights, fp32 noise for dL/dx and dL/dlog_T_*; bf16
noise under autocast — same as autograd of reference). Path is pure-PyTorch
compile-fused: forward = sign-pack + embedding_bag (TinyMHLut-fast), backward =
recomputed soft pipeline in @torch.compile body (no [B, T, K] activations saved
across the fwd/bwd boundary except `index`).

Goal: reproduce exp251's 1.6026 bpb result while validating the new soft backward
path. All other knobs identical: V2D learnable T init=0.1, learnable T_soft/T_sel
init=0.5/0.5, NAPs unchanged, 8K steps.

Launch:
    PYTHONPATH=/home/starost/nanochat \\
        /home/starost/spiky/.venv/bin/python -u \\
        nanochat_exps/exp252_tinymhlut_soft/train.py \\
        > nanochat_exps/exp252_tinymhlut_soft/stdout.log 2>&1 &
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

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut  # noqa: F401  (kept for compat)
from spiky.lutorch.soft_multi_head_lut import SoftMultiHeadLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import VectorToDominance, DominanceToVector

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


# --- LUT block helpers --------------------------------------------------------
_TINY_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
)

# exp252: swap SoftMultiHeadLUT for TinyMultiHeadLut(backward_mode='soft').
# Gradients are identical to SoftMHLut(hard=True) but path is pure-PyTorch
# (compile-fused) with TinyMHLut-fast forward.
_TINY_SOFT_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    backward_mode='soft',
    soft_score_temp=cfg.get('soft_score_temp', 0.5),
    select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', False),
    use_bf16=cfg.get('soft_use_bf16', True),
    argmax_noise_eps=cfg.get('argmax_noise_eps', 0.0),
    bf16_argmax=cfg.get('bf16_argmax', False),
)

def _make_qk_joint(layer_idx, seed_offset):
    """Joint Q/K TinyMHLut(soft): input=E, n_heads=H, n_outputs=2*d_qk."""
    n_inputs = E + (_pos_emb_dim(layer_idx) if _POS_EMB_ACTIVE else 0)
    qk_kwargs = dict(_TINY_SOFT_KWARGS)
    qk_kwargs['initial_weights_noise'] = cfg.get('qk_lut_init_std',
                                                 cfg.get('mhlut_init_std', 0.001))
    return TinyMultiHeadLut(
        input_dim=n_inputs,
        n_heads=H,
        n_outputs=2 * d_qk,
        n_anchor_pairs=cfg['qk_input_nap'],
        tables_per_head=cfg['qk_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **qk_kwargs,
    )

def _make_v(layer_idx, seed_offset):
    return TinyMultiHeadLut(
        input_dim=E,
        n_heads=H,
        n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'],
        tables_per_head=cfg['v_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **_TINY_SOFT_KWARGS,
    )

_OUT_TPH_PER_LAYER = cfg.get('out_tph_per_layer')
def _make_out(layer_idx, seed_offset):
    tph = _OUT_TPH_PER_LAYER[layer_idx] if _OUT_TPH_PER_LAYER is not None else cfg['out_tph']
    return TinyMultiHeadLut(
        input_dim=H * d_v,
        n_heads=1,
        n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'],
        tables_per_head=tph,
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **_TINY_SOFT_KWARGS,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qk_joint = _make_qk_joint(layer_idx, layer_idx)
        self.v_lut    = _make_v(layer_idx, 200 + layer_idx)
        self.out_proj = _make_out(layer_idx, 400 + layer_idx)

        canon_t = cfg.get('canon_temperature', 0.1)
        v2d_init_t = cfg.get('out_v2d_temperature_init', canon_t)
        v2d_learnable = cfg.get('out_v2d_learnable_temperature', False)
        # NOTE: qk_v2d removed — Q and K are passed directly to SDPA after q_norm/k_norm.
        # out_v2d / out_d2v preserved (rank-canonicalize out_proj output).
        self.out_v2d = VectorToDominance(
            E,
            smooth_mode=False,
            temperature=v2d_init_t,
            learnable_temperature=v2d_learnable,
        )
        self.out_d2v = DominanceToVector(E, normalise=True)

        self.pos_dim = _pos_emb_dim(layer_idx)
        if _POS_EMB_ACTIVE:
            self.qk_input_ln = nn.LayerNorm(E + self.pos_dim)

        # QK-norm: LayerNorm on d_qk dim before SDPA (Qwen/Gemma style).
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)

    def forward(self, x, pos_emb):
        B, T, _ = x.shape
        if _POS_EMB_ACTIVE:
            pos = pos_emb.unsqueeze(0).expand(B, -1, -1)
            xp = torch.cat([x, pos], dim=-1)
            xp = self.qk_input_ln(xp).reshape(B * T, E + self.pos_dim)
        else:
            xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, E)
        x_flat = x.reshape(B * T, E)

        qk_out = self.qk_joint(xp)                                     # [B*T, H, 2*d_qk]
        q_vec = qk_out[..., :d_qk]                                     # [B*T, H, d_qk]
        k_vec = qk_out[..., d_qk:]                                     # [B*T, H, d_qk]
        q_vec = self.q_norm(q_vec)
        k_vec = self.k_norm(k_vec)
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)

        v_vec = self.v_lut(x_flat)                                     # [B*T, H, d_v]
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_real = self.out_proj(out_in).squeeze(1)                    # [B*T, E]
        out_dom  = self.out_v2d(out_real)                              # [B*T, E*(E-1)/2] in ±1
        out_rank = self.out_d2v(out_dom).reshape(B, T, E)
        return out_rank


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
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        concat_dim = N_LAYERS * E
        unembed_hidden = cfg.get('unembed_hidden', concat_dim * 8)
        self.unembedder = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, unembed_hidden, bias=False),
            nn.GELU(),
            nn.Linear(unembed_hidden, VOCAB_SIZE, bias=False),
        )

    def get_device(self):
        return self.token_embedder.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        x = self.token_embedder(tokens)
        outs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            outs.append(x)
        concat = torch.cat(outs, dim=-1)
        logits = self.unembedder(concat)
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

print(f'Q/K Joint TinyMHLut(soft): in_nap={cfg["qk_input_nap"]} tph={cfg["qk_tph"]} d_qk={d_qk} (n_outputs=2*d_qk={2*d_qk}) init_std={cfg.get("qk_lut_init_std", cfg.get("mhlut_init_std", 0.001))} T_soft={cfg.get("soft_score_temp",0.5)} T_sel={cfg.get("select_temp",0.5)} [no V2D, q_norm+k_norm -> SDPA]')
print(f'V TinyMHLut(soft):       in_nap={cfg["v_input_nap"]} tph={cfg["v_tph"]} d_v={d_v} (n_outputs=d_v)')
_tph_str = str(_OUT_TPH_PER_LAYER) if _OUT_TPH_PER_LAYER is not None else str(cfg['out_tph'])
print(f'out_proj TinyMHLut(soft): in_nap={cfg["out_input_nap"]} tph={_tph_str} (n_outputs=E={E}) + out_v2d/out_d2v')
print(f'V2D out:         init_T={cfg.get("out_v2d_temperature_init", cfg.get("canon_temperature",0.1))} learnable={cfg.get("out_v2d_learnable_temperature", False)}')
print(f'Attention:       SDPA (causal); plain default 1/sqrt(d_qk={d_qk})={1.0/math.sqrt(d_qk):.4f}, no learnable attn_scale')


# --- Temperature tracking -----------------------------------------------------
def collect_temperature_specs(model):
    """Return ordered list of (column_name, getter_fn) for all learnable temps.

    Iterates layers in order and produces stable, descriptive column names so
    rows can be appended over the run without re-deriving the schema.
    """
    specs = []
    for li, blk in enumerate(model.layers):
        for slut_name in ('qk_joint', 'v_lut', 'out_proj'):
            mod = getattr(blk, slut_name, None)
            if mod is None:
                continue
            if getattr(mod, 'learnable_temps', False):
                specs.append((f'L{li}.{slut_name}.T_soft',
                              (lambda m=mod: float(m.log_soft_score_temp.detach().exp()))))
                specs.append((f'L{li}.{slut_name}.T_sel',
                              (lambda m=mod: float(m.log_select_temp.detach().exp()))))
        v2d = getattr(blk, 'out_v2d', None)
        if v2d is not None and getattr(v2d, 'learnable_temperature', False):
            specs.append((f'L{li}.out_v2d.T',
                          (lambda m=v2d: float(m.log_temperature.detach().exp()))))
    return specs

temp_specs = collect_temperature_specs(model)
temp_path = os.path.join(EXP_DIR, 'temperatures.csv')
temp_f = open(temp_path, 'w', newline='')
temp_w = csv.writer(temp_f)
temp_w.writerow(['step'] + [name for name, _ in temp_specs])
print(f'Tracking {len(temp_specs)} learnable temperatures in temperatures.csv')


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

# Log step-0 temperatures (before any optimizer steps) for reference.
temp_w.writerow([0] + [f'{getter():.6f}' for _, getter in temp_specs])
temp_f.flush()

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
        temp_w.writerow([step] + [f'{getter():.6f}' for _, getter in temp_specs])
        temp_f.flush()
        model.train()

csv_f.close()
temp_f.close()
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
}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
