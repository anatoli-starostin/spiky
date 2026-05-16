"""nanochat_exps/exp272_dual_stream_residual_lut — fork of exp265.

Dual-stream architecture (sketch v4):

  - Two streams:
      x_lut   (B, T, E=96)   — LUT-compute carry between blocks
      x_resid (B, T, D=384)  — vanilla-baseline-dim residual stream, accumulated
                              by adding residual_lut output each layer.

  - Per-layer block (same exp265 internals minus V2D/D2V):
      qk_joint (TinyMHL soft, NAP=6) + v_lut (TinyMHL multi-alt, NAP=8)
      + q_norm + k_norm + SDPA + out_proj (TinyMHL soft, NAP=6 -> E).
      pos_emb is *added* to x for the QK input only (V uses raw x). No FFN.

  - Block end (post-norm): ln_e on out_e; the LN'd value is BOTH the next-block
    x_lut input AND the input to residual_lut (TinyMHL, out_proj settings,
    E -> D). residual_lut output is the layer's D-dim contribution.

  - Model loop: x_resid += r_out per layer (no per-layer LN on the stream).
    Final: ln_final(D) before tied unembed.

  - Tied embedding/unembedder at D=384: W_D shared between tok_emb_D and the
    final linear (single matmul, matches vanilla complexity).

  - Separate small E-dim token embedding (tok_emb_E) feeds the LUT carry; same
    per-layer pos_embs as exp265.

Removed from exp265: out_v2d, out_d2v, the MLP unembedder
(LayerNorm + Linear + GELU + Linear over concat'd layer outputs).
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

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut  # noqa: F401
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E           = cfg['embedding_dim']
D           = cfg['residual_dim']
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


# --- LUT block helpers (verbatim from exp265, plus residual_lut) --------------
_TINY_SOFT_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    backward_mode='soft',
    soft_score_temp=cfg.get('soft_score_temp', 0.5),
    select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True),
    use_bf16=cfg.get('soft_use_bf16', True),
    argmax_noise_eps=cfg.get('argmax_noise_eps', 0.0),
)

_TINY_MULTIALT_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    backward_mode='ste',
    n_alternatives=3,
    argmax_noise_eps=cfg.get('argmax_noise_eps', 0.0),
    learnable_temps=cfg.get('multialt_learnable_temps', False),
    uncertainty_T_init=cfg.get('uncertainty_T_init', 1.0),
)

def _make_qkv_joint(layer_idx, seed_offset):
    """Joint qkv LUT — n_outputs = 2*d_qk + d_v. Q/K come from this LUT exclusively
    (replaces qk_joint). The last d_v outputs are ADDED to v_lut's output, giving
    v a parallel shallow contribution that shares anchor decisions with q/k.
    """
    mode = cfg.get('qkv_backward_mode', 'soft')
    kwargs = _TINY_SOFT_KWARGS if mode == 'soft' else _TINY_MULTIALT_KWARGS
    kwargs = dict(kwargs)
    kwargs['initial_weights_noise'] = cfg.get('qkv_lut_init_std',
                                              cfg.get('mhlut_init_std', 0.001))
    return TinyMultiHeadLut(
        input_dim=E,
        n_heads=H,
        n_outputs=2 * d_qk + d_v,
        n_anchor_pairs=cfg.get('qkv_input_nap', cfg['qk_input_nap']),
        tables_per_head=cfg.get('qkv_tph', cfg['qk_tph']),
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **kwargs,
    )

def _make_v(layer_idx, seed_offset):
    v_mode = cfg.get('v_backward_mode', 'ste')
    v_kwargs = _TINY_SOFT_KWARGS if v_mode == 'soft' else _TINY_MULTIALT_KWARGS
    return TinyMultiHeadLut(
        input_dim=E,
        n_heads=H,
        n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'],
        tables_per_head=cfg['v_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **v_kwargs,
    )

_OUT_TPH_PER_LAYER = cfg.get('out_tph_per_layer')
def _make_out(layer_idx, seed_offset):
    """out_proj. Backward mode selectable via cfg['out_backward_mode'] (default 'soft').
    Use 'ste' (multi-alt n_alt=3) when NAP=8 to avoid soft's [B*T, K=2^NAP] memory blow-up."""
    tph = _OUT_TPH_PER_LAYER[layer_idx] if _OUT_TPH_PER_LAYER is not None else cfg['out_tph']
    mode = cfg.get('out_backward_mode', 'soft')
    kwargs = _TINY_SOFT_KWARGS if mode == 'soft' else _TINY_MULTIALT_KWARGS
    return TinyMultiHeadLut(
        input_dim=H * d_v,
        n_heads=1,
        n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'],
        tables_per_head=tph,
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **kwargs,
    )

def _make_residual_lut(layer_idx, seed_offset):
    """E -> D residual projection. TinyMHLut with out_proj settings (soft, NAP=6)."""
    return TinyMultiHeadLut(
        input_dim=E,
        n_heads=1,
        n_outputs=D,
        n_anchor_pairs=cfg.get('residual_input_nap', cfg['out_input_nap']),
        tables_per_head=cfg.get('residual_tph', cfg['out_tph']),
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **_TINY_SOFT_KWARGS,
    )


# --- RoPE (replaces additive learned pos_emb) -------------------------------
# Standard half-rotation form applied on q, k inside the LUT attention path,
# right before SDPA. d_qk=64 is the head dimension. v is NOT rotated.
_ROPE_BASE = cfg.get('rope_base', 10000.0)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
        ))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)


def _rotate_half(t):
    a, b = t.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


class LUTBlock(nn.Module):
    """exp321 block + joint qkv_lut (NAP=6, tph=256, n_out = 2*d_qk + d_v).
    Q and K come exclusively from qkv_lut (replaces qk_joint). v_lut keeps its
    own NAP=8 capacity and gets an ADDITIVE contribution from qkv_lut's last
    d_v outputs — sharing anchor decisions between qk and a parallel v branch.
    """
    def __init__(self, layer_idx):
        super().__init__()
        self.qkv_lut      = _make_qkv_joint(layer_idx, layer_idx)
        self.v_lut        = _make_v(layer_idx, 200 + layer_idx)
        self.out_proj     = _make_out(layer_idx, 400 + layer_idx)
        self.residual_lut = _make_residual_lut(layer_idx, 600 + layer_idx)

        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)

        # Post-norm at block end on E-stream output. Same LN'd value feeds
        # both the next-block x_lut and residual_lut.
        self.ln_e = nn.LayerNorm(E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)

        # Joint qkv_lut produces q, k, and a v-branch contribution.
        qkv_out = self.qkv_lut(x_flat)                                # [B*T, H, 2*d_qk + d_v]
        q_vec = self.q_norm(qkv_out[..., :d_qk])
        k_vec = self.k_norm(qkv_out[..., d_qk:2 * d_qk])
        v_branch = qkv_out[..., 2 * d_qk:]                            # [B*T, H, d_v]
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])

        v_lut_out = self.v_lut(x_flat)                                # [B*T, H, d_v]
        v_vec = v_lut_out + v_branch                                  # additive
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e  = self.out_proj(out_in).squeeze(1)                     # [B*T, E]

        # Post-norm on E-stream output; same value feeds residual_lut.
        out_e_norm = self.ln_e(out_e)                                 # [B*T, E]
        x_lut_next = out_e_norm.reshape(B, T, E)                      # -> next block
        r_out      = self.residual_lut(out_e_norm).squeeze(1).reshape(B, T, D)
        return x_lut_next, r_out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # E-dim token embedding for the LUT carry (separate, small).
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        # NO tok_emb_D: x_resid starts at zeros, only accumulates residual_lut contributions.
        # Separate UNTIED unembedder (no weight sharing).
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)

        # Shared RoPE buffers for all blocks (no learned positional params).
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])

        # Final LN on the residual stream before the unembedder.
        self.ln_final = nn.LayerNorm(D)

    def get_device(self):
        return self.tok_emb_E.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x_resid = torch.zeros(B, T, D, device=tokens.device, dtype=self.tok_emb_E.weight.dtype)
        x_lut   = self.tok_emb_E(tokens)              # (B, T, E)
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r                     # accumulate
        x_resid = self.ln_final(x_resid)              # LN before unembed
        logits = self.unembedder(x_resid)             # untied Linear(D, V)
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

lut_params     = []
tok_emb_params = []
pos_emb_params = []
decay_params   = []
nodecay_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim >= 3:
        lut_params.append(p)             # TinyMHLut.weights — no wd
    elif name.startswith('tok_emb_E.'):
        tok_emb_params.append(p)         # token embedding — no wd (standard practice)
    elif name.startswith('pos_embs.'):
        pos_emb_params.append(p)         # position embeddings — no wd (standard practice)
    elif p.ndim == 2:
        decay_params.append(p)           # only the unembedder Linear(D, V) keeps wd
    else:
        nodecay_params.append(p)         # LN affine, biases, log_temps
_LUT_LR = cfg.get('lut_lr', cfg['adam_lr'])
adam_groups = [
    dict(params=lut_params,     lr=_LUT_LR,        betas=(0.9, 0.95), eps=1e-8,
         weight_decay=0.0),
    dict(params=decay_params,   lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=cfg.get('weight_decay', 0.0)),
    dict(params=tok_emb_params + pos_emb_params + nodecay_params,
         lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
]
print(
    f'optimizer groups: '
    f'lut={sum(p.numel() for p in lut_params):,} (wd=0, lr={_LUT_LR}) | '
    f'decay(unembed)={sum(p.numel() for p in decay_params):,} (wd={cfg.get("weight_decay", 0.0)}) | '
    f'tok_emb={sum(p.numel() for p in tok_emb_params):,} (wd=0) | '
    f'pos_emb={sum(p.numel() for p in pos_emb_params):,} (wd=0) | '
    f'nodecay_other={sum(p.numel() for p in nodecay_params):,} (wd=0)'
)
optimizer = torch.optim.AdamW(adam_groups)
for g in optimizer.param_groups:
    g['initial_lr'] = g['lr']

# --- Lookahead (optional) -----------------------------------------------------
_USE_LOOKAHEAD   = bool(cfg.get('use_lookahead', False))
_LOOKAHEAD_K     = int(cfg.get('lookahead_k', 5))
_LOOKAHEAD_ALPHA = float(cfg.get('lookahead_alpha', 0.5))
_LOOKAHEAD_GROUPS = set(cfg.get('lookahead_groups', ['lut']))
_LOOKAHEAD_PARAMS = []
if _USE_LOOKAHEAD:
    _id_to_group = {}
    for p in lut_params:        _id_to_group[id(p)] = 'lut'
    for p in decay_params:      _id_to_group[id(p)] = 'decay'
    for p in tok_emb_params:    _id_to_group[id(p)] = 'tok_emb'
    for p in pos_emb_params:    _id_to_group[id(p)] = 'pos_emb'
    for p in nodecay_params:    _id_to_group[id(p)] = 'nodecay'
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if _id_to_group.get(id(p)) in _LOOKAHEAD_GROUPS:
            _LOOKAHEAD_PARAMS.append((p, p.detach().clone()))
    print(f'Lookahead enabled: {len(_LOOKAHEAD_PARAMS)} tensors in groups={sorted(_LOOKAHEAD_GROUPS)}, k={_LOOKAHEAD_K}, alpha={_LOOKAHEAD_ALPHA}')

_NOISE_EPS = cfg.get('argmax_noise_eps', 0.0)
print(f'D=residual_dim={D}, E=embedding_dim={E}, H={H}, d_qk={d_qk}, d_v={d_v}, L={N_LAYERS}')
print(f'QKV joint TinyMHLut({cfg.get("qkv_backward_mode", "soft")}, noise_eps={_NOISE_EPS}): in_nap={cfg.get("qkv_input_nap", cfg["qk_input_nap"])} tph={cfg.get("qkv_tph", cfg["qk_tph"])} n_out=2*d_qk+d_v={2*d_qk+d_v} (q,k from this LUT; last d_v added to v_lut)')
print(f'V_lut    TinyMHLut({cfg.get("v_backward_mode", "ste")}, noise_eps={_NOISE_EPS}): in_nap={cfg["v_input_nap"]} tph={cfg["v_tph"]} d_v={d_v}')
_out_tph_str = str(_OUT_TPH_PER_LAYER) if _OUT_TPH_PER_LAYER is not None else str(cfg['out_tph'])
print(f'out_proj TinyMHLut({cfg.get("out_backward_mode", "soft")}, noise_eps={_NOISE_EPS}): in_nap={cfg["out_input_nap"]} tph={_out_tph_str} n_out=E={E}')
print(f'residual_lut TinyMHLut(soft, noise_eps={_NOISE_EPS}): in_nap={cfg.get("residual_input_nap", cfg["out_input_nap"])} tph={cfg.get("residual_tph", cfg["out_tph"])} n_out=D={D}  [E -> D]')
print(f'NO tok_emb_D (x_resid starts at zeros); UNTIED unembedder Linear(D={D}, V={VOCAB_SIZE}); tok_emb_E separate at E={E}; ln_final(D) before unembed')


# --- Temperature tracking -----------------------------------------------------
def collect_temperature_specs(model):
    specs = []
    for li, blk in enumerate(model.layers):
        for slut_name in ('qkv_lut', 'v_lut', 'out_proj', 'residual_lut'):
            mod = getattr(blk, slut_name, None)
            if mod is None:
                continue
            if getattr(mod, 'learnable_temps', False):
                specs.append((f'L{li}.{slut_name}.T_soft',
                              (lambda m=mod: float(m.log_soft_score_temp.detach().exp()))))
                specs.append((f'L{li}.{slut_name}.T_sel',
                              (lambda m=mod: float(m.log_select_temp.detach().exp()))))
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

    # --- Lookahead outer update: every k steps, pull slow toward fast, reset fast to slow.
    if _USE_LOOKAHEAD and step % _LOOKAHEAD_K == 0:
        with torch.no_grad():
            for p_fast, p_slow in _LOOKAHEAD_PARAMS:
                p_slow.mul_(1.0 - _LOOKAHEAD_ALPHA).add_(p_fast.data, alpha=_LOOKAHEAD_ALPHA)
                p_fast.data.copy_(p_slow)

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
