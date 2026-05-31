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

_TINY_HYBRID_SMOOTH_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    backward_mode='hybrid_smooth',
    hybrid_smooth_n_alt=cfg.get('hybrid_smooth_n_alt', 1),
    soft_score_temp=cfg.get('soft_score_temp', 0.5),
    select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True),
    use_bf16=cfg.get('soft_use_bf16', True),
    argmax_noise_eps=cfg.get('argmax_noise_eps', 0.0),
)

def _kwargs_for_mode(mode):
    if mode == 'soft':
        return _TINY_SOFT_KWARGS
    if mode == 'hybrid_smooth':
        return _TINY_HYBRID_SMOOTH_KWARGS
    return _TINY_MULTIALT_KWARGS

def _make_qk(layer_idx, seed_offset):
    """qk LUT — n_outputs = 2*d_qk. Q and K come from this LUT only.
    v_branch removed (exp643): v_lut alone provides v.
    """
    mode = cfg.get('qkv_backward_mode', 'soft')
    kwargs = dict(_kwargs_for_mode(mode))
    kwargs['initial_weights_noise'] = cfg.get('qkv_lut_init_std',
                                              cfg.get('mhlut_init_std', 0.001))
    return TinyMultiHeadLut(
        input_dim=E,
        n_heads=H,
        n_outputs=2 * d_qk,
        n_anchor_pairs=cfg.get('qkv_input_nap', cfg['qk_input_nap']),
        tables_per_head=cfg.get('qkv_tph', cfg['qk_tph']),
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **kwargs,
    )

def _make_v(layer_idx, seed_offset):
    v_mode = cfg.get('v_backward_mode', 'ste')
    v_kwargs = _kwargs_for_mode(v_mode)
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
    Use 'ste' (multi-alt n_alt=3) when NAP=8 to avoid soft's [B*T, K=2^NAP] memory blow-up.

    If cfg['out_proj_n_sparse'] is set and < E, enables sparse_scatter: each table
    has per-table n_outputs=out_proj_n_sparse, scatter-adds into the wider E-dim
    output. Each table picks `out_proj_n_sparse` random slots out of E.
    """
    tph = _OUT_TPH_PER_LAYER[layer_idx] if _OUT_TPH_PER_LAYER is not None else cfg['out_tph']
    mode = cfg.get('out_backward_mode', 'soft')
    kwargs = dict(_kwargs_for_mode(mode))   # COPY — we may mutate below
    n_sparse = cfg.get('out_proj_n_sparse')
    if n_sparse is not None and n_sparse < E:
        per_table_out = n_sparse
        kwargs['sparse_scatter_n_outputs'] = E
        kwargs['sparse_scatter_seed'] = (cfg.get('out_proj_sparse_scatter_seed_base', 7777)
                                          + cfg['random_seed'] + seed_offset)
    else:
        per_table_out = E
    return TinyMultiHeadLut(
        input_dim=H * d_v,
        n_heads=1,
        n_outputs=per_table_out,
        n_anchor_pairs=cfg['out_input_nap'],
        tables_per_head=tph,
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **kwargs,
    )

def _make_residual_lut(layer_idx, seed_offset):
    """E -> D residual projection."""
    res_mode = cfg.get('residual_backward_mode', 'soft')
    return TinyMultiHeadLut(
        input_dim=E,
        n_heads=1,
        n_outputs=D,
        n_anchor_pairs=cfg.get('residual_input_nap', cfg['out_input_nap']),
        tables_per_head=cfg.get('residual_tph', cfg['out_tph']),
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **_kwargs_for_mode(res_mode),
    )

def _make_final_residual_lut(seed_offset):
    """Single E -> D projection at end of model (exp649). tph = final_residual_tph
    (typically num_layers * residual_tph, so total LUT params match the per-layer
    accumulation in earlier exps)."""
    res_mode = cfg.get('residual_backward_mode', 'soft')
    return TinyMultiHeadLut(
        input_dim=E,
        n_heads=1,
        n_outputs=D,
        n_anchor_pairs=cfg.get('residual_input_nap', cfg['out_input_nap']),
        tables_per_head=cfg['final_residual_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **_kwargs_for_mode(res_mode),
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


class MeanAbsNorm(nn.Module):
    """L1 mean-abs norm: x / (mean(|x|) + eps). No centering, no affine.
    Cheapest magnitude norm; for LUTs the per-token scale mu is absorbed by
    the temperature, so we just need consistent magnitude across tokens."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)


class LUTBlock(nn.Module):
    """exp649: pure E-stream block — no per-layer residual_lut. A single
    final_residual_lut at the end of the model (in Model) projects the
    accumulated E-stream into D.
    """
    def __init__(self, layer_idx):
        super().__init__()
        self.qk_lut       = _make_qk(layer_idx, layer_idx)
        self.v_lut        = _make_v(layer_idx, 200 + layer_idx)
        self.out_proj     = _make_out(layer_idx, 400 + layer_idx)

        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)

        self.ln_pre  = MeanAbsNorm(E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)

        x_pre = self.ln_pre(x_flat)                                   # MeanAbsNorm pre-norm

        qk_out = self.qk_lut(x_pre)                                   # [B*T, H, 2*d_qk]
        q_vec = self.q_norm(qk_out[..., :d_qk])
        k_vec = self.k_norm(qk_out[..., d_qk:2 * d_qk])
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])

        v_lut_out = self.v_lut(x_pre)                                 # [B*T, H, d_v]
        v = v_lut_out.reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e  = self.out_proj(out_in).squeeze(1)                     # [B*T, E]

        # E-residual skip around attention/LUT block.
        return (x_flat + out_e).reshape(B, T, E)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # E-dim token embedding for the LUT carry (separate, small).
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        # exp653: no residual_lut at all. Final E embedding feeds the unembedder
        # directly through ln_final(E). D is unused.
        self.unembedder = nn.Linear(E, VOCAB_SIZE, bias=False)

        # Shared RoPE buffers for all blocks (no learned positional params).
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])

        # Final LN on E directly (no residual_lut projection to D).
        self.ln_final = nn.LayerNorm(E)

    def get_device(self):
        return self.tok_emb_E.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)                # (B, T, E)
        for layer in self.layers:
            x_lut = layer(x_lut, self.rope.cos, self.rope.sin)
        # No residual_lut: feed final E embedding directly to unembedder.
        x_lut  = self.ln_final(x_lut)                 # LN(E) before unembed
        logits = self.unembedder(x_lut)               # untied Linear(E, V)
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

class Lion(torch.optim.Optimizer):
    """EvoLved Sign Momentum (Lion). Sign-based with single momentum buffer."""
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, (b1, b2), wd = grp['lr'], grp['betas'], grp['weight_decay']
            for p in grp['params']:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if 'exp_avg' not in st:
                    st['exp_avg'] = torch.zeros_like(p)
                m = st['exp_avg']
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                update = (m * b1 + g * (1.0 - b1)).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(g, alpha=1.0 - b2)


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
_LUT_LR    = cfg.get('lut_lr', cfg['adam_lr'])
_LUT_BETAS = tuple(cfg.get('lut_betas', (0.9, 0.95)))
adam_groups = [
    dict(params=decay_params,   lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=cfg.get('weight_decay', 0.0)),
    dict(params=tok_emb_params + pos_emb_params + nodecay_params,
         lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
]
adam_opt = torch.optim.AdamW(adam_groups)
lion_opt = Lion([dict(params=lut_params, lr=_LUT_LR, weight_decay=0.0)],
                lr=_LUT_LR, betas=_LUT_BETAS)
optimizers = [adam_opt, lion_opt]
for opt in optimizers:
    for g in opt.param_groups:
        g['initial_lr'] = g['lr']
print(
    f'LUT optimizer = Lion (lut_lr={_LUT_LR}, betas={_LUT_BETAS}) | '
    f'lut={sum(p.numel() for p in lut_params):,} | '
    f'adam_decay(unembed)={sum(p.numel() for p in decay_params):,} (wd={cfg.get("weight_decay", 0.0)}) | '
    f'adam_tok_emb={sum(p.numel() for p in tok_emb_params):,} (wd=0) | '
    f'adam_pos_emb={sum(p.numel() for p in pos_emb_params):,} (wd=0) | '
    f'adam_nodecay_other={sum(p.numel() for p in nodecay_params):,} (wd=0)'
)

_NOISE_EPS = cfg.get('argmax_noise_eps', 0.0)
print(f'D=residual_dim={D}, E=embedding_dim={E}, H={H}, d_qk={d_qk}, d_v={d_v}, L={N_LAYERS}')
print(f'QK TinyMHLut({cfg.get("qkv_backward_mode", "soft")}, noise_eps={_NOISE_EPS}): in_nap={cfg.get("qkv_input_nap", cfg["qk_input_nap"])} tph={cfg.get("qkv_tph", cfg["qk_tph"])} n_out=2*d_qk={2*d_qk} (no v_branch)')
print(f'V_lut    TinyMHLut({cfg.get("v_backward_mode", "ste")}, noise_eps={_NOISE_EPS}): in_nap={cfg["v_input_nap"]} tph={cfg["v_tph"]} d_v={d_v}')
_out_tph_str = str(_OUT_TPH_PER_LAYER) if _OUT_TPH_PER_LAYER is not None else str(cfg['out_tph'])
_out_sparse_str = (f' SPARSE(per_table={cfg["out_proj_n_sparse"]} -> wide=E={E})'
                   if cfg.get('out_proj_n_sparse') is not None and cfg["out_proj_n_sparse"] < E
                   else f' n_out=E={E}')
print(f'out_proj TinyMHLut({cfg.get("out_backward_mode", "soft")}, noise_eps={_NOISE_EPS}): in_nap={cfg["out_input_nap"]} tph={_out_tph_str}{_out_sparse_str}')
print(f'NO residual_lut at all (exp653). Final E embedding -> ln_final(E={E}) -> Linear(E, V={VOCAB_SIZE}). D dim unused.')


# --- Temperature tracking -----------------------------------------------------
def collect_temperature_specs(model):
    specs = []
    for li, blk in enumerate(model.layers):
        for slut_name in ('qk_lut', 'v_lut', 'out_proj'):
            mod = getattr(blk, slut_name, None)
            if mod is None:
                continue
            if getattr(mod, 'learnable_temps', False):
                specs.append((f'L{li}.{slut_name}.T_soft',
                              (lambda m=mod: float(m.log_soft_score_temp.detach().exp()))))
                specs.append((f'L{li}.{slut_name}.T_sel',
                              (lambda m=mod: float(m.log_select_temp.detach().exp()))))
    final_mod = getattr(model, 'final_residual_lut', None)
    if final_mod is not None and getattr(final_mod, 'learnable_temps', False):
        specs.append(('final_residual_lut.T_soft',
                      (lambda m=final_mod: float(m.log_soft_score_temp.detach().exp()))))
        specs.append(('final_residual_lut.T_sel',
                      (lambda m=final_mod: float(m.log_select_temp.detach().exp()))))
    return specs

temp_specs = collect_temperature_specs(model)
temp_path = os.path.join(EXP_DIR, 'temperatures.csv')
temp_f = open(temp_path, 'w', newline='')
temp_w = csv.writer(temp_f)
temp_w.writerow(['step'] + [name for name, _ in temp_specs])
print(f'Tracking {len(temp_specs)} learnable temperatures in temperatures.csv')

# --- Per-parameter weight-delta tracking --------------------------------------
# Snapshot every trainable parameter; at each eval step write (step, name, weight_norm,
# delta_norm, rel_delta) to a long-format CSV. delta is measured from the previous
# snapshot (= since last eval step), then the snapshot is refreshed.
_PARAM_SNAPSHOT = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
_weight_csv_path = os.path.join(EXP_DIR, 'weight_deltas.csv')
_weight_csv_f = open(_weight_csv_path, 'w', newline='')
_weight_csv_w = csv.writer(_weight_csv_f)
_weight_csv_w.writerow(['step', 'param_name', 'weight_norm', 'delta_norm', 'rel_delta'])
print(f'Tracking weight deltas of {len(_PARAM_SNAPSHOT)} parameters in weight_deltas.csv')

def _log_weight_deltas(step_):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            w = p.detach()
            w_norm = float(w.norm())
            prev = _PARAM_SNAPSHOT.get(n)
            if prev is None or prev.shape != w.shape:
                _PARAM_SNAPSHOT[n] = w.clone()
                continue
            d_norm = float((w - prev).norm())
            rel = (d_norm / w_norm) if w_norm > 0 else 0.0
            _weight_csv_w.writerow([step_, n, f'{w_norm:.6e}', f'{d_norm:.6e}', f'{rel:.6e}'])
            _PARAM_SNAPSHOT[n] = w.clone()
        _weight_csv_f.flush()


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
    for opt in optimizers:
        for g in opt.param_groups:
            g['lr'] = g['initial_lr'] * lr_scale

    for opt in optimizers:
        opt.zero_grad()
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, targets=y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum

    for opt in optimizers:
        opt.step()

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
        _log_weight_deltas(step)
        model.train()

csv_f.close()
temp_f.close()
_weight_csv_f.close()
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
