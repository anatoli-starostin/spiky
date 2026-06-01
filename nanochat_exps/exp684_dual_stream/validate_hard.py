"""Validate exp684 checkpoint in HARD mode.

Hard mode = each TinyMHLut forward returns W[argmax] with coefficient 1.0
(single-row lookup, no 2-row hybrid_smooth blend, no soft mixing).

Compares against the training-mode val_bpb (hybrid_smooth 2-row blend = 1.4262).
"""
import sys, os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
sys.path.insert(0, '/home/starost/spiky/src')

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

from spiky.lutorch.tiny_multi_head_lut import (
    TinyMultiHeadLut, _soft_lut_fwd_body_einsum,
)
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E    = cfg['embedding_dim']
D    = cfg['residual_dim']
H    = cfg['n_heads']
d_qk = cfg['d_qk']
d_v  = cfg['d_v']
N_LAYERS = cfg['num_layers']
DEVICE_BS = cfg['device_batch_size']
_ROPE_BASE = cfg.get('rope_base', 10000.0)


BASE_DIR = get_base_dir()
tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, 'tokenizer'))
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f'Vocab: {VOCAB_SIZE}')
token_bytes = get_token_bytes(device=DEVICE)


_TINY_SOFT_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    backward_mode=cfg.get('backward_mode', 'soft'),
    soft_score_temp=cfg.get('soft_score_temp', 0.5),
    select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True),
    use_bf16=cfg.get('soft_use_bf16', True),
    argmax_noise_eps=cfg.get('argmax_noise_eps', 0.0),
)

def _make_qk(seed_offset):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2*d_qk,
        n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, **_TINY_SOFT_KWARGS)

def _make_v(seed_offset):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, **_TINY_SOFT_KWARGS)

def _make_out(seed_offset):
    return TinyMultiHeadLut(input_dim=H*d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, **_TINY_SOFT_KWARGS)

def _make_residual(seed_offset):
    return TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
        n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, **_TINY_SOFT_KWARGS)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)


def _rotate_half(t):
    a, b = t.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class MeanAbsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qk_lut = _make_qk(layer_idx)
        self.v_lut  = _make_v(200 + layer_idx)
        self.out_proj = _make_out(400 + layer_idx)
        self.residual_lut = _make_residual(600 + layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.ln_pre = MeanAbsNorm(E)
        self.ln_resid = MeanAbsNorm(E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B*T, E)
        x_pre = self.ln_pre(x_flat)
        qk_out = self.qk_lut(x_pre)
        q_vec = self.q_norm(qk_out[..., :d_qk])
        k_vec = self.k_norm(qk_out[..., d_qk:2*d_qk])
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v_vec = self.v_lut(x_pre)
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)
        out_e = self.out_proj(out_in).squeeze(1)
        x_lut_next_flat = x_flat + out_e
        r_in = self.ln_resid(x_lut_next_flat)
        r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, D)
        return x_lut_next_flat.reshape(B, T, E), r_out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)

    def get_device(self):
        return self.tok_emb_E.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)
        x_resid = torch.zeros(B, T, D, device=x_lut.device, dtype=x_lut.dtype)
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid)
        logits = self.unembedder(x_resid)
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1,
            )
        return logits


# Build model, load checkpoint
print('Building model...')
model = Model().to(DEVICE)
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f'Params: {n_params/1e6:.2f}M  |  ckpt step={ckpt["step"]}  final_val_bpb(train mode)={ckpt["final_val_bpb"]:.4f}')


def hard_forward_factory(mod: TinyMultiHeadLut):
    """Return a forward closure that does pure argmax single-row lookup with coeff=1.0.
    Uses `_soft_lut_fwd_body_einsum` which computes argmax(einsum(p, bit_matrix))
    and then embedding_bag-sums W[argmax] across tables — no 2-row blend, no scaling.
    """
    def hard_forward(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != mod.input_dim:
            raise ValueError(f'x shape must be [B,{mod.input_dim}], got {tuple(x.shape)}')
        T_soft = mod.log_soft_score_temp.exp()
        autocast_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
                        if mod.use_bf16 and x.is_cuda
                        else torch.amp.autocast('cpu', enabled=False))
        with autocast_ctx:
            out, _ = _soft_lut_fwd_body_einsum(
                x, mod.weights, mod.soft_anchor_a_long, mod.soft_anchor_b_long,
                mod.soft_bit_matrix, T_soft,
                mod.n_heads, mod.tables_per_head, mod.table_dim,
            )
        return out
    return hard_forward


def patch_hard(model: nn.Module):
    """Override forward on every TinyMHLut to use hard argmax single-row lookup."""
    n_patched = 0
    for m in model.modules():
        if isinstance(m, TinyMultiHeadLut):
            m.forward = hard_forward_factory(m)
            n_patched += 1
    return n_patched


# === SOFT (training-mode hybrid_smooth) baseline re-eval =====================
EVAL_STEPS = 50  # heavier than train-time eval_steps=10 for a stable estimate
print(f'\n=== Eval with EVAL_STEPS={EVAL_STEPS}, device_batch_size={DEVICE_BS} ===')

print('\n--- SOFT (hybrid_smooth, 2-row blend) ---')
val_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE
)
t0 = time.time()
bpb_soft = evaluate_bpb(model, val_loader, EVAL_STEPS, token_bytes)
t_soft = time.time() - t0
print(f'  val_bpb (soft) = {bpb_soft:.4f}   [eval time {t_soft:.1f}s]')


# === HARD (single-row argmax with coeff=1.0) =================================
print('\n--- HARD (argmax single row, coeff=1.0) ---')
n_patched = patch_hard(model)
print(f'  patched {n_patched} TinyMHLut modules to hard forward')

val_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE
)
t0 = time.time()
bpb_hard = evaluate_bpb(model, val_loader, EVAL_STEPS, token_bytes)
t_hard = time.time() - t0
print(f'  val_bpb (hard) = {bpb_hard:.4f}   [eval time {t_hard:.1f}s]')


# === Report ==================================================================
delta = bpb_hard - bpb_soft
print('\n========================================================')
print(f'  SOFT (hybrid_smooth 2-row blend):  {bpb_soft:.4f} bpb')
print(f'  HARD (argmax single row coeff=1):  {bpb_hard:.4f} bpb')
print(f'  delta (hard - soft):               {delta:+.4f} bpb  ({delta*1000:+.1f} mb)')
print(f'  training-mode final (from ckpt):   {ckpt["final_val_bpb"]:.4f} bpb')
print('========================================================')

with open(os.path.join(EXP_DIR, 'validate_hard_summary.json'), 'w') as f:
    json.dump(dict(
        soft_bpb=float(bpb_soft),
        hard_bpb=float(bpb_hard),
        delta_bpb=float(delta),
        ckpt_final_val_bpb=float(ckpt['final_val_bpb']),
        eval_steps=EVAL_STEPS,
        device_batch_size=DEVICE_BS,
        context_size=CONTEXT_SIZE,
    ), f, indent=2)
print(f'wrote {os.path.join(EXP_DIR, "validate_hard_summary.json")}')
