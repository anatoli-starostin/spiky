"""Hard-eval the exp720 checkpoint.

exp720 was trained with TinyMultiHeadLut(backward_mode='hybrid_smooth'),
which dispatches forward to the soft-top-K blended path. To get the
deployment (hard) number, we set every TinyMultiHeadLut.backward_mode to
'ste' at eval time, which routes the forward through the standard
embedding_bag hard lookup (assumes n_alternatives==1, which is the default
for exp720's hybrid_smooth setup).

Reports:
  - soft eval (backward_mode='hybrid_smooth', matches training-time eval) -> should ~ match 1.2052
  - hard eval (backward_mode='ste')                                       -> deployment number
  - gap (mb)
"""
import sys, os, json
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = torch.device('cuda:0')
EXP_DIR = '/home/starost/spiky/nanochat_exps/exp720_pure_soft_bs96_16k'
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E = cfg['embedding_dim']; D = cfg['residual_dim']
H = cfg['n_heads']; d_qk = cfg['d_qk']; d_v = cfg['d_v']
N_LAYERS = cfg['num_layers']
DEVICE_BS = cfg['device_batch_size']
ROPE_BASE = cfg.get('rope_base', 10000.0)
EVAL_STEPS = cfg.get('eval_steps', 10)
_NOISE_EPS = cfg.get('argmax_noise_eps', 0.0)

BASE_DIR = get_base_dir()
tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, 'tokenizer'))
VOCAB_SIZE = tokenizer.get_vocab_size()
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
    argmax_noise_eps=_NOISE_EPS,
)


def _make_qk(s):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2 * d_qk,
                            n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
                            random_seed=cfg['random_seed'] + s, device=DEVICE, **_TINY_SOFT_KWARGS)
def _make_v(s):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
                            n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
                            random_seed=cfg['random_seed'] + s, device=DEVICE, **_TINY_SOFT_KWARGS)
def _make_out(s):
    return TinyMultiHeadLut(input_dim=H * d_v, n_heads=1, n_outputs=E,
                            n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
                            random_seed=cfg['random_seed'] + s, device=DEVICE, **_TINY_SOFT_KWARGS)
def _make_residual_lut(s):
    return TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
                            n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
                            random_seed=cfg['random_seed'] + s, device=DEVICE, **_TINY_SOFT_KWARGS)
def _make_emb_resid_lut(s):
    return TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
                            n_anchor_pairs=cfg['emb_resid_input_nap'], tables_per_head=cfg['emb_resid_tph'],
                            random_seed=cfg['random_seed'] + s, device=DEVICE, **_TINY_SOFT_KWARGS)


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
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin)


class MeanAbsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps = eps
    def forward(self, x):
        return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)


class LUTBlock(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.qk_lut = _make_qk(i)
        self.v_lut = _make_v(200 + i)
        self.out_proj = _make_out(400 + i)
        self.residual_lut = _make_residual_lut(600 + i)
        self.q_norm = nn.LayerNorm(d_qk); self.k_norm = nn.LayerNorm(d_qk)
        self.ln_pre = MeanAbsNorm(E); self.ln_resid = MeanAbsNorm(E)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)
        x_pre = self.ln_pre(x_flat)
        qk_out = self.qk_lut(x_pre)
        q_vec = self.q_norm(qk_out[..., :d_qk]); k_vec = self.k_norm(qk_out[..., d_qk:2 * d_qk])
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v = self.v_lut(x_pre).reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e = self.out_proj(out_in).squeeze(1)
        x_lut_next = x_flat + out_e
        r_in = self.ln_resid(x_lut_next)
        r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, D)
        return x_lut_next.reshape(B, T, E), r_out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.emb_resid_lut = _make_emb_resid_lut(800)
        self.ln_emb_resid = MeanAbsNorm(E)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, CONTEXT_SIZE, base=ROPE_BASE, device=DEVICE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)
    def get_device(self): return self.tok_emb_E.weight.device
    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)
        x_emb_pre = self.ln_emb_resid(x_lut.reshape(B * T, E))
        x_resid = self.emb_resid_lut(x_emb_pre).squeeze(1).reshape(B, T, D)
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid)
        logits = self.unembedder(x_resid)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), reduction=loss_reduction, ignore_index=-1)
        return logits


model = Model().to(DEVICE)
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), weights_only=False, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
print(f"Loaded exp720 checkpoint @ step {ckpt.get('step', '?')}, "
      f"training-best soft={ckpt.get('best_val_bpb', float('nan')):.4f}")

luts = [m for m in model.modules() if isinstance(m, TinyMultiHeadLut)]
print(f'Found {len(luts)} TinyMultiHeadLut modules')
print(f'current backward_mode (sample): {luts[0].backward_mode}, n_alternatives={luts[0].n_alternatives}')


def make_val_loader():
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE)


# --- Eval 1: soft mode (matches training) ------------------------------------
for m in luts: m.backward_mode = 'hybrid_smooth'
model.eval()
with torch.no_grad():
    bpb_soft = evaluate_bpb(model, make_val_loader(), EVAL_STEPS, token_bytes)
print(f"\n[backward_mode='hybrid_smooth']  val_bpb_soft = {bpb_soft:.4f}")

# --- Eval 2: hard mode (deployment) ------------------------------------------
# Bit-pack mismatch: hybrid_smooth uses MSB-first powers; STE/TAPL uses
# LSB-first. The trained row at hybrid_smooth-index `msb_pack(bits)` lives at
# row `lsb_pack(bits) = bit_reverse(msb_pack(bits), NAP)` for hard reads.
# Permute weight rows so new_weights[lsb_idx] == old_weights[msb_idx].
def _bit_reverse_perm(nap):
    K = 1 << nap
    out = [0] * K
    for k in range(K):
        r = 0
        for i in range(nap):
            if k & (1 << i):
                r |= 1 << (nap - 1 - i)
        out[k] = r
    return torch.tensor(out, dtype=torch.long)

with torch.no_grad():
    for m in luts:
        nap = m.n_anchor_pairs
        perm = _bit_reverse_perm(nap).to(m.weights.device)
        # weights shape: (n_heads*tph, 2^NAP, n_outputs) — row axis is dim=1.
        m.weights.data = m.weights.data.index_select(1, perm).contiguous()
        m.backward_mode = 'ste'
model.eval()
with torch.no_grad():
    bpb_hard = evaluate_bpb(model, make_val_loader(), EVAL_STEPS, token_bytes)
print(f"[backward_mode='ste'   (hard, +bit-reverse perm)]   val_bpb_hard = {bpb_hard:.4f}")

print(f"\nsoft -> hard gap: {(bpb_hard - bpb_soft) * 1000:+.2f} mb")
print(f"training-time final: {ckpt.get('final_val_bpb', float('nan')):.4f} (sanity vs soft above)")
