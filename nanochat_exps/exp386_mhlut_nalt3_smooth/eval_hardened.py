"""Load exp386 checkpoint and re-evaluate val bpb under HARDENED inference:
smooth_mode=False, n_alternatives=1. Tests whether the training-only gradient
mechanism translates to inference cost, or if we can drop it at eval."""
import os, sys, json
import torch

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda'
torch.manual_seed(cfg['random_seed'])

# Replicate model build from train.py (minimal subset).
import torch.nn as nn
import torch.nn.functional as F
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

CONTEXT_SIZE = cfg['context_size']
E = cfg['embedding_dim']; D = cfg['residual_dim']; H = cfg['n_heads']
d_qk = cfg['d_qk']; d_v = cfg['d_v']; N_LAYERS = cfg['num_layers']
DEVICE_BS = cfg['device_batch_size']
_ROPE_BASE = cfg.get('rope_base', 10000.0)

BASE_DIR = get_base_dir()
tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, 'tokenizer'))
VOCAB_SIZE = tokenizer.get_vocab_size()
token_bytes = get_token_bytes(device=DEVICE)


def _make_lut(input_dim, n_heads, n_outputs, n_anchor_pairs, tables_per_head, seed_offset, init_std=None):
    return MultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=n_anchor_pairs, tables_per_head=tables_per_head,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        initial_weights_noise=init_std or cfg.get('mhlut_init_std', 0.001),
        n_alternatives=int(cfg.get('n_alternatives', 3)),
        smooth_mode=bool(cfg.get('smooth_mode', True)),
        argmax_noise_eps=cfg.get('argmax_noise_eps', 0.0),
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
    )


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)


def _rotate_half(t): a, b = t.chunk(2, dim=-1); return torch.cat([-b, a], dim=-1)
def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return (q*cos + _rotate_half(q)*sin, k*cos + _rotate_half(k)*sin)


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qkv_lut = _make_lut(E, H, 2*d_qk + d_v,
                                  cfg.get('qkv_input_nap', cfg['qk_input_nap']),
                                  cfg.get('qkv_tph', cfg['qk_tph']),
                                  layer_idx,
                                  init_std=cfg.get('qkv_lut_init_std', 0.001))
        self.v_lut = _make_lut(E, H, d_v, cfg['v_input_nap'], cfg['v_tph'], 200 + layer_idx)
        _OUT = cfg['out_tph_per_layer']
        self.out_proj = _make_lut(H * d_v, 1, E, cfg['out_input_nap'], _OUT[layer_idx], 400 + layer_idx)
        self.residual_lut = _make_lut(E, 1, D,
                                       cfg.get('residual_input_nap', cfg['out_input_nap']),
                                       cfg.get('residual_tph', cfg['out_tph']),
                                       600 + layer_idx)
        self.q_norm = nn.LayerNorm(d_qk); self.k_norm = nn.LayerNorm(d_qk); self.ln_e = nn.LayerNorm(E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B*T, E)
        qkv_out = self.qkv_lut(x_flat)
        q_vec = self.q_norm(qkv_out[..., :d_qk])
        k_vec = self.k_norm(qkv_out[..., d_qk:2*d_qk])
        v_branch = qkv_out[..., 2*d_qk:]
        q = q_vec.reshape(B, T, H, d_qk).permute(0,2,1,3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0,2,1,3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v_lut_out = self.v_lut(x_flat)
        v = (v_lut_out + v_branch).reshape(B, T, H, d_v).permute(0,2,1,3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0,2,1,3).reshape(B*T, H*d_v)
        out_e = self.out_proj(out_in).squeeze(1)
        out_e_norm = self.ln_e(out_e)
        x_lut_next = out_e_norm.reshape(B, T, E)
        r_out = self.residual_lut(out_e_norm).squeeze(1).reshape(B, T, D)
        return x_lut_next, r_out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)

    def get_device(self): return self.tok_emb_E.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x_resid = torch.zeros(B, T, D, device=tokens.device, dtype=self.tok_emb_E.weight.dtype)
        x_lut = self.tok_emb_E(tokens)
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid)
        logits = self.unembedder(x_resid)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction=loss_reduction, ignore_index=-1)
        return logits


print('Loading exp386 checkpoint...')
model = Model().to(DEVICE)
state = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE, weights_only=True)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f'  missing={len(missing)}, unexpected={len(unexpected)}')


def run_eval(label):
    model.eval()
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE)
    bpb = evaluate_bpb(model, loader, 50, token_bytes)
    print(f'  [{label}] val bpb = {bpb:.4f}')
    return bpb


# Baseline: as-trained config
print('\n=== AS-TRAINED (n_alt=3, smooth=True) ===')
b_trained = run_eval('trained')

# Harden: smooth_mode=False (drops smooth interpolation)
print('\n=== HARDENED: smooth_mode=False ===')
for mod in model.modules():
    if isinstance(mod, MultiHeadLut):
        mod.smooth_mode = False
        # Propagate to the underlying projection module too
        if hasattr(mod, 'projection'):
            mod.projection.smooth_mode = False
b_smooth_off = run_eval('smooth=False')

# Even harder: smooth=False AND n_alternatives=1 (no alt lookups at all)
print('\n=== FULLY HARDENED: smooth_mode=False, n_alternatives=1 ===')
for mod in model.modules():
    if isinstance(mod, MultiHeadLut):
        mod.smooth_mode = False
        mod.n_alternatives = 1
        if hasattr(mod, 'projection'):
            mod.projection.smooth_mode = False
            mod.projection.n_alternatives = 1
b_full_hard = run_eval('n_alt=1, smooth=False')

print('\n=== SUMMARY ===')
print(f'  as-trained                       : {b_trained:.4f}')
print(f'  smooth=False                     : {b_smooth_off:.4f} (Δ={b_smooth_off - b_trained:+.4f})')
print(f'  n_alt=1, smooth=False (single)   : {b_full_hard:.4f} (Δ={b_full_hard - b_trained:+.4f})')
