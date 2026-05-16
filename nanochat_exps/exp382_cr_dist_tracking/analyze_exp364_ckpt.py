"""Load the exp364 (bs=192 SOTA, 1.3769) checkpoint into a freshly-built
model and analyse the c_r touch distribution WITHOUT training. Runs at both
bs=16 (match exp382) and bs=192 (production)."""
import os, sys, json
import torch

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

EXP_DIR = '/home/starost/spiky/nanochat_exps/exp382_cr_dist_tracking'
EXP364_DIR = '/home/starost/spiky/nanochat_exps/exp364_bs192'
sys.path.insert(0, EXP_DIR)

with open(os.path.join(EXP364_DIR, 'config.json')) as f:
    cfg = json.load(f)

# Build the same Model class from exp382 train.py by importing the model
# directly. Easiest: replicate the small helper from c_r_stats.py.

import torch.nn as nn
import torch.nn.functional as F
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from visit_tracker import install_visit_trackers

DEVICE = 'cuda'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E = cfg['embedding_dim']; D = cfg['residual_dim']; H = cfg['n_heads']
d_qk = cfg['d_qk']; d_v = cfg['d_v']; N_LAYERS = cfg['num_layers']
_ROPE_BASE = cfg.get('rope_base', 10000.0)

BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()

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
    a, b = t.chunk(2, dim=-1); return torch.cat([-b, a], dim=-1)
def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return (q*cos + _rotate_half(q)*sin, k*cos + _rotate_half(k)*sin)


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        kwargs_qkv = dict(_TINY_SOFT_KWARGS)
        kwargs_qkv['initial_weights_noise'] = cfg.get('qkv_lut_init_std', 0.001)
        self.qkv_lut = TinyMultiHeadLut(
            input_dim=E, n_heads=H, n_outputs=2*d_qk + d_v,
            n_anchor_pairs=cfg.get('qkv_input_nap', cfg['qk_input_nap']),
            tables_per_head=cfg.get('qkv_tph', cfg['qk_tph']),
            random_seed=cfg['random_seed'] + layer_idx, device=DEVICE, **kwargs_qkv,
        )
        self.v_lut = TinyMultiHeadLut(
            input_dim=E, n_heads=H, n_outputs=d_v,
            n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
            random_seed=cfg['random_seed'] + 200 + layer_idx, device=DEVICE, **_TINY_SOFT_KWARGS,
        )
        _OUT_TPH = cfg['out_tph_per_layer']
        self.out_proj = TinyMultiHeadLut(
            input_dim=H * d_v, n_heads=1, n_outputs=E,
            n_anchor_pairs=cfg['out_input_nap'], tables_per_head=_OUT_TPH[layer_idx],
            random_seed=cfg['random_seed'] + 400 + layer_idx, device=DEVICE, **_TINY_SOFT_KWARGS,
        )
        self.residual_lut = TinyMultiHeadLut(
            input_dim=E, n_heads=1, n_outputs=D,
            n_anchor_pairs=cfg.get('residual_input_nap', cfg['out_input_nap']),
            tables_per_head=cfg.get('residual_tph', cfg['out_tph']),
            random_seed=cfg['random_seed'] + 600 + layer_idx, device=DEVICE, **_TINY_SOFT_KWARGS,
        )
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
        v_vec = v_lut_out + v_branch
        v = v_vec.reshape(B, T, H, d_v).permute(0,2,1,3)
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
    def forward(self, tokens):
        B, T = tokens.shape
        x_resid = torch.zeros(B, T, D, device=tokens.device, dtype=self.tok_emb_E.weight.dtype)
        x_lut = self.tok_emb_E(tokens)
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid)
        return self.unembedder(x_resid)


print('Loading exp364 checkpoint...')
model = Model().to(DEVICE)
ckpt_path = os.path.join(EXP364_DIR, 'checkpoint.pt')
state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f'  missing keys: {len(missing)}, unexpected: {len(unexpected)}')
if missing:
    print('  missing[:5]:', missing[:5])
if unexpected:
    print('  unexpected[:5]:', unexpected[:5])

install_visit_trackers(model)
model.eval()

# Collect c_r stats at bs=16 (match exp382) and bs=192 (production bs)
for BS in (16, 192):
    print(f'\n=== bs={BS} ===')
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, BS, CONTEXT_SIZE, split='train', device=DEVICE,
    )
    N_BATCHES = 3
    module_specs = []
    for li, blk in enumerate(model.layers):
        for nm in ('qkv_lut', 'v_lut', 'out_proj', 'residual_lut'):
            module_specs.append((f'L{li}.{nm}', getattr(blk, nm)))
    agg = {name: [] for name, _ in module_specs}
    for b in range(N_BATCHES):
        x, y = next(loader)
        with torch.no_grad():
            _ = model(x)
        for name, mod in module_specs:
            agg[name].append(mod.weights._visit_counts.detach().cpu().clone())

    print(f'\n{"module":25s} {"touch_frac":>11s} {"c_r mean":>10s} {"c_r med":>9s} '
          f'{"c_r max":>9s} {"frac_eq0 %":>10s} {"frac_eq1 %":>10s} {"frac_ge8 %":>10s} {"frac_ge64 %":>11s}')
    for name, _mod in module_specs:
        stack = torch.stack(agg[name], dim=0).float()
        flat = stack.flatten()
        touched = flat > 0
        touch_frac = touched.float().mean().item()
        if touched.any():
            pos = flat[touched]
            c_mean = pos.mean().item(); c_med = pos.median().item(); c_max = pos.max().item()
        else:
            c_mean = c_med = c_max = 0.0
        frac_eq0 = (flat == 0).float().mean().item()
        frac_eq1 = (flat == 1).float().mean().item()
        frac_ge8 = (flat >= 8).float().mean().item()
        frac_ge64= (flat >= 64).float().mean().item()
        print(f'{name:25s} {touch_frac:>10.1%} {c_mean:>10.1f} {c_med:>9.0f} {c_max:>9.0f} '
              f'{frac_eq0:>9.1%} {frac_eq1:>9.1%} {frac_ge8:>9.1%} {frac_ge64:>10.1%}')
