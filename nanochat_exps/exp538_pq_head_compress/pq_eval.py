"""exp538: offline Product-Quantization compression of exp513's trained Linear
unembedder. No retraining -- k-means the [V, D] head weight into m sub-codebooks
(2^b entries each), reconstruct, and measure val bpb vs the 1.4825 baseline.

logit(v) = x . w_v,  w_v = concat_j C_j[code_v[j]]
         = sum_j  x_j . C_j[code_v[j]]   (ADC table trick -> matmul-free V-side)
Here we just reconstruct the dense W_pq and eval normally; the table impl is an
inference detail and gives identical logits. This measures QUALITY of the
quantization only.
"""
import sys, os, json, time
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

BASE = '/home/starost/spiky/nanochat_exps/exp513_qk_argmax_nap4_tph256'
with open(os.path.join(BASE, 'config.json')) as f:
    cfg = json.load(f)
DEVICE = 'cuda'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']; E = cfg['embedding_dim']; D = cfg['residual_dim']
H = cfg['n_heads']; d_qk = cfg['d_qk']; d_v = cfg['d_v']; N_LAYERS = cfg['num_layers']
DEVICE_BS = cfg['device_batch_size']; _ROPE_BASE = cfg.get('rope_base', 10000.0)
_NOISE_EPS = cfg.get('argmax_noise_eps', 0.0)

# --- tokenizer + val loader ---
tokenizer = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
VOCAB_SIZE = tokenizer.get_vocab_size()
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE)
EVAL_STEPS = 100  # more than training's 10 for low-noise comparison

# --- model defs (identical to exp513/530 train.py) ---
_TINY_SOFT_KWARGS = dict(
    weight_dtype=torch.float32, anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001), backward_mode='soft',
    soft_score_temp=cfg.get('soft_score_temp', 0.5), select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True), use_bf16=cfg.get('soft_use_bf16', True),
    argmax_noise_eps=_NOISE_EPS)

def _mk(input_dim, n_heads, n_outputs, nap, tph, so):
    return TinyMultiHeadLut(input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph, random_seed=cfg['random_seed'] + so,
        device=DEVICE, **_TINY_SOFT_KWARGS)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

def _rotate_half(t):
    a, b = t.chunk(2, dim=-1); return torch.cat([-b, a], dim=-1)
def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

class MeanAbsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps = eps
    def forward(self, x): return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)

class LUTBlock(nn.Module):
    def __init__(self, li):
        super().__init__()
        self.qk_lut = _mk(E, H, 2 * d_qk, cfg['qkv_input_nap'], cfg['qkv_tph'], li)
        self.v_lut = _mk(E, H, d_v, cfg['v_input_nap'], cfg['v_tph'], 200 + li)
        self.out_proj = _mk(H * d_v, 1, E, cfg['out_input_nap'], cfg['out_tph'], 400 + li)
        self.residual_lut = _mk(E, 1, D, cfg['residual_input_nap'], cfg['residual_tph'], 600 + li)
        self.q_norm = nn.LayerNorm(d_qk); self.k_norm = nn.LayerNorm(d_qk)
        self.ln_pre = MeanAbsNorm(E); self.ln_post = MeanAbsNorm(E)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape; xf = x.reshape(B * T, E); xp = self.ln_pre(xf)
        qk = self.qk_lut(xp)
        q = self.q_norm(qk[..., :d_qk]).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_norm(qk[..., d_qk:2 * d_qk]).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v = self.v_lut(xp).reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        oe = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)).squeeze(1)
        xn = xf + oe; xnf = xn.reshape(B, T, E)
        r = self.residual_lut(self.ln_post(xn)).squeeze(1).reshape(B, T, D)
        return xnf, r

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)
    def get_device(self): return self.tok_emb_E.weight.device
    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x_resid = torch.zeros(B, T, D, device=tokens.device, dtype=self.tok_emb_E.weight.dtype)
        x_lut = self.tok_emb_E(tokens)
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin); x_resid = x_resid + r
        logits = self.unembedder(self.ln_final(x_resid))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction=loss_reduction, ignore_index=-1)
        return logits

# --- build + load checkpoint ---
model = Model().to(DEVICE)
ckpt = torch.load(os.path.join(BASE, 'checkpoint.pt'), map_location=DEVICE)
missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
print(f'loaded exp513 ckpt | missing={len(missing)} unexpected={len(unexpected)}')
model.eval()

# --- PQ (k-means in torch on GPU) ---
def torch_kmeans(X, k, iters=30, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    N = X.shape[0]
    C = X[torch.randperm(N, generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        newC = torch.zeros_like(C); cnt = torch.zeros(k, device=X.device)
        newC.index_add_(0, a, X); cnt.index_add_(0, a, torch.ones(N, device=X.device))
        empty = cnt == 0
        cnt = cnt.clamp_min(1.0); newC = newC / cnt.unsqueeze(1)
        if empty.any():
            ridx = torch.randint(0, N, (int(empty.sum()),), generator=g, device=X.device)
            newC[empty] = X[ridx]
        C = newC
    a = torch.cdist(X, C).argmin(1)
    return C, a

@torch.no_grad()
def pq_reconstruct(W, m, b, iters=30):
    V, Dd = W.shape; sub = Dd // m; k = 1 << b
    Wpq = torch.empty_like(W)
    for j in range(m):
        Xj = W[:, j * sub:(j + 1) * sub].contiguous()
        Cj, aj = torch_kmeans(Xj, k, iters, seed=100 + j)
        Wpq[:, j * sub:(j + 1) * sub] = Cj[aj]
    return Wpq

W0 = model.unembedder.weight.data.clone()  # [V, D]
V = W0.shape[0]
LINEAR_PARAMS = V * D

print(f'\n=== exp513 PQ-head compression (V={V}, D={D}, eval_steps={EVAL_STEPS}) ===')
print(f'baseline Linear head: {LINEAR_PARAMS/1e6:.2f}M params, {LINEAR_PARAMS*4/1e6:.1f}MB/token bandwidth\n')

def report(tag, bpb, codebook_floats=None, code_bits=None, recon=None):
    if codebook_floats is None:
        print(f'{tag:28s} bpb={bpb:.4f}')
    else:
        eff = codebook_floats + code_bits / 32.0  # float-equivalent
        print(f'{tag:28s} bpb={bpb:.4f}  Δ={bpb-base_bpb:+.4f}  '
              f'params={eff/1e6:.3f}M ({LINEAR_PARAMS/eff:.0f}x less)  '
              f'codeBW={code_bits/8/1e3:.0f}KB/tok  recon_err={recon:.3f}')

t0 = time.time()
base_bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
report(f'baseline (orig Linear)', base_bpb)
print(f'  (training-time reported 1.4825; this {EVAL_STEPS}-step eval = {base_bpb:.4f})\n')

CONFIGS = [(24, 8), (48, 8), (96, 8), (192, 8)]  # sub = D/m = 16, 8, 4, 2 dims
for (m, b) in CONFIGS:
    Wpq = pq_reconstruct(W0, m, b)
    recon = float((W0 - Wpq).norm() / W0.norm())
    model.unembedder.weight.data.copy_(Wpq)
    bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
    codebook_floats = (1 << b) * D
    code_bits = V * m * b
    report(f'PQ m={m} b={b} (k={1<<b})', bpb, codebook_floats, code_bits, recon)
    model.unembedder.weight.data.copy_(W0)  # restore

print(f'\ntotal {time.time()-t0:.0f}s')
