"""Flexible FFN-slot sweep trainer (shared across the exp043+ CompressionMHL sweep).

MinimalGPT + RoPE, config-driven. The FFN slot of every block is one of:
  * ffn_type="dense"        -> the vanilla 384->1536->384 GELU MLP (baselines).
  * ffn_type="compression"  -> x = x + gamma*Linear(384->384)(h) + CompressionMultiHeadLUT(h),
                               h = ln2(x); gamma in {0,1}; CompressionMHL params in config.
Unembedder is tied (lm_head.weight = tok_emb.weight) when tie_unembedder=True, else untied.

Outputs alongside: metrics.csv, summary.json, loss.png, checkpoint.pt.
"""
import sys, os, json, math, time, csv
# Live-log fix: force line-buffered stdout so `step N |` / [VAL] lines reach
# run.log immediately even when stdout is redirected to a file (block-buffered by
# default). Belt-and-suspenders with `python -u` / PYTHONUNBUFFERED=1 at launch.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
# The local module lives beside this file. `python train.py` puts it on sys.path
# implicitly, but an exec()-based smoke harness would not -- make it explicit.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut       # optimizer isinstance route
from spiky.lutorch.hyperplane_multi_head_lut import HyperplaneMultiHeadLUT
from spiky.lutorch.ternary_hyperplane_multi_head_lut import (
    TernaryHyperplaneMultiHeadLUT, max_entropy_temp, expected_nonzero_divisor)
# exp_g_0030: identical to exp_g_0029 (pure HyperplaneMHL at full model dim, no
# compress/decompress, 384-dim cells) with ONE change -- the LUT class is
# TernaryHyperplaneMultiHeadLUT, whose hyperplane weights are quantized to {-1, 0, +1}
# by a straight-through estimator. See local_ternary_ffn.py.
from local_ternary_ffn import PureTernaryHyperplaneMHL as CompressionMultiHeadLUT

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

DEPTH, N_EMBD, N_HEAD, SEQ_LEN = cfg['depth'], cfg['n_embd'], cfg['n_head'], cfg['seq_len']
DEVICE_BS, TOTAL_BS, N_STEPS = cfg['device_batch_size'], cfg['total_batch_size'], cfg['n_steps']
LR, WD, WARMUP_FRAC = cfg['lr'], cfg['weight_decay'], cfg['lr_warmup_fraction']
# LR-schedule length, decoupled from how many steps we actually run. This is a
# 4,000-step screen, but the schedule is held at exp_n_0121's 16,000 (warmup 1,600,
# cosine over 16,000) rather than recompressed into 4,000. Consequence, stated
# plainly: the run stops mid-anneal at LR ~0.86 of peak, so its final number is a
# trajectory probe, not a converged result, and is NOT comparable to exp_n_0121's
# 16k final of 1.19145.
SCHED_STEPS = cfg.get('lr_schedule_steps', N_STEPS)
EVAL_EVERY, EVAL_STEPS = cfg['eval_every'], cfg['eval_steps']

FFN_TYPE = cfg.get('ffn_type', 'compression')          # "dense" | "compression"
GAMMA    = int(cfg.get('gamma', 0))                    # 0/1: parallel Linear(384->384) path
TIE      = bool(cfg.get('tie_unembedder', False))

# CompressionMHL knobs (only used when ffn_type == "compression")
LUT_IN     = cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim'))    # -1 => no compress
LUT_OUT    = cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim'))   # -1 => no decompress
LUT_NAP    = cfg.get('lut_n_anchor_pairs')
LUT_TPH    = cfg.get('lut_tables_per_head')
LUT_HEADS  = cfg.get('lut_n_heads', 1)
LUT_JOINT  = cfg.get('lut_joint_head_compression', False)
LUT_FWD    = cfg.get('lut_forward_mode', 'hard')
LUT_BF16   = cfg.get('lut_use_bf16', False)
LUT_NOISE  = cfg.get('lut_init_weights_noise', 1e-3)
LUT_LEARNTEMP = bool(cfg.get('lut_learnable_temps', False))   # learnable T_soft/T_sel per head
LUT_SEED   = cfg.get('lut_base_seed', 1000)
LUT_HP_INIT = cfg.get('lut_hyperplane_init', 'anchor_pairs')
# T init for the ternary straight-through routing. The dead zone is |w| <= T*ln3.
# exp_g_0032 pairs T=0.5 (band 0.5493) with hyperplane_init="balanced_ternary", which
# draws w ~ N(0, sigma^2) with sigma = band / Phi^-1(2/3) ~= 2.5504*T ~= 1.2753, so the
# step-0 routing quantizes to ~equal thirds of -1 / 0 / +1. That is the whole point of
# this run: exp_g_0030's anchor_pairs init put EVERY component maximally far from the
# boundary (zeros at w=0, +-1s at |w|=1, against a boundary at 0.5493) and its ternary
# values did not move at all early on. A balanced random init spreads components across
# the boundary so a real fraction of them are within reach of flipping.
_T_CFG = cfg.get('lut_ternary_temp_init', 0.5)
# "max_entropy" is DERIVED, not a magic number: with unit-std weights the zero fraction
# is set by the band alone, so T = Phi^-1((1+f)/2) / ln3 gives exactly f zeros. At
# f = 1/3 that is 0.43073/1.09861 = 0.392065 -- equal thirds of -1 / 0 / +1.
LUT_TERNARY_T = (max_entropy_temp() if _T_CFG == 'max_entropy' else float(_T_CFG))
# exp_g_0033: give each hyperplane one trainable scalar threshold b, so the routing
# DECISION is <q, x> + b > 0 instead of <q, x> > 0 -- the dead zone can sit off-centre
# on that hyperplane's own projection axis. b is zeroed at construction, so this run
# starts from exactly the function exp_g_0032 starts from and diverges only as b
# learns. Note b cannot change q itself (q = ternary(stanh(w, T)) depends on w alone);
# it moves which row a table selects.
LUT_TRAINABLE_BIAS = bool(cfg.get('lut_trainable_bias', False))
# Divide the routing projection by input_dim before the comparison and before the
# temperature-scaled soft score. The balanced init gives ~256 nonzeros per hyperplane,
# so raw <q,x> has std ~16 -- roughly 11x the anchor-pairs scale the soft_score /
# select temperatures (init 0.5) were chosen for, which saturates the surrogate.
# The bias lives in the NORMALIZED space: the decision is <q,x>/input_dim + b > 0.
# Note the hard routing at b=0 is unchanged by this (sign is scale-invariant); what
# the divisor changes is the gradient path.
LUT_NORMALIZE_PROJ = cfg.get('lut_normalize_projection', False)
# Standardize each hyperplane's weight vector to unit std every forward, BEFORE
# ternarization. Removes the overall-magnitude degree of freedom -- only the direction
# pattern trains -- and pins the ternary density, which is what makes a fixed derived
# divisor correct rather than a guess.
LUT_NORMALIZE_W = bool(cfg.get('lut_normalize_weights', False))
# Per-head output decompression: each LUT table stores an inner_out vector instead of a
# full n_embd one, the heads are CONCATENATED, and a learned Linear projects back to
# n_embd. Shrinks the table axis, which is where nearly all the parameters live.
LUT_DECOMPRESS_HEADS = bool(cfg.get('lut_decompress_heads', False))
LUT_INNER_OUT = cfg.get('lut_inner_out', None)
# Sparsity regularization. The penalty is mean(|tanh(w'/(2T))|) with T DETACHED, so it
# pushes the (already unit-std) weights toward the dead zone rather than widening the
# band. lambda is on the MEAN, which is dimension-invariant -- measured share of the
# task gradient is ~8.2e-4 per unit lambda, so O(100) is the working range.
LUT_PENALTY = float(cfg.get('lut_nonzero_penalty_weight', 0.0))

BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f'Vocab size: {VOCAB_SIZE}')
assert VOCAB_SIZE == cfg['tokenizer_vocab_size']
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='train', device=DEVICE)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='val', device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin)


class MinimalAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MinimalBlock(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn_type = FFN_TYPE
        if FFN_TYPE == 'dense':
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd, bias=False), nn.GELU(),
                nn.Linear(4 * n_embd, n_embd, bias=False))
        else:
            self.lin = nn.Linear(n_embd, n_embd, bias=True) if GAMMA == 1 else None
            self.ffn = CompressionMultiHeadLUT(
                input_dim=n_embd, output_dim=n_embd,
                inner_in_dim=LUT_IN, inner_out_dim=LUT_OUT,
                nap=LUT_NAP, tph=LUT_TPH, n_heads=LUT_HEADS,
                joint_head_compression=LUT_JOINT, forward_mode=LUT_FWD,
                use_bf16=LUT_BF16, initial_weights_noise=LUT_NOISE,
                learnable_temps=LUT_LEARNTEMP,
                hyperplane_init=LUT_HP_INIT,
                ternary_temp_init=LUT_TERNARY_T,
                trainable_bias=LUT_TRAINABLE_BIAS,
                normalize_projection=LUT_NORMALIZE_PROJ,
                normalize_weights=LUT_NORMALIZE_W,
                decompress_heads=LUT_DECOMPRESS_HEADS,
                inner_out=LUT_INNER_OUT,
                nonzero_penalty_weight=LUT_PENALTY,
                random_seed=LUT_SEED + layer_idx)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        h = self.ln2(x)
        if self.ffn_type == 'dense':
            return x + self.mlp(h)
        B, T, C = h.shape
        out = self.ffn(h.reshape(B * T, C)).reshape(B, T, C).to(h.dtype)
        if self.lin is not None:
            out = out + self.lin(h)
        return x + out


class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head, max_seq_len=seq_len)
        self.blocks = nn.ModuleList([MinimalBlock(n_embd, n_head, i) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            if FFN_TYPE == 'dense':
                nn.init.zeros_(block.mlp[-1].weight)
            else:
                if block.ffn.has_decompress:
                    nn.init.zeros_(block.ffn.decompress.weight)   # FFN-slot output proj
                if block.lin is not None:
                    nn.init.zeros_(block.lin.weight)              # parallel Linear (gamma=1)
        if TIE:
            self.head.weight = self.tok_emb.weight                # weight sharing

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def get_device(self):
        return self.tok_emb.weight.device

    def forward(self, idx, targets=None, loss_reduction='mean'):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope.cos, self.rope.sin)
        logits = self.head(self.ln_f(x))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction=loss_reduction, ignore_index=-1)
        return logits


# LUT module types whose OWN parameters go in the no-decay group. exp_n_0121's rule
# was "the LUT module's parameters are not weight-decayed" (tables + temps);
# HyperplaneMHL fills the same role, so its tables, temps, and the hyperplane w/b that
# replace 0121's fixed anchor buffers follow it too. Under wd=0.1 the w/b would shrink
# over training: harmless to the hard forward (sign(<w,x>+b) is scale-invariant) but it
# steadily softens the temperature-scaled soft backward -- a second variable on top of
# the module swap.
# TernaryHyperplaneMultiHeadLUT subclasses HyperplaneMultiHeadLUT, so isinstance
# already matches; listed explicitly so the no-decay intent is readable, and so
# log_ternary_temp is visibly covered alongside the tables and the other temps.
_LUT_TYPES = (FastMultiHeadLut, HyperplaneMultiHeadLUT,
              TernaryHyperplaneMultiHeadLUT)


def setup_optimizer(model, lr, weight_decay):
    lut_ids = {id(p) for m in model.modules() if isinstance(m, _LUT_TYPES)
               for p in m.parameters(recurse=False)}
    decay, nodecay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (nodecay if (id(p) in lut_ids or p.ndim < 2) else decay).append(p)
    groups = [dict(params=decay, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay),
              dict(params=nodecay, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)]
    opt = torch.optim.AdamW(groups)
    for g in opt.param_groups:
        g['initial_lr'] = g['lr']
    n_lut = sum(m.weights.numel() for m in model.modules() if isinstance(m, _LUT_TYPES))
    print(f"AdamW (0033 grouping) | decay(2-D weights)={sum(p.numel() for p in decay):,} wd={weight_decay} | "
          f"nodecay(LUT tables+temps+1-D)={sum(p.numel() for p in nodecay):,} wd=0 | lr={lr} betas=(0.9, 0.95) "
          f"eps=1e-8 [LUT tables={n_lut:,} in nodecay]")
    return opt


def get_lr_scale(step, n_steps, warmup_frac):
    w = int(warmup_frac * n_steps)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n_steps - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


model = MinimalGPT(vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD, n_layer=DEPTH, seq_len=SEQ_LEN).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f'MinimalGPT: depth={DEPTH}, dim={N_EMBD}, heads={N_HEAD}, seq_len={SEQ_LEN} | ffn={FFN_TYPE} tie={TIE} gamma={GAMMA}')
_f0 = model.blocks[0].ffn
print(f'FFN LUT: {type(_f0).__name__} | compress={_f0.has_compress} '
      f'decompress={_f0.has_decompress} | cells store {_f0.lut.n_outputs}-dim outputs '
      f'| {_f0.lut.n_lookup_tables} tables x {_f0.lut.table_dim} cells')
print(f'  hyperplanes {tuple(_f0.lut.hyperplane_weight.shape)} on the FULL '
      f'{_f0.input_dim}-dim input, init={_f0.hyperplane_init}')
_ts = _f0.lut.ternary_stats()
print(f'  TERNARY routing: init={_f0.lut.hyperplane_init} T init {LUT_TERNARY_T} '
      f'(dead zone |w| <= {_ts["T_max"] * 1.0986:.4f}, sigma {_f0.lut.balanced_sigma}) '
      f'| +1 {_ts["frac_pos"]:.4f}  0 {_ts["frac_zero"]:.4f}  -1 {_ts["frac_neg"]:.4f} '
      f'| {_ts["nonzero_per_hyperplane"]:.2f} nonzeros/hyperplane')
_bias_is_param = any(nm == 'hyperplane_bias'
                     for nm, _ in _f0.lut.named_parameters())
print(f'  trainable_bias={LUT_TRAINABLE_BIAS} -> hyperplane_bias is a '
      f'{"PARAMETER" if _bias_is_param else "frozen zero buffer"}, '
      f'shape {tuple(_f0.lut.hyperplane_bias.shape)}')
assert _bias_is_param == LUT_TRAINABLE_BIAS, 'trainable_bias did not take effect'
assert _f0.lut.normalize_weights == LUT_NORMALIZE_W, \
    'normalize_weights did not take effect'
assert _f0.decompress_heads == LUT_DECOMPRESS_HEADS, \
    'decompress_heads did not take effect'
if LUT_DECOMPRESS_HEADS:
    _wt = _f0.lut.weights
    _dc = _f0.decompress
    print(f'  decompress_heads=True -> LUT tables {tuple(_wt.shape)} '
          f'(third dim = inner_out = {LUT_INNER_OUT}), heads CONCATENATED then projected')
    print(f'  decompress Linear {_dc.in_features} -> {_dc.out_features} '
          f'(= n_heads*inner_out = {LUT_HEADS}*{LUT_INNER_OUT} -> n_embd), '
          f'bias={_dc.bias is not None}')
    assert _wt.shape[2] == LUT_INNER_OUT, \
        f'LUT table output axis is {_wt.shape[2]}, expected inner_out {LUT_INNER_OUT}'
    assert _dc.in_features == LUT_HEADS * LUT_INNER_OUT and _dc.out_features == N_EMBD, \
        f'decompress is {_dc.in_features}->{_dc.out_features}, expected ' \
        f'{LUT_HEADS * LUT_INNER_OUT}->{N_EMBD}'
    print(f'  decompress.weight zero-initialized (slot is identity at step 0): '
          f'{bool((_dc.weight == 0).all())}')
if LUT_NORMALIZE_W:
    _wn = _f0.lut.normalized_weight()
    print(f'  normalize_weights=True -> per-hyperplane standardized w: '
          f'mean {float(_wn.mean().detach()):+.2e}, std {float(_wn.std().detach()):.6f} '
          f'(expect 0 and 1)')
    assert abs(float(_wn.std().detach()) - 1.0) < 0.02, 'normalized w is not unit std'
    _Tstar = max_entropy_temp()
    print(f'  T init {LUT_TERNARY_T:.6f} (derived max_entropy_temp() = {_Tstar:.6f}), '
          f'band = T*ln3 = {LUT_TERNARY_T * 1.0986122886681098:.6f}')
    assert abs(LUT_TERNARY_T - _Tstar) < 1e-6, 'T init is not the max-entropy value'
    _Dstar = expected_nonzero_divisor(N_EMBD)
    print(f'  divisor {_f0.lut.projection_divisor:.6f} (derived '
          f'expected_nonzero_divisor({N_EMBD}) = sqrt({N_EMBD}*2/3) = {_Dstar:.6f})')
    assert abs(_f0.lut.projection_divisor - _Dstar) < 1e-9, 'divisor is not the derived value'
_dyn = _f0.lut.dynamic_divisor
_proj = ('<q,x>/sqrt(nnz_h)' if _dyn
         else '<q,x>' if _f0.lut.projection_divisor == 1.0
         else f'<q,x>/{_f0.lut.projection_divisor:.3f}')
_divtxt = ('PER-HYPERPLANE sqrt(nonzero count), recomputed every forward '
           f'(range {float(_f0.lut.nonzero_divisor().min()):.2f}..'
           f'{float(_f0.lut.nonzero_divisor().max()):.2f} at init)'
           if _dyn else f'{_f0.lut.projection_divisor:.4f}')
print(f'  normalize_projection={LUT_NORMALIZE_PROJ!r} -> mode '
      f'{_f0.lut.projection_norm!r}, divisor {_divtxt} (input_dim {_f0.lut.input_dim})')
print(f'  ROUTING DECISION: {_proj}{" + b" if _bias_is_param else ""} > 0')
assert _f0.lut.normalize_projection == (LUT_NORMALIZE_PROJ not in (False, 'none')), \
    'normalize_projection did not take effect'
assert _f0.lut.projection_norm == (
    'sqrt_input_dim' if LUT_NORMALIZE_PROJ is True else
    ('none' if LUT_NORMALIZE_PROJ is False else LUT_NORMALIZE_PROJ)), \
    f'projection_norm is {_f0.lut.projection_norm!r}, expected {LUT_NORMALIZE_PROJ!r}'
# Measure the quantity the soft-score temperature is actually compared against, so the
# divisor choice is judged on a number rather than assumed. LayerNorm'd input, so a
# unit-ish Gaussian probe is representative of what the slot sees.
_probe = torch.randn(256, N_EMBD, device=DEVICE)
_pstd = _f0.lut.projection_std(_probe)
_Ts = float(_f0.lut.log_soft_score_temp.detach().exp())
print(f'  projection std at init: {_pstd:.6f}   vs soft_score_temp {_Ts:.3f} '
      f'-> score/T = {_pstd / _Ts:.4f}')
print(f'  mean nonzeros/hyperplane at init: {_f0.lut.mean_nonzero_count():.4f}')
assert 0.2 <= _pstd / _Ts <= 10.0, (
    f'score/T {_pstd/_Ts:.4f} is outside the healthy band 0.2-10 at init -- the '
    f'surrogate would be flat or saturated before training even starts')
print(f'  score/T is inside the healthy band 0.2-10: OK')
# exp_g_0034 starts DEGENERATE by design, so the equal-thirds check does not apply.
# Assert the intended property instead: nearly everything must quantize to zero, but
# the routing must not be entirely dead.
print(f'  step-0 split: +1 {_ts["frac_pos"]:.6f}  0 {_ts["frac_zero"]:.6f}  '
      f'-1 {_ts["frac_neg"]:.6f}  ({_ts["nonzero_per_hyperplane"]:.4f} nonzeros/hyperplane)')
if LUT_HP_INIT == 'near_zero_ternary':
    assert _ts['frac_zero'] > 0.99, (
        f'near_zero init should start >99% zeros, got {_ts["frac_zero"]:.4f}')
    assert _ts['frac_zero'] < 1.0, 'routing is ENTIRELY zero -- nothing could ever learn'
    _q0 = _f0.lut.hard_ternary_weight()
    _rows = _q0.shape[0] * _q0.shape[1]
    _nzr = int((_q0 != 0).any(-1).sum())
    print(f'  DEGENERATE START confirmed: {_ts["frac_zero"]*100:.3f}% of components are 0; '
          f'{_nzr:,}/{_rows:,} hyperplanes ({100*_nzr/_rows:.1f}%) have any nonzero')
elif LUT_HP_INIT == 'balanced_ternary':
    _dev = max(abs(_ts[k] - 1/3) for k in ('frac_pos', 'frac_zero', 'frac_neg'))
    print(f'  step-0 split is within {_dev:.4f} of equal thirds '
          f'({"BALANCED" if _dev < 0.02 else "OFF TARGET -- check sigma/T"})')
print(f'  bias: {"PARAMETER" if any(n == "hyperplane_bias" for n, _ in _f0.lut.named_parameters()) else "none (frozen zero buffer) — routing test is <q,x> > 0"}')
print(f'Run length {N_STEPS:,} steps on a {SCHED_STEPS:,}-step LR schedule '
      f'(warmup {int(WARMUP_FRAC * SCHED_STEPS):,})')
print(f'Params: {total_params:,}')

if os.environ.get('SMOKE'):
    print('SMOKE OK'); sys.exit(0)

optimizer = setup_optimizer(model, lr=LR, weight_decay=WD)
tokens_per_step = DEVICE_BS * SEQ_LEN
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'Tokens/micro-batch: {tokens_per_step:,} | grad_accum: {grad_accum} | effective batch: {grad_accum * tokens_per_step:,} tokens')

csv_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f); csv_w.writerow(['step', 'train_loss', 'val_bpb'])

# ---------------------------------------------------------------------------
# TERNARY DRIFT INSTRUMENTATION
#
# The question this run exists to answer: do the DISCRETE ternary values q actually
# change, or does only the continuous shadow weight w wiggle without ever crossing the
# |w| <= T*ln3 dead-zone boundary or flipping sign? If q never moves, this run is just
# exp_g_0031's fixed anchor-pair routing carrying dead parameters.
#
# w moving is expected and is NOT the signal. We track q itself, against the step-0
# init and against the previous eval, plus T, since T moves the boundary.
# All no_grad, eval-time only -- the training math is untouched.
_tern_luts = [_m for _m in model.modules()
              if isinstance(_m, TernaryHyperplaneMultiHeadLUT)]
_n_hyperplanes = sum(_m.hyperplane_weight.shape[0] * _m.hyperplane_weight.shape[1]
                     for _m in _tern_luts)
print(f'ternary drift tracking: {len(_tern_luts)} LUT slots, '
      f'{_n_hyperplanes:,} hyperplanes, '
      f'{sum(_m.hyperplane_weight.numel() for _m in _tern_luts):,} components')


@torch.no_grad()
def _q_all():
    """Every slot's hard ternary routing, flattened into one vector."""
    return torch.cat([_m.hard_ternary_weight().reshape(-1) for _m in _tern_luts])


@torch.no_grad()
def _T_all():
    return torch.cat([_m.ternary_temp.reshape(-1) for _m in _tern_luts])


_q_init = _q_all().clone()
_q_prev = _q_init.clone()
_n_comp = _q_init.numel()

drift_f = open(os.path.join(EXP_DIR, 'ternary_drift.csv'), 'w', newline='')
drift_w = csv.writer(drift_f)
drift_w.writerow([
    'step', 'frac_zero', 'frac_pos', 'frac_neg', 'nonzero_per_hyperplane',
    'hamming_vs_init', 'hamming_frac', 'sign_flips_vs_init',
    'zero_to_nonzero_vs_init', 'nonzero_to_zero_vs_init', 'churn_vs_prev_eval',
    'T_min', 'T_mean', 'T_max',
    # exp_g_0033: does b actually move? It starts at exactly 0 everywhere, so
    # b_absmean > 0 at all is the signal, and b_min/b_max show the spread.
    'b_min', 'b_mean', 'b_max', 'b_absmean', 'b_frac_nonzero',
    # exp_g_0035: the dynamic divisor's whole job is to hold score/T in band as the
    # routing densifies. Log both so we can see it working (or not) rather than assume.
    'mean_nonzero_count', 'proj_std', 'score_over_temp', 'divisor_mean',
    # exp_g_0039..41: the sparsity sweep. sparsity_surrogate is the quantity the
    # penalty minimises; frac_zero and mean_nonzero_count are what it buys.
    'sparsity_surrogate', 'penalty_value',
])

# One FIXED probe batch, drawn once, so the score/T series is comparable across evals
# rather than reflecting a different random input each time.
_probe_g = torch.Generator(device='cpu').manual_seed(20350)
_DRIFT_PROBE = torch.randn(256, N_EMBD, generator=_probe_g).to(DEVICE)


@torch.no_grad()
def _b_all():
    return torch.cat([_m.hyperplane_bias.reshape(-1) for _m in _tern_luts])


@torch.no_grad()
def _drift_row(step):
    global _q_prev
    q = _q_all()
    T = _T_all()
    diff = q != _q_init
    both_nz = (_q_init != 0) & (q != 0)
    row = [
        step,
        f'{float((q == 0).float().mean()):.6f}',
        f'{float((q > 0).float().mean()):.6f}',
        f'{float((q < 0).float().mean()):.6f}',
        f'{float((q != 0).sum()) / _n_hyperplanes:.4f}',
        int(diff.sum()),
        f'{float(diff.float().mean()):.8f}',
        int((both_nz & diff).sum()),                       # +1 <-> -1
        int(((_q_init == 0) & (q != 0)).sum()),            # entered the routing
        int(((_q_init != 0) & (q == 0)).sum()),            # fell into the dead zone
        int((q != _q_prev).sum()),                         # step-over-step churn
        f'{float(T.min()):.6f}', f'{float(T.mean()):.6f}', f'{float(T.max()):.6f}',
    ]
    b = _b_all()
    row += [f'{float(b.min()):.6f}', f'{float(b.mean()):.6f}',
            f'{float(b.max()):.6f}', f'{float(b.abs().mean()):.6f}',
            f'{float((b != 0).float().mean()):.6f}']
    _l0 = _tern_luts[0]
    _nzc = sum(m_.mean_nonzero_count() for m_ in _tern_luts) / len(_tern_luts)
    _ps = _l0.projection_std(_DRIFT_PROBE)
    _sot = _l0.score_over_temp(_DRIFT_PROBE)
    _dm = (float(_l0.nonzero_divisor().mean()) if _l0.dynamic_divisor
           else float(_l0.projection_divisor))
    row += [f'{_nzc:.4f}', f'{_ps:.6f}', f'{_sot:.4f}', f'{_dm:.4f}']
    _sur = sum(float(m_.sparsity_surrogate().detach()) for m_ in _tern_luts) / len(_tern_luts)
    _pen = sum(float(m_.sparsity_penalty().detach()) for m_ in _tern_luts)
    row += [f'{_sur:.6f}', f'{_pen:.6f}']
    drift_w.writerow(row); drift_f.flush()
    _q_prev = q.clone()
    print(f'[TERNARY] step {step}: hamming_vs_init {row[5]:,} '
          f'({float(diff.float().mean())*100:.4f}%) | sign_flips {row[7]:,} | '
          f'0->nz {row[8]:,} | nz->0 {row[9]:,} | churn {row[10]:,} | '
          f'frac_zero {row[1]} | T {row[11]}..{row[13]} | '
          f'b {row[14]}..{row[16]} absmean {row[17]} nonzero {row[18]} | '
          f'nnz/hp {row[19]} score/T {row[21]} divisor {row[22]} | '
          f'surrogate {row[23]} penalty {row[24]}')


_drift_row(0)   # the step-0 baseline: all-zero drift by construction
train_losses_logged, val_bpbs, val_steps = [], [], []
ema, best_bpb, t0 = None, float('inf'), time.time()

model.train()
for step in range(1, N_STEPS + 1):
    lr_scale = get_lr_scale(step, SCHED_STEPS, WARMUP_FRAC)
    for g in optimizer.param_groups:
        g['lr'] = g['initial_lr'] * lr_scale
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum
    if LUT_PENALTY > 0.0:
        # Added once per optimizer step, NOT once per micro-batch: the accumulation
        # loop contributes sum(loss/grad_accum) = the mean loss, so the total objective
        # is mean_loss + pen. Guarded, so lambda=0 computes nothing at all and the run
        # is bitwise identical to exp_g_0037.
        pen = sum(m_.sparsity_penalty() for m_ in model.modules()
                  if isinstance(m_, TernaryHyperplaneMultiHeadLUT))
        pen.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    ema = accum_loss if ema is None else 0.99 * ema + 0.01 * accum_loss
    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | loss={ema:.4f} | lr={lr_scale * LR:.2e}')
    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval()
        bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
        best_bpb = min(best_bpb, bpb)
        print(f'[VAL] step {step}: bpb={bpb:.4f}')
        train_losses_logged.append(ema); val_bpbs.append(bpb); val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}']); csv_f.flush()
        _drift_row(step)
        model.train()

csv_f.close()
drift_f.close()
elapsed = time.time() - t0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, train_losses_logged, label='train (ema)'); ax1.set(xlabel='step', ylabel='ce loss', title='Training Loss'); ax1.legend(); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, 'o-', color='tab:orange', label='val bpb'); ax2.set(xlabel='step', ylabel='bpb', title='Validation BPB'); ax2.legend(); ax2.grid(True)
plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120); plt.close()

summary = {'exp_name': cfg['exp_name'], 'best_val_bpb': best_bpb,
           'final_val_bpb': val_bpbs[-1] if val_bpbs else None,
           'total_params': total_params, 'training_time_hours': round(elapsed / 3600, 3)}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ==='); print(json.dumps(summary, indent=2))
