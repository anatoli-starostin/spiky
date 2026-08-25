"""exp024 — exp023 (single-stream + Linear unembedder) with hyperplane_init=anchor_pairs.

Exact clone of exp023 with ONE change: hyperplane_init="anchor_pairs" (anchor-pair init)
for all backbone LUTs (qk/v/out_proj HyperplaneMultiHeadLUT), instead of exp023's
random/scale 0.05. The anchor-vs-random A/B at the single-stream + Linear architecture.
Everything else byte-identical to exp023 (single stream, LayerNorm, plain Linear unembedder,
6L/E384/6h×64, LUT qk4/256 v6/256 out7/512 hard, seed 42, 16k recipe). Baselines: exp010
dual-stream 1.1940, exp023 single-stream random-init.

ORIGINAL exp023 header follows:
================================================================================
exp023 — single-stream champion lineage with a PLAIN Linear unembedder (the CONTROL).

Exact clone of exp022 with ONE change: the HyperplaneCodeLUT unembedder is replaced by a
plain nn.Linear(E=384 -> V=32768, bias=False) — the standard linear unembedder (same as
exp010's). Everything else byte-identical to exp022: single residual stream (no dual stream,
no residual LUTs), LayerNorm everywhere incl. ln_final before the unembedder,
hyperplane_init=random scale 0.05 for all backbone LUTs, backbone 6L/E384/6h x64/ctx512/
vocab32768/RoPE1e4/untied emb, LUT qk4/256 v6/256 out7/512 (hard), seed 42, 16k recipe.

Isolates the single-stream reduction FROM the unembedder: exp023 = single-stream + Linear.
vs exp010 (dual-stream + Linear, 1.1940) -> the cost of removing the dual stream/residual LUTs;
vs exp019/020/022 (single-stream + code-LUT) -> the cost of the code-LUT unembedder.

ORIGINAL exp022 header follows:
================================================================================
exp022 — exp020 (single-stream random-init code-LUT) with the unembedder T raised 16 -> 64.

Exact copy of exp020 with ONE change: HyperplaneCodeLUT n_tables T = 64 (was 16); nap=15,
input_dim E=384, V=32768 unchanged. Tests whether more voting recovers quality at champion
E=384 scale. Everything else byte-identical to exp020 (single stream, LayerNorm everywhere,
hyperplane_init=random scale 0.05 for all LUTs incl. the code-LUT hyperplanes, w_cell 0.02).
The code-LUT loops/accumulates over T tables with gradient checkpointing, so peak memory
stays ~O([N,V]) even at T=64. Baselines: exp010 dual-stream 1.1940, exp020 T=16.

ORIGINAL exp020 header follows:
================================================================================
exp020 — exp019 (single-stream code-LUT) with hyperplane_init=random (scale 0.05) for ALL LUTs.

Exact copy of exp019 with ONE change: every LUT's hyperplane init is random Gaussian
N(0, 0.05^2), bias 0 — the qk/v/out_proj HyperplaneMultiHeadLUT sites AND the
HyperplaneCodeLUT unembedder's hyperplane_weight/bias (via the new hyperplane_init_scale
support in hyperplane_code_lut.py). w_cell keeps its near-uniform const 0.02 init.
Motivation: exp012 showed random init ties anchor for the backbone, and random may suit
the code-LUT unembedder better. Baselines: exp010 dual-stream 1.1940, exp019 anchor-init
single-stream.

ORIGINAL exp019 header follows:
================================================================================
exp019 — phase 2: champion lineage reduced to SINGLE STREAM + HyperplaneCodeLUT unembedder.

From exp018 (LayerNorm version of the E=384 champion) with two changes:
 (1) REMOVE the dual stream: no emb_resid_lut, no per-layer residual_lut, no D-stream.
     Single residual stream x (dim E=384). Each block: LayerNorm(x) -> qk/v LUT -> RoPE
     softmax attention -> out_proj LUT -> x += out_proj. After layers: ln_final(x) ->
     unembedder -> logits (the final LayerNorm is KEPT).
 (2) REPLACE the Linear(384->32768) unembedder with HyperplaneCodeLUT (nap=15 -> V=32768,
     T=16): soft-sign per-code scores gated by w_cell, voted over 16 tables.
Everything else identical to exp010/exp018: 6 layers, 6 heads d_qk=d_v=64, ctx512, vocab
32768, RoPE1e4, untied tok_emb, LUT qk nap4/tph256, v nap6/tph256, out_proj nap7/tph512
(HyperplaneMHL, anchor-pairs, hard), LayerNorm everywhere, seed 42, 16k recipe. Much
smaller/cheaper than the exp010 dual-stream champion (baseline 1.1940).

ORIGINAL header follows:
================================================================================
exp018 — champion exp010 with the SINGLE change MeanAbsNorm -> LayerNorm at every norm site.

Byte-identical to exp010 (E=384 dual-stream, 6 layers, 6 heads x64, ctx 512, vocab 32768,
RoPE 1e4, untied embeddings, LUT qk/v/out/residual/emb_resid HyperplaneMHL, 16000 steps,
same optimizer/recipe) EXCEPT the three MeanAbsNorm(E) pre-norms (ln_pre, ln_resid,
ln_emb_resid) are replaced by nn.LayerNorm(E) (standard affine). ln_final was already
nn.LayerNorm and is unchanged. Baseline for comparison: exp010 best_val_bpb = 1.1940.

ORIGINAL exp010 header follows:
================================================================================
exp010 — exp752 recipe, LUT layers swapped for HyperplaneMultiHeadLUT.

A/B clone of exp752 (full ~277M-param LUT-GPT, 16K-step recipe). The ONLY
change vs exp752 is the LUT layer TYPE: every FastMultiHeadLUT site (qk_lut,
v_lut, out_proj, residual_lut, emb_resid_lut) is replaced by
HyperplaneMultiHeadLUT, which generalizes each fixed anchor-pair sign test to a
LEARNED affine hyperplane 1[<w_i,x> + b_i > 0]. Geometry (NAP/tph per site),
optimizer recipe (Lion on LUT weights, AdamW elsewhere), bf16 LUT + fp32 master
Lion, grad clip 1.0, 6 layers / dim 384 / 6 heads, 16000 steps — all IDENTICAL
to exp752.

Init: hyperplane_init="anchor_pairs" (w_i = e_p1 - e_p2, b_i = 0), which
reproduces the fixed-anchor FastMultiHeadLUT bit-for-bit at step 0, so this is a
strict, A/B-able generalization: it starts identical to exp752 and diverges only
as the hyperplanes learn. NOTE: the learned hyperplanes add ~48M trainable
params (hyperplane_weight [n_tables, NAP, input_dim] per site), so this is NOT
param-matched to exp752 (~277M -> ~325M); it tests learned-hyperplane vs
fixed-anchor front-ends, not iso-param.

The new hyperplane params (hyperplane_weight, hyperplane_bias) are routed to the
Adam-no-weight-decay group (lr=adam_lr), matching the hyperplane_ffn recipe
(exp006-009); LUT table weights stay on Lion exactly as in exp752. Set
lut_layer_type="fast" in config.json to recover exp752 behaviour verbatim.

ORIGINAL exp752 docstring follows:
================================================================================
exp752 — exp750 recipe extended to 16K steps.

Fork of exp750 (4K SOTA 1.3863, bf16 LUT storage + fp32 master Lion +
fp32 head + global clip(1.0)). Single change: n_steps 4000 -> 16000.
Same eff bs=48, same architecture, same optimizer recipe. Tests whether
today's clip(1.0) lever improves the 16K horizon too.

Closest 16K references:
  exp731 = 1.2178 @ 276.8M (same arch, no clip, fp32 master Lion)
  exp735 = 1.2138 @ 314.6M (exp731 + v_lut NAP=7 widening, +37.7M params)

If exp752 < 1.2178 cleanly, clip is a free win at long horizon too.
If exp752 < 1.2138, it beats the current 16K SOTA without the v_lut widening.

Expected wallclock ~3.8-4.0 h (4x exp750 since bf16 storage already gives
the speed). Eff bs = 48 sequences/step matches all prior 16K runs.

ORIGINAL exp737 docstring follows:

exp737 — bf16 weight_dtype on every FastMultiHeadLUT + fp32 master Lion.

Fork of exp732 (4K, val=1.3912 @ 276.8M). Changes:
 1. weight_dtype fp32 -> bf16 on qk_lut, v_lut, out_proj, residual_lut,
    emb_resid_lut (HBM bandwidth halved on weight reads).
 2. Lion optimizer now keeps an fp32 master copy + fp32 momentum for any
    bf16 param. Updates apply to the master in fp32, then copy.cast(bf16)
    back to the param so the forward still reads bf16 from HBM.
 3. FORCE_BMM_WGRAD=1 env-var: route v_lut wgrad through bmm-sparse-S
    instead of bf16 atomic-add scatter (bench shows -8 ms / step).

First exp737 attempt (without master) showed +5-9 mb persistent drift vs
exp732, growing from +0.005 nats at step 1 to +0.010 at step 100, then
holding +6.5 mb val_bpb at step 1400 — diagnosed as Lion-update rounding
landing in bf16 storage. This run tests whether fp32 master closes the gap.

Expected wallclock: ~62-65 min (-15% vs exp732's 73 min). Master copy adds
~50 % memory to LUT params (bf16+fp32+fp32-momentum = 10 B/weight vs
exp732's fp32+fp32-momentum = 8 B/weight).
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.hyperplane_multi_head_lut import HyperplaneMultiHeadLUT
from spiky.lutorch.hyperplane_code_lut import HyperplaneCodeLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
# exp101: multihead (no-collapse) CompressionMHL subclass, defined LOCALLY in this
# experiment dir — the shared src/spiky/lutorch/compression_mhl.py is NOT modified.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mh_compression import CompressionMultiHeadLUTMH

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

# --- exp101 out_proj sweep: env-driven per-run parameterization ---------------
# Sweeps ONLY the out_proj CompressionMHL (inner_in=OUT_IN, tph=OUT_TPH); q/k/v are
# held at the exp101 baseline (compress_inner_in=48 / compress_tph=32). Each run
# writes ALL its outputs into a per-run subdir OUT_DIR = EXP_DIR/RUN_TAG so 3
# concurrent runs never collide.
OUT_IN  = int(os.environ.get('OUT_IN', cfg.get('out_inner_in', 48)))
OUT_TPH = int(os.environ.get('OUT_TPH', cfg.get('out_tph', 32)))
RUN_TAG = os.environ.get('RUN_TAG', f'in{OUT_IN}_tph{OUT_TPH}')
OUT_DIR = os.path.join(EXP_DIR, RUN_TAG)
os.makedirs(OUT_DIR, exist_ok=True)
if os.environ.get('N_STEPS'):
    cfg['n_steps'] = int(os.environ['N_STEPS'])
cfg['exp_name'] = f'exp101_sweep_outproj_{RUN_TAG}'

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
_ROPE_BASE  = cfg.get('rope_base', 10000.0)


# --- Tokenizer + dataloader ---------------------------------------------------
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f'Vocab size: {VOCAB_SIZE}')

train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='train', device=DEVICE
)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE
)
token_bytes = get_token_bytes(device=DEVICE)


# --- LUT factories ------------------------------------------------------------
_WEIGHT_DTYPE = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
}[cfg.get('weight_dtype', 'fp32')]
_HYPER_DTYPE = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
}[cfg.get('hyperplane_dtype', 'fp32')]

# --- LUT layer TYPE selector (the ONE knob that differs from exp752) ----------
# 'fast'       -> FastMultiHeadLUT (fixed anchor-pair sign tests) == exp752.
# 'hyperplane' -> HyperplaneMultiHeadLUT (learned affine hyperplanes). With
#                 hyperplane_init="anchor_pairs" it reproduces 'fast' bit-for-bit
#                 at init, then learns the hyperplanes.
_LUT_LAYER = cfg.get('lut_layer_type', 'fast')

if _LUT_LAYER == 'hyperplane':
    _LUT_CLS = HyperplaneMultiHeadLUT
    # HyperplaneMultiHeadLUT has NO backward_mode (always full-K softmax
    # surrogate backward) and adds hyperplane_init / hyperplane_dtype.
    _LUT_KWARGS = dict(
        forward_mode=cfg.get('forward_mode', 'hard'),
        weight_dtype=_WEIGHT_DTYPE,
        hyperplane_dtype=_HYPER_DTYPE,
        hyperplane_init=cfg.get('hyperplane_init', 'anchor_pairs'),
        hyperplane_init_scale=cfg.get('hyperplane_init_scale', None),
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
        soft_score_temp=cfg.get('soft_score_temp', 0.5),
        select_temp=cfg.get('select_temp', 0.5),
        learnable_temps=cfg.get('soft_learnable_temps', True),
        use_bf16=cfg.get('soft_use_bf16', True),
    )
elif _LUT_LAYER == 'fast':
    _LUT_CLS = FastMultiHeadLut
    _LUT_KWARGS = dict(
        forward_mode=cfg.get('forward_mode', 'hard'),
        backward_mode=cfg.get('backward_mode', 'ball'),
        weight_dtype=_WEIGHT_DTYPE,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
        soft_score_temp=cfg.get('soft_score_temp', 0.5),
        select_temp=cfg.get('select_temp', 0.5),
        learnable_temps=cfg.get('soft_learnable_temps', True),
        use_bf16=cfg.get('soft_use_bf16', True),
    )
else:
    raise ValueError(f"lut_layer_type must be 'fast' or 'hyperplane', got {_LUT_LAYER!r}")

print(f'LUT layer type = {_LUT_LAYER} ({_LUT_CLS.__name__})'
      + (f", hyperplane_init={_LUT_KWARGS['hyperplane_init']}, "
         f"hyperplane_dtype={cfg.get('hyperplane_dtype', 'fp32')}"
         if _LUT_LAYER == 'hyperplane' else ''))

# --- exp101: SEPARATE q/k/v + out_proj built from CompressionMultiHeadLUT ------
# (NOT Hyperplane/FastMHL directly; the _LUT_CLS/_LUT_KWARGS plumbing below is left
#  only for the dead-code lineage factories and is unused by these attention sites.)
# q/k/v: compress E -> H*inner_in, per-head HARD FastMHL routing, NO decompress,
#        heads kept SEPARATE via multihead_output -> [N, H, d]. Each replaces one
#        of exp024's Hyperplane qk/v projections; qk is now split into q and k.
# out_proj: standard compress -> FastMHL -> decompress (heads collapse) -> [N, E].
_C_IN   = cfg.get('compress_inner_in', 48)     # LUT input width per head (compression target)
_C_NAP  = cfg.get('compress_nap', 6)           # routing bits per table (2**nap cells)
_C_TPH  = cfg.get('compress_tph', 32)          # tables per head
_OUT_H  = cfg.get('out_lut_heads', 8)          # out_proj internal LUT head count

def _make_q(seed_offset):
    return CompressionMultiHeadLUTMH(
        input_dim=E, output_dim=d_qk, inner_in_dim=_C_IN, inner_out_dim=-1,
        nap=_C_NAP, tph=_C_TPH, n_heads=H, multihead_output=True,
        forward_mode="hard", use_bf16=True,
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE)   # -> [N, H, d_qk]

def _make_k(seed_offset):
    return CompressionMultiHeadLUTMH(
        input_dim=E, output_dim=d_qk, inner_in_dim=_C_IN, inner_out_dim=-1,
        nap=_C_NAP, tph=_C_TPH, n_heads=H, multihead_output=True,
        forward_mode="hard", use_bf16=True,
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE)   # -> [N, H, d_qk]

def _make_v(seed_offset):
    return CompressionMultiHeadLUTMH(
        input_dim=E, output_dim=d_v, inner_in_dim=_C_IN, inner_out_dim=-1,
        nap=_C_NAP, tph=_C_TPH, n_heads=H, multihead_output=True,
        forward_mode="hard", use_bf16=True,
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE)   # -> [N, H, d_v]

def _make_out(seed_offset):
    # SWEPT: inner_in_dim=OUT_IN, tph=OUT_TPH. HELD: inner_out_dim=48, n_heads=8, nap=6.
    return CompressionMultiHeadLUT(
        input_dim=H * d_v, output_dim=E, inner_in_dim=OUT_IN, inner_out_dim=48,
        nap=_C_NAP, tph=OUT_TPH, n_heads=_OUT_H,
        forward_mode="hard", use_bf16=True,
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE)   # -> [N, E]

def _make_residual_lut(seed_offset):
    """Per-layer residual_lut: E -> D, accumulated into the D-stream."""
    return _LUT_CLS(
        input_dim=E, n_heads=1, n_outputs=D,
        n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
        **_LUT_KWARGS,
    )

def _make_emb_resid_lut(seed_offset):
    """UNUSED in exp019 (single stream). Kept for lineage."""
    return _LUT_CLS(
        input_dim=E, n_heads=1, n_outputs=D,
        n_anchor_pairs=cfg['emb_resid_input_nap'], tables_per_head=cfg['emb_resid_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
        **_LUT_KWARGS,
    )

def _make_unembedder(seed_offset):
    """exp019 unembedder = HyperplaneCodeLUT mapping the E-dim hidden DIRECTLY to
    V=2^nap logits (nap must satisfy 2^nap == VOCAB_SIZE). Per-code soft-sign scores
    gated by w_cell, voted over code_unemb_tables (T) tables; no stored per-cell rows.
    hyperplane_weight/bias -> AdamW no-wd (by name); w_cell -> AdamW wd0.1 group."""
    return HyperplaneCodeLUT(
        input_dim=E, nap=cfg['code_unemb_nap'], n_tables=cfg['code_unemb_tables'],
        n_outputs=VOCAB_SIZE,
        T_soft=cfg.get('soft_score_temp', 0.5),
        hyperplane_init=cfg.get('hyperplane_init', 'anchor_pairs'),
        hyperplane_init_scale=cfg.get('hyperplane_init_scale', None),
        w_cell_init=cfg.get('code_unemb_wcell_init', 0.02),
        initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
    )


# --- RoPE on (q, k) -----------------------------------------------------------
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
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)


class LUTBlock(nn.Module):
    """exp019 SINGLE-STREAM block (LayerNorm): LayerNorm -> qk/v LUT -> RoPE softmax
    attention -> out_proj LUT -> added back to the single residual stream x. No
    residual_lut, no D-stream (both removed vs exp010/exp018)."""
    def __init__(self, layer_idx):
        super().__init__()
        # exp101: SEPARATE q and k (was one joint qk_lut), all CompressionMHL.
        self.q_lut    = _make_q(layer_idx)
        self.k_lut    = _make_k(100 + layer_idx)
        self.v_lut    = _make_v(200 + layer_idx)
        self.out_proj = _make_out(400 + layer_idx)

        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.ln_pre = nn.LayerNorm(E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)

        x_pre = self.ln_pre(x_flat)

        # q_lut / k_lut each return [N, H, d_qk] (heads separate) via multihead_output.
        q_vec = self.q_norm(self.q_lut(x_pre).float())
        k_vec = self.k_norm(self.k_lut(x_pre).float())
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])

        v_vec = self.v_lut(x_pre).float()                 # [N, H, d_v]
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e = self.out_proj(out_in).squeeze(1).float()

        x_next = x_flat + out_e                     # single residual stream
        return x_next.reshape(B, T, E)


class Model(nn.Module):
    """exp019: single residual stream (dim E=384), LayerNorm everywhere. No
    emb_resid_lut, no D-stream. Readout keeps the final LayerNorm: after all layers,
    ln_final(x) -> HyperplaneCodeLUT unembedder -> 32768 logits (replaces the linear
    unembedder)."""
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(E)             # KEPT: norm before the unembedder
        self.unembedder = nn.Linear(E, VOCAB_SIZE, bias=False)   # exp023 control: plain Linear (was HyperplaneCodeLUT)

    def get_device(self):
        # Required by nanochat.loss_eval.evaluate_bpb.
        return self.tok_emb_E.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x = self.tok_emb_E(tokens)                       # [B,T,E] single stream
        for layer in self.layers:
            x = layer(x, self.rope.cos, self.rope.sin)   # [B,T,E]
        x_flat = self.ln_final(x.reshape(B * T, E))      # LayerNorm(E) before unembedder
        # exp023: plain Linear unembedder [B*T, E] -> [B*T, VOCAB]
        logits = self.unembedder(x_flat).float().view(B, T, VOCAB_SIZE)
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

if os.environ.get('SMOKE'):
    # Wiring/shape check only — no training. Verify per-head q/k/v shapes and logits.
    model.eval()
    blk = model.layers[0]
    xr = torch.randn(4, E, device=DEVICE)
    with torch.no_grad():
        q_dbg = blk.q_lut(model.layers[0].ln_pre(xr))
        v_dbg = blk.v_lut(model.layers[0].ln_pre(xr))
        o_dbg = blk.out_proj(torch.randn(4, H * d_v, device=DEVICE))
    print(f'[SMOKE] q_lut out shape = {tuple(q_dbg.shape)} (expect [4,{H},{d_qk}])')
    print(f'[SMOKE] v_lut out shape = {tuple(v_dbg.shape)} (expect [4,{H},{d_v}])')
    print(f'[SMOKE] out_proj out shape = {tuple(o_dbg.shape)} (expect [4,{E}])')
    assert tuple(q_dbg.shape) == (4, H, d_qk) and tuple(v_dbg.shape) == (4, H, d_v)
    assert tuple(o_dbg.shape) == (4, E)
    toks = torch.randint(0, VOCAB_SIZE, (2, CONTEXT_SIZE), device=DEVICE)
    with torch.no_grad():
        logits = model(toks)
    print(f'[SMOKE] full-model logits shape = {tuple(logits.shape)} (expect [2,{CONTEXT_SIZE},{VOCAB_SIZE}])')
    assert tuple(logits.shape) == (2, CONTEXT_SIZE, VOCAB_SIZE)
    print('[SMOKE] forward OK — separate q/k/v CompressionMHL wiring verified')
    model.train()
    # fall through to build the optimizers (verifies param grouping), then exit below.

def get_lr_scale(step):
    n = N_STEPS
    w = int(WARMUP_FRAC * n)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

lut_params        = []
hyperplane_params = []
tok_emb_params    = []
decay_params      = []
nodecay_params    = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    # HyperplaneMultiHeadLUT adds hyperplane_weight (ndim 3) and hyperplane_bias
    # (ndim 2). Route them by NAME (before the ndim rules) to the Adam-no-wd
    # group, matching the hyperplane_ffn recipe (exp006-009). Without this,
    # hyperplane_weight would fall into lut_params (Lion) and hyperplane_bias
    # into decay_params (AdamW wd=0.1) — both wrong. Empty for lut_layer_type
    # 'fast', so the 'fast' path stays byte-identical to exp752's grouping.
    if name.endswith('hyperplane_weight') or name.endswith('hyperplane_bias'):
        hyperplane_params.append(p)
    elif p.ndim >= 3:
        lut_params.append(p)
    elif name.startswith('tok_emb_E.'):
        tok_emb_params.append(p)
    elif p.ndim == 2:
        decay_params.append(p)
    else:
        nodecay_params.append(p)


class Lion(torch.optim.Optimizer):
    """Lion with fp32 master copy for bf16 params.

    When the LUT parameter is stored in bf16 (for HBM bandwidth), per-step Lion
    updates land in bf16 directly and lose precision over iterations (~5-9 mb
    bpb drift observed in the first exp737 attempt). Mitigation: keep an fp32
    master copy + fp32 momentum; apply the update in fp32 to the master, then
    copy.cast(bf16) back to the param so the forward still reads bf16 from HBM.
    """
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, (b1, b2), wd = grp['lr'], grp['betas'], grp['weight_decay']
            for p in grp['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                is_low = p.dtype != torch.float32
                if 'exp_avg' not in st:
                    st['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    if is_low:
                        st['master'] = p.detach().to(torch.float32).clone()
                m = st['exp_avg']
                g_f = p.grad if p.grad.dtype == torch.float32 else p.grad.to(torch.float32)
                if is_low:
                    master = st['master']
                    if wd != 0:
                        master.mul_(1.0 - lr * wd)
                    update = (m * b1 + g_f * (1.0 - b1)).sign_()
                    master.add_(update, alpha=-lr)
                    m.mul_(b2).add_(g_f, alpha=1.0 - b2)
                    p.data.copy_(master)
                else:
                    if wd != 0:
                        p.mul_(1.0 - lr * wd)
                    update = (m * b1 + g_f * (1.0 - b1)).sign_()
                    p.add_(update, alpha=-lr)
                    m.mul_(b2).add_(g_f, alpha=1.0 - b2)

_LUT_LR  = cfg.get('lut_lr', cfg['adam_lr'])
_LUT_OPT = cfg.get('lut_optimizer', 'adamw')

adam_groups = [
    dict(params=decay_params,   lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=cfg.get('weight_decay', 0.0)),
    dict(params=tok_emb_params + nodecay_params + hyperplane_params,
         lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
]
optimizer = torch.optim.AdamW(adam_groups)

if _LUT_OPT == 'lion':
    lut_optimizer = Lion([dict(params=lut_params, lr=_LUT_LR, weight_decay=0.0)],
                         lr=_LUT_LR, betas=tuple(cfg.get('lut_betas', (0.9, 0.99))))
else:
    lut_optimizer = torch.optim.AdamW(
        [dict(params=lut_params, lr=_LUT_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)])

all_optimizers = [optimizer, lut_optimizer]
for o in all_optimizers:
    for g in o.param_groups:
        g['initial_lr'] = g['lr']
print(
    f'LUT optimizer = {_LUT_OPT} (lut_lr={_LUT_LR}) | '
    f'lut={sum(p.numel() for p in lut_params):,} | '
    f'hyperplane={sum(p.numel() for p in hyperplane_params):,} (AdamW no-wd, lr={cfg["adam_lr"]}) | '
    f'decay(unembed)={sum(p.numel() for p in decay_params):,} (wd={cfg.get("weight_decay", 0.0)}) | '
    f'tok_emb={sum(p.numel() for p in tok_emb_params):,} | '
    f'nodecay_other={sum(p.numel() for p in nodecay_params):,} | non-LUT on AdamW'
)

print(f'D=residual_dim={D}, E=embedding_dim={E}, H={H}, d_qk={d_qk}, d_v={d_v}, L={N_LAYERS}')
print(f'exp101: SEPARATE q_lut/k_lut/v_lut = CompressionMHL(in={E}->out={d_qk}, inner_in={_C_IN}, '
      f'inner_out=-1, nap={_C_NAP}, tph={_C_TPH}, n_heads={H}, multihead_output=True) -> [N,{H},{d_qk}]')
print(f'[SWEEP {RUN_TAG}] out_proj = CompressionMHL(in={H*d_v}->out={E}, inner_in={OUT_IN} (SWEPT), '
      f'inner_out=48, nap={_C_NAP}, tph={OUT_TPH} (SWEPT), n_heads={_OUT_H}) -> [N,{E}] | out_dir={OUT_DIR}')
print(f'exp101 SINGLE-STREAM E={E} (no residual_lut / emb_resid / D-stream); LayerNorm everywhere')
if os.environ.get('SMOKE'):
    print('[SMOKE] optimizer grouping built OK — exiting before training'); sys.exit(0)
print(f'UNEMBEDDER plain Linear(E={E} -> V={VOCAB_SIZE}, bias=False) [exp023 control; was HyperplaneCodeLUT]; ln_final(E) KEPT before unembedder')
print(f'UNTIED unembedder Linear(D={D}, V={VOCAB_SIZE}); tok_emb_E at E={E}; ln_final(D); RoPE base={_ROPE_BASE}')


# --- Temperature tracking -----------------------------------------------------
def collect_temperature_specs(model):
    specs = []
    for li, blk in enumerate(model.layers):
        for slut_name in ('q_lut', 'k_lut', 'v_lut', 'out_proj'):
            mod = getattr(blk, slut_name, None)
            if mod is not None and getattr(mod, 'learnable_temps', False):
                specs.append((f'L{li}.{slut_name}.T_soft',
                              (lambda m=mod: float(m.log_soft_score_temp.detach().exp()))))
                specs.append((f'L{li}.{slut_name}.T_sel',
                              (lambda m=mod: float(m.log_select_temp.detach().exp()))))
    return specs

temp_specs = collect_temperature_specs(model)
temp_path = os.path.join(OUT_DIR, 'temperatures.csv')
temp_f = open(temp_path, 'w', newline='')
temp_w = csv.writer(temp_f)
temp_w.writerow(['step'] + [name for name, _ in temp_specs])
print(f'Tracking {len(temp_specs)} learnable temperatures in temperatures.csv')

# --- Per-parameter weight-delta tracking --------------------------------------
_PARAM_SNAPSHOT = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
_weight_csv_path = os.path.join(OUT_DIR, 'weight_deltas.csv')
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

csv_path = os.path.join(OUT_DIR, 'metrics.csv')
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
    for o in all_optimizers:
        for g in o.param_groups:
            g['lr'] = g['initial_lr'] * lr_scale

    for o in all_optimizers:
        o.zero_grad()
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, targets=y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum

    # Global grad clip across all parameters (Lion's sign-step ignores
    # magnitude; clip mainly affects the momentum buffer and the AdamW
    # update on non-LUT params).
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    for o in all_optimizers:
        o.step()

    ema = accum_loss if ema is None else 0.99 * ema + 0.01 * accum_loss

    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | loss={ema:.4f} | lr={lr_scale * cfg["adam_lr"]:.2e}')

    if step in (1, 5) and DEVICE == 'cuda':
        print(f'[MEM] step {step} alloc_peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB '
              f'reserved={torch.cuda.max_memory_reserved()/1e9:.1f}GB')

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
ax2.plot(val_steps, val_bpbs, label='val bpb', color='red')
ax2.set(xlabel='step', ylabel='bpb', title='Validation BPB')
ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'loss.png'), dpi=120)
plt.close(fig)

summary = dict(
    exp_name=cfg['exp_name'],
    best_val_bpb=best_bpb,
    final_val_bpb=val_bpbs[-1] if val_bpbs else float('nan'),
    n_params=n_params,
    training_time_hours=round(elapsed / 3600, 3),
)
with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

ckpt_path = os.path.join(OUT_DIR, 'checkpoint.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'lut_optimizer_state_dict': lut_optimizer.state_dict(),
    'config': cfg,
    'step': N_STEPS,
    'best_val_bpb': best_bpb,
    'final_val_bpb': val_bpbs[-1] if val_bpbs else float('nan'),
}, ckpt_path)
print(f'saved checkpoint -> {ckpt_path}')

print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
