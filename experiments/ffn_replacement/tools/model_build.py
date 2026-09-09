"""Config-driven model builder for the ffn_replacement runs — shared by the fixed-eval
trainer (`train_fixed.py`) and the standalone scorer (`tools/score_checkpoint.py`) so both
rebuild the *identical* architecture from a run's `config.json`.

MinimalGPT + RoPE. The FFN slot of every block is one of:
  * ffn_type="dense"       -> vanilla 384->1536->384 GELU MLP (baselines).
  * ffn_type="compression" -> CompressionMultiHeadLUT (+ optional parallel Linear when gamma=1).
      - ffn_lut_kind (default "compression") may be "fastmhl_raw" -> a raw FastMultiHeadLut
        FFN driven by the config's raw_nap / raw_tph / raw_n_heads keys (used by
        exp_n_0136; note its top-level lut_* keys are inert in that case).
Unembedder is tied (head.weight = tok_emb.weight) when tie_unembedder=True, else untied.

The module structure / attribute names match the original per-run train.py exactly, so a
checkpoint saved by those trainers loads into this model with 0 missing / 0 unexpected keys.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut          # noqa: E402
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT       # noqa: E402
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy              # noqa: E402


DEFAULT_READ_TAU = 0.1

# delta_m measured on exp_g_0193 (the standard config: margin, no z_norm, nap8/tph128) over
# 8,192 real val tokens by diag_margin_gap.py -- the per-layer median of m_(1), the smallest
# of the nap anchor margins, which IS the cost gap the n=2 blend has to discriminate.
#
# THESE ARE delta_m ITSELF, not a multiple of it. The blend-weight sensitivity
# |dw/dtau| ~ (2c/tau^2) w(1-w) is maximised at tau* = 2c/z*, where z* = 2.399379 maximises
# z^2 sigmoid(z) sigmoid(-z) -- i.e. tau* = 0.8335 * delta_m, a pure number independent of
# scale. So tau = delta_m sits at 95.6% of peak sensitivity, which is why the plain measured
# gap is the right init and no correction factor is applied. For contrast, tau = 2*delta_m
# would give 44.8% -- a factor-of-2 slip here halves the routing gradient.
#
# PER-LAYER, and the measurement forces it. delta_m spans 3.3x across depth
# (0.0331 -> 0.1079), so the current flat default of 0.1 sits at only 22.4% of peak
# sensitivity at layer 0 while reaching 98.5% at layer 5 -- badly mismatched exactly where
# the routing deficit was measured. Within a layer it is flat (per-head medians span
# 1.01-1.22x, across-table CV 0.017-0.021), so one scalar per layer is enough and per-head
# or per-table would buy nothing.
MEASURED_TAU_G0193 = [0.03309, 0.07241, 0.07801, 0.08158, 0.09190, 0.10786]


def _read_tau_for_layer(cfg, layer_idx: int) -> float:
    """Resolve `lut_read_tau` for one layer: float | per-layer list | "auto".

    "auto" means "initialise from the measured margin gap" and resolves against
    `lut_read_tau_measured` in the config when present, else the exp_g_0193 measurement
    above. A maker SHOULD resolve "auto" to explicit numbers at config-creation time so
    config.json records what actually ran; this path is the fallback, and the trainer
    writes the resolved per-layer values into summary.json either way.
    """
    v = cfg.get('lut_read_tau', DEFAULT_READ_TAU)
    if isinstance(v, str):
        if v != 'auto':
            raise ValueError(f"lut_read_tau must be a number, a per-layer list, or 'auto'; "
                             f"got {v!r}")
        table = cfg.get('lut_read_tau_measured', MEASURED_TAU_G0193)
        if len(table) != cfg['depth']:
            raise ValueError(f"lut_read_tau='auto' needs one measured tau per layer: got "
                             f"{len(table)} for depth {cfg['depth']}")
        return float(table[layer_idx])
    if isinstance(v, (list, tuple)):
        if len(v) != cfg['depth']:
            raise ValueError(f"lut_read_tau list must have one entry per layer: got "
                             f"{len(v)} for depth {cfg['depth']}")
        return float(v[layer_idx])
    return float(v)


def resolved_read_taus(cfg):
    """The per-layer taus this config will actually build with -- for summary.json."""
    return [_read_tau_for_layer(cfg, i) for i in range(cfg['depth'])]


def _anchor_policy(cfg):
    """Map optional config key 'lut_anchor_policy' (string) to an AnchorSamplingPolicy,
    or None to keep each module's default (CANONICAL_FULL_COVERAGE)."""
    v = cfg.get('lut_anchor_policy')
    return AnchorSamplingPolicy(v) if v else None


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
    def __init__(self, n_embd, n_head, layer_idx, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn_type = cfg.get('ffn_type', 'compression')
        gamma = int(cfg.get('gamma', 0))
        if self.ffn_type == 'dense':
            # Activation configurable; default 'gelu' == the historical dense baseline
            # (byte-identical), 'relu' for the ReLU ablation. Same widths / no bias / no
            # extra params either way.
            _act = {'gelu': nn.GELU, 'relu': nn.ReLU}[cfg.get('dense_activation', 'gelu')]
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd, bias=False), _act(),
                nn.Linear(4 * n_embd, n_embd, bias=False))
        else:
            self.lin = nn.Linear(n_embd, n_embd, bias=True) if gamma == 1 else None
            fwd = cfg.get('lut_forward_mode', 'hard')
            bf16 = cfg.get('lut_use_bf16', False)
            noise = cfg.get('lut_init_weights_noise', 1e-3)
            learn = bool(cfg.get('lut_learnable_temps', False))
            seed = cfg.get('lut_base_seed', 1000) + layer_idx
            if cfg.get('ffn_lut_kind', 'compression') == 'fastmhl_raw':
                # Raw FastMHL: no compress/decompress; tables emit full n_embd. (exp_n_0136)
                self.ffn = FastMultiHeadLut(
                    input_dim=n_embd, n_heads=int(cfg['raw_n_heads']), n_outputs=n_embd,
                    n_anchor_pairs=int(cfg['raw_nap']), tables_per_head=int(cfg['raw_tph']),
                    forward_mode=fwd, backward_topk=cfg.get('lut_backward_topk', 0),
                    anchor_sampling_policy=_anchor_policy(cfg),
                    use_bf16=bf16, initial_weights_noise=noise,
                    learnable_temps=learn, random_seed=seed)
            else:
                self.ffn = CompressionMultiHeadLUT(
                    input_dim=n_embd, output_dim=n_embd,
                    inner_in_dim=cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim')),
                    inner_out_dim=cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim')),
                    nap=cfg['lut_n_anchor_pairs'], tph=cfg['lut_tables_per_head'],
                    n_heads=cfg.get('lut_n_heads', 1),
                    joint_head_compression=cfg.get('lut_joint_head_compression', False),
                    forward_mode=fwd, backward_topk=cfg.get('lut_backward_topk', 0),
                    anchor_sampling_policy=_anchor_policy(cfg),
                    use_bf16=bf16, initial_weights_noise=noise,
                    learnable_temps=learn, random_seed=seed,
                    # LookupFFN-line knobs; both default to the pre-existing behaviour
                    lut_impl=cfg.get('lut_impl', 'fast'),
                    forward_confidence=cfg.get('lut_forward_confidence', False),
                    # DEFAULT CHANGED to 'margin' (was 'bounded'). Verified safe against
                    # the committed record: of 89 run configs, 28 have the gate on and 21
                    # are on the light path, and EVERY one of them sets
                    # lut_confidence_form explicitly -- zero rely on this fallback, so no
                    # historical run rebuilds differently. See LIGHTMHL_SURVEY.md
                    # "Current standard configuration".
                    confidence_form=cfg.get('lut_confidence_form', 'margin'),
                    confidence_gain=cfg.get('lut_confidence_gain', 1.0),
                    # Optional skip INSIDE the FFN: decompress(lut(z) + z). Adds no
                    # parameters and requires eff_in == eff_out. Default False, so every
                    # existing config builds a bit-identical model to before this line
                    # existed (verified by param/buffer sha256 on exp_n_0185's config).
                    inner_residual=cfg.get('lut_inner_residual', False),
                    # LayerNorm on the compressed code before the lookup; default False so
                    # every existing config builds a bit-identical model.
                    z_norm=cfg.get('lut_z_norm', False),
                    # BH4 shape, read only when lut_impl == 'bh4'. The defaults are the
                    # reference implementation's n_factors=4 and our parity-matched block.
                    bh4_block=cfg.get('lut_bh4_block', 4),
                    bh4_factors=cfg.get('lut_bh4_factors', 4),
                    # Top-n blended read-out on the light path (default 1 = single cell,
                    # so every existing config builds a bit-identical model). n>1 makes
                    # the blend weights differentiable in the margins, which is a
                    # DIRECTIONAL routing gradient plain Light does not have.
                    read_top_n=cfg.get('lut_read_top_n', 1),
                    read_tau=_read_tau_for_layer(cfg, layer_idx),
                    read_tau_learnable=bool(cfg.get('lut_read_tau_learnable', False)),
                    # Single-anchor addressing (light path). Default 'pair' == unchanged, so
                    # every existing config builds a bit-identical model. 'single' makes each
                    # address bit the sign of ONE pooled coordinate instead of a pair diff.
                    anchor_mode=cfg.get('lut_anchor_mode', 'pair'),
                    pool_size=cfg.get('lut_pool_size', None),
                    anchor_unique_partition=bool(cfg.get('lut_anchor_unique_partition', False)),
                    # Lookup-gated cell: 'constant' (default, unchanged) | 'gated_affine'
                    # (u+v⊙x) | 'gated_multiply' (v⊙x).
                    cell_mode=cfg.get('lut_cell_mode', 'constant'),
                    margin_signed=bool(cfg.get('lut_margin_signed', True)))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        h = self.ln2(x)
        if self.ffn_type == 'dense':
            return x + self.mlp(h)
        B, T, C = h.shape
        o = self.ffn(h.reshape(B * T, C))
        if o.dim() == 3:                 # raw FastMHL returns [N, n_heads, C]; sum heads
            o = o.sum(dim=1)
        o = o.reshape(B, T, C).to(h.dtype)
        if self.lin is not None:
            o = o + self.lin(h)
        return x + o


class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        n_embd, n_head, n_layer, seq_len = cfg['n_embd'], cfg['n_head'], cfg['depth'], cfg['seq_len']
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head, max_seq_len=seq_len)
        self.blocks = nn.ModuleList([MinimalBlock(n_embd, n_head, i, cfg) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            if block.ffn_type == 'dense':
                nn.init.zeros_(block.mlp[-1].weight)
            else:
                if getattr(block.ffn, 'has_decompress', False) \
                        and hasattr(block.ffn.decompress, 'weight'):
                    # codebook read-out makes decompress an Identity (its M decode lives in
                    # LightMHL, deliberately init'd small rather than zeroed) -> nothing to zero.
                    nn.init.zeros_(block.ffn.decompress.weight)
                    # On the BH4 path the decompress BIAS must be zeroed too. Zeroing only
                    # the weight leaves the branch emitting its bias (norm ~0.82 at init),
                    # a token-INDEPENDENT constant that accumulates down the residual
                    # stream; LayerNorm removes each token's mean across dimensions, not a
                    # direction shared by every token. Anchor-pair addressing cancels that
                    # offset in d = z[a] - z[b] and never noticed, but BH4 signs the
                    # coordinates themselves and dies on it. Measured at init on real
                    # tokens: with the bias left alone the fraction of code coordinates
                    # whose sign never flips runs 0.00/0.14/0.28/0.38/0.43/0.49 by depth
                    # and layer 5 reaches only 9.8 of its 128 addresses; with it zeroed
                    # every layer reads 0.0000 constant and 127.6/128 addresses. Scoped to
                    # lut_impl='bh4' so every existing config still builds bit-identically.
                    if cfg.get('lut_impl', 'fast') == 'bh4' \
                            and block.ffn.decompress.bias is not None:
                        nn.init.zeros_(block.ffn.decompress.bias)
                if getattr(block, 'lin', None) is not None:
                    nn.init.zeros_(block.lin.weight)
        if bool(cfg.get('tie_unembedder', False)):
            self.head.weight = self.tok_emb.weight

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


def build_model(cfg, vocab_size, device='cuda'):
    """Build a MinimalGPT from a run's config dict and move it to `device`."""
    return MinimalGPT(vocab_size, cfg).to(device)
