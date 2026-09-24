"""LUT out_proj variant (BODY_TASK edae7f0f).

Both the attention out_proj AND the FFN of every block are CompressionMultiHeadLUT modules:

  block(x):  # standard two-norm pre-norm
    a  = attn_no_proj(ln1(x))            # ln1 pre-attention; QKV + RoPE + causal SDPA, NO out_proj Linear;
                                         # output kept head-separated -> [B,T,H*D] (head-major)
    op = out_proj_lut(a)                 # CompressionMultiHeadLUT, input_multi_head (no compress):
                                         # each head routes its own D=E/H slice -> [B,T,E]
    x  = x + op
    f  = ffn_lut(ln2(x))                 # CompressionMultiHeadLUT, compress E->H*D -> [B,T,E]
    x  = x + f

Wrapper: token embeddings -> L blocks -> final LayerNorm -> UNTIED unembedder (not weight-tied).

Reuses the tested RoPE from model_build.py. build_model(cfg, vocab_size, device) mirrors
MinimalGPT's forward interface exactly: forward(idx, targets=None, loss_reduction='mean') returns
the cross-entropy loss (per-token when loss_reduction='none', as tools/fixed_eval expects), or the
logits when targets is None.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_build import RotaryEmbedding, apply_rope          # tested RoPE (same tools/ dir)
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT


def _lut_cfg(cfg, role, key, default=None):
    """Per-LUT hyperparameter with independent config groups: `lut_<role>_<key>` (role in
    {'outproj','ffn'}) overrides the shared `lut_<key>`, which in turn falls back to `default`.
    The out_proj LUT and the FFN LUT thus have SEPARATE, INDEPENDENT hyperparameter sets; a config
    that sets only the shared keys keeps both LUTs identical (byte-identical to the pre-split build).
    Applies to: n_heads, inner_in_dim, inner_out_dim, tables_per_head, n_anchor_pairs, read_top_n,
    confidence_form, light_forward_mode, head_dropout_rate."""
    return cfg.get(f'lut_{role}_{key}', cfg.get(f'lut_{key}', default))


def _make_lut(n_embd, cfg, role, seed, *, input_multi_head, inner_in_dim, n_heads):
    """A CompressionMultiHeadLUT sized E -> E, hyperparameters resolved for `role` ('outproj'|'ffn').

    input_multi_head=True  (out_proj): inner_in_dim=-1 (no compress); each head routes its own
                                       E//n_heads slice of the pre-split attention output. n_heads is
                                       PINNED to the attention head count H (the slices must line up).
    input_multi_head=False (ffn):      inner_in_dim=D (compress E -> n_heads*D); n_heads is free.

    Regularisers (config-toggleable, default OFF, per-LUT via the lut_<role>_* override): LUT table
    dropout lut[_<role>]_head_dropout_rate -> LightMHL whole-table Bernoulli score-mask (train-only,
    off at eval); TV via cfg['lut_cell_smoothness'] applied by the trainer over every LightMHL table.
    """
    conf = _lut_cfg(cfg, role, 'confidence_form', 'margin')
    # learned_margin needs its (g, beta, gamma) init, in LightMHL's order (shared keys, as in abl_04/09).
    lm_init = None
    if conf == 'learned_margin':
        lm_init = (float(cfg.get('lut_learned_margin_g_init', 0.0)),
                   float(cfg.get('lut_learned_margin_beta_init', 2.0)),
                   float(cfg.get('lut_learned_margin_gamma_init', 1.0)))
    return CompressionMultiHeadLUT(
        input_dim=n_embd, output_dim=n_embd,
        inner_in_dim=inner_in_dim, inner_out_dim=int(_lut_cfg(cfg, role, 'inner_out_dim')),
        nap=int(_lut_cfg(cfg, role, 'n_anchor_pairs')), tph=int(_lut_cfg(cfg, role, 'tables_per_head')),
        n_heads=n_heads, lut_impl='light',
        forward_confidence=True,
        confidence_form=conf,
        confidence_gain=float(cfg.get('lut_confidence_gain', 1.0)),
        learned_margin_init=lm_init,
        learned_margin_freeze_g=bool(_lut_cfg(cfg, role, 'learned_margin_freeze_g', False)),
        light_forward_mode=_lut_cfg(cfg, role, 'light_forward_mode', 'scored'),
        read_top_n=int(_lut_cfg(cfg, role, 'read_top_n', 2)),       # "2 alternatives"
        read_tau=float(_lut_cfg(cfg, role, 'read_tau', 0.1)),
        read_tau_learnable=bool(_lut_cfg(cfg, role, 'read_tau_learnable', False)),
        z_norm=False,
        input_multi_head=input_multi_head,
        head_dropout_rate=float(_lut_cfg(cfg, role, 'head_dropout_rate', 0.0)),  # LUT table dropout (off by default)
        random_seed=seed,
    )


class AttentionNoProj(nn.Module):
    """Standard multi-head attention up to (and excluding) out_proj; NO normalization.

    Returns the head-separated output flattened to [B, T, H*D] (head-major), ready to be fed
    straight into the out_proj-replacement LUT.
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # [B, H, T, D]
        return y.transpose(1, 2).contiguous().view(B, T, C)          # [B, T, H*D] head-major


class OutProjLUTBlock(nn.Module):
    """Standard two-norm pre-norm block, with attention out_proj and the FFN both LUTs:
        x = x + out_proj_lut( attn( ln1(x) ) )     # ln1 pre-attention; covers the whole attn->out_proj branch
        x = x + ffn_lut( ln2(x) )                   # ln2 pre-FFN
    There is NO separate norm directly before the out_proj LUT (ln1 sits before attention) and NO
    out_proj Linear (the attention output stays head-separated and feeds the out_proj LUT)."""
    def __init__(self, n_embd, n_head, layer_idx, cfg):
        super().__init__()
        base = int(cfg.get('lut_base_seed', 1000))
        self.ln1 = nn.LayerNorm(n_embd)   # pre-attention norm (covers attention -> out_proj LUT)
        self.attn = AttentionNoProj(n_embd, n_head)
        # out_proj LUT: its n_heads is PINNED to the attention head count H (input_multi_head splits
        # the E-dim attention output into H equal per-head slices). Its params come from the
        # lut_outproj_* / shared keys, INDEPENDENT of the FFN LUT.
        self.out_proj_lut = _make_lut(n_embd, cfg, 'outproj', seed=base + 2 * layer_idx,
                                      input_multi_head=True, inner_in_dim=-1, n_heads=n_head)
        # ffn LUT: LayerNorm then compress E -> n_heads*inner_in. n_heads is free (lut_ffn_n_heads).
        self.ln2 = nn.LayerNorm(n_embd)   # pre-FFN norm
        self.ffn_lut = _make_lut(n_embd, cfg, 'ffn', seed=base + 2 * layer_idx + 1,
                                 input_multi_head=False,
                                 inner_in_dim=int(_lut_cfg(cfg, 'ffn', 'inner_in_dim')),
                                 n_heads=int(_lut_cfg(cfg, 'ffn', 'n_heads')))

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        a = self.attn(self.ln1(x), cos, sin)                          # ln1 -> attn, head-major [B,T,C]
        op = self.out_proj_lut(a.reshape(B * T, C)).view(B, T, C)     # out_proj replacement
        x = x + op
        f = self.ffn_lut(self.ln2(x).reshape(B * T, C)).view(B, T, C)
        x = x + f
        return x


class MinimalGPTOutProj(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        n_embd, n_head, n_layer, seq_len = cfg['n_embd'], cfg['n_head'], cfg['depth'], cfg['seq_len']
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head, max_seq_len=seq_len)
        self.blocks = nn.ModuleList([OutProjLUTBlock(n_embd, n_head, i, cfg) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)          # UNTIED unembedder
        self.apply(self._init_weights)
        # Decompress (output-projection) init per sub-block:
        #  * out_proj LUT: SMALL NON-ZERO so the attention signal propagates through the out_proj LUT
        #    from step 1 (muting it at init starved attention). GPT-style residual/output-projection
        #    scaling std = 0.02 / sqrt(2 * n_layer) keeps the residual variance controlled and the
        #    init loss near ln(vocab).
        #  * FFN LUT: keep decompress.weight zeroed (FFN sub-block starts muted, trains up).
        #  * BOTH: zero the decompress BIAS -- the default nn.Linear bias (uniform ±1/sqrt(fan_in))
        #    would otherwise add a token-INDEPENDENT constant down the residual stream at init (cf. the
        #    bh4-path bias-zeroing note in model_build.py). Zeroing it gives a clean start.
        resid_std = 0.02 / math.sqrt(2 * n_layer)
        for block in self.blocks:
            if hasattr(block.out_proj_lut.decompress, 'weight'):
                nn.init.normal_(block.out_proj_lut.decompress.weight, std=resid_std)
            if hasattr(block.ffn_lut.decompress, 'weight'):
                nn.init.zeros_(block.ffn_lut.decompress.weight)
            for lut in (block.out_proj_lut, block.ffn_lut):
                if getattr(lut.decompress, 'bias', None) is not None:
                    nn.init.zeros_(lut.decompress.bias)
        # Deliberately NOT weight-tied: self.head stays independent of self.tok_emb.

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def get_device(self):
        return self.tok_emb.weight.device

    # --- TV (Hamming-1 cell-smoothness) regulariser, mirrors MinimalGPT's wiring. Covers every
    #     LightMultiHeadLUT table (both the out_proj LUT and the FFN LUT). The trainer applies
    #     cfg['lut_cell_smoothness'] * lut_tv_penalty() once per step, only when that knob > 0. ---
    def lut_tv_modules(self):
        from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
        return [m for m in self.modules() if isinstance(m, LightMultiHeadLUT)]

    def lut_tv_penalty(self):
        ms = self.lut_tv_modules()
        if not ms:
            return torch.zeros((), device=self.get_device())
        return torch.stack([m.cell_tv() for m in ms]).mean()

    @torch.no_grad()
    def lut_tv_by_layer(self):
        covered = {id(m) for m in self.lut_tv_modules()}
        out = []
        for b in self.blocks:
            ms = [m for m in b.modules() if id(m) in covered]
            if ms:
                out.append(float(torch.stack([m.cell_tv() for m in ms]).mean()))
        return out

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
    return MinimalGPTOutProj(vocab_size, cfg).to(device)
