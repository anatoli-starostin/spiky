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
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_build import RotaryEmbedding, apply_rope          # tested RoPE (same tools/ dir)
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT


def _make_lut(n_embd, cfg, seed, *, input_multi_head, inner_in_dim):
    """A CompressionMultiHeadLUT sized E -> E with the shared light-path config.

    input_multi_head=True  (out_proj): inner_in_dim=-1 (no compress); each head routes its own
                                       E//n_heads slice of the pre-split attention output.
    input_multi_head=False (ffn):      inner_in_dim=D (compress E -> n_heads*D).

    Regularisers (config-toggleable, default OFF, same wiring as the FFN-replacement runs):
      * LUT table dropout: cfg['lut_head_dropout_rate'] (default 0.0) -> LightMHL whole-table
        Bernoulli dropout on the per-table confidence score (keep 1-p, survivors /(1-p), train-only,
        off at eval). Reaches BOTH the out_proj LUT and the FFN LUT.
      * TV (Hamming-1 cell smoothness): cfg['lut_cell_smoothness'] (default 0.0) is applied by the
        trainer as lambda * model.lut_tv_penalty() over every LightMHL table (see lut_tv_* below).
    """
    return CompressionMultiHeadLUT(
        input_dim=n_embd, output_dim=n_embd,
        inner_in_dim=inner_in_dim, inner_out_dim=int(cfg['lut_inner_out_dim']),
        nap=int(cfg['lut_n_anchor_pairs']), tph=int(cfg['lut_tables_per_head']),
        n_heads=int(cfg['lut_n_heads']), lut_impl='light',
        forward_confidence=True,
        confidence_form=cfg.get('lut_confidence_form', 'margin'),
        light_forward_mode=cfg.get('lut_light_forward_mode', 'scored'),
        read_top_n=int(cfg.get('lut_read_top_n', 2)),       # "2 alternatives"
        z_norm=False,
        input_multi_head=input_multi_head,
        head_dropout_rate=float(cfg.get('lut_head_dropout_rate', 0.0)),   # LUT table dropout (off by default)
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
        self.out_proj_lut = _make_lut(n_embd, cfg, seed=base + 2 * layer_idx,
                                      input_multi_head=True, inner_in_dim=-1)
        # ffn replacement: LayerNorm then the LUT (compress E -> H*D).
        self.ln2 = nn.LayerNorm(n_embd)   # pre-FFN norm
        self.ffn_lut = _make_lut(n_embd, cfg, seed=base + 2 * layer_idx + 1,
                                 input_multi_head=False, inner_in_dim=int(cfg['lut_inner_in_dim']))

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
        # Zero every LUT decompress at init so each sub-block emits 0 -> block is identity at the
        # start and the residual stream is clean (same idea as MinimalGPT zeroing attn.proj / the
        # FFN decompress). The LUT tables train up from there.
        for block in self.blocks:
            for lut in (block.out_proj_lut, block.ffn_lut):
                if getattr(lut, 'has_decompress', False) and hasattr(lut.decompress, 'weight'):
                    nn.init.zeros_(lut.decompress.weight)
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
