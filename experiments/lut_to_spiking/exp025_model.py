"""Reconstruct the exp025 model (single-stream, Linear unembedder, FIXED FastMHL anchors)
from its checkpoint, so we can run real data through it and tap a real LUT table.

Architecture read off the checkpoint's own config + state-dict keys:
  tok_emb_E [V,E] -> N_LAYERS x LUTBlock -> ln_final(LayerNorm E) -> unembedder Linear(E,V)
  LUTBlock: ln_pre(LayerNorm E) -> qk_lut / v_lut -> q_norm/k_norm + RoPE -> SDPA
            -> out_proj(FastMHL, input = H*d_v) -> x = x + out_proj(attn)
(single stream: the checkpoint has no residual_lut / emb_resid_lut and ln_pre carries
 affine parameters, so it is a LayerNorm, not MeanAbsNorm.)

Acceptance test: val bpb must reproduce the checkpoint's recorded 1.2408.
"""
import os, sys, math, torch
import torch.nn as nn
import torch.nn.functional as F

from paths import NANOCHAT_ROOT, EXP025_CKPT
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

CKPT = EXP025_CKPT
DEVICE = "cuda"


def _rope_tables(head_dim, max_seq_len, base, device):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Exp025(nn.Module):
    def __init__(self, cfg, device=DEVICE):
        super().__init__()
        self.cfg = cfg
        E = cfg["embedding_dim"]; H = cfg["n_heads"]
        self.E, self.H = E, H
        self.d_qk, self.d_v = cfg["d_qk"], cfg["d_v"]
        V = 32768
        wd = {"fp32": torch.float32, "bf16": torch.bfloat16}[cfg.get("weight_dtype", "bf16")]
        kw = dict(weight_dtype=wd,
                  anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
                  initial_weights_noise=cfg.get("mhlut_init_std", 0.001),
                  forward_mode=cfg.get("forward_mode", "hard"),
                  soft_score_temp=cfg.get("soft_score_temp", 0.5),
                  select_temp=cfg.get("select_temp", 0.5),
                  learnable_temps=cfg.get("soft_learnable_temps", True),
                  use_bf16=cfg.get("soft_use_bf16", True))
        self.tok_emb_E = nn.Embedding(V, E)
        self.layers = nn.ModuleList()
        for i in range(cfg["num_layers"]):
            blk = nn.Module()
            blk.qk_lut = FastMultiHeadLut(
                input_dim=E, n_heads=H, n_outputs=2 * self.d_qk,
                n_anchor_pairs=cfg["qkv_input_nap"], tables_per_head=cfg["qkv_tph"],
                random_seed=cfg["random_seed"] + i, device=device, **kw)
            blk.v_lut = FastMultiHeadLut(
                input_dim=E, n_heads=H, n_outputs=self.d_v,
                n_anchor_pairs=cfg["v_input_nap"], tables_per_head=cfg["v_tph"],
                random_seed=cfg["random_seed"] + 200 + i, device=device, **kw)
            blk.out_proj = FastMultiHeadLut(
                input_dim=H * self.d_v, n_heads=1, n_outputs=E,
                n_anchor_pairs=cfg["out_input_nap"], tables_per_head=cfg["out_tph"],
                random_seed=cfg["random_seed"] + 400 + i, device=device, **kw)
            blk.q_norm = nn.LayerNorm(self.d_qk)
            blk.k_norm = nn.LayerNorm(self.d_qk)
            blk.ln_pre = nn.LayerNorm(E)
            self.layers.append(blk)
        self.ln_final = nn.LayerNorm(E)
        self.unembedder = nn.Linear(E, V, bias=False)
        cos, sin = _rope_tables(self.d_qk, cfg["context_size"], cfg["rope_base"], device)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def get_device(self):
        return self.tok_emb_E.weight.device

    def block_forward(self, blk, x):
        B, T, E = x.shape
        H, d_qk, d_v = self.H, self.d_qk, self.d_v
        xf = x.reshape(B * T, E)
        x_pre = blk.ln_pre(xf)
        qk = blk.qk_lut(x_pre).float()
        q = blk.q_norm(qk[..., :d_qk]).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = blk.k_norm(qk[..., d_qk:2 * d_qk]).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, self.rope_cos[..., :T, :], self.rope_sin[..., :T, :])
        v = blk.v_lut(x_pre).float().reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e = blk.out_proj(out_in).squeeze(1).float()
        return (xf + out_e).reshape(B, T, E)

    def forward(self, tokens, targets=None, loss_reduction="mean"):
        x = self.tok_emb_E(tokens)
        for blk in self.layers:
            x = self.block_forward(blk, x)
        logits = self.unembedder(self.ln_final(x))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction=loss_reduction, ignore_index=-1)
        return logits


def load_exp025(device=DEVICE, ckpt=CKPT):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = Exp025(d["config"], device=device).to(device)
    missing, unexpected = m.load_state_dict(d["model_state_dict"], strict=False)
    real_missing = [k for k in missing if not k.startswith("rope_")]
    assert not real_missing, f"missing keys: {real_missing[:8]}"
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"
    m.eval()
    return m, d
