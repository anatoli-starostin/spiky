"""Extract (concat[384], logits[vocab]) pairs from exp174's unembedder.

Loads the exp174 checkpoint, runs forward on training data, captures the input
to `unembedder` (= concat of layer outputs) and the corresponding logits.
Saves N pairs to parquet for offline distillation experiments.

Defaults: 100K samples, parquet at ./pairs.parquet.

Usage:
    python nanochat_exps/distill_unembedder/extract_pairs.py [--n 100000]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import VectorToDominance, DominanceToVector
from spiky.lutorch.bit_attention import BitAttention


# ============================================================
# Config / globals (replicate exp174)
# ============================================================
EXP174_DIR = '/home/starost/spiky/nanochat_exps/exp174_lut_qk_big_unembed'
with open(os.path.join(EXP174_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E           = cfg['embedding_dim']
H           = cfg['n_heads']
d_qk        = cfg['d_qk']
d_v         = cfg['d_v']
N_LAYERS    = cfg['num_layers']
D_QK_P      = d_qk * (d_qk - 1) // 2
DEVICE_BS   = cfg['device_batch_size']

_POS_EMB_CFG = cfg.get('pos_emb_dim', 0)
_POS_EMB_ACTIVE = isinstance(_POS_EMB_CFG, int) and _POS_EMB_CFG > 0
def _pos_emb_dim(layer_idx):
    return _POS_EMB_CFG if _POS_EMB_ACTIVE else E

# ============================================================
# Tokenizer + dataloader (need VOCAB_SIZE / iter)
# ============================================================
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f'Vocab size: {VOCAB_SIZE}')

train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='train', device=DEVICE
)


# ============================================================
# Model (copied from exp174)
# ============================================================
_TINY_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
)

def _make_qk_joint(layer_idx, seed_offset):
    n_inputs = E + (_pos_emb_dim(layer_idx) if _POS_EMB_ACTIVE else 0)
    return TinyMultiHeadLut(
        input_dim=n_inputs, n_heads=H, n_outputs=2 * d_qk,
        n_anchor_pairs=cfg['qk_input_nap'], tables_per_head=cfg['qk_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_TINY_KWARGS,
    )

def _make_v(layer_idx, seed_offset):
    return TinyMultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_TINY_KWARGS,
    )

_OUT_TPH_PER_LAYER = cfg.get('out_tph_per_layer')
def _make_out(layer_idx, seed_offset):
    tph = _OUT_TPH_PER_LAYER[layer_idx] if _OUT_TPH_PER_LAYER is not None else cfg['out_tph']
    return TinyMultiHeadLut(
        input_dim=H * d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'], tables_per_head=tph,
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_TINY_KWARGS,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qk_joint = _make_qk_joint(layer_idx, layer_idx)
        self.v_lut    = _make_v(layer_idx, 200 + layer_idx)
        self.out_proj = _make_out(layer_idx, 400 + layer_idx)
        canon_t = cfg.get('canon_temperature', 0.1)
        self.qk_v2d = VectorToDominance(d_qk, smooth_mode=False, temperature=canon_t)
        self.out_v2d = VectorToDominance(E, smooth_mode=False, temperature=canon_t)
        self.out_d2v = DominanceToVector(E, normalise=True)
        self.pos_dim = _pos_emb_dim(layer_idx)
        if _POS_EMB_ACTIVE:
            self.qk_input_ln = nn.LayerNorm(E + self.pos_dim)
        self.attn_scale = nn.Parameter(torch.tensor(
            float(cfg.get('learnable_attn_scale_init', 0.25))
        ))
        self.bit_attn = BitAttention(D_QK_P)
        self._inv_sqrt_p = float(1.0 / math.sqrt(D_QK_P))

    def forward(self, x, pos_emb):
        B, T, _ = x.shape
        if _POS_EMB_ACTIVE:
            pos = pos_emb.unsqueeze(0).expand(B, -1, -1)
            xp = torch.cat([x, pos], dim=-1)
            xp = self.qk_input_ln(xp).reshape(B * T, E + self.pos_dim)
        else:
            xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, E)
        x_flat = x.reshape(B * T, E)
        qk_out = self.qk_joint(xp)
        q_vec = qk_out[..., :d_qk]; k_vec = qk_out[..., d_qk:]
        q_dom = self.qk_v2d(q_vec); k_dom = self.qk_v2d(k_vec)
        q = q_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        k = k_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        v_vec = self.v_lut(x_flat)
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        effective_scale = self.attn_scale * self._inv_sqrt_p
        attn = self.bit_attn(q, k, v, is_causal=True, scale=effective_scale)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_real = self.out_proj(out_in).squeeze(1)
        out_dom  = self.out_v2d(out_real)
        out_rank = self.out_d2v(out_dom).reshape(B, T, E)
        return out_rank


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        _pos_dim_fn = _pos_emb_dim if _POS_EMB_ACTIVE else (lambda i: E)
        _pos_init_scale = cfg.get('pos_emb_init_scale', 0.1)
        self.pos_embs = nn.ParameterList([
            nn.Parameter(torch.randn(CONTEXT_SIZE, _pos_dim_fn(i)) * _pos_init_scale)
            for i in range(N_LAYERS)
        ])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        concat_dim = N_LAYERS * E
        unembed_hidden = cfg.get('unembed_hidden', concat_dim * 8)
        self.unembedder = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, unembed_hidden, bias=False),
            nn.GELU(),
            nn.Linear(unembed_hidden, VOCAB_SIZE, bias=False),
        )

    def forward_capture(self, tokens):
        """Forward returning (concat, logits) — both pre/post unembedder."""
        x = self.token_embedder(tokens)
        outs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            outs.append(x)
        concat = torch.cat(outs, dim=-1)               # [B, T, N_LAYERS*E]
        logits = self.unembedder(concat)               # [B, T, vocab]
        return concat, logits


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100_000, help='number of samples to extract')
    p.add_argument('--out', type=str,
                   default='/home/starost/spiky/nanochat_exps/distill_unembedder/pairs.parquet')
    p.add_argument('--ckpt', type=str,
                   default=os.path.join(EXP174_DIR, 'checkpoint.pt'))
    args = p.parse_args()

    print(f'Building model & loading checkpoint: {args.ckpt}')
    model = Model().to(DEVICE)
    sd = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded model — {n_params/1e6:.1f}M params')

    concat_dim = N_LAYERS * E
    samples_per_batch = DEVICE_BS * CONTEXT_SIZE  # 8 * 512 = 4096
    n_batches = (args.n + samples_per_batch - 1) // samples_per_batch
    print(f'Will run {n_batches} batches → ~{n_batches * samples_per_batch} samples '
          f'(target {args.n})')

    # Build parquet schema
    schema = pa.schema([
        ('input',  pa.list_(pa.float32(), concat_dim)),
        ('logits', pa.list_(pa.float16(), VOCAB_SIZE)),
    ])

    writer = pq.ParquetWriter(args.out, schema, compression='zstd')

    t0 = time.time()
    samples_written = 0
    with torch.no_grad():
        for bi in range(n_batches):
            tokens, _targets = next(train_loader)
            concat, logits = model.forward_capture(tokens)        # [B, T, *]
            # flatten across (B, T)
            inputs_np = concat.reshape(-1, concat_dim).float().cpu().numpy()    # [B*T, 384]
            logits_np = logits.reshape(-1, VOCAB_SIZE).to(torch.float16).cpu().numpy()  # [B*T, vocab]
            n_this = inputs_np.shape[0]
            if samples_written + n_this > args.n:
                n_this = args.n - samples_written
                inputs_np = inputs_np[:n_this]
                logits_np = logits_np[:n_this]
            # build pyarrow table for this batch
            input_arr  = pa.FixedSizeListArray.from_arrays(
                pa.array(inputs_np.flatten(), type=pa.float32()), concat_dim)
            logits_arr = pa.FixedSizeListArray.from_arrays(
                pa.array(logits_np.flatten(), type=pa.float16()), VOCAB_SIZE)
            tbl = pa.Table.from_arrays([input_arr, logits_arr], schema=schema)
            writer.write_table(tbl)
            samples_written += n_this
            if (bi + 1) % 5 == 0 or bi == n_batches - 1:
                dt = time.time() - t0
                rate = samples_written / dt
                print(f'  batch {bi+1}/{n_batches}: {samples_written}/{args.n} '
                      f'samples ({rate:.0f}/s, elapsed {dt:.1f}s)')
            if samples_written >= args.n:
                break

    writer.close()
    sz = os.path.getsize(args.out) / (1024**3)
    dt = time.time() - t0
    print(f'\nDone. Saved {samples_written} samples → {args.out} ({sz:.2f} GB, {dt:.1f}s)')


if __name__ == '__main__':
    main()
