"""
exp004_lut_baseline — LUTTransformer matching lutorch_transformer.ipynb exactly.
Warmup-cosine lr schedule, 100k steps.
"""
import sys, os, json
import torch
import torch.nn as nn
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import (
    make_sampler, Trainer,
    CONTEXT_SIZE, VOCAB_SIZE,
)
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_attention import LUTAttention, PairProcessingConfig
from spiky.lutorch.lut_helpers import UncertaintyMode

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
torch.manual_seed(cfg['random_seed'])

# ── Model ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LUTTransformerConfig:
    vocab_size: int = 257
    embedding_dim: int = 64
    num_layers: int = 6
    num_heads: int = 4
    hidden_dim_attn: int = 16
    hidden_dim_value: int = 256
    hidden_dim_ffn: int = 256
    n_anchor_pairs_attn: int = 10
    n_anchor_pairs_ffn: int = 12
    n_positional_buckets: int = 1
    tables_per_head_attn: int = 96
    tables_per_head_value: int = 96
    ffn_tables: int = 96
    dropout: float = 0.0
    smooth_mode: bool = False
    n_alternatives: int = 1
    connected_anchors_mode: bool = False
    attention_temperature: float = 0.25
    initial_weights_noise: float = 0.001
    normalise_weights: bool = False
    calibrate_output: bool = False
    uncertainty_mode: UncertaintyMode = UncertaintyMode.INVERSE_L1
    pair_config: PairProcessingConfig = PairProcessingConfig(c1=1.0, c2=-2.0)
    device: object = DEVICE
    random_seed: object = 42

    def __post_init__(self):
        assert self.embedding_dim % self.num_heads == 0


class LUTTransformer(nn.Module):

    class Block(nn.Module):
        def __init__(self, c: LUTTransformerConfig):
            super().__init__()
            self.cross_attn = LUTAttention(
                MultiHeadLut(
                    input_dim=c.embedding_dim,
                    n_heads=c.num_heads,
                    n_outputs=1,
                    n_anchor_pairs=c.n_anchor_pairs_attn,
                    tables_per_head=c.tables_per_head_attn,
                    n_buckets=c.n_positional_buckets,
                    smooth_mode=c.smooth_mode,
                    n_alternatives=c.n_alternatives,
                    normalize_weights=c.normalise_weights,
                    calibrate_output=False,
                    device=c.device,
                    connected_anchors_mode=c.connected_anchors_mode,
                    random_seed=c.random_seed,
                    initial_weights_noise=c.initial_weights_noise,
                    uncertainty_mode=c.uncertainty_mode,
                ),
                causal=True,
                include_diagonal=False,
                attention_temperature=c.attention_temperature,
                n_positional_buckets=c.n_positional_buckets,
                pair_config=c.pair_config,
            )
            self.value_lut = MultiHeadLut(
                input_dim=c.embedding_dim,
                n_heads=c.num_heads,
                n_outputs=c.embedding_dim // c.num_heads,
                n_anchor_pairs=c.n_anchor_pairs_attn,
                tables_per_head=c.tables_per_head_value,
                smooth_mode=c.smooth_mode,
                n_alternatives=c.n_alternatives,
                normalize_weights=c.normalise_weights,
                calibrate_output=c.calibrate_output,
                device=c.device,
                connected_anchors_mode=c.connected_anchors_mode,
                random_seed=c.random_seed,
                initial_weights_noise=c.initial_weights_noise,
                uncertainty_mode=c.uncertainty_mode,
            )
            self.attn_dropout = nn.Dropout(c.dropout)
            self.ffn = MultiHeadLut(
                input_dim=c.embedding_dim,
                n_heads=1,
                n_outputs=c.embedding_dim,
                n_anchor_pairs=c.n_anchor_pairs_ffn,
                tables_per_head=c.ffn_tables,
                smooth_mode=c.smooth_mode,
                n_alternatives=c.n_alternatives,
                normalize_weights=c.normalise_weights,
                calibrate_output=c.calibrate_output,
                device=c.device,
                connected_anchors_mode=c.connected_anchors_mode,
                random_seed=c.random_seed,
                initial_weights_noise=c.initial_weights_noise,
                uncertainty_mode=c.uncertainty_mode,
                n_post_processor_inputs=c.hidden_dim_ffn,
            )
            self.ffn_dropout = nn.Dropout(c.dropout)

        def forward(self, z):
            B, S, E = z.shape
            attn_weights = self.cross_attn(z, z)           # [B, S, S, H]
            v = self.value_lut(z.reshape(-1, E))           # [B*S, H, E//H]
            H = v.shape[1]
            v = v.reshape(B, S, H, -1)                     # [B, S, H, E//H]
            attn_out = attn_weights.permute(0,3,1,2) @ v.permute(0,2,1,3)  # [B, H, S, E//H]
            attn_out = attn_out.permute(0,2,1,3).reshape(B, S, E)
            z = z + self.attn_dropout(attn_out)
            ffn_out = self.ffn(z.reshape(-1, E)).reshape(B, S, -1)
            z = z + self.ffn_dropout(ffn_out)
            return z

    def __init__(self, c: LUTTransformerConfig, maxlen=CONTEXT_SIZE):
        super().__init__()
        self.config = c
        with torch.no_grad():
            self.token_embedder = nn.Embedding(c.vocab_size, c.embedding_dim // 2, device=c.device)
            self.token_embedder.weight.copy_(
                torch.randn(self.token_embedder.weight.shape, device=c.device) * 0.1
            )
            self.token_unembedder = nn.Embedding(c.vocab_size, c.embedding_dim, device=c.device)
            self.token_unembedder.weight.copy_(
                torch.randn(self.token_unembedder.weight.shape, device=c.device) * 0.1
            )
        self.layers = nn.ModuleList([LUTTransformer.Block(c) for _ in range(c.num_layers)])
        self.register_buffer(
            'pos_emb', torch.randn([1, maxlen, c.embedding_dim // 2], device=c.device) * 0.1
        )

    def forward(self, tokens):
        z = self.token_embedder(tokens)
        z = torch.cat([z, self.pos_emb[:, :tokens.shape[1]].repeat(tokens.shape[0], 1, 1)], dim=-1)
        for layer in self.layers:
            z = layer(z)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        return z @ self.token_unembedder.weight.T


# ── Run ────────────────────────────────────────────────────────────────────────
sampler = make_sampler(DEVICE, random_seed=1)

c = LUTTransformerConfig(
    vocab_size=cfg['vocab_size'],
    embedding_dim=cfg['embedding_dim'],
    num_layers=cfg['num_layers'],
    num_heads=cfg['num_heads'],
    hidden_dim_attn=cfg['hidden_dim_attn'],
    hidden_dim_value=cfg['hidden_dim_value'],
    hidden_dim_ffn=cfg['hidden_dim_ffn'],
    n_anchor_pairs_attn=cfg['n_anchor_pairs_attn'],
    n_anchor_pairs_ffn=cfg['n_anchor_pairs_ffn'],
    n_positional_buckets=cfg['n_positional_buckets'],
    tables_per_head_attn=cfg['tables_per_head_attn'],
    tables_per_head_value=cfg['tables_per_head_value'],
    ffn_tables=cfg['ffn_tables'],
    dropout=cfg['dropout'],
    smooth_mode=cfg['smooth_mode'],
    n_alternatives=cfg['n_alternatives'],
    connected_anchors_mode=cfg['connected_anchors_mode'],
    attention_temperature=cfg['attention_temperature'],
    initial_weights_noise=cfg['initial_weights_noise'],
    normalise_weights=cfg['normalise_weights'],
    calibrate_output=cfg['calibrate_output'],
    random_seed=cfg['random_seed'],
    device=DEVICE,
)
model = LUTTransformer(c).to(DEVICE)

Trainer(model, sampler, cfg, EXP_DIR).run()
