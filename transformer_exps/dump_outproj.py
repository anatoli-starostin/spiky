"""
Dump out_proj input/output activity from exp181 checkpoint (last token only).
Produces CSV: layer, x1..x64, y1..y32
One row per (sample, layer). 10000 batches × 128 samples × 6 layers = 7.68M rows.
"""
import sys, os, csv, time
import torch
import torch.nn.functional as F
import torch.nn as nn
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer_exps.common import make_sampler, BOS_ID, CONTEXT_SIZE
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.lut_attention import LUTAttentionV3

EXP_DIR = os.path.join(os.path.dirname(__file__), 'exp181_v3_v256_op768nap5')
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
SEQ_LEN = CONTEXT_SIZE
torch.manual_seed(cfg['random_seed'])


def make_score_attn(seed_offset=0):
    E, P, H = cfg['embedding_dim'], cfg['positional_dim'], cfg['n_heads']
    lut = MultiHeadLut(
        input_dim=2*E+P, n_heads=H, n_outputs=1,
        n_anchor_pairs=cfg['attention_nap'], tables_per_head=cfg['attention_tph'],
        smooth_mode=False, n_alternatives=1, normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, recompute_in_backward=False,
    )
    return LUTAttentionV3(lut, seq_len=SEQ_LEN, causal=True, include_diagonal=True)


def make_value_lut(seed_offset=200):
    E, H, d_v = cfg['embedding_dim'], cfg['n_heads'], cfg['d_v']
    return MultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['value_nap'], tables_per_head=cfg['value_tph'],
        smooth_mode=False, n_alternatives=1, normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, recompute_in_backward=False,
    )


def make_out_proj(seed_offset=400):
    E, H, d_v = cfg['embedding_dim'], cfg['n_heads'], cfg['d_v']
    return MultiHeadLut(
        input_dim=H*d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_proj_nap'], tables_per_head=cfg['out_proj_tph'],
        smooth_mode=False, n_alternatives=1, normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, recompute_in_backward=False,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        E, H, d_v = cfg['embedding_dim'], cfg['n_heads'], cfg['d_v']
        self.score_attn = make_score_attn(seed_offset=layer_idx)
        self.value_lut = make_value_lut(seed_offset=200+layer_idx)
        self.attn_norm = nn.LayerNorm(H*d_v)
        self.out_proj = make_out_proj(seed_offset=400+layer_idx)
        self.norm = nn.LayerNorm(E)
        self.H, self.d_v = H, d_v
        self._captured_input = None
        self._captured_output = None

    def forward(self, x, rel_pe):
        B, T, E = x.shape
        H, d_v = self.H, self.d_v
        raw_scores = self.score_attn(x, rel_pe).squeeze(-1).permute(0, 3, 1, 2)
        attn_weights = F.softmax(raw_scores, dim=-1)
        v = self.value_lut(x.reshape(B*T, E)).reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn_out = (attn_weights @ v).permute(0, 2, 1, 3).reshape(B, T, H*d_v)
        attn_out = self.attn_norm(attn_out)

        op_input = attn_out.reshape(B*T, H*d_v)
        proj_out = self.out_proj(op_input).squeeze(1)

        # Capture last token only: indices T-1, 2T-1, 3T-1, ...
        last_tok_idx = torch.arange(T-1, B*T, T, device=x.device)
        self._captured_input = op_input[last_tok_idx].detach()   # [B, 64]
        self._captured_output = proj_out[last_tok_idx].detach()  # [B, 32]

        return x + self.norm(proj_out.reshape(B, T, E))


class LUTTransformerV3Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        E, P = cfg['embedding_dim'], cfg['positional_dim']
        self.token_embedder = nn.Embedding(cfg['vocab_size'], E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.rel_pe = nn.Parameter(torch.randn(SEQ_LEN, P) * 0.1)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(cfg['num_layers'])])
        self.unembedder = nn.Linear(E, cfg['vocab_size'], bias=False)

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer in self.layers:
            x = layer(x, self.rel_pe)
        return self.unembedder(x)


model = LUTTransformerV3Softmax().to(DEVICE)
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print("Loaded checkpoint from exp181")

N_BATCHES = 10000
BATCH_SIZE = 128
N_LAYERS = cfg['num_layers']
H_DV = cfg['n_heads'] * cfg['d_v']
E = cfg['embedding_dim']

sampler = make_sampler(DEVICE, random_seed=1)

header = ['layer'] + [f'x{i+1}' for i in range(H_DV)] + [f'y{i+1}' for i in range(E)]
out_path = os.path.join(EXP_DIR, 'outproj_dump.csv')
total_rows = N_BATCHES * BATCH_SIZE * N_LAYERS
print(f"Writing to {out_path}")
print(f"Expected rows: {total_rows:,}")

t0 = time.time()
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    with torch.no_grad():
        for batch_idx in range(N_BATCHES):
            tokens = sampler.sample_training_batch(BATCH_SIZE).long()
            inp = torch.empty_like(tokens)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = tokens[:, :-1]
            _ = model(inp)

            for layer_idx, layer in enumerate(model.layers):
                xi = layer._captured_input.cpu().numpy()  # [B, 64]
                yi = layer._captured_output.cpu().numpy()  # [B, 32]
                for row in range(xi.shape[0]):
                    writer.writerow([layer_idx] + xi[row].tolist() + yi[row].tolist())

            if (batch_idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed
                eta = (N_BATCHES - batch_idx - 1) / rate
                print(f"batch {batch_idx+1}/{N_BATCHES}  ({rate:.1f} batch/s, ETA {eta/60:.1f}min)")

elapsed = time.time() - t0
print(f"Done in {elapsed/60:.1f}min, {total_rows:,} rows written")
