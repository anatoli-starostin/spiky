"""
Profile LUT transformer forward+backward with varying tph for OutProj.
Uses exp239-style architecture: Q/K nap=5, V nap=6, OutProj nap=6, no FFN.
"""
import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy

DEVICE = 'cuda:0'
E = 32
H = 4
d_qk = 8
d_v = 8
NAP_QK = 5
NAP_V = 6
NAP_OUT = 6
TPH_QKV = 128
N_LAYERS = 6
BATCH = 128
SEQ_LEN = 32
SEED = 42
WARMUP_ITERS = 5
BENCH_ITERS = 20


def _make_lut(n_heads, n_outputs, nap, tph, seed_offset):
    return MultiHeadLut(
        input_dim=E, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
        initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=SEED+seed_offset, device=DEVICE, recompute_in_backward=True,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx, tph_out):
        super().__init__()
        self.q_lut = _make_lut(H, d_qk, NAP_QK, TPH_QKV, layer_idx)
        self.k_lut = _make_lut(H, d_qk, NAP_QK, TPH_QKV, 100+layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v_lut = _make_lut(H, d_v, NAP_V, TPH_QKV, 200+layer_idx)
        self.out_proj = _make_lut(1, E, NAP_OUT, tph_out, 400+layer_idx)
        self.norm1 = nn.LayerNorm(E)

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = x + pos_emb.unsqueeze(0)
        xp_flat = xp.reshape(B*T, _E)
        x_flat = x.reshape(B*T, _E)

        q = self.q_lut(xp_flat).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_lut(xp_flat).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_lut(x_flat).reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        proj = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)).squeeze(1).reshape(B, T, _E)
        x = x + self.norm1(proj)
        return x


class Model(nn.Module):
    def __init__(self, tph_out):
        super().__init__()
        self.token_embedder = nn.Embedding(257, E)
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.randn(SEQ_LEN, E) * 0.1) for _ in range(N_LAYERS)])
        self.layers = nn.ModuleList([LUTBlock(i, tph_out) for i in range(N_LAYERS)])
        self.unembedder = nn.Sequential(
            nn.Linear(E, 128), nn.GELU(), nn.Linear(128, 257, bias=False),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
        return self.unembedder(x)


def bench(tph_out):
    torch.manual_seed(SEED)
    model = Model(tph_out).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_params = sum(p.numel() for p in model.parameters())
    inp = torch.randint(0, 257, (BATCH, SEQ_LEN), device=DEVICE)
    tgt = torch.randint(0, 257, (BATCH, SEQ_LEN), device=DEVICE)

    # Warmup
    model.train()
    for _ in range(WARMUP_ITERS):
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, 257), tgt.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(BENCH_ITERS):
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, 257), tgt.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    ms_per_step = elapsed / BENCH_ITERS * 1000
    del model, optimizer
    torch.cuda.empty_cache()
    return total_params, ms_per_step


print(f'{"tph_out":>8} | {"params":>12} | {"ms/step":>10} | {"relative":>10}')
print('-' * 50)

results = []
for tph_out in [64, 128, 256, 512, 1024, 2048, 4096]:
    try:
        params, ms = bench(tph_out)
        results.append((tph_out, params, ms))
        rel = ms / results[0][2] if results else 1.0
        print(f'{tph_out:>8} | {params:>12,} | {ms:>9.1f}ms | {rel:>9.2f}x')
    except RuntimeError as e:
        print(f'{tph_out:>8} | OOM: {e}')
        break
