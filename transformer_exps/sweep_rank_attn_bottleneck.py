"""
Bottleneck sweep: RankAttention + FFN LUT + GELU unembedder.
3000 steps, bs=256, constant lr=0.001.
Base: Q/K(nap=4,tph=128), V(nap=4,tph=256), OutProj(nap=5,tph=768), FFN(nap=6,tph=768).
"""
import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer_exps.common import make_sampler, BOS_ID, CONTEXT_SIZE
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy

DEVICE = 'cuda:0'
SEQ_LEN = CONTEXT_SIZE
N_STEPS = 3000
BATCH_SIZE = 256
LR = 0.001

E = 32
P = 16
H = 4
D_QK = 16
D_V = 16


def _make_lut(input_dim, n_heads, n_outputs, nap, tph, seed):
    return MultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
        initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=seed, device=DEVICE, recompute_in_backward=True,
    )


def make_model(cfg):
    class Block(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.q = _make_lut(E+P, H, D_QK, cfg['qk_nap'], cfg['qk_tph'], 42+idx)
            self.k = _make_lut(E+P, H, D_QK, cfg['qk_nap'], cfg['qk_tph'], 142+idx)
            self.v = _make_lut(E, H, D_V, cfg['v_nap'], cfg['v_tph'], 242+idx)
            self.out_proj = _make_lut(H*D_V, 1, E, cfg['op_nap'], cfg['op_tph'], 442+idx)
            self.norm1 = nn.LayerNorm(E)
            self.has_ffn = cfg.get('ffn_nap', 0) > 0
            if self.has_ffn:
                self.ffn = _make_lut(E, 1, E, cfg['ffn_nap'], cfg['ffn_tph'], 642+idx)
                self.norm2 = nn.LayerNorm(E)

        def forward(self, x, pos_emb):
            B, T, _E = x.shape
            xp = torch.cat([x, pos_emb.unsqueeze(0).expand(B, -1, -1)], dim=-1)
            q = self.q(xp.reshape(B*T, -1)).reshape(B, T, H, D_QK).permute(0,2,1,3)
            k = self.k(xp.reshape(B*T, -1)).reshape(B, T, H, D_QK).permute(0,2,1,3)
            v = self.v(x.reshape(B*T, _E)).reshape(B, T, H, D_V).permute(0,2,1,3)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            proj = self.out_proj(attn.permute(0,2,1,3).reshape(B*T, H*D_V)).squeeze(1).reshape(B, T, _E)
            x = x + self.norm1(proj)
            if self.has_ffn:
                ffn_out = self.ffn(x.reshape(B*T, _E)).squeeze(1).reshape(B, T, _E)
                x = x + self.norm2(ffn_out)
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(257, E)
            self.emb.weight.data.uniform_(-0.1, 0.1)
            self.pos_emb = nn.Parameter(torch.randn(SEQ_LEN, P) * 0.1)
            self.layers = nn.ModuleList([Block(i) for i in range(6)])
            self.unemb = nn.Sequential(
                nn.Linear(E, 128), nn.GELU(), nn.Linear(128, 257, bias=False),
            )
        def forward(self, tokens):
            x = self.emb(tokens)
            for layer in self.layers:
                x = layer(x, self.pos_emb)
            return self.unemb(x)

    return Model().to(DEVICE)


def run_config(cfg):
    torch.manual_seed(42)
    model = make_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    if n_params > 25_000_000:
        return n_params, None, "SKIP (>25M)"

    sampler = make_sampler(DEVICE, random_seed=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ema = None

    model.train()
    t0 = time.time()
    for step in range(N_STEPS):
        x = sampler.sample_training_batch(BATCH_SIZE).long()
        inp = torch.empty_like(x); inp[:, 0] = BOS_ID; inp[:, 1:] = x[:, :-1]
        logits = model(inp)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        lv = loss.item()
        ema = lv if ema is None else 0.99*ema + 0.01*lv

    elapsed = time.time() - t0
    return n_params, ema, f"{elapsed:.0f}s"


# Base config
BASE = dict(qk_nap=4, qk_tph=128, v_nap=4, v_tph=256, op_nap=5, op_tph=768, ffn_nap=6, ffn_tph=768)

CONFIGS = [
    dict(**BASE, label="base"),

    # Scale Q/K
    dict(**{**BASE, 'qk_nap': 5, 'qk_tph': 128}, label="qk_nap5"),
    dict(**{**BASE, 'qk_nap': 5, 'qk_tph': 256}, label="qk_nap5_tph256"),
    dict(**{**BASE, 'qk_nap': 6, 'qk_tph': 128}, label="qk_nap6"),
    dict(**{**BASE, 'qk_tph': 256}, label="qk_tph256"),

    # Scale V
    dict(**{**BASE, 'v_nap': 5, 'v_tph': 256}, label="v_nap5"),
    dict(**{**BASE, 'v_nap': 6, 'v_tph': 128}, label="v_nap6"),
    dict(**{**BASE, 'v_tph': 512}, label="v_tph512"),

    # Scale OutProj
    dict(**{**BASE, 'op_nap': 6, 'op_tph': 512}, label="op_nap6"),
    dict(**{**BASE, 'op_nap': 6, 'op_tph': 768}, label="op_nap6_tph768"),
    dict(**{**BASE, 'op_tph': 1024}, label="op_tph1024"),

    # Scale FFN
    dict(**{**BASE, 'ffn_nap': 5, 'ffn_tph': 768}, label="ffn_nap5"),
    dict(**{**BASE, 'ffn_tph': 1024}, label="ffn_tph1024"),
    dict(**{**BASE, 'ffn_nap': 5, 'ffn_tph': 1024}, label="ffn_nap5_tph1024"),

    # Scale everything
    dict(qk_nap=5, qk_tph=256, v_nap=5, v_tph=256, op_nap=6, op_tph=768, ffn_nap=6, ffn_tph=1024, label="all_up"),

    # Minimal Q/K, max FFN+OutProj
    dict(qk_nap=4, qk_tph=64, v_nap=4, v_tph=256, op_nap=6, op_tph=1024, ffn_nap=6, ffn_tph=1024, label="min_qk_max_rest"),

    # Scale DOWN Q/K
    dict(**{**BASE, 'qk_nap': 3, 'qk_tph': 128}, label="qk_nap3"),
    dict(**{**BASE, 'qk_tph': 64}, label="qk_tph64"),

    # Scale DOWN V
    dict(**{**BASE, 'v_nap': 3, 'v_tph': 256}, label="v_nap3"),
    dict(**{**BASE, 'v_tph': 128}, label="v_tph128"),

    # Scale DOWN OutProj
    dict(**{**BASE, 'op_nap': 4, 'op_tph': 768}, label="op_nap4"),
    dict(**{**BASE, 'op_tph': 512}, label="op_tph512"),
    dict(**{**BASE, 'op_tph': 256}, label="op_tph256"),

    # Scale DOWN FFN
    dict(**{**BASE, 'ffn_tph': 512}, label="ffn_tph512"),
    dict(**{**BASE, 'ffn_tph': 256}, label="ffn_tph256"),

    # No FFN at all
    dict(**{**BASE, 'ffn_nap': 0}, label="no_ffn"),
]


if __name__ == '__main__':
    print(f"RankAttn + FFN LUT + GELU unembedder, {N_STEPS} steps, bs={BATCH_SIZE}")
    print(f"{'label':<22s} {'params':>10s} {'loss@3k':>8s} {'time':>6s}")
    print("-" * 50)
    results = []
    for cfg in CONFIGS:
        label = cfg.pop('label')
        n_params, ema, info = run_config(cfg)
        loss_str = f"{ema:.4f}" if ema is not None else "N/A"
        print(f"{label:<22s} {n_params:>10,d} {loss_str:>8s} {info:>6s}")
        results.append((label, n_params, ema))
        sys.stdout.flush()

    print(f"\n=== SORTED BY LOSS ===")
    results.sort(key=lambda x: x[2] if x[2] is not None else 999)
    for label, n_params, ema in results:
        loss_str = f"{ema:.4f}" if ema is not None else "N/A"
        print(f"{label:<22s} {n_params:>10,d} {loss_str:>8s}")
