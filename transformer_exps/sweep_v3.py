"""
Quick parameter sweep for V3 softmax architecture.
5000 steps, batch_size=32, lr=0.001. Reports final EMA loss.
"""
import sys, os, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer_exps.common import make_sampler, BOS_ID, CONTEXT_SIZE
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.lut_attention import LUTAttentionV3

DEVICE = 'cuda:0'
SEQ_LEN = CONTEXT_SIZE
N_STEPS = 5000
BATCH_SIZE = 32
LR = 0.001


def make_model(cfg):
    E = cfg['E']
    P = cfg['P']
    H = cfg['H']
    d_v = cfg['d_v']

    def _excl_sets():
        if not cfg.get('excl', False):
            return None
        return [list(range(E)), list(range(E, 2*E)), list(range(2*E, 2*E+P))]

    def _make_score(seed_offset):
        lut = MultiHeadLut(
            input_dim=2*E+P, n_heads=H, n_outputs=1,
            n_anchor_pairs=cfg['a_nap'], tables_per_head=cfg['a_tph'],
            smooth_mode=False, n_alternatives=1,
            normalize_weights=False, calibrate_output=False,
            anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
            initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
            random_seed=42+seed_offset, device=DEVICE, recompute_in_backward=True,
            exclusion_sets=_excl_sets(),
        )
        return LUTAttentionV3(lut, seq_len=SEQ_LEN, causal=True, include_diagonal=True)

    def _make_value(seed_offset):
        return MultiHeadLut(
            input_dim=E, n_heads=H, n_outputs=d_v,
            n_anchor_pairs=cfg['v_nap'], tables_per_head=cfg['v_tph'],
            smooth_mode=False, n_alternatives=1,
            normalize_weights=False, calibrate_output=False,
            anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
            initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
            random_seed=42+seed_offset, device=DEVICE, recompute_in_backward=True,
        )

    def _make_outproj(seed_offset):
        return MultiHeadLut(
            input_dim=H*d_v, n_heads=1, n_outputs=E,
            n_anchor_pairs=cfg['o_nap'], tables_per_head=cfg['o_tph'],
            smooth_mode=False, n_alternatives=1,
            normalize_weights=False, calibrate_output=False,
            anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
            initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
            random_seed=42+seed_offset, device=DEVICE, recompute_in_backward=True,
        )

    class Block(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.score_attn = _make_score(idx)
            self.value_lut = _make_value(200+idx)
            self.out_proj = _make_outproj(400+idx)
            self.attn_norm = nn.LayerNorm(H*d_v)
            self.norm = nn.LayerNorm(E)

        def forward(self, x, rel_pe):
            B, T, _E = x.shape
            scores = self.score_attn(x, rel_pe).squeeze(-1).permute(0,3,1,2)
            weights = F.softmax(scores, dim=-1)
            v = self.value_lut(x.reshape(B*T, _E)).reshape(B, T, H, d_v).permute(0,2,1,3)
            attn_out = (weights @ v).permute(0,2,1,3).reshape(B, T, H*d_v)
            attn_out = self.attn_norm(attn_out)
            proj = self.out_proj(attn_out.reshape(B*T, H*d_v)).squeeze(1).reshape(B, T, _E)
            return x + self.norm(proj)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(257, E)
            self.emb.weight.data.uniform_(-0.1, 0.1)
            self.rel_pe = nn.Parameter(torch.randn(SEQ_LEN, P) * 0.1)
            self.layers = nn.ModuleList([Block(i) for i in range(6)])
            self.unemb = nn.Linear(E, 257, bias=False)

        def forward(self, tokens):
            x = self.emb(tokens)
            for layer in self.layers:
                x = layer(x, self.rel_pe)
            return self.unemb(x)

    return Model().to(DEVICE)


def run_config(cfg):
    torch.manual_seed(42)
    model = make_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > 9_000_000:
        return n_params, None, "SKIP (>9M)"

    sampler = make_sampler(DEVICE, random_seed=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ema = None
    alpha = 0.01

    model.train()
    t0 = time.time()
    for step in range(N_STEPS):
        x = sampler.sample_training_batch(BATCH_SIZE).long()
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        logits = model(inp)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lv = loss.item()
        ema = lv if ema is None else (1-alpha)*ema + alpha*lv

    elapsed = time.time() - t0
    return n_params, ema, f"{elapsed:.0f}s"


CONFIGS = [
    # --- Reference from previous sweep ---
    dict(E=32, P=16, H=4, d_v=16, a_nap=6, a_tph=128, v_nap=4, v_tph=128, o_nap=4, o_tph=768, excl=False,
         label="prev_winner"),

    # --- Scale out_proj (main lever) ---
    dict(E=32, P=16, H=4, d_v=16, a_nap=6, a_tph=128, v_nap=4, v_tph=128, o_nap=4, o_tph=1536, excl=False,
         label="op1536"),
    dict(E=32, P=16, H=4, d_v=16, a_nap=6, a_tph=128, v_nap=4, v_tph=128, o_nap=4, o_tph=2048, excl=False,
         label="op2048"),
    dict(E=32, P=16, H=4, d_v=16, a_nap=6, a_tph=128, v_nap=4, v_tph=256, o_nap=4, o_tph=1024, excl=False,
         label="v256_op1024"),
    dict(E=32, P=16, H=4, d_v=16, a_nap=6, a_tph=128, v_nap=4, v_tph=256, o_nap=4, o_tph=1536, excl=False,
         label="v256_op1536"),

    # --- Scale d_v (wider value head) ---
    dict(E=32, P=16, H=4, d_v=32, a_nap=6, a_tph=128, v_nap=4, v_tph=128, o_nap=4, o_tph=768, excl=False,
         label="dv32_light"),
    dict(E=32, P=16, H=4, d_v=32, a_nap=6, a_tph=128, v_nap=4, v_tph=256, o_nap=4, o_tph=512, excl=False,
         label="dv32_balanced"),
    dict(E=32, P=16, H=4, d_v=32, a_nap=6, a_tph=128, v_nap=4, v_tph=128, o_nap=4, o_tph=1024, excl=False,
         label="dv32_heavy_op"),

    # --- Scale attention ---
    dict(E=32, P=16, H=4, d_v=16, a_nap=6, a_tph=256, v_nap=4, v_tph=128, o_nap=4, o_tph=768, excl=False,
         label="attn256"),
    dict(E=32, P=16, H=4, d_v=16, a_nap=6, a_tph=256, v_nap=4, v_tph=128, o_nap=4, o_tph=1024, excl=False,
         label="attn256_op1024"),

    # --- All nap=6 scaled up ---
    dict(E=32, P=16, H=4, d_v=16, a_nap=6, a_tph=128, v_nap=6, v_tph=128, o_nap=6, o_tph=512, excl=False,
         label="all_nap6_op512"),
    dict(E=32, P=16, H=4, d_v=16, a_nap=6, a_tph=128, v_nap=6, v_tph=64, o_nap=6, o_tph=768, excl=False,
         label="all_nap6_heavy_op"),

    # --- More heads ---
    dict(E=32, P=16, H=8, d_v=16, a_nap=6, a_tph=128, v_nap=4, v_tph=128, o_nap=4, o_tph=768, excl=False,
         label="8heads"),

    # --- Wider embedding ---
    dict(E=64, P=16, H=4, d_v=16, a_nap=6, a_tph=128, v_nap=4, v_tph=128, o_nap=4, o_tph=512, excl=False,
         label="E64"),
    dict(E=64, P=16, H=4, d_v=16, a_nap=6, a_tph=128, v_nap=4, v_tph=256, o_nap=4, o_tph=768, excl=False,
         label="E64_heavy"),

]


if __name__ == '__main__':
    print(f"{'label':<22s} {'params':>8s} {'loss@5k':>8s} {'time':>6s}")
    print("-" * 48)
    results = []
    for cfg in CONFIGS:
        label = cfg.pop('label')
        n_params, ema, info = run_config(cfg)
        loss_str = f"{ema:.4f}" if ema is not None else "N/A"
        print(f"{label:<22s} {n_params:>8,d} {loss_str:>8s} {info:>6s}")
        results.append((label, n_params, ema))
        sys.stdout.flush()

    print("\n=== SORTED BY LOSS ===")
    results.sort(key=lambda x: x[2] if x[2] is not None else 999)
    for label, n_params, ema in results:
        loss_str = f"{ema:.4f}" if ema is not None else "N/A"
        print(f"{label:<22s} {n_params:>8,d} {loss_str:>8s}")
