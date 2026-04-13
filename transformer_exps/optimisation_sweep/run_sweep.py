"""
Optimisation sweep: continue from exp184 checkpoint for 10K steps.
6 configs: {Adam, SGD, SGD+momentum} x {n_alternatives=1, n_alternatives=3}
All with constant lr=0.0001, batch_size=128.
"""
import sys, os, json, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import make_sampler, BOS_ID, CONTEXT_SIZE
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.lut_attention import LUTAttentionV3

DEVICE = 'cuda:0'
SEQ_LEN = CONTEXT_SIZE
N_STEPS = 10000
BATCH_SIZE = 128
LR = 0.0001
CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', 'exp184_v3_v256_op768nap5_100k', 'checkpoint.pt')

# exp184 config
BASE_CFG = {
    'embedding_dim': 32, 'positional_dim': 16, 'num_layers': 6,
    'n_heads': 4, 'd_v': 16,
    'attention_nap': 6, 'attention_tph': 128,
    'value_nap': 4, 'value_tph': 256,
    'out_proj_nap': 5, 'out_proj_tph': 768,
    'smooth_mode': False, 'normalise_weights': False,
    'calibrate_output': False, 'initial_weights_noise': 0.001,
    'anchor_sampling_policy': 'full_coverage',
}


def build_model(n_alternatives=1):
    cfg = {**BASE_CFG, 'n_alternatives': n_alternatives}
    E, P, H, d_v = cfg['embedding_dim'], cfg['positional_dim'], cfg['n_heads'], cfg['d_v']

    def _make_score(seed_offset):
        lut = MultiHeadLut(
            input_dim=2*E+P, n_heads=H, n_outputs=1,
            n_anchor_pairs=cfg['attention_nap'], tables_per_head=cfg['attention_tph'],
            smooth_mode=False, n_alternatives=n_alternatives,
            normalize_weights=False, calibrate_output=False,
            anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
            initial_weights_noise=cfg['initial_weights_noise'],
            uncertainty_mode=UncertaintyMode.INVERSE_L1,
            random_seed=42+seed_offset, device=DEVICE, recompute_in_backward=True,
        )
        return LUTAttentionV3(lut, seq_len=SEQ_LEN, causal=True, include_diagonal=True)

    def _make_value(seed_offset):
        return MultiHeadLut(
            input_dim=E, n_heads=H, n_outputs=d_v,
            n_anchor_pairs=cfg['value_nap'], tables_per_head=cfg['value_tph'],
            smooth_mode=False, n_alternatives=n_alternatives,
            normalize_weights=False, calibrate_output=False,
            anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
            initial_weights_noise=cfg['initial_weights_noise'],
            uncertainty_mode=UncertaintyMode.INVERSE_L1,
            random_seed=42+seed_offset, device=DEVICE, recompute_in_backward=True,
        )

    def _make_outproj(seed_offset):
        return MultiHeadLut(
            input_dim=H*d_v, n_heads=1, n_outputs=E,
            n_anchor_pairs=cfg['out_proj_nap'], tables_per_head=cfg['out_proj_tph'],
            smooth_mode=False, n_alternatives=n_alternatives,
            normalize_weights=False, calibrate_output=False,
            anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
            initial_weights_noise=cfg['initial_weights_noise'],
            uncertainty_mode=UncertaintyMode.INVERSE_L1,
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
            self.H, self.d_v = H, d_v

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
            self.token_embedder = nn.Embedding(257, E)
            self.rel_pe = nn.Parameter(torch.randn(SEQ_LEN, P) * 0.1)
            self.layers = nn.ModuleList([Block(i) for i in range(6)])
            self.unembedder = nn.Linear(E, 257, bias=False)

        def forward(self, tokens):
            x = self.token_embedder(tokens)
            for layer in self.layers:
                x = layer(x, self.rel_pe)
            return self.unembedder(x)

    return Model().to(DEVICE)


def eval_model(model, sampler, batch_size):
    """Val loss without generation."""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in sampler.testing_batches_iterator(batch_size):
            inp = torch.empty(batch.shape[0], batch.shape[1], dtype=torch.long, device=batch.device)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1].long()
            logits = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), batch.long().reshape(B*T))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def run_config(opt_name, n_alt):
    torch.manual_seed(42)
    model = build_model(n_alternatives=n_alt)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt)

    if opt_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    elif opt_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    elif opt_name == 'SGD_mom':
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    sampler = make_sampler(DEVICE, random_seed=1)
    val_before = eval_model(model, sampler, BATCH_SIZE)

    ema = None
    model.train()
    t0 = time.time()
    best_val = val_before
    label = f"{opt_name}_nalt{n_alt}"

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
        ema = lv if ema is None else 0.99*ema + 0.01*lv

        if (step + 1) % 1000 == 0:
            print(f"  {label} step {step+1}/{N_STEPS} train={ema:.4f}")

        if (step + 1) % 2000 == 0:
            val = eval_model(model, sampler, BATCH_SIZE)
            if val < best_val:
                best_val = val
            print(f"  {label} step {step+1}/{N_STEPS} val={val:.4f}")

    elapsed = time.time() - t0
    val_after = eval_model(model, sampler, BATCH_SIZE)
    if val_after < best_val:
        best_val = val_after

    return val_before, best_val, val_after, f"{elapsed:.0f}s"


CONFIGS = [
    ('Adam', 1),
    ('Adam', 3),
    ('SGD', 1),
    ('SGD', 3),
    ('SGD_mom', 1),
    ('SGD_mom', 3),
]


if __name__ == '__main__':
    print(f"Starting from exp184 checkpoint (val_loss ~1.4411)")
    print(f"10K steps, lr=0.0001, batch_size=128")
    print(f"{'config':<20s} {'before':>8s} {'best':>8s} {'after':>8s} {'time':>6s}")
    print("-" * 56)
    results = []
    for opt_name, n_alt in CONFIGS:
        label = f"{opt_name}_nalt{n_alt}"
        val_before, best_val, val_after, info = run_config(opt_name, n_alt)
        delta = best_val - val_before
        print(f"{label:<20s} {val_before:>8.4f} {best_val:>8.4f} {val_after:>8.4f} {info:>6s}  ({delta:+.4f})")
        results.append((label, val_before, best_val, val_after))
        sys.stdout.flush()

    print(f"\n=== SORTED BY BEST VAL ===")
    results.sort(key=lambda x: x[2])
    for label, vb, best, va in results:
        print(f"{label:<20s} {vb:.4f} → {best:.4f} ({best-vb:+.4f})")
