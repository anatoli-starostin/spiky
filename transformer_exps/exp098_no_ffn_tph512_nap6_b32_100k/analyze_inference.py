"""
Inference-time analytics: track which lookup indices are actually used
during text generation to verify whether tables are truly dead.
"""
import sys, os, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

# ── Rebuild model ──────────────────────────────────────────────────────────────
import torch.nn as nn
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import RankAttention
from transformer_exps.common import CONTEXT_SIZE, VOCAB_SIZE, make_sampler

DEVICE = 'cuda:0'

def make_lut(input_dim, n_heads, n_outputs, tables_per_head, cfg, seed_offset=0):
    return MultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=cfg['n_anchor_pairs'], tables_per_head=tables_per_head,
        smooth_mode=cfg['smooth_mode'], n_alternatives=cfg['n_alternatives'],
        normalize_weights=cfg['normalise_weights'], calibrate_output=cfg['calibrate_output'],
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
    )

class LUTBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        d, p, h = cfg['embedding_dim'], cfg['positional_dim'], cfg['num_heads']
        d_qk, d_v = cfg['d_qk'], cfg['d_v']
        tph_qkv, tph_op = cfg['qkv_tables_per_head'], cfg['out_proj_tables_per_head']
        s = layer_idx * 10
        self.q_lut    = make_lut(d + p, h, d_qk, tph_qkv, cfg, s + 0)
        self.k_lut    = make_lut(d + p, h, d_qk, tph_qkv, cfg, s + 1)
        self.v_lut    = make_lut(d + p, h, d_v,  tph_qkv, cfg, s + 2)
        self.out_proj = make_lut(h * d_v, 1, d,  tph_op,  cfg, s + 3)
        self.rank_attn = RankAttention(d_qk, d_v, smooth_mode=False, temperature=cfg['rank_attn_temperature'])
        self.n_heads, self.d_qk, self.d_v, self.d = h, d_qk, d_v, d

    def forward(self, x, pos):
        B, T, E = x.shape
        x_pos_flat = torch.cat([x, pos], dim=-1).reshape(-1, E + pos.shape[-1])
        q = self.q_lut(x_pos_flat).permute(1, 0, 2).reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        k = self.k_lut(x_pos_flat).permute(1, 0, 2).reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        v = self.v_lut(x_pos_flat).permute(1, 0, 2).reshape(self.n_heads, B, T, self.d_v).permute(1, 0, 2, 3)
        attn = self.rank_attn(q, k, v, is_causal=True).permute(0, 2, 1, 3).reshape(B * T, self.n_heads * self.d_v)
        return x + self.out_proj(attn)[:, 0, :].reshape(B, T, E)

class LUTTransformerV2(nn.Module):
    def __init__(self, cfg, maxlen=CONTEXT_SIZE):
        super().__init__()
        d, p = cfg['embedding_dim'], cfg['positional_dim']
        self.token_embedder = nn.Embedding(cfg['vocab_size'], d)
        self.register_buffer('pos_emb', torch.randn(1, maxlen, p) * 0.1)
        self.layers = nn.ModuleList([LUTBlock(cfg, i) for i in range(cfg['num_layers'])])
        self.unembedder = MultiHeadLut(
            input_dim=d, n_heads=1, n_outputs=cfg['vocab_size'],
            n_anchor_pairs=cfg['n_anchor_pairs'], tables_per_head=cfg['tables_per_head'],
            smooth_mode=cfg['smooth_mode'], n_alternatives=cfg['n_alternatives'],
            normalize_weights=False, calibrate_output=False,
            connected_anchors_mode=cfg['connected_anchors_mode'],
            initial_weights_noise=cfg['initial_weights_noise'],
            uncertainty_mode=UncertaintyMode.INVERSE_L1,
            random_seed=cfg['random_seed'] + 999, device=DEVICE,
        )
    def forward(self, tokens):
        B, T = tokens.shape
        x = self.token_embedder(tokens)
        pos = self.pos_emb[:, :T].expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, pos)
        return self.unembedder(x.reshape(-1, x.shape[-1]))[:, 0, :].reshape(B, T, -1)

model = LUTTransformerV2(cfg)
model.load_state_dict(torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded.")

# ── Hook into AnchorPairsLookup to capture lookup_indices ────────────────────
from spiky.lutorch.anchor_pairs_lookup import AnchorPairsLookup

# hit_counts[lut_name][table_idx] = set of bin indices seen
hit_counts = defaultdict(lambda: defaultdict(set))

hooks = []
def make_hook(lut_name):
    def hook(module, input, output):
        # output = (lookup_indices, lookup_alt_indices, lookup_alt_deltas) or just lookup_indices
        # In eval mode (smooth_mode=False, n_alternatives=1): returns (lookup_indices, None, None)
        if isinstance(output, tuple):
            indices = output[0]  # [B, n_tables]
        else:
            indices = output
        if indices is None:
            return
        # indices: [B, n_tables] — accumulate unique bin per table
        indices_np = indices.detach().cpu().numpy()  # [B, n_tables]
        n_tables = indices_np.shape[1]
        for t in range(n_tables):
            hit_counts[lut_name][t].update(indices_np[:, t].tolist())
    return hook

# Register hooks on all AnchorPairsLookup modules
for name, module in model.named_modules():
    if isinstance(module, AnchorPairsLookup):
        # strip '.lookup' suffix to get lut name
        lut_name = name.replace('.lookup', '')
        hooks.append(module.register_forward_hook(make_hook(lut_name)))

print(f"Registered {len(hooks)} hooks.")

# ── Generate text ─────────────────────────────────────────────────────────────
N_PROMPTS = 50
GEN_LEN   = 200
RAW_VOCAB  = 256
BOS_ID     = 256

prefixes = [
    "Once upon a time ",
    "The history of science ",
    "In the year 2024 ",
    "The best way to learn ",
    "Scientists have discovered ",
    "The city of London ",
    "A new study shows ",
    "The economy is ",
    "Technology has changed ",
    "Children learn best when ",
]

print(f"Generating {N_PROMPTS} sequences of length {GEN_LEN}...")
with torch.no_grad():
    for i in range(N_PROMPTS):
        prefix = prefixes[i % len(prefixes)]
        ctx = [BOS_ID] + list(prefix.encode('utf-8', errors='replace'))
        for _ in range(GEN_LEN):
            trunc = ctx[-(CONTEXT_SIZE - 1):]
            x = torch.zeros([1, CONTEXT_SIZE], dtype=torch.long, device=DEVICE)
            x[0, -len(trunc):] = torch.tensor(trunc, dtype=torch.long, device=DEVICE)
            logits = model(x)
            probs = torch.softmax(logits[:, -1, :RAW_VOCAB], dim=-1)[0]
            token = torch.multinomial(probs, 1).item()
            ctx.append(token)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_PROMPTS} done")

for h in hooks:
    h.remove()

# ── Analyze hit counts ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("INFERENCE BIN COVERAGE per LUT")
print("(how many of 64 possible bins each table actually hit during generation)")
print("=" * 70)

lut_names = sorted(hit_counts.keys())
n_entries = 2 ** cfg['n_anchor_pairs']  # 64

summary = []
print(f"{'LUT':<35}  {'n_tables':>8}  {'mean_bins':>10}  {'med_bins':>9}  {'1-bin%':>7}  {'full%':>7}")
for lut_name in lut_names:
    td = hit_counts[lut_name]
    n_tables = max(td.keys()) + 1
    bin_counts = [len(td[t]) for t in range(n_tables)]
    mean_bins = np.mean(bin_counts)
    med_bins  = np.median(bin_counts)
    one_bin   = np.mean([c == 1 for c in bin_counts]) * 100
    full      = np.mean([c == n_entries for c in bin_counts]) * 100
    print(f"{lut_name:<35}  {n_tables:>8}  {mean_bins:>10.2f}  {med_bins:>9.1f}  {one_bin:>6.1f}%  {full:>6.1f}%")
    summary.append((lut_name, n_tables, bin_counts, mean_bins))

# ── Per-LUT bin distribution histogram ────────────────────────────────────────
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = axes.flatten()

for idx, (lut_name, n_tables, bin_counts, mean_bins) in enumerate(summary):
    ax = axes[idx]
    ax.hist(bin_counts, bins=range(1, n_entries + 2), color='steelblue', edgecolor='white', linewidth=0.3)
    ax.set_title(lut_name.replace('layers.', 'L').replace('.', '/'), fontsize=8)
    ax.set_xlabel('bins used', fontsize=7)
    ax.set_ylabel('# tables', fontsize=7)
    ax.axvline(mean_bins, color='red', linestyle='--', linewidth=1, label=f'mean={mean_bins:.1f}')
    ax.legend(fontsize=6)

for idx in range(len(summary), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'exp098: Bins used per table during inference\n({N_PROMPTS} prompts × {GEN_LEN} tokens)', fontsize=13)
plt.tight_layout()
out_path = os.path.join(EXP_DIR, 'inference_bin_coverage.png')
plt.savefig(out_path, dpi=100, bbox_inches='tight')
print(f"\nPlot saved to {out_path}")

# ── Overall summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("OVERALL: fraction of tables that use only 1 bin (trivially dead)")
print("=" * 70)
all_counts = []
for lut_name, n_tables, bin_counts, _ in summary:
    all_counts.extend(bin_counts)
print(f"Total tables: {len(all_counts)}")
print(f"Mean bins used per table: {np.mean(all_counts):.2f} / {n_entries}")
print(f"Tables using exactly 1 bin:  {np.mean([c==1 for c in all_counts])*100:.1f}%")
print(f"Tables using <= 2 bins:      {np.mean([c<=2 for c in all_counts])*100:.1f}%")
print(f"Tables using >= 8 bins:      {np.mean([c>=8 for c in all_counts])*100:.1f}%")
print(f"Tables using all {n_entries} bins:    {np.mean([c==n_entries for c in all_counts])*100:.1f}%")
