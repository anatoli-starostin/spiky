"""Routing entropy analysis for exp364 (or any TinyMHLut-based LUT-LM).

For each LUT module in each layer:
  - Hook the forward of TinyMultiHeadLut to capture `index` (selected row per
    table per token).
  - Aggregate row-occupancy histograms across many val tokens.
  - Compute per-table entropy H(p) = -Σ p_k log2 p_k.
  - Compute global metrics: mean entropy, dead-row count, top-1 mass.

Usage:
  python analyze_routing_entropy.py [EXP_DIR]

Loads <EXP_DIR>/checkpoint.pt and <EXP_DIR>/train.py via importlib to
reconstruct the model exactly.
"""
import os, sys, json, math, importlib.util, csv
import torch
import torch.nn.functional as F

EXP_DIR = sys.argv[1] if len(sys.argv) > 1 else '/home/starost/spiky/nanochat_exps/exp364_bs192'

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

# Suppress the train.py's "if __name__ == '__main__'" issue — we just want
# the classes. We'll import train as a module and re-instantiate.
sys.path.insert(0, EXP_DIR)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda'
torch.manual_seed(cfg['random_seed'])

# Quickest: exec the train.py up to Model class definition, then load weights.
# We'll redefine just enough to instantiate the model.
# A simpler way: import the model classes by execing the file.
print(f'Loading model from {EXP_DIR}/train.py and checkpoint.pt...')

# Read train.py source up to the build/train section so we get class definitions.
with open(os.path.join(EXP_DIR, 'train.py')) as f:
    src = f.read()

# Cut off at training loop (find marker)
markers = ['# --- Build + optimiser', '# --- Build + train', 'model = Model()', 'model = Model(']
cut = None
for m in markers:
    idx = src.find(m)
    if idx > 0:
        cut = idx
        break
if cut is None:
    print('Could not find model build marker — using full train.py (will train!).')
    cut = len(src)
header_src = src[:cut]

# Replace the loaders to avoid initializing right now
namespace = {'__name__': '__analyze__', '__file__': os.path.join(EXP_DIR, 'train.py')}
# Pre-populate to skip the loader init lines? Easier: just run header.
# But the train.py defines train_loader at top — let's just exec and tolerate.
exec(compile(header_src, '<analyze>', 'exec'), namespace)

Model = namespace['Model']
model = Model().to(DEVICE)
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE,
                  weights_only=False)
# checkpoint may be model.state_dict() directly or a wrapper
if isinstance(ckpt, dict) and 'model' in ckpt:
    state = ckpt['model']
else:
    state = ckpt
missing, unexpected = model.load_state_dict(state, strict=False)
print(f'Loaded: missing={len(missing)}, unexpected={len(unexpected)}')
model.eval()

# Find all TinyMHLut modules
modules = []
for name, m in model.named_modules():
    if isinstance(m, TinyMultiHeadLut):
        modules.append((name, m))
print(f'Found {len(modules)} TinyMHLut modules')
for name, m in modules:
    K = 1 << m.n_anchor_pairs
    print(f'  {name}: n_heads={m.n_heads} tph={m.tables_per_head} NAP={m.n_anchor_pairs} K={K} n_out={m.n_outputs}')

# Hook each module to capture the `index` tensor from the forward pass
captured = {name: [] for name, _ in modules}

def make_hook(name, K):
    def hook(mod, inputs, output):
        # The TinyMHLut forward through autograd.Function does sign-pack + argmax
        # internally. We can re-derive index from inputs if needed, but the
        # cleanest way is to call the lookup explicitly.
        # However: the public forward already computed `index`. We can recompute
        # by replicating the sign-pack logic. Simpler: re-run the lookup which
        # is fast.
        x = inputs[0]   # [B*T, input_dim]
        # Replicate sign-pack: anchor diff -> sign -> bit pack
        # TinyMHLut stores anchor_a_long / anchor_b_long at module level.
        a_idx = mod.soft_anchor_a_long   # [n_tables, NAP]
        b_idx = mod.soft_anchor_b_long
        # Diff
        x_a = x[:, a_idx]   # [B*T, n_tables, NAP]
        x_b = x[:, b_idx]
        d = x_a - x_b      # [B*T, n_tables, NAP]
        bits = (d > 0).long()  # [B*T, n_tables, NAP], MSB-first per bit_matrix
        # Pack into row index
        powers = 2 ** torch.arange(mod.n_anchor_pairs - 1, -1, -1, device=x.device)
        index = (bits * powers[None, None, :]).sum(dim=-1)  # [B*T, n_tables]
        # Histogram (sum over tokens) per table
        n_tables = index.shape[1]
        flat = index + torch.arange(n_tables, device=x.device)[None, :] * K
        hist_flat = torch.bincount(flat.reshape(-1), minlength=n_tables * K)
        captured[name].append(hist_flat.view(n_tables, K).cpu())
    return hook

for name, m in modules:
    K = 1 << m.n_anchor_pairs
    m.register_forward_hook(make_hook(name, K))

# Build val loader & run a number of batches
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)

CONTEXT_SIZE = cfg['context_size']
DEVICE_BS = 16
N_BATCHES = 32  # 32 * 16 * 512 = 262K tokens

val_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE)

print(f'Running {N_BATCHES} val batches (~{N_BATCHES * DEVICE_BS * CONTEXT_SIZE:,} tokens)...')
with torch.no_grad():
    for _ in range(N_BATCHES):
        x, _ = next(val_loader)
        model(x)

# Aggregate hists per module, compute entropy
print(f'\n=== Per-module routing entropy ({N_BATCHES * DEVICE_BS * CONTEXT_SIZE:,} tokens) ===')
print(f'{"module":<24} {"NAP":<4} {"max H":<7} {"mean H":<8} {"med H":<8} {"min H":<8} {"dead rows %":<12} {"top-1 mass %":<13}')

summary_rows = []
for name, m in modules:
    K = 1 << m.n_anchor_pairs
    NAP = m.n_anchor_pairs
    hist = sum(captured[name])  # [n_tables, K], total counts
    total = hist.sum(dim=-1, keepdim=True).clamp(min=1).to(torch.float32)
    p = hist.to(torch.float32) / total   # [n_tables, K]
    # Entropy per table
    log_p = torch.where(p > 0, p.log2(), torch.zeros_like(p))
    H = -(p * log_p).sum(dim=-1)   # [n_tables], in bits
    # Dead rows (visited < 1% of expected uniform = visited fraction < 0.01/K)
    dead_threshold = 1e-6
    dead_frac = (p < dead_threshold).to(torch.float32).mean().item()
    # Top-1 mass
    top1 = p.max(dim=-1).values.mean().item()
    summary_rows.append({
        'module': name, 'NAP': NAP, 'K': K,
        'max_H': float(NAP),
        'mean_H': H.mean().item(),
        'median_H': H.median().item(),
        'min_H': H.min().item(),
        'dead_frac_pct': dead_frac * 100,
        'top1_pct': top1 * 100,
    })
    print(f'{name:<24} {NAP:<4} {NAP:<7} {H.mean().item():<8.3f} {H.median().item():<8.3f} {H.min().item():<8.3f} {dead_frac*100:<12.2f} {top1*100:<13.2f}')

# Save CSV
out_csv = os.path.join(EXP_DIR, 'routing_entropy.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    for r in summary_rows:
        w.writerow(r)
print(f'\nSaved {out_csv}')
