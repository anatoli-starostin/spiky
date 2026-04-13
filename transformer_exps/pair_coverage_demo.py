"""
Demo: pair coverage analytics for a MultiHeadLut with input_dim=32, tph=256, nap=4.
Shows how many of the 496 unique pairs are covered, distribution of pair frequency,
and how many tables are fully connected vs disconnected.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from collections import Counter
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode

INPUT_DIM = 32
N_HEADS = 1
N_OUTPUTS = 32
NAP = 4
TPH = 256
DEVICE = 'cpu'

lut = MultiHeadLut(
    input_dim=INPUT_DIM,
    n_heads=N_HEADS,
    n_outputs=N_OUTPUTS,
    n_anchor_pairs=NAP,
    tables_per_head=TPH,
    smooth_mode=False,
    n_alternatives=1,
    normalize_weights=False,
    calibrate_output=False,
    connected_anchors_mode=False,
    initial_weights_noise=0.001,
    uncertainty_mode=UncertaintyMode.INVERSE_L1,
    random_seed=42,
    device=DEVICE,
)

apl = lut.lookup  # AnchorPairsLookup
a = apl.anchor_pairs_a  # [n_tables, nap]
b = apl.anchor_pairs_b  # [n_tables, nap]
n_tables = a.shape[0]

print(f"input_dim={INPUT_DIM}, nap={NAP}, tph={TPH}")
print(f"n_tables (tph * n_heads): {n_tables}")
print(f"total unique pairs possible: {INPUT_DIM * (INPUT_DIM - 1) // 2}")
print(f"total pair slots: {n_tables * NAP}")
print()

# Normalize pairs so i < j
all_pairs = []
for t in range(n_tables):
    for p in range(NAP):
        i, j = a[t, p].item(), b[t, p].item()
        all_pairs.append((min(i, j), max(i, j)))

pair_counts = Counter(all_pairs)
n_unique_possible = INPUT_DIM * (INPUT_DIM - 1) // 2
n_covered = len(pair_counts)
n_missing = n_unique_possible - n_covered

print(f"=== Pair coverage ===")
print(f"Unique pairs covered: {n_covered} / {n_unique_possible} ({100*n_covered/n_unique_possible:.1f}%)")
print(f"Pairs never seen:     {n_missing} ({100*n_missing/n_unique_possible:.1f}%)")
print(f"Max times a pair appears: {max(pair_counts.values())}")
print(f"Mean times a pair appears: {sum(pair_counts.values())/len(pair_counts):.2f}")
print()

# Per-table connectivity: how many indices are shared across the nap pairs?
n_disconnected = 0  # all 2*nap indices distinct
n_partial = 0
n_fully_connected = 0

for t in range(n_tables):
    indices = set()
    for p in range(NAP):
        indices.add(a[t, p].item())
        indices.add(b[t, p].item())
    n_distinct = len(indices)
    if n_distinct == 2 * NAP:
        n_disconnected += 1
    elif n_distinct <= NAP + 1:
        n_fully_connected += 1
    else:
        n_partial += 1

print(f"=== Table connectivity ===")
print(f"Fully disconnected (8 distinct indices): {n_disconnected} / {n_tables} ({100*n_disconnected/n_tables:.1f}%)")
print(f"Partially connected:                     {n_partial} / {n_tables} ({100*n_partial/n_tables:.1f}%)")
print(f"Highly connected (≤5 distinct indices):  {n_fully_connected} / {n_tables} ({100*n_fully_connected/n_tables:.1f}%)")
