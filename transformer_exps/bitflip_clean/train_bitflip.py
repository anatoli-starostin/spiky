"""
Clean BitFlip experiment: train a single PermutationalLut to match a trained out_proj.

The target function was produced by a trained PermLut, so it's perfectly representable.
We train a fresh PermLut with binary ±1 weights using BitFlipOptimizer to match it.
"""
import sys, os, json, time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from spiky.lutorch.permutational_lut import PermutationalLut
from spiky.lutorch.bit_flip_optimizer import BitFlipOptimizer

DEVICE = 'cuda:0'
EXP_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
data = torch.load(os.path.join(EXP_DIR, 'dataset.pt'), weights_only=True)
x_all = data['x'].to(DEVICE)  # [10000, 32]
y_all = data['y'].to(DEVICE)  # [10000, 32]

N_TRAIN = 8000
N_TEST = 2000
x_train, x_test = x_all[:N_TRAIN], x_all[N_TRAIN:]
y_train, y_test = y_all[:N_TRAIN], y_all[N_TRAIN:]

# Create fresh PermLut with same architecture
PERM_KWARGS = dict(
    pair_mode='scrambled', soft_mode='ste', temperature=0.1,
    device=DEVICE, recompute_in_backward=True, initial_weights_noise=0.001,
)

model = PermutationalLut(
    n_inputs=32, n_outputs=32, n_heads=1,
    input_nap=6, output_nap=32, tph=2048,
    random_seed=999,  # different seed than target
    **PERM_KWARGS,
)

# BitFlip optimizer
bit_opt = BitFlipOptimizer([model], lr=0.01)

n_steps = 5000
batch_size = 128
test_every = 100

print(f'Weights: {model.inner.projection.weights.numel():,} (binary ±1)')
print(f'Train: {N_TRAIN}, Test: {N_TEST}, BS: {batch_size}')
print(f'BitFlip lr=0.01')

torch.manual_seed(42)
t0 = time.time()
best_test = float('inf')

for step in range(n_steps):
    # Sample batch
    idx = torch.randint(0, N_TRAIN, (batch_size,), device=DEVICE)
    xb = x_train[idx]
    yb = y_train[idx]

    bit_opt.zero_grad()
    out = model(xb).squeeze(1)  # [B, 32]
    loss = ((out - yb) ** 2).mean()
    loss.backward()
    bit_opt.step()

    if step % test_every == 0:
        model.eval()
        with torch.no_grad():
            test_out = model(x_test).squeeze(1)
            test_loss = ((test_out - y_test) ** 2).mean().item()
        model.train()
        if test_loss < best_test:
            best_test = test_loss
        elapsed = time.time() - t0
        print(f'step {step:5d} | train={loss.item():.4f} | test={test_loss:.4f} | best={best_test:.4f} | {elapsed:.1f}s')

elapsed = time.time() - t0
print(f'\nDone in {elapsed:.1f}s. Best test loss: {best_test:.4f}')
