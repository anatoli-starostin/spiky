"""
Generate synthetic dataset from exp282's trained out_proj PermutationalLut.

Samples random inputs, runs through the trained out_proj, saves (input, output) pairs.
This gives a target function that a PermLut CAN represent perfectly (since it was
produced by one), making it ideal for testing discrete-weight training convergence.
"""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from spiky.lutorch.permutational_lut import PermutationalLut

DEVICE = 'cuda:0'
SRC_EXP = 'transformer_exps/exp282_fullperm_sumpos'
OUT_DIR = 'transformer_exps/bitflip_clean'

# Recreate the out_proj with same config
PERM_KWARGS = dict(
    pair_mode='scrambled', soft_mode='ste', temperature=0.1,
    device=DEVICE, recompute_in_backward=True, initial_weights_noise=0.001,
)

out_proj = PermutationalLut(
    n_inputs=4 * 8, n_outputs=32, n_heads=1,
    input_nap=6, output_nap=32, tph=2048,
    random_seed=42 + 400,  # same seed_offset as layer 0 out_proj
    **PERM_KWARGS,
)

# Load trained weights from checkpoint
ckpt = torch.load(os.path.join(SRC_EXP, 'checkpoint.pt'), map_location=DEVICE, weights_only=True)
# Find layer 0 out_proj weights
out_proj_key = 'layers.0.out_proj.inner.projection.weights'
out_proj.inner.projection.weights.data.copy_(ckpt[out_proj_key])
print(f'Loaded out_proj weights from {out_proj_key}')
print(f'Weight range: [{out_proj.inner.projection.weights.data.min():.4f}, {out_proj.inner.projection.weights.data.max():.4f}]')

# Generate dataset
out_proj.eval()
N = 10000
torch.manual_seed(123)

with torch.no_grad():
    x = torch.randn(N, 4 * 8, device=DEVICE)
    y = out_proj(x)  # [N, 1, 32]
    y = y.squeeze(1)  # [N, 32]

print(f'Generated {N} samples')
print(f'x shape: {x.shape}, y shape: {y.shape}')
print(f'y range: [{y.min():.4f}, {y.max():.4f}], y std: {y.std():.4f}')

# Save
os.makedirs(OUT_DIR, exist_ok=True)
torch.save({'x': x.cpu(), 'y': y.cpu()}, os.path.join(OUT_DIR, 'dataset.pt'))
print(f'Saved to {OUT_DIR}/dataset.pt')
