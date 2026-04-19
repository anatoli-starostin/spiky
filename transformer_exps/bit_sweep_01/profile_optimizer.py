"""Profile the BitPermutationLUTOptimizer step on a large out_proj-sized LUT.

Runs N warmup + N measured steps using CUDA events timing. Reports:
  - per-step wall time
  - peak-allocated memory delta per step
  - which kernel path (legacy or fused_fp8_adam_latent_inplace) is active
"""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.bit_permutation_lut_optimizer import BitPermutationLUTOptimizer

device = torch.device('cuda:0')
torch.manual_seed(0)

# out_proj-sized LUT from exp299: n_heads=1, tph=1024, input_nap=10 (td=1024), output_nap=32
lut = BitPermutationLUT(
    n_inputs=64, n_outputs=32, n_heads=1,
    input_nap=10, output_nap=32, tph=1024,
    random_seed=0, initial_weights_noise=0.001,
    device=device,
)
opt = BitPermutationLUTOptimizer([lut], lr=1e-3)

try:
    from lutorch_cuda import get_lutorch_manager
    mgr = get_lutorch_manager()
    has_inplace = hasattr(mgr, "fused_fp8_adam_latent_inplace")
    print(f'fused_fp8_adam_latent_inplace available: {has_inplace}')
except Exception as e:
    print(f'error importing lutorch_cuda: {e}')
    has_inplace = False

N_WARMUP = 20
N_MEASURE = 200
BS = 64

x = torch.randn(BS, 64, device=device)

def do_step():
    xb = x.detach().requires_grad_(True)
    out = lut(xb)
    out.sum().backward()
    opt.step()

# Warmup
for _ in range(N_WARMUP):
    do_step()

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(N_MEASURE):
    do_step()
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
per_step_ms = elapsed_ms / N_MEASURE
peak_bytes = torch.cuda.max_memory_allocated()
print(f'per_step: {per_step_ms:.3f} ms, total: {elapsed_ms:.1f} ms over {N_MEASURE} steps')
print(f'peak allocated: {peak_bytes / 1e6:.1f} MB')

opt.close()
