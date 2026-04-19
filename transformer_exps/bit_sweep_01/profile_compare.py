"""Compare per-step wall time of three kernel paths for the optimizer."""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
import spiky.lutorch.bit_permutation_lut_optimizer as optmod
from spiky.lutorch.bit_permutation_lut_optimizer import BitPermutationLUTOptimizer

device = torch.device('cuda:0')
torch.manual_seed(0)

# out_proj-sized LUT.
CFG = dict(
    n_inputs=64, n_outputs=32, n_heads=1,
    input_nap=10, output_nap=32, tph=1024,
    random_seed=0, initial_weights_noise=0.001,
    device=device,
)
N_WARMUP = 20
N_MEASURE = 200
BS = 64
x = torch.randn(BS, 64, device=device)

import lutorch_cuda
mgr = lutorch_cuda.get_lutorch_manager()


def bench(path_name: str, disable: set):
    """Monkey-patch lutorch manager to hide kernels in `disable`,
    force the optimizer down a specific path."""
    saved = {}
    for name in disable:
        saved[name] = getattr(type(mgr), name, None)
    # Patch the get_lutorch_manager-returned object via wrapper.
    class HidingMgr:
        def __init__(self, inner, hidden):
            self._inner = inner
            self._hidden = hidden
        def __getattr__(self, n):
            if n in self._hidden:
                raise AttributeError(n)
            return getattr(self._inner, n)
    hider = HidingMgr(mgr, disable)

    # Swap the module-level _get_bit_permlut_native to return the hider.
    original = optmod._get_bit_permlut_native
    optmod._get_bit_permlut_native = lambda: hider
    try:
        torch.manual_seed(0)
        lut = BitPermutationLUT(**CFG)
        opt = BitPermutationLUTOptimizer([lut], lr=1e-3)
        # warmup
        for _ in range(N_WARMUP):
            xb = x.detach().requires_grad_(True)
            lut(xb).sum().backward()
            opt.step()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(N_MEASURE):
            xb = x.detach().requires_grad_(True)
            lut(xb).sum().backward()
            opt.step()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / N_MEASURE
        opt.close()
    finally:
        optmod._get_bit_permlut_native = original
    print(f'  {path_name:35s}: {ms:.3f} ms/step')


print('Out_proj-sized LUT (N=1024, td=1024, nap=32)')
bench('legacy fused_fp8_adam',
      disable={'fused_fp8_adam_full_inkernel', 'fused_fp8_adam_latent_inplace'})
bench('latent-inplace + Python m/v quant',
      disable={'fused_fp8_adam_full_inkernel'})
bench('fully fused (2 kernels, all in-kernel)',
      disable=set())
