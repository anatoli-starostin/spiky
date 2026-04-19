"""Profile fwd/bwd/opt for q/k, v, out_proj LUTs across {fp8, bf16, fp32}
latent modes with soft backward. Reports per-step ms for each phase and
aggregates the per-layer cost.
"""
import sys, os, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.bit_permutation_lut_optimizer import BitPermutationLUTOptimizer

DEVICE = 'cuda:0'
cfg = json.load(open(os.path.join(
    os.path.dirname(__file__), '..', 'exp312_bf16_soft', 'config.json'
)))
E  = cfg['embedding_dim']            # 32
H  = cfg['n_heads']                  # 4
d_qk = cfg['d_qk']                   # 24
d_v  = cfg['d_v']                    # 16
B    = cfg['batch_size']             # 8
T    = cfg['context_size']           # 128
BT   = B * T                         # 1024

LATENT_MODES = ['fp8', 'bf16', 'fp32']

SPECS = [
    ('q/k',      dict(n_inputs=E,      n_outputs=d_qk, n_heads=H, input_nap=cfg['qk_input_nap'],  output_nap=cfg['qk_output_nap'],  tph=cfg['qk_tph']),      BT, E),
    ('v',        dict(n_inputs=E,      n_outputs=d_v,  n_heads=H, input_nap=cfg['v_input_nap'],   output_nap=cfg['v_output_nap'],   tph=cfg['v_tph']),       BT, E),
    ('out_proj', dict(n_inputs=H*d_v,  n_outputs=E,    n_heads=1, input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'], tph=cfg['out_tph']),     BT, H*d_v),
]

N_WARMUP = 20
N_MEASURE = 200

def time_ms(fn):
    """Average ms/step over N_MEASURE iterations."""
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N_MEASURE):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / N_MEASURE


def bench(name, lut_kwargs, batch_size, in_dim, latent_dtype):
    torch.manual_seed(0)
    lut = BitPermutationLUT(
        **lut_kwargs, random_seed=0,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=latent_dtype,
        soft_backward=True,
        device=DEVICE,
    )
    opt = BitPermutationLUTOptimizer([lut], lr=1e-3)
    x = torch.randn(batch_size, in_dim, device=DEVICE)

    # Forward only.
    def _fwd_only():
        with torch.no_grad():
            _ = lut(x)
    # Forward + backward (with graph retained for reuse of x).
    # Each call re-runs forward + backward from scratch.
    def _fwd_bwd():
        xb = x.detach().requires_grad_(True)
        out = lut(xb)
        out.sum().backward()
    # Full step (forward + backward + optimizer).
    def _full():
        xb = x.detach().requires_grad_(True)
        out = lut(xb)
        out.sum().backward()
        opt.step()

    # Measure wall time for three nested workloads, then subtract.
    t_fwd  = time_ms(_fwd_only)
    t_fb   = time_ms(_fwd_bwd)
    t_full = time_ms(_full)

    t_bwd  = t_fb   - t_fwd
    t_opt  = t_full - t_fb
    opt.close()

    return dict(fwd=t_fwd, bwd=t_bwd, opt=t_opt, total=t_full)


N_LAYERS = cfg['num_layers']
mult = {'q/k': 2, 'v': 1, 'out_proj': 1}

for dtype in LATENT_MODES:
    print(f'\n=== latent_dtype={dtype} (soft backward, B×T={BT}, '
          f'{N_MEASURE} iters after {N_WARMUP} warmup) ===')
    print(f'{"lut":10s}  {"fwd":>8s}  {"bwd":>8s}  {"opt":>8s}  {"total":>8s}   ms/step')
    results = {}
    for name, kwargs, bs, in_dim in SPECS:
        r = bench(name, kwargs, bs, in_dim, dtype)
        results[name] = r
        print(f'{name:10s}  {r["fwd"]:8.3f}  {r["bwd"]:8.3f}  {r["opt"]:8.3f}  {r["total"]:8.3f}')
    agg = {k: 0.0 for k in ('fwd', 'bwd', 'opt', 'total')}
    for name, r in results.items():
        for k in agg:
            agg[k] += r[k] * mult[name] * N_LAYERS
    print(f'{"all":10s}  {agg["fwd"]:8.3f}  {agg["bwd"]:8.3f}  {agg["opt"]:8.3f}  '
          f'{agg["total"]:8.3f}   ({N_LAYERS} layers × (2 q/k + 1 v + 1 out_proj))')
