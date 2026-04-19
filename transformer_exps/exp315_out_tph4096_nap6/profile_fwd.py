"""Profile bit_perm_lut_dom_gather_fwd_kernel for exp315's out_proj shape.

Runs forward-only for q/k, v, out_proj with exp315's config, reports ms/step.
Purpose: baseline timing BEFORE kernel rewrite (warp-cooperative + atomicAdd);
run again AFTER rebuild to measure speedup.
"""
import os, sys, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT

DEVICE = 'cuda:0'
cfg = json.load(open(os.path.join(os.path.dirname(__file__), 'config.json')))
E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
B = cfg['batch_size']
T = cfg['context_size']
BT = B * T   # 32 * 128 = 4096

SPECS = [
    ('q/k',      dict(n_inputs=E, n_outputs=d_qk, n_heads=H,
                      input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
                      tph=cfg['qk_tph']), BT, E),
    ('v',        dict(n_inputs=E, n_outputs=d_v, n_heads=H,
                      input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'],
                      tph=cfg['v_tph']), BT, E),
    ('out_proj', dict(n_inputs=H*d_v, n_outputs=E, n_heads=1,
                      input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'],
                      tph=cfg['out_tph']), BT, H*d_v),
]

N_WARMUP = 20
N_MEASURE = 200


def time_ms(fn):
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


def bench(name, kwargs, batch_size, in_dim):
    torch.manual_seed(0)
    lut = BitPermutationLUT(
        **kwargs, random_seed=0,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg['bit_lut_latent_dtype'],
        soft_backward=True,
        device=DEVICE,
    )
    x = torch.randn(batch_size, in_dim, device=DEVICE)

    # K and inv_idx shape — for reporting.
    n_heads = kwargs['n_heads']
    tph = kwargs['tph']
    output_nap = kwargs['output_nap']
    n_outputs = kwargs['n_outputs']
    P = n_outputs * (n_outputs - 1) // 2
    avg_K = tph * output_nap / P
    K_actual = lut.inv_idx.size(-1)

    def _fwd():
        with torch.no_grad():
            _ = lut(x)

    t = time_ms(_fwd)
    return t, K_actual, avg_K, P


N_LAYERS = cfg['num_layers']
mult = {'q/k': 2, 'v': 1, 'out_proj': 1}

print(f'=== exp315 forward profile, B·T={BT}, {N_MEASURE} iters ===')
print(f'{"lut":10s}  {"tph":>5s}  {"in_nap":>6s}  {"out_nap":>7s}  '
      f'{"P":>5s}  {"K":>5s}  {"avg_K":>7s}  {"ms/step":>8s}')
total_layer_cost = 0.0
for name, kwargs, bs, in_dim in SPECS:
    t, K, avg_K, P = bench(name, kwargs, bs, in_dim)
    print(f'{name:10s}  {kwargs["tph"]:>5d}  {kwargs["input_nap"]:>6d}  '
          f'{kwargs["output_nap"]:>7d}  {P:>5d}  {K:>5d}  {avg_K:>7.1f}  {t:>8.3f}')
    total_layer_cost += t * mult[name]
print(f'{"all/layer":10s}  {total_layer_cost:>72.3f}')
print(f'{"all/model":10s}  {total_layer_cost * N_LAYERS:>72.3f}   '
      f'({N_LAYERS} layers x (2 q/k + 1 v + 1 out_proj))')
