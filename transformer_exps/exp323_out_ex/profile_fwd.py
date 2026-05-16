"""Profile fwd+bwd for q/k, v, out_proj(Ex) at exp323 config. Mirrors exp315's
profile but with BitPermutationLUTEx for out_proj. Reports ms/step."""
import os, sys, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.bit_permutation_lut_ex import BitPermutationLUTEx

DEVICE = 'cuda:0'
cfg = json.load(open(os.path.join(os.path.dirname(__file__), 'config.json')))
E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
B = cfg['batch_size']
T = cfg['context_size']
BT = B * T

N_WARMUP = 20
N_MEASURE = 200


def time_ms(fn):
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N_MEASURE):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / N_MEASURE


def bench(name, lut, x):
    def _fwd_only():
        with torch.no_grad():
            _ = lut(x)
    def _fb():
        xb = x.detach().requires_grad_(True)
        out = lut(xb)
        out.sum().backward()
    tf = time_ms(_fwd_only)
    tfb = time_ms(_fb)
    return tf, tfb - tf


def make_qk():
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_qk, n_heads=H,
        input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
        tph=cfg['qk_tph'], random_seed=0,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg['bit_lut_latent_dtype'],
        soft_backward=True, device=DEVICE,
    )

def make_v():
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_v, n_heads=H,
        input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'],
        tph=cfg['v_tph'], random_seed=0,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg['bit_lut_latent_dtype'],
        soft_backward=True, device=DEVICE,
    )

def make_out():
    return BitPermutationLUTEx(
        n_inputs=H * d_v, n_outputs=E, n_heads=1,
        input_nap=cfg['out_input_nap'], input_tph=cfg['out_input_tph'],
        voting_nap=cfg['out_voting_nap'],
        output_nap=cfg['out_output_nap'], output_tph=cfg['out_output_tph'],
        random_seed=0,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        soft_backward=cfg.get('bit_lut_soft_backward', True),
        latent_dtype=cfg['bit_lut_latent_dtype'],
        device=DEVICE,
    )


torch.manual_seed(0)
q = make_qk().to(DEVICE)
v = make_v().to(DEVICE)
o = make_out().to(DEVICE)

x_E = torch.randn(BT, E, device=DEVICE)
x_Hdv = torch.randn(BT, H * d_v, device=DEVICE)

N_LAYERS = cfg['num_layers']
mult = {'q/k': 2, 'v': 1, 'out_proj': 1}

print(f'=== exp323 profile, B·T={BT}, {N_MEASURE} iters ===')
print(f'{"lut":10s}  {"fwd":>8s}  {"bwd":>8s}  {"total":>8s}   ms/step')
agg_fwd, agg_bwd = 0.0, 0.0
for name, lut, x, m in [
    ('q/k', q, x_E, 2),
    ('v', v, x_E, 1),
    ('out_proj', o, x_Hdv, 1),
]:
    tf, tb = bench(name, lut, x)
    print(f'{name:10s}  {tf:8.3f}  {tb:8.3f}  {tf+tb:8.3f}')
    agg_fwd += tf * m
    agg_bwd += tb * m
print(f'{"per-layer":10s}  {agg_fwd:8.3f}  {agg_bwd:8.3f}  {agg_fwd+agg_bwd:8.3f}')
print(f'{"per-model":10s}  {agg_fwd*N_LAYERS:8.3f}  {agg_bwd*N_LAYERS:8.3f}  {(agg_fwd+agg_bwd)*N_LAYERS:8.3f}   '
      f'({N_LAYERS} layers × (2 q/k + 1 v + 1 out_proj))')
