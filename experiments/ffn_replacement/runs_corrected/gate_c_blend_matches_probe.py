"""Gate (c): the TRAINING-CAPABLE blend must reproduce the eval-only probe's number.

probe_soft_readout.py monkey-patched a blended forward from outside the layer and, being
eval-only, computed its weights from `d.detach()`. The in-layer implementation
(LightMultiHeadLUT.read_top_n) computes them from `d` so they carry gradient. Under
`no_grad` those are numerically the same quantity, so the two must agree -- if they do not,
the in-layer version differs from the thing that produced the -0.0107 result and I need to
know how.

Expected on exp_n_0200 at n=2, tau=0.1: 1.123830 (probe), against an unpatched 1.134523.

    python gate_c_blend_matches_probe.py [run_dir]
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'tools')))
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir                          # noqa: E402
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes  # noqa: E402
from model_build import build_model                               # noqa: E402
from fixed_eval import evaluate_bpb_fixed, eval_config            # noqa: E402
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402

RUN = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1
                   else 'exp_n_0200_light_margin_znorm_nap8_tph256_48k_seed1')
PROBE = {'exp_n_0200_light_margin_znorm_nap8_tph256_48k_seed1': (1.13452257, 1.12383047),
         'exp_n_0192_repro0191_seed1': (1.17707452, 1.17128616),
         'exp_n_0196_light_margin_znorm_nap8_tph256_seed1': (1.16391187, 1.15775836)}
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = json.load(open(os.path.join(RUN, 'config.json')))
ec = eval_config(cfg)
tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
tb = get_token_bytes(device=DEV)
model = build_model(cfg, tok.get_vocab_size(), device=DEV)
model.load_state_dict(torch.load(os.path.join(RUN, 'checkpoint.pt'), map_location=DEV),
                      strict=False)
model.eval()
lights = [m for m in model.modules() if isinstance(m, LightMultiHeadLUT)]
print(f'{os.path.basename(RUN)}: {len(lights)} Light modules, device {DEV}')

base = evaluate_bpb_fixed(model, tok, tb, cfg['seq_len'], DEV, **ec)
print(f'  in-layer, read_top_n=1 (default)     bpb = {base:.8f}')

for m in lights:
    m.read_top_n, m.read_tau = 2, 0.1
got = evaluate_bpb_fixed(model, tok, tb, cfg['seq_len'], DEV, **ec)
print(f'  in-layer, read_top_n=2 tau=0.1       bpb = {got:.8f}   '
      f'delta {got - base:+.8f}')

key = os.path.basename(RUN)
if key in PROBE:
    p_base, p_blend = PROBE[key]
    print(f'\n  probe_soft_readout.py reference       n=1 {p_base:.8f}   n=2 {p_blend:.8f}')
    d1, d2 = abs(base - p_base), abs(got - p_blend)
    print(f'  |in-layer - probe|                   n=1 {d1:.3e}       n=2 {d2:.3e}')
    ok = d1 < 1e-6 and d2 < 1e-6
    print(f'\nGATE C: {"PASS" if ok else "*** FAIL ***"}'
          f'  (the in-layer blend is the same computation as the probe)')
    sys.exit(0 if ok else 1)
