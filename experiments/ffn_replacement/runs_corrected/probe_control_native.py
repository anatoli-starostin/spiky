"""Control for probe_soft_readout.py: isolate kernel-vs-torch from the blend.

The unpatched module under `no_grad` takes `_fused_eval` -- a native CUDA kernel that
computes the address AND the score in one pass and never materialises the margins
(light_multi_head_lut.py:285-312). The probe's patched forward cannot use that kernel: it
needs the margins to choose the neighbouring cells. So the raw "unpatched vs n=1"
comparison mixes two changes at once:

  (a) kernel arithmetic vs torch arithmetic for the SCORE, and
  (b) the blend itself, which at n=1 is the identity.

This script removes (a) by disabling the native handles on every Light module, so the
unpatched forward also runs the torch path. If the resulting number equals the probe's
n=1 number, then (b) is exactly the identity as designed and the whole of the observed
drift is float ordering inside the score.
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
                   else 'exp_g_0190_B16k_light_bnorm_tph128_znorm_seed1')
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
n_native = sum(1 for m in lights if m._native_msb_scored is not None)
print(f'{len(lights)} Light modules; {n_native} have the native scored-eval kernel bound')

a = evaluate_bpb_fixed(model, tok, tb, cfg['seq_len'], DEV, **ec)
print(f'[A] unpatched, native kernel ENABLED   bpb = {a:.10f}')

for m in lights:
    m._native_msb_scored = None      # forces forward() past _fused_eval
    m._native_msb = None             # and _pack_index onto the torch expression
b = evaluate_bpb_fixed(model, tok, tb, cfg['seq_len'], DEV, **ec)
print(f'[B] unpatched, native kernel DISABLED  bpb = {b:.10f}')
print(f'    |A - B| = {abs(a - b):.3e}   <- pure kernel-vs-torch arithmetic')
print()
print('Compare [B] against probe_soft_readout.py n=1. If they agree to ~1e-9 the blend is')
print('the exact identity at n=1 and the probe is sound.')
