"""Checks on the --dropout knob that the runs depend on being true.

1. p = 0 is an exact identity: the model built with dropout=0.0 gives bit-identical outputs to the same
   model in train and eval mode, so every run made before the knob existed is unaffected.
2. p > 0 is ACTIVE in train mode: two forward passes on the same input differ.
3. p > 0 is OFF in eval mode: two forward passes on the same input are bit-identical, and so is a pass
   after switching back and forth.
4. evaluate() measures the eval-mode model even when called mid-training, and leaves the module in the
   mode it found it in. This is the one that would silently corrupt the logged curves if wrong.
5. p = 0 keeps the ORIGINAL state_dict key layout, so checkpoints written before the knob existed still
   load. An nn.Dropout carries no parameters, but putting one inside the FFN's Sequential renumbers the
   second Linear -- which is exactly how this broke the first time.

Run directly (not under pytest, per the branch convention for these files):
    python test_dropout_active_and_off.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from vit_autoencoder import ViTAutoencoder, evaluate, make_eval  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ARGS = dict(n_tokens=196, patch_dim=4, d_model=64, n_heads=4, enc_layers=4, dec_layers=4,
            latent=64, latent_tokens=8, ffn_mult=4, ffn_kind='mlp', n_tables=64, device=DEV, seed=0)


def main():
    x = torch.randn(8, 196, 4, device=DEV)
    fails = []

    m0 = ViTAutoencoder(**ARGS, dropout=0.0)
    m0.train()
    a, b = m0(x), m0(x)
    m0.eval()
    c = m0(x)
    print(f'1. p=0 train-vs-train identical: {torch.equal(a, b)}; train-vs-eval identical: '
          f'{torch.equal(a, c)}')
    if not (torch.equal(a, b) and torch.equal(a, c)):
        fails.append('p=0 is not an identity -- pre-existing runs would change')

    m = ViTAutoencoder(**ARGS, dropout=0.2)
    m.train()
    t1, t2 = m(x), m(x)
    differ = not torch.equal(t1, t2)
    spread = float((t1 - t2).abs().mean().detach())
    print(f'2. p=0.2 train mode stochastic: {differ}  (mean |diff| {spread:.4f})')
    if not differ:
        fails.append('dropout is NOT active in train mode')

    m.eval()
    e1, e2 = m(x), m(x)
    same = torch.equal(e1, e2)
    m.train()
    m.eval()
    e3 = m(x)
    print(f'3. p=0.2 eval mode deterministic: {same}; stable across a train/eval round trip: '
          f'{torch.equal(e1, e3)}')
    if not (same and torch.equal(e1, e3)):
        fails.append('dropout is still firing in eval mode')

    m.train()
    fwd = make_eval(m, 'vit', 28, 2)
    flat = torch.randn(64, 784, device=DEV)
    v1 = evaluate(fwd, flat, model=m)
    restored = m.training
    v2 = evaluate(fwd, flat, model=m)
    print(f'4. evaluate() from train mode: {v1:.6f} then {v2:.6f} -- identical: {v1 == v2}; '
          f'left the module in train mode: {restored}')
    if v1 != v2:
        fails.append('evaluate() is measuring a dropped-out network')
    if not restored:
        fails.append('evaluate() did not restore train mode')

    k0, k2 = set(m0.state_dict()), set(m.state_dict())
    legacy = {'enc.0.ffn.0.weight', 'enc.0.ffn.2.weight', 'dec.3.ffn.2.bias'}
    print(f'5. p=0 keeps the pre-dropout key layout: {legacy <= k0}; p=0.2 has its own '
          f'({"enc.0.ffn.3.weight" in k2}); same number of tensors: {len(k0) == len(k2)}')
    if not legacy <= k0:
        fails.append('p=0 state_dict keys moved -- older checkpoints will not load')

    print('FAIL: ' + '; '.join(fails) if fails else 'all five checks pass')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
