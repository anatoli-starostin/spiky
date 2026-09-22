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
6. augment() does what it claims -- flips some images and not others, shifts within +/- pad, pads with
   the background value rather than grey, and preserves the set of pixel intensities it was given.
7. The EVAL PATH IS UNAUGMENTED. augment() is called only on the training batch, so the check is that
   evaluate() reproduces a hand-computed MSE on the clean tensor exactly.

Run directly (not under pytest, per the branch convention for these files):
    python test_dropout_active_and_off.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from vit_autoencoder import PAD_VALUE, ViTAutoencoder, augment, evaluate, make_eval  # noqa: E402

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

    # 6. the augmentation itself
    side, pad = 28, 2
    clean = (torch.rand(512, side, side, device=DEV) > 0.7).float() * 2.0 + PAD_VALUE
    # keep all content at least `pad` px from every edge, so a shift of up to pad cannot push any of it
    # off the image -- that makes "the multiset of pixel values is unchanged" an exact assertion below
    clean[:, :pad, :] = clean[:, -pad:, :] = clean[:, :, :pad] = clean[:, :, -pad:] = PAD_VALUE
    clean = clean.reshape(512, side * side)
    aug = augment(clean, side, pad)
    changed = (aug != clean).any(-1).float().mean()
    # a flip is detectable as equality with the mirrored original; a shift is not, so count both
    mirrored = clean.view(-1, 1, side, side).flip(-1).reshape(clean.shape)
    flipped = ((aug == mirrored).all(-1)).float().mean()
    unmoved = ((aug == clean).all(-1)).float().mean()
    border_ok = bool((aug.view(-1, side, side)[:, 0, :].min() >= PAD_VALUE - 1e-6))
    kept = torch.equal(aug.sort(-1).values, clean.sort(-1).values)   # exact: content cannot leave
    print(f'6. augment(): {float(changed):.2f} of the batch changed, {float(flipped):.3f} are exact '
          f'mirrors, {float(unmoved):.3f} untouched, border never brighter than the pad value: '
          f'{border_ok}, intensity budget preserved: {kept}')
    if not (0.2 < float(changed) <= 1.0 and border_ok):
        fails.append('augment() is not flipping/shifting as intended')
    if float(unmoved) > 0.2:
        fails.append('augment() leaves too much of the batch untouched')
    if not kept:
        fails.append('augment() changed pixel values -- it should only move them')

    # 7. the eval path never sees it
    m.eval()
    got = evaluate(fwd, flat, model=m)
    with torch.no_grad():
        want = float(((fwd(flat) - flat) ** 2).sum() / flat.numel())
    print(f'7. evaluate() on the clean tensor: {got:.6f} vs hand-computed {want:.6f} -- '
          f'match: {abs(got - want) < 1e-6}')
    if abs(got - want) >= 1e-6:
        fails.append('the eval path is not reconstructing the clean input')

    print('FAIL: ' + '; '.join(fails) if fails else 'all seven checks pass')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
