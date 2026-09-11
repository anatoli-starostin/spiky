"""Bit-identity regression for the confidence-form code: capture outputs, change the code, compare.

    python regress_confidence_forms.py capture before   # on the UNCHANGED code
    python regress_confidence_forms.py capture after    # after adding a form
    python regress_confidence_forms.py compare          # every tensor must be torch.equal

Covers the three pre-existing forms only (bounded, bounded_norm, margin):
  * _confidence_score and _confidence_score_and_dscore, fp32 and fp64, gain 1.0 and 2.5
  * LightMultiHeadLUT forward + backward (CPU fp64), read_top_n 1 and 2, margin and bounded_norm
  * LightMultiHeadLUT no-grad CUDA eval (the native fused scored kernel), margin, n=1
  * FastMultiHeadLut hard forward with the confidence gate (margin), forward + backward
Everything is rebuilt from fixed seeds, so the before/after runs construct identical modules.
"""
import os
import sys

os.environ.setdefault('LUT_DISABLE_COMPILE', '1')
import torch                                                                       # noqa: E402
from spiky.lutorch.fast_multi_head_lut import (                                    # noqa: E402
    FastMultiHeadLut, _confidence_score, _confidence_score_and_dscore)
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT                   # noqa: E402

FORMS = ('bounded', 'bounded_norm', 'margin')
OUT = '/tmp/regress_confidence_forms_{}.pt'


def capture(tag):
    out = {}
    g = torch.Generator().manual_seed(123)
    base = torch.randn(64, 32, 8, generator=g, dtype=torch.float64)
    for dt in (torch.float32, torch.float64):
        d = base.to(dt)
        for form in FORMS:
            for gain in (1.0, 2.5):
                out[f'score/{form}/{gain}/{dt}'] = _confidence_score(d, form, gain)
                s, ds = _confidence_score_and_dscore(d, form, gain)
                out[f'score_and_dscore.s/{form}/{gain}/{dt}'] = s
                out[f'score_and_dscore.ds/{form}/{gain}/{dt}'] = ds

    for n in (1, 2):
        for form in ('margin', 'bounded_norm'):
            m = LightMultiHeadLUT(input_dim=48, n_tables=4 * 32, output_dim=48, n_anchor_pairs=8,
                                  confidence_form=form, random_seed=1000, n_heads=4,
                                  multi_head_input=True, read_top_n=n, read_tau=0.5)
            m._compile_enabled = False
            m = m.double()
            gz = torch.Generator().manual_seed(7)
            z = torch.randn(16, 4, 48, generator=gz, dtype=torch.float64).requires_grad_(True)
            y = m(z)
            go = torch.randn(y.shape, generator=gz, dtype=torch.float64)
            (y * go).sum().backward()
            out[f'light/{form}/n{n}/y'] = y.detach()
            out[f'light/{form}/n{n}/zgrad'] = z.grad.detach()
            out[f'light/{form}/n{n}/tablegrad'] = m.tables.grad.detach()
            if n == 2:
                out[f'light/{form}/n{n}/taugrad'] = (m.log_tau.grad.detach()
                                                     if m.log_tau.grad is not None else torch.zeros(()))

    if torch.cuda.is_available():
        m = LightMultiHeadLUT(input_dim=48, n_tables=4 * 32, output_dim=48, n_anchor_pairs=8,
                              confidence_form='margin', random_seed=1000, n_heads=4,
                              multi_head_input=True, read_top_n=1, device='cuda')
        gz = torch.Generator().manual_seed(9)
        z = torch.randn(16, 4, 48, generator=gz).cuda()
        with torch.no_grad():
            out['light_cuda_native_eval/margin/y'] = m(z).cpu()
        out['light_cuda_native_eval/used_native'] = torch.tensor(
            int(m._fused_eval(z.reshape(16, -1).contiguous()) is not None))

    try:
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        f = FastMultiHeadLut(input_dim=48, n_heads=4, n_outputs=48, n_anchor_pairs=8,
                             tables_per_head=32, forward_mode='hard', forward_confidence=True,
                             confidence_form='margin', multi_head_input=True, random_seed=1000,
                             device=dev, use_bf16=False)
        gz = torch.Generator().manual_seed(11)
        x = torch.randn(16, 4 * 48, generator=gz).to(dev).requires_grad_(True)
        y = f(x)
        go = torch.randn(y.shape, generator=gz).to(dev)
        (y * go).sum().backward()
        out['fast/margin/y'] = y.detach().cpu()
        out['fast/margin/xgrad'] = x.grad.detach().cpu()
        out['fast/margin/wgrad'] = next(p for p in f.parameters() if p.dim() >= 2).grad.detach().cpu()
    except Exception as e:                                   # recorded, not hidden
        out['fast/skipped'] = repr(e)

    torch.save(out, OUT.format(tag))
    print(f'captured {len(out)} entries -> {OUT.format(tag)}'
          + (f"  (fast skipped: {out['fast/skipped']})" if 'fast/skipped' in out else ''))


def compare():
    a, b = torch.load(OUT.format('before')), torch.load(OUT.format('after'))
    bad = []
    for k in sorted(a):
        if k not in b:
            bad.append(f'missing after: {k}')
        elif isinstance(a[k], str) or isinstance(b[k], str):
            if a[k] != b[k]:
                bad.append(f'{k}: {a[k]!r} vs {b[k]!r}')
        elif not torch.equal(a[k], b[k]):
            bad.append(f'{k}: max |diff| {(a[k].double() - b[k].double()).abs().max().item():.3e}')
    print(f'compared {len(a)} entries: ' + ('ALL BIT-IDENTICAL' if not bad else f'{len(bad)} DIFFER'))
    for x in bad:
        print('   ', x)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    if sys.argv[1] == 'capture':
        capture(sys.argv[2])
    else:
        compare()
