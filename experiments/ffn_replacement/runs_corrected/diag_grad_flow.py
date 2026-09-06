"""Per-layer gradient flow, Light vs Fast, on the trained checkpoints. CPU only.

Backs the LayerNorm-placement assessment with a measurement rather than an argument: run a
real batch with the real loss, backprop, and report the gradient arriving at each block --
at the FFN input (ln2's output), at the residual stream, and at the LUT's own parameters.

If LN placement starved shallow layers we would expect the ACTIVATION gradients to shrink
toward layer 0 in Light and not in Fast. If instead the activation gradients look similar in
both, LN placement is not the discriminator and the depth-graded margin profile has another
cause.

Parameter gradients are reported RELATIVE to the parameter norm, since that is what sets the
effective update size under AdamW, and absolute norms across differently-sized tensors are
not comparable.

    python diag_grad_flow.py
"""
import json
import os
import sys

import torch

FR = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
RC = os.path.join(FR, 'runs_corrected')
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))

from nanochat.common import get_base_dir                                  # noqa: E402
from nanochat.tokenizer import RustBPETokenizer                           # noqa: E402
from nanochat.dataloader import (                                         # noqa: E402
    tokenizing_distributed_data_loader_bos_bestfit)
from model_build import build_model                                       # noqa: E402

MODELS = [('LIGHT', 'exp_n_0184_B16k_light_bnorm_seed1'),
          ('FAST ', 'exp_n_0129_grid_H4d48_nap8_tph256')]
torch.manual_seed(0)


def main():
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    ld = tokenizing_distributed_data_loader_bos_bestfit(tok, 48, 512, split='val',
                                                        device='cpu')
    x_all, y_all = next(iter(ld))
    x_ids, y_ids = x_all[12:14].clone(), y_all[12:14].clone()
    print(f'{x_ids.numel():,} tokens (rows [12,14) of the corrected eval slice), real loss\n')

    for label, run in MODELS:
        D = os.path.join(RC, run)
        cfg = json.load(open(os.path.join(D, 'config.json')))
        sd = torch.load(os.path.join(D, 'checkpoint.pt'), map_location='cpu')
        m = build_model(cfg, tok.get_vocab_size(), device='cpu')
        m.load_state_dict(sd, strict=False)
        m.train()

        acts = {}
        hooks = []
        for li, blk in enumerate(m.blocks):
            def mk(li):
                def h(mod, inp, out, li=li):
                    # BOTH are non-leaf tensors, so both need retain_grad() before backward
                    # or .grad silently stays None (it warns, but returns nan-looking output).
                    out.retain_grad()
                    inp[0].retain_grad()
                    acts.setdefault(li, {})['ffn_in'] = out
                    acts[li]['resid'] = inp[0]
                return h
            blk.ln2.register_forward_hook(mk(li))

        loss = m(x_ids, y_ids)
        loss.backward()

        print(f'=== {label}  ({run})   loss {loss.item():.5f}')
        print(f'   {"layer":<6}{"grad@FFN-in":>14}{"grad@resid":>13}'
              f'{"|g|/|W| compress":>19}{"|g|/|W| tables":>17}{"|g|/|W| decompress":>20}'
              f'{"ln2.w grad":>13}')
        for li, blk in enumerate(m.blocks):
            ffn = blk.ffn
            lut = ffn.lut_light if hasattr(ffn, 'lut_light') else ffn.lut_batched
            tbl = lut.tables if hasattr(lut, 'tables') else lut.weights
            gi = acts[li]['ffn_in'].grad
            gr = acts[li]['resid'].grad

            def rel(p):
                return (p.grad.norm() / p.norm().clamp_min(1e-30)).item() \
                    if p.grad is not None else float('nan')

            print(f'   {li:<6}{gi.norm().item():>14.6g}'
                  f'{(gr.norm().item() if gr is not None else float("nan")):>13.6g}'
                  f'{rel(ffn.compress.weight):>19.6g}{rel(tbl):>17.6g}'
                  f'{rel(ffn.decompress.weight):>20.6g}'
                  f'{blk.ln2.weight.grad.norm().item():>13.6g}')
        print()
        del m


if __name__ == '__main__':
    main()
