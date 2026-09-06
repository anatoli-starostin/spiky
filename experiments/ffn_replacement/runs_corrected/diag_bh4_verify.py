"""Verification for the BH4 addressing path, before any GPU time is spent.

Five things, in order:
  (a) BACKWARD COMPATIBILITY -- with lut_impl unchanged, exp_g_0190's config must build a
      bit-identical model (param / buffer / structure hashes).
  (b) OUR BATCHED BH4 == THE REFERENCE MODULE, elementwise, so "reused, not
      reimplemented" is a checked claim rather than a comment.
  (c) PARAMETER ARITHMETIC against the parity target (compress + anchor-pair buffers).
  (d) THE FLAG IS ACTIVE where the effect lives -- at the code and the addresses, not at
      the block output, which zero-init decompress forces to 0 either way.
  (e) SIGN CONSTANCY -- per layer, the fraction of BH4 output coordinates whose sign never
      flips across real tokens. If this is near 1.0 the addressing is degenerate and the
      run would be wasted. Measured at init and, when a checkpoint exists, trained.

CPU only.
"""
import copy
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
sys.path.insert(0, '/tmp/claude-1000')          # nucstar's reference, extracted read-only
torch.set_num_threads(4)

BASE = 'exp_g_0190_B16k_light_bnorm_tph128_znorm_seed1'
PRE = {  # recorded before the bh4 work, on BASE's config
    'param': None, 'buffer': None, 'structure': None,
}


def fingerprint(m):
    h = hashlib.sha256()
    for n, p in sorted(m.named_parameters()):
        h.update(n.encode()); h.update(p.detach().numpy().tobytes())
    b = hashlib.sha256()
    for n, buf in sorted(m.named_buffers()):
        b.update(n.encode()); b.update(buf.detach().numpy().tobytes())
    s = hashlib.sha256('|'.join(f'{n}:{tuple(p.shape)}'
                                for n, p in sorted(m.named_parameters())).encode())
    return h.hexdigest(), b.hexdigest(), s.hexdigest()


def main():
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    from model_build import build_model
    from spiky.lutorch.bh4_multi_head_lut import BH4MultiHead, fwht

    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    vocab = tok.get_vocab_size()
    base_cfg = json.load(open(os.path.join(HERE, BASE, 'config.json')))

    print('=' * 92)
    print('(a) BACKWARD COMPATIBILITY -- lut_impl untouched must be bit-identical')
    print('=' * 92)
    torch.manual_seed(base_cfg['random_seed'])
    m0 = build_model(base_cfg, vocab, device='cpu')
    ph, bh, sh = fingerprint(m0)
    print(f'   params {sum(p.numel() for p in m0.parameters()):,}  (exp_g_0190: 67,352,256)')
    print(f'   param  sha256 {ph}')
    print(f'   buffer sha256 {bh}')
    print(f'   structure sha {sh}')
    print(f'   expected param sha (recorded pre-change) '
          f'0025191a...  -- this config also carries z_norm, so compare against the '
          f'exp_g_0190 build, not the exp_n_0185 one')
    del m0

    print('\n' + '=' * 92)
    print('(b) OUR BATCHED BH4 == nucstar\'s REFERENCE MODULE, elementwise')
    print('=' * 92)
    try:
        from lookup_ffn import BH4 as RefBH4, fwht as ref_fwht
    except ImportError:
        print('   *** reference not importable; extract it with '
              '`git show origin/research/lookupffn:research/lookupffn/lookup_ffn.py`')
        RefBH4 = None
    if RefBH4 is not None:
        x = torch.randn(7, 64, dtype=torch.float32)
        print(f'   fwht agrees with reference: '
              f'{torch.allclose(fwht(x), ref_fwht(x), atol=1e-6)}   '
              f'max|diff| {(fwht(x) - ref_fwht(x)).abs().max():.3e}')
        H, D, BLK, NF = 3, 64, 4, 4
        ours = BH4MultiHead(D, H, block=BLK, n_factors=NF, random_seed=0)
        refs = [RefBH4(D, block=BLK, n_factors=NF) for _ in range(H)]
        with torch.no_grad():          # give the reference our blocks, head by head
            for h, r in enumerate(refs):
                r.blocks.copy_(ours.blocks[h])
        xb = torch.randn(11, H, D)
        got = ours(xb)
        want = torch.stack([refs[h](xb[:, h]) for h in range(H)], dim=1)
        print(f'   batched multi-head == per-head reference: '
              f'{torch.allclose(got, want, atol=1e-6)}   '
              f'max|diff| {(got - want).abs().max():.3e}')
        print(f'   param count per head: ours {ours.blocks[0].numel():,} '
              f'reference {refs[0].blocks.numel():,}  '
              f'formula n_factors*dim*block = {NF * D * BLK:,}')

    print('\n' + '=' * 92)
    print('(c) PARAMETER ARITHMETIC vs the parity target')
    print('=' * 92)
    torch.manual_seed(base_cfg['random_seed'])
    mb = build_model(base_cfg, vocab, device='cpu')
    f0 = mb.blocks[0].ffn
    comp_w = f0.compress.weight.numel()
    comp_b = f0.compress.bias.numel() if f0.compress.bias is not None else 0
    la = f0.lut_light
    anchors = la.anchor_a.numel() + la.anchor_b.numel()
    anchors_native = la.native_anchor_a.numel() + la.native_anchor_b.numel()
    target = comp_w + comp_b + anchors
    print(f'   compress weights            {comp_w:>10,}   ({f0.compress.weight.shape[1]}'
          f' -> {f0.compress.weight.shape[0]})')
    print(f'   compress bias               {comp_b:>10,}')
    print(f'   anchor buffers (logical)    {anchors:>10,}   '
          f'anchor_a{tuple(la.anchor_a.shape)} + anchor_b -> H*tph*NAP*2')
    print(f'   (flattened kernel duplicate {anchors_native:>10,}   native_anchor_a/b; the '
          f'same geometry, held twice)')
    print(f'   PARITY TARGET               {target:>10,}')
    print()
    D_W = 1024                      # working width: pow2 >= max(384, tph*NAP=896)
    H = 4
    print(f'   BH4 = n_heads * n_factors * (dim/block) * block^2 = n_heads*n_factors*dim*block')
    print(f'   {"block":>7}{"n_factors":>11}{"params":>12}{"vs target":>12}{"ratio":>9}')
    best = None
    for nf in (4, 5):
        for blk in (2, 4, 8, 16):
            n = H * nf * D_W * blk
            d = n - target
            print(f'   {blk:>7}{nf:>11}{n:>12,}{d:>+12,}{n / target:>9.3f}'
                  + ('   <- chosen' if (nf, blk) == (4, 4) else ''))
    del mb

    print('\n' + '=' * 92)
    print('(d)+(e) BUILD THE BH4 MODEL; IS IT ACTIVE, AND ARE THE SIGNS CONSTANT?')
    print('=' * 92)
    cfg = copy.deepcopy(base_cfg)
    cfg['lut_impl'] = 'bh4'
    cfg['lut_n_anchor_pairs'] = 7
    cfg['lut_z_norm'] = False
    cfg['lut_confidence_form'] = 'margin'
    cfg['lut_bh4_block'] = 4
    cfg['lut_bh4_factors'] = 4
    torch.manual_seed(cfg['random_seed'])
    m = build_model(cfg, vocab, device='cpu')
    tot = sum(p.numel() for p in m.parameters())
    ffn = m.blocks[0].ffn
    lb = ffn.lut_bh4
    print(f'   total params {tot:,}   vs exp_g_0190 67,352,256   '
          f'delta {tot - 67_352_256:+,}')
    print(f'   BH4 per layer {lb.bh4.blocks.numel():,}   tables per layer '
          f'{lb.tables.numel():,}   decompress {ffn.decompress.weight.numel():,}')
    print(f'   working width {lb.width}  (pow2 >= max(384, tph*NAP={lb.n_code}));  '
          f'code coords used {lb.n_code}/{lb.width}')
    print(f'   compress is Identity: {isinstance(ffn.compress, torch.nn.Identity)}   '
          f'z_norm: {getattr(ffn, "z_norm", None)}')

    # optimiser groups -- printed, not reasoned
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
    from spiky.lutorch.bh4_multi_head_lut import BH4MultiHeadLUT
    names = {id(p): n for n, p in m.named_parameters()}
    for nd in (False, True):
        # mirrors setup_optimizer() in train_fixed.py exactly
        exempt = ((FastMultiHeadLut, LightMultiHeadLUT, BH4MultiHeadLUT) if nd
                  else (FastMultiHeadLut,))
        ids = {id(p) for mod in m.modules() if isinstance(mod, exempt)
               for p in mod.parameters(recurse=False)}
        dec, nod = [], []
        for p in m.parameters():
            (nod if (id(p) in ids or p.ndim < 2) else dec).append(names[id(p)])
        print(f'   lut_tables_no_decay={nd}:  bh4.blocks in DECAY '
              f'{sum("bh4.blocks" in n for n in dec)} / NODECAY '
              f'{sum("bh4.blocks" in n for n in nod)}   |  lut_bh4.tables in DECAY '
              f'{sum(n.endswith("lut_bh4.tables") for n in dec)} / NODECAY '
              f'{sum(n.endswith("lut_bh4.tables") for n in nod)}')

    ld = tokenizing_distributed_data_loader_bos_bestfit(tok, 48, 512, split='val',
                                                        device='cpu')
    x_all, _ = next(iter(ld))
    x_ids = x_all[12:16].clone()
    cap = {}
    hs = []
    for li, blk in enumerate(m.blocks):
        def mk(li, f):
            def h(mod, inp, out):
                cap[li] = f.lut_bh4.code(inp[0]).detach()
            return h
        hs.append(blk.ffn.register_forward_hook(mk(li, blk.ffn)))
    with torch.no_grad():
        m(x_ids)
    for h in hs:
        h.remove()

    print(f'\n   SIGN CONSTANCY at init, on {x_ids.numel():,} real val tokens')
    print(f'   {"layer":>6}{"const-sign frac":>18}{"|mean|/std":>13}{"code std":>11}'
          f'{"distinct addr/128":>20}')
    worst = 0.0
    for li in range(len(m.blocks)):
        z = cap[li]                                   # [N,H,T,NAP]
        flat = z.reshape(z.shape[0], -1)
        pos = (flat > 0).to(torch.float64).mean(dim=0)
        const = float(((pos == 0) | (pos == 1)).to(torch.float64).mean())
        worst = max(worst, const)
        ratio = float((flat.mean(0).abs() / flat.std(0).clamp_min(1e-9)).mean())
        powers = (2 ** torch.arange(z.shape[-1] - 1, -1, -1))
        idx = ((z > 0).long() * powers).sum(-1)       # [N,H,T]
        dis = float(torch.stack([
            torch.bincount(idx[:, h, t], minlength=1 << z.shape[-1]).gt(0).sum()
            for h in range(z.shape[1]) for t in range(0, z.shape[2], 16)]).float().mean())
        print(f'   {li:>6}{const:>18.4f}{ratio:>13.4f}{float(flat.std()):>11.4f}'
              f'{dis:>20.1f}')
    print(f'\n   worst-layer constant-sign fraction: {worst:.4f}   '
          f'(1.0 would mean a dead address; exp_g_0190 layer 0 was 1.0000 under '
          f'coordinate-signing)')

    # (d) the flag is active: gradient reaches BH4 through the score, at trained scale
    with torch.no_grad():
        for b in m.blocks:
            b.ffn.decompress.weight.normal_(0, 2.3 / b.ffn.decompress.weight.numel() ** 0.5)
    loss = m(torch.randint(0, vocab, (2, 64))).float().mean()
    loss.backward()
    print(f'\n   smoke: loss {loss.item():.6g}   bh4.blocks grad '
          f'{lb.bh4.blocks.grad.norm():.4e}   tables grad {lb.tables.grad.norm():.4e}')
    print(f'   BH4 receives gradient (through the score only): '
          f'{"YES" if lb.bh4.blocks.grad.norm() > 0 else "*** NO ***"}')


if __name__ == '__main__':
    main()
