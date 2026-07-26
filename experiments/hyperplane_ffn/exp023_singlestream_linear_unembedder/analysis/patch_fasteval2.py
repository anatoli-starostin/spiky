import os
src="/home/astarostin/projects/spiky/experiments/hyperplane_ffn/exp023_singlestream_linear_unembedder/train.py"
dst="/tmp/exp023_fe2/train.py"; os.makedirs("/tmp/exp023_fe2",exist_ok=True)
s=open(src).read()
s=s.replace("        backward_mode=cfg.get('backward_mode', 'ball'),\n","")
anchor="n_params = sum(p.numel() for p in model.parameters())\n"
block=anchor+'''
if os.environ.get('SWAP_EVAL'):
    import numpy as _np, collections as _c
    _ck = torch.load(os.environ['SWAP_CKPT'], map_location=DEVICE)['model_state_dict']
    _mode = os.environ.get('SWAP_MODE','baseline')
    if _mode == 'baseline':
        _m,_u = model.load_state_dict(_ck, strict=True)
        print('LOAD baseline strict OK (missing=%d unexpected=%d)' % (len(_m),len(_u)))
    else:
        hw = {k.removesuffix('.hyperplane_weight'):v for k,v in _ck.items() if k.endswith('.hyperplane_weight')}
        hb = {k.removesuffix('.hyperplane_bias'):v for k,v in _ck.items() if k.endswith('.hyperplane_bias')}
        sd = {k:v for k,v in _ck.items() if not (k.endswith('.hyperplane_weight') or k.endswith('.hyperplane_bias'))}
        _m,_u = model.load_state_dict(sd, strict=False)
        print('LOAD argmaxmin after-strip missing=%d unexpected=%d' % (len(_m),len(_u)))
        cap={}; hooks=[]
        for name,mod in model.named_modules():
            if name in hw:
                W = hw[name].float()
                mod.soft_anchor_a_long.copy_(W.argmax(-1).to(mod.soft_anchor_a_long.dtype))
                mod.soft_anchor_b_long.copy_(W.argmin(-1).to(mod.soft_anchor_b_long.dtype))
                def _mk(nm):
                    def _h(m,inp): cap[nm]=inp[0].detach()
                    return _h
                hooks.append(mod.register_forward_pre_hook(_mk(name)))
        model.eval()
        _x,_y = next(val_loader_factory())
        with torch.no_grad(): _=model(_x, targets=_y)
        for h in hooks: h.remove()
        sf=_c.defaultdict(list)
        for name in cap:
            X=cap[name].float(); N=min(X.shape[0],256); X=X[:N]
            W=hw[name].float(); B=hb[name].float()
            at=torch.einsum('nd,tpd->ntp', X, W)+B.unsqueeze(0)
            ai=W.argmax(-1); bi=W.argmin(-1)
            dv=X[:,ai]-X[:,bi]
            flip=((at>0)!=(dv>0)).float().mean().item()
            sf[name.split('.')[-1]].append(flip)
        for site in sf: print('FLIP_%s=%.4f' % (site, float(_np.mean(sf[site]))))
    model.eval()
    with torch.no_grad():
        _bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
    print('RESULT mode=%s bpb=%.4f' % (_mode, _bpb))
    sys.exit(0)
'''
assert anchor in s
open(dst,"w").write(s.replace(anchor,block,1))
print("patched",dst)
