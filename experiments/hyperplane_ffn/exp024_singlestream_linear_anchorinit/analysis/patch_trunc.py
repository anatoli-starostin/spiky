import os
src="/home/astarostin/projects/spiky/experiments/hyperplane_ffn/exp024_singlestream_linear_anchorinit/train.py"
dst="/tmp/exp024_trunc/train.py"; os.makedirs("/tmp/exp024_trunc",exist_ok=True)
s=open(src).read()
anchor="n_params = sum(p.numel() for p in model.parameters())\n"
block=anchor+'''
if os.environ.get('TRUNC_EVAL'):
    import numpy as _np, collections as _c
    _ck = torch.load(os.environ['TRUNC_CKPT'], map_location=DEVICE)['model_state_dict']
    _m,_u = model.load_state_dict(_ck, strict=True)
    print('LOAD strict missing=%d unexpected=%d' % (len(_m),len(_u)))
    _k = os.environ.get('TRUNC_K','full')
    sites = [(n,mod) for n,mod in model.named_modules()
             if isinstance(getattr(mod,'hyperplane_weight',None), torch.nn.Parameter)]
    print('NUM_SITES=%d' % len(sites))
    full_W = {n: mod.hyperplane_weight.detach().clone() for n,mod in sites}
    if _k != 'full':
        kk = int(_k)
        with torch.no_grad():
            for n,mod in sites:
                W = mod.hyperplane_weight  # [T, nap, D]
                idx = W.abs().topk(kk, dim=-1).indices
                mask = torch.zeros_like(W, dtype=torch.bool)
                mask.scatter_(-1, idx, True)
                W.mul_(mask.to(W.dtype))
        # bit-flip diagnostic: truncated address vs full trained address
        cap={}; hooks=[]
        for n,mod in sites:
            def _mk(nm):
                def _h(m,inp): cap[nm]=inp[0].detach()
                return _h
            hooks.append(mod.register_forward_pre_hook(_mk(n)))
        model.eval()
        _x,_y = next(val_loader_factory())
        with torch.no_grad(): _=model(_x, targets=_y)
        for h in hooks: h.remove()
        sf=_c.defaultdict(list)
        for n,mod in sites:
            D=mod.hyperplane_weight.shape[-1]
            X=cap[n].float().reshape(-1, D); N=min(X.shape[0],256); X=X[:N]
            Wt=mod.hyperplane_weight.float(); Wf=full_W[n].float(); B=mod.hyperplane_bias.float()
            af=torch.einsum('nd,tpd->ntp', X, Wf)+B.unsqueeze(0)
            at=torch.einsum('nd,tpd->ntp', X, Wt)+B.unsqueeze(0)
            flip=((af>0)!=(at>0)).float().mean().item()
            sf[n.split('.')[-1]].append(flip)
        for site in sf: print('FLIP_%s=%.4f' % (site, float(_np.mean(sf[site]))))
    model.eval()
    with torch.no_grad():
        _bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
    print('RESULT k=%s bpb=%.4f' % (_k, _bpb))
    sys.exit(0)
'''
assert anchor in s
open(dst,"w").write(s.replace(anchor,block,1))
print("patched",dst)
