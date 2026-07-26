import re
src="/home/astarostin/projects/spiky/experiments/hyperplane_ffn/exp023_singlestream_linear_unembedder/train.py"
dst="/tmp/exp023_fasteval/train.py"
import os; os.makedirs("/tmp/exp023_fasteval",exist_ok=True)
s=open(src).read()
# 1) fix the 'fast' branch: drop the unsupported backward_mode kwarg
s=s.replace("        backward_mode=cfg.get('backward_mode', 'ball'),\n","")
# 2) insert a swap-eval block right after the n_params line
anchor="n_params = sum(p.numel() for p in model.parameters())\n"
block=anchor+'''
if os.environ.get('SWAP_EVAL'):
    _ck = torch.load(os.environ['SWAP_CKPT'], map_location=DEVICE)
    _miss, _unexp = model.load_state_dict(_ck['model_state_dict'], strict=False)
    print('LOAD missing=%d unexpected=%d' % (len(_miss), len(_unexp)))
    print('sample_missing=%s' % (_miss[:5],))
    print('sample_unexpected=%s' % (_unexp[:5],))
    model.eval()
    with torch.no_grad():
        _bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
    print('SWAP_%s_BPB=%.4f' % (cfg['lut_layer_type'].upper(), _bpb))
    sys.exit(0)
'''
assert anchor in s, "anchor not found"
s=s.replace(anchor, block, 1)
open(dst,"w").write(s)
print("patched -> ",dst, "| backward_mode remaining:", s.count("backward_mode=cfg.get"))
