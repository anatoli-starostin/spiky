"""Module/layer transplant ablation: localize WHERE exp486's -0.117 bpb lives.

Base = exp475 (bs=16, worse). For each group of parameters, copy exp486's
(bs=48, better) weights into the base model, eval bpb on a fixed shared val set,
then restore. The bpb IMPROVEMENT from each transplant = how much upgrading that
group to bs=48-quality matters. The group that recovers the most is the
bottleneck to target.

Shared seed+anchors make the swap valid (identical row identity / shapes).

Groups tested:
  - whole model (sanity: should recover exp486's bpb)
  - each module-type across all layers (qkv_lut/v_lut/out_proj/residual_lut)
  - unembedder, tok_emb, norms
  - each layer (all its modules)
  - out_proj per single layer (the suspected bottleneck)

Run: /home/starost/spiky/.venv/bin/python transplant.py
"""
import os, sys, json, math
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from model_def import build_model
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

DEVICE = 'cuda'
CKPT_475 = '/home/starost/spiky/nanochat_exps/exp475_meanabs_nocenter/checkpoint.pt'
CKPT_486 = '/home/starost/spiky/nanochat_exps/exp486_bs48_8k/checkpoint.pt'
N_BATCHES = 16
LUT_NAMES = ('qkv_lut', 'v_lut', 'out_proj', 'residual_lut')


@torch.no_grad()
def eval_bpb_fixed(model, batches, token_bytes):
    total_nats, total_bytes = 0.0, 0
    for x, y in batches:
        logits = model(x)
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = F.nll_loss(logp.view(-1, logp.size(-1)), y.view(-1),
                         ignore_index=-1, reduction='sum')
        valid = (y.view(-1) != -1)
        total_bytes += token_bytes[y.view(-1)[valid]].sum().item()
        total_nats += nll.item()
    return total_nats / total_bytes / math.log(2)


def keys_for_group(sd, group):
    """Return state_dict keys matching a group spec."""
    if group == 'ALL':
        return list(sd.keys())
    if group.startswith('module:'):
        name = group.split(':', 1)[1]
        return [k for k in sd if f'.{name}.' in k or k.endswith(f'.{name}.weights')
                or f'.{name}.' in k]
    if group.startswith('layer:'):
        li = group.split(':', 1)[1]
        return [k for k in sd if k.startswith(f'layers.{li}.')]
    if group.startswith('layermod:'):
        _, li, name = group.split(':')
        return [k for k in sd if k.startswith(f'layers.{li}.{name}.')]
    if group.startswith('top:'):           # exact top-level prefix
        pfx = group.split(':', 1)[1]
        return [k for k in sd if k.startswith(pfx)]
    raise ValueError(group)


def main():
    ck475 = torch.load(CKPT_475, map_location='cpu', weights_only=False)
    ck486 = torch.load(CKPT_486, map_location='cpu', weights_only=False)
    cfg = dict(ck475['config']); cfg['vocab_size'] = 32768
    sd475 = ck475['model_state_dict']
    sd486 = {k: v.to(DEVICE) for k, v in ck486['model_state_dict'].items()}

    base = build_model(cfg, device=DEVICE)
    base.load_state_dict(sd475, strict=False)
    base.eval()

    # fixed val batches
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, cfg['device_batch_size'], cfg['context_size'], split='val', device=DEVICE)
    batches = [next(loader) for _ in range(N_BATCHES)]
    token_bytes = get_token_bytes(device=DEVICE)

    base_bpb = eval_bpb_fixed(base, batches, token_bytes)
    # full transplant sanity
    base.load_state_dict({k: sd486[k] for k in sd486}, strict=False)
    full_bpb = eval_bpb_fixed(base, batches, token_bytes)
    base.load_state_dict(sd475, strict=False)   # restore
    print(f'exp475 base bpb = {base_bpb:.4f}')
    print(f'full exp486 bpb = {full_bpb:.4f}  (gap to recover = {base_bpb-full_bpb:.4f})')

    N_LAYERS = cfg['num_layers']
    groups = ['ALL']
    groups += [f'module:{n}' for n in LUT_NAMES]
    groups += ['top:unembedder', 'top:tok_emb_E', 'top:ln_final']
    groups += [f'layer:{i}' for i in range(N_LAYERS)]
    groups += [f'layermod:{i}:out_proj' for i in range(N_LAYERS)]

    sd_base = {k: v.to(DEVICE) for k, v in sd475.items()}
    results = {}
    print(f'\n{"group":28s} {"bpb":>8s} {"improvement":>12s}')
    print('-' * 52)
    for g in groups:
        keys = keys_for_group(sd486, g)
        keys = [k for k in keys if k in sd_base and sd_base[k].shape == sd486[k].shape]
        if not keys:
            continue
        base.load_state_dict({**sd_base, **{k: sd486[k] for k in keys}}, strict=False)
        bpb = eval_bpb_fixed(base, batches, token_bytes)
        base.load_state_dict(sd_base, strict=False)   # restore
        results[g] = bpb
        print(f'{g:28s} {bpb:8.4f} {base_bpb - bpb:+12.4f}')

    out = dict(base_bpb=base_bpb, full_bpb=full_bpb, transplants=results)
    with open(os.path.join(HERE, 'transplant_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\nwrote transplant_results.json')


if __name__ == '__main__':
    main()
