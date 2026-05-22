"""Basin-invariant 'where/what improves' analysis: exp475 vs exp486.

The transplant ablation showed the two models are holistically coadapted (no
single module carries the gain), so weight-space comparison is uninformative.
Here we compare them in FUNCTION space on identical val data:

  1. Logit-lens over depth: apply ln_final + the model's own unembedder to the
     PARTIAL residual sum after each layer. bpb-vs-depth curve shows WHERE in
     depth exp486's advantage appears (early = shallow features better; late =
     only final layers).
  2. Per-position loss: mean next-token bits at each context position 0..T-1.
     Tells if the gain is in-context (later positions) or base modeling.
  3. Per-frequency loss: bucket target tokens by empirical frequency; mean bits
     per bucket. Tells if exp486 improves common or rare tokens.

Run: /home/starost/spiky/.venv/bin/python probe.py
"""
import os, sys, json, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from model_def import load_checkpoint
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

DEVICE = 'cuda'
CKPT_475 = '/home/starost/spiky/nanochat_exps/exp475_meanabs_nocenter/checkpoint.pt'
CKPT_486 = '/home/starost/spiky/nanochat_exps/exp486_bs48_8k/checkpoint.pt'
N_BATCHES = 24
N_FREQ_BUCKETS = 8


@torch.no_grad()
def forward_partials(model, tokens):
    """Replicate Model.forward but return per-depth partial logits via logit-lens
    on the accumulating D-stream. Returns list length N_LAYERS+1 of [B,T,V]
    (index k = residual sum of layers 0..k-1; index 0 = zero stream)."""
    B, T = tokens.shape
    D = model.D
    x_resid = torch.zeros(B, T, D, device=tokens.device, dtype=model.tok_emb_E.weight.dtype)
    x_lut = model.tok_emb_E(tokens)
    partials_resid = [x_resid.clone()]
    for layer in model.layers:
        x_lut, r = layer(x_lut, model.rope.cos, model.rope.sin)
        x_resid = x_resid + r
        partials_resid.append(x_resid.clone())
    # logit-lens: apply ln_final + unembedder to each partial (skip the zero one)
    logits_list = []
    for k in range(1, len(partials_resid)):
        z = model.ln_final(partials_resid[k])
        logits_list.append(model.unembedder(z))
    return logits_list           # length N_LAYERS, each [B,T,V]


@torch.no_grad()
def logit_lens_bpb(model, batches, token_bytes):
    """bpb at each depth k (cumulative residual after k layers)."""
    nlay = len(model.layers)
    nats = [0.0] * nlay
    nbytes = 0
    for x, y in batches:
        logits_list = forward_partials(model, x)
        yv = y.view(-1)
        valid = (yv != -1)
        nbytes += token_bytes[yv[valid]].sum().item()
        for k, lg in enumerate(logits_list):
            logp = F.log_softmax(lg.float(), dim=-1)
            nats[k] += F.nll_loss(logp.view(-1, logp.size(-1)), yv,
                                  ignore_index=-1, reduction='sum').item()
    return [n / nbytes / math.log(2) for n in nats]


@torch.no_grad()
def per_token_bits(model, batches):
    """Return (bits_flat [Nvalid], targets_flat [Nvalid], positions_flat [Nvalid])
    for the FINAL-layer logits — bits/token (nats/ln2)."""
    bits_all, tgt_all, pos_all = [], [], []
    for x, y in batches:
        B, T = x.shape
        logits = model(x)
        logp = F.log_softmax(logits.float(), dim=-1)
        ce = F.nll_loss(logp.view(-1, logp.size(-1)), y.view(-1),
                        ignore_index=-1, reduction='none').view(B, T)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        valid = (y != -1)
        bits_all.append((ce[valid] / math.log(2)).cpu())
        tgt_all.append(y[valid].cpu())
        pos_all.append(pos[valid].cpu())
    return (torch.cat(bits_all), torch.cat(tgt_all), torch.cat(pos_all))


def per_position_curve(bits, positions, T):
    out = np.full(T, np.nan)
    for t in range(T):
        m = (positions == t)
        if m.any():
            out[t] = bits[m].mean().item()
    return out


def per_frequency_curve(bits, targets, freq_rank, n_buckets):
    """Bucket target tokens by frequency rank (0=rarest .. 1=commonest);
    return (bucket_mean_bits, bucket_edges_label)."""
    r = freq_rank[targets]                       # [N] in [0,1)
    edges = np.quantile(r.numpy(), np.linspace(0, 1, n_buckets + 1))
    means, labels = [], []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        m = (r >= lo) & (r <= hi) if i == n_buckets - 1 else (r >= lo) & (r < hi)
        means.append(bits[m].mean().item() if m.any() else float('nan'))
        labels.append(f'{lo:.2f}-{hi:.2f}')
    return means, labels


def main():
    ck = torch.load(CKPT_475, map_location='cpu', weights_only=False)
    cfg = dict(ck['config'])
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, cfg['device_batch_size'], cfg['context_size'], split='val', device=DEVICE)
    batches = [next(loader) for _ in range(N_BATCHES)]
    token_bytes = get_token_bytes(device=DEVICE)
    T = cfg['context_size']

    # empirical token frequency over all targets -> frequency RANK in [0,1)
    all_tgt = torch.cat([y.view(-1)[y.view(-1) != -1].cpu() for _, y in batches])
    V = 32768
    counts = torch.bincount(all_tgt, minlength=V).float()
    order = counts.argsort()                        # ascending freq
    rank = torch.empty(V); rank[order] = torch.arange(V).float() / V   # 0=rarest

    m475, _ = load_checkpoint(CKPT_475, device=DEVICE)
    m486, _ = load_checkpoint(CKPT_486, device=DEVICE)

    res = {}
    for tag, m in [('exp475', m475), ('exp486', m486)]:
        lens = logit_lens_bpb(m, batches, token_bytes)
        bits, tgts, pos = per_token_bits(m, batches)
        ppos = per_position_curve(bits, pos, T)
        pfreq, flabels = per_frequency_curve(bits, tgts, rank, N_FREQ_BUCKETS)
        res[tag] = dict(logit_lens=lens, per_position=ppos.tolist(),
                        per_freq=pfreq, freq_labels=flabels,
                        overall_bits=bits.mean().item())
        print(f'{tag}: final-bpb-lens={lens[-1]:.4f}  overall_bits/tok={bits.mean():.4f}')

    with open(os.path.join(HERE, 'probe_results.json'), 'w') as f:
        json.dump(res, f, indent=2)

    # ---- prints ----
    print('\nLOGIT-LENS bpb vs depth (cumulative residual):')
    print(f'  depth:  ' + '  '.join(f'L{k+1}' for k in range(len(res["exp475"]["logit_lens"]))))
    print(f'  exp475: ' + '  '.join(f'{v:.3f}' for v in res['exp475']['logit_lens']))
    print(f'  exp486: ' + '  '.join(f'{v:.3f}' for v in res['exp486']['logit_lens']))
    print(f'  delta:  ' + '  '.join(f'{b-a:+.3f}' for a, b in
          zip(res['exp475']['logit_lens'], res['exp486']['logit_lens'])))

    print('\nPER-FREQUENCY bits/token (rarest -> commonest):')
    print('  bucket: ' + '  '.join(res['exp475']['freq_labels']))
    print('  exp475: ' + '  '.join(f'{v:.2f}' for v in res['exp475']['per_freq']))
    print('  exp486: ' + '  '.join(f'{v:.2f}' for v in res['exp486']['per_freq']))
    print('  delta:  ' + '  '.join(f'{b-a:+.2f}' for a, b in
          zip(res['exp475']['per_freq'], res['exp486']['per_freq'])))

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    nl = len(res['exp475']['logit_lens'])
    axes[0].plot(range(1, nl + 1), res['exp475']['logit_lens'], 'o-', label='exp475 bs16')
    axes[0].plot(range(1, nl + 1), res['exp486']['logit_lens'], 's-', label='exp486 bs48')
    axes[0].set(xlabel='depth (cumulative layers)', ylabel='bpb (logit-lens)',
                title='Logit-lens bpb vs depth')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    pa = np.array(res['exp475']['per_position']); pb = np.array(res['exp486']['per_position'])
    axes[1].plot(pa, label='exp475', alpha=0.8)
    axes[1].plot(pb, label='exp486', alpha=0.8)
    axes[1].set(xlabel='context position', ylabel='bits/token', title='Per-position loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    x = np.arange(N_FREQ_BUCKETS); w = 0.38
    axes[2].bar(x - w/2, res['exp475']['per_freq'], w, label='exp475')
    axes[2].bar(x + w/2, res['exp486']['per_freq'], w, label='exp486')
    axes[2].set_xticks(x); axes[2].set_xticklabels(res['exp475']['freq_labels'], rotation=45, fontsize=7)
    axes[2].set(xlabel='token freq rank bucket (rare->common)', ylabel='bits/token',
                title='Per-frequency loss')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'probe.png'), dpi=120)
    plt.close(fig)
    print('\nwrote probe_results.json, probe.png')


if __name__ == '__main__':
    main()
