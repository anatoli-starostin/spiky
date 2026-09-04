"""Diagnostic: why does the same checkpoint report 1.15144 at bs48x10 but ~1.117 at bs48x50?

Not an experiment — no training, no model changes. Evaluates ONE untouched checkpoint on
consecutive DISJOINT windows of the val stream and decomposes bpb into its two factors.

`evaluate_bpb` computes  bpb = total_nats / (ln2 * total_bytes), summing nats only over
tokens whose byte length is > 0 (BOS and other specials contribute neither nats nor bytes).
That factors exactly:

    bpb = (nats / counted_tokens) / (ln2 * (bytes / counted_tokens))
        =   per-token loss          /  (ln2 * bytes-per-token)

so a window can read high either because the model is genuinely more surprised (numerator)
or because its tokens are shorter in bytes (denominator). Both are reported per window.

METHOD NOTE. Each window's 10 batches are pulled once from a SINGLE sequential loader and
cached, then handed to the project's own `evaluate_bpb` as an iterator. The decomposition is
computed from those same cached tensors. This matters: driving the canonical function over
identical cached data removes any question of a reimplementation drifting from it, and lets
the script assert the two agree (`bpb_matches_evaluate_bpb`) instead of assuming it.

    python eval_windows.py        # writes results.json alongside
"""
import json
import math
import os
import statistics as st
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
for p in (os.path.join(REPO, 'experiments', 'ffn_replacement', 'benchmark'),
          os.path.join(REPO, 'src'), NANOCHAT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as M                                                    # noqa: E402
from nanochat.common import get_base_dir                             # noqa: E402
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes     # noqa: E402
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit  # noqa: E402
from nanochat.loss_eval import evaluate_bpb                          # noqa: E402

TEACHER = os.path.join(REPO, 'experiments', 'ffn_replacement', 'runs',
                       'exp_n_0151_long48k_untied_vanilla')
EVAL_BS = 48
WINDOW = 10          # batches per window -- the published protocol's eval_steps
N_WINDOWS = 20       # 20 disjoint windows = 200 batches = 4.9M tokens
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class EvalAdapter(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt

    def get_device(self):
        return self.gpt.tok_emb.weight.device

    def forward(self, idx, targets=None, loss_reduction='mean'):
        logits = self.gpt(idx)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                               targets.reshape(-1), ignore_index=-1,
                               reduction=loss_reduction)
        return loss.view(targets.shape) if loss_reduction == 'none' else loss


def main():
    cfg, teacher = M.build(TEACHER, load_checkpoint=True, dev=DEVICE)
    teacher.eval()
    SEQ = cfg['seq_len']
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    token_bytes = get_token_bytes(device=DEVICE)
    bos_id = tok.get_bos_token_id()
    adapter = EvalAdapter(teacher).to(DEVICE)

    # ONE loader consumed sequentially -> windows are disjoint by construction.
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, EVAL_BS, SEQ, split='val', device=DEVICE)

    rows = []
    for w in range(N_WINDOWS):
        # CLONE. The loader yields VIEWS into one reused GPU buffer (`inputs`/`targets` are
        # the same two tensors every iteration, overwritten in place), so caching the
        # yielded objects would store WINDOW references to the last batch and silently
        # evaluate it WINDOW times. Verified: without the clone, window 0 reads 1.15273
        # instead of the correct 1.15144.
        batches = [(x.clone(), y.clone()) for x, y in
                   (next(loader) for _ in range(WINDOW))]
        with torch.no_grad():
            bpb_canon = float(evaluate_bpb(adapter, iter(batches), WINDOW, token_bytes))
        # decomposition from the SAME cached tensors
        nats = torch.zeros((), dtype=torch.float64, device=DEVICE)
        nbytes = torch.zeros((), dtype=torch.int64, device=DEVICE)
        counted = torch.zeros((), dtype=torch.int64, device=DEVICE)
        nbos = torch.zeros((), dtype=torch.int64, device=DEVICE)
        ntok = 0
        with torch.no_grad():
            for x, y in batches:
                l2 = adapter(x, y, loss_reduction='none')
                b2d = token_bytes[y]
                keep = b2d > 0
                nats += (l2 * keep).sum().double()
                nbytes += b2d.sum()
                counted += keep.sum()
                nbos += (y == bos_id).sum()
                ntok += y.numel()
        nats_f, bytes_f, cnt = nats.item(), nbytes.item(), counted.item()
        bpb_dec = nats_f / (math.log(2) * bytes_f)
        rows.append(dict(
            window=w, first_batch=w * WINDOW,
            bpb=bpb_canon, bpb_from_decomposition=bpb_dec,
            bpb_matches_evaluate_bpb=abs(bpb_canon - bpb_dec) < 1e-6,
            per_token_nats=nats_f / cnt, bytes_per_token=bytes_f / cnt,
            counted_tokens=cnt, total_tokens=ntok,
            zero_byte_tokens=ntok - cnt, bos_tokens=nbos.item(),
            bos_per_1k=1000.0 * nbos.item() / ntok))
        r = rows[-1]
        print(f"  w{w:>2} (batches {r['first_batch']:>3}-{r['first_batch']+WINDOW-1:>3}): "
              f"bpb {r['bpb']:.5f}  per-tok-nats {r['per_token_nats']:.5f}  "
              f"bytes/tok {r['bytes_per_token']:.4f}  BOS/1k {r['bos_per_1k']:.2f}"
              f"{'' if r['bpb_matches_evaluate_bpb'] else '   <-- MISMATCH'}", flush=True)

    bpbs = [r['bpb'] for r in rows]
    ptn = [r['per_token_nats'] for r in rows]
    bpt = [r['bytes_per_token'] for r in rows]
    rest = bpbs[1:]

    def blk(v):
        return dict(mean=st.mean(v), stdev=st.stdev(v) if len(v) > 1 else 0.0,
                    min=min(v), max=max(v))

    summary = dict(
        teacher=os.path.relpath(TEACHER, REPO), eval_bs=EVAL_BS,
        window_batches=WINDOW, n_windows=N_WINDOWS,
        published_bpb=1.151444329946291,
        all_decompositions_match_evaluate_bpb=all(r['bpb_matches_evaluate_bpb'] for r in rows),
        windows=rows,
        all_windows=dict(bpb=blk(bpbs), per_token_nats=blk(ptn), bytes_per_token=blk(bpt)),
        windows_excl_first=dict(bpb=blk(rest), per_token_nats=blk(ptn[1:]),
                                bytes_per_token=blk(bpt[1:])),
        window0_z_vs_rest=(bpbs[0] - st.mean(rest)) / st.stdev(rest),
        n_windows_at_or_above_window0=sum(1 for b in bpbs if b >= bpbs[0]),
        cumulative_first_n=[
            dict(n_windows=k,
                 bpb=sum(r['per_token_nats'] * r['counted_tokens'] for r in rows[:k])
                     / (math.log(2) * sum(r['bytes_per_token'] * r['counted_tokens']
                                          for r in rows[:k])))
            for k in (1, 2, 5, 10, 20)])
    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    a, e = summary['all_windows'], summary['windows_excl_first']
    print(f"\ndecomposition agrees with evaluate_bpb on every window: "
          f"{summary['all_decompositions_match_evaluate_bpb']}")
    print(f"all {N_WINDOWS} windows : mean {a['bpb']['mean']:.5f} sd {a['bpb']['stdev']:.5f} "
          f"range [{a['bpb']['min']:.5f}, {a['bpb']['max']:.5f}]")
    print(f"excluding w0     : mean {e['bpb']['mean']:.5f} sd {e['bpb']['stdev']:.5f}")
    print(f"window 0 = {bpbs[0]:.5f} -> z vs rest {summary['window0_z_vs_rest']:.2f} sigma; "
          f"{summary['n_windows_at_or_above_window0']} of {N_WINDOWS} windows >= it")
    print(f"per-token nats: w0 {ptn[0]:.5f} vs rest {e['per_token_nats']['mean']:.5f}  |  "
          f"bytes/token: w0 {bpt[0]:.4f} vs rest {e['bytes_per_token']['mean']:.4f}")
    print("cumulative bpb over first N windows:",
          {d['n_windows']: round(d['bpb'], 5) for d in summary['cumulative_first_n']})
    print(f"wrote {HERE}/results.json")


if __name__ == '__main__':
    main()
