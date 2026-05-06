"""Stacked LUT residual unembedder.

Architecture:
  x ← LayerNorm(x)
  for k in N_BLOCKS:
      x ← x + LUT_k(x)  where LUT_k: 384→384, n_heads=1, nap=6, tph=NlogN/nap
  logits ← Linear(x, vocab) + bias

Each LUT block covers ~N log N pairs (sparse — much fewer than full coverage's
N(N-1)/2). Residual stacking is meant to compose simple per-block transforms
into more expressive overall function.

Compare against:
  - fit_baseline_mlp.py: KL ≈ 0.012 at 20 ep
  - fit_lut.py (exp175 setup): KL ≈ 0.439 at 20 ep / 0.346 at 40 ep
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IN_DIM = 384
VOCAB = 32768


class StackedLutUnembedder(nn.Module):
    def __init__(self, in_dim, vocab, n_blocks, block_tph, block_nap,
                 init_std, seed=42, dense_concat=False,
                 expand_dim=0, expand_tph=0, block_n_heads=1,
                 share_weights=False):
        super().__init__()
        self.in_dim = in_dim
        self.vocab = vocab
        self.dense_concat = dense_concat
        self.n_blocks = n_blocks
        self.expand_dim = expand_dim
        self.block_n_heads = block_n_heads
        self.share_weights = share_weights
        if in_dim % block_n_heads != 0:
            raise ValueError(f'in_dim={in_dim} not divisible by block_n_heads={block_n_heads}')
        per_head_out = in_dim // block_n_heads
        self.ln = nn.LayerNorm(in_dim)

        def _make_lut(i):
            return TinyMultiHeadLut(
                input_dim=in_dim,
                n_heads=block_n_heads,
                n_outputs=per_head_out,
                n_anchor_pairs=block_nap,
                tables_per_head=block_tph,
                anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
                random_seed=seed + i * 100 + 1,
                initial_weights_noise=init_std,
                weight_dtype=torch.float32,
                device=torch.device(DEVICE),
            )

        if share_weights:
            shared = _make_lut(0)
            self.blocks = nn.ModuleList([shared] * n_blocks)
        else:
            self.blocks = nn.ModuleList([_make_lut(i) for i in range(n_blocks)])

        body_out_dim = (n_blocks + 1) * in_dim if dense_concat else in_dim

        if expand_dim > 0:
            # Wide LUT expansion: body_out_dim → expand_dim (no residual)
            self.expander = TinyMultiHeadLut(
                input_dim=body_out_dim,
                n_heads=1,
                n_outputs=expand_dim,
                n_anchor_pairs=block_nap,
                tables_per_head=expand_tph if expand_tph > 0 else block_tph,
                anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
                random_seed=seed + 9999,
                initial_weights_noise=init_std,
                weight_dtype=torch.float32,
                device=torch.device(DEVICE),
            )
            self.final = nn.Linear(expand_dim, vocab, bias=True)
        else:
            self.expander = None
            self.final = nn.Linear(body_out_dim, vocab, bias=True)

    def _block_forward(self, block, x):
        # block returns [B, n_heads, n_outputs]; reshape to [B, in_dim]
        out = block(x)
        return out.reshape(out.shape[0], -1)

    def forward(self, x):
        x = self.ln(x)
        if self.dense_concat:
            stages = [x]
            for block in self.blocks:
                x = x + self._block_forward(block, x)
                stages.append(x)
            x = torch.cat(stages, dim=-1)
        else:
            for block in self.blocks:
                x = x + self._block_forward(block, x)
        if self.expander is not None:
            x = self.expander(x).squeeze(1)
        return self.final(x)


class ParquetPairsDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.tbl = pq.read_table(path, memory_map=True)
        self.n = self.tbl.num_rows
        print(f'Materializing {self.n} rows from parquet...')
        t0 = time.time()
        inputs_chunks = self.tbl.column('input').to_numpy(zero_copy_only=False)
        self.inputs = np.stack([np.asarray(a, dtype=np.float32) for a in inputs_chunks])
        logits_chunks = self.tbl.column('logits').to_numpy(zero_copy_only=False)
        self.logits = np.stack([np.asarray(a, dtype=np.float16) for a in logits_chunks])
        print(f'  inputs:  {self.inputs.shape} {self.inputs.dtype}')
        print(f'  logits:  {self.logits.shape} {self.logits.dtype}')
        print(f'  load time: {time.time()-t0:.1f}s')

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.inputs[i]),
            torch.from_numpy(self.logits[i]).to(torch.float32),
        )


def kl_loss(student_logits, teacher_logits):
    log_p_student = F.log_softmax(student_logits, dim=-1)
    p_teacher     = F.softmax(teacher_logits, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction='batchmean')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', type=str,
                   default='/home/starost/spiky/nanochat_exps/distill_unembedder/pairs.parquet')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--n_blocks', type=int, default=3)
    p.add_argument('--block_tph', type=int, default=0,  # 0 → auto NlogN/nap
                   help='Tables per block. 0 → ceil(N*log2(N) / nap)')
    p.add_argument('--block_nap', type=int, default=6)
    p.add_argument('--init_std', type=float, default=0.1)
    p.add_argument('--dense_concat', action='store_true',
                   help='DenseNet-style: concat y0..y_n_blocks for final layer')
    p.add_argument('--expand_dim', type=int, default=0,
                   help='Wide LUT expansion before Linear final (0 = none)')
    p.add_argument('--expand_tph', type=int, default=0,
                   help='Expander tph (0 = use block_tph)')
    p.add_argument('--block_n_heads', type=int, default=1,
                   help='Multi-head LUT: each block uses K parallel LUTs '
                        '(in_dim split across heads, output reshape to in_dim).')
    p.add_argument('--share_weights', action='store_true',
                   help='Use ONE shared LUT applied n_blocks times (recurrent).')
    p.add_argument('--lr_schedule', type=str, default='constant',
                   choices=['constant', 'cosine'],
                   help='LR schedule: constant or cosine (lr → 0 at end)')
    p.add_argument('--warmup_steps', type=int, default=0,
                   help='Linear LR warmup over first N steps')
    args = p.parse_args()

    if args.block_tph == 0:
        args.block_tph = math.ceil(IN_DIM * math.log2(IN_DIM) / args.block_nap)
        print(f'Auto block_tph = ceil({IN_DIM} * log2({IN_DIM}) / {args.block_nap}) = {args.block_tph}')

    torch.manual_seed(args.seed)
    ds = ParquetPairsDataset(args.pairs)
    n = len(ds)
    print(f'\nUsing all {n} samples for training (capacity test).')
    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    model = StackedLutUnembedder(
        in_dim=IN_DIM, vocab=VOCAB,
        n_blocks=args.n_blocks, block_tph=args.block_tph, block_nap=args.block_nap,
        init_std=args.init_std, seed=args.seed, dense_concat=args.dense_concat,
        expand_dim=args.expand_dim, expand_tph=args.expand_tph,
        block_n_heads=args.block_n_heads,
        share_weights=args.share_weights,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: StackedLutUnembedder — {n_params/1e6:.2f}M params')
    print(f'  LN({IN_DIM})')
    for k in range(args.n_blocks):
        print(f'  block {k}: TinyMultiHeadLut({IN_DIM}→{IN_DIM}, '
              f'n_heads=1, tph={args.block_tph}, nap={args.block_nap}) + residual')
    print(f'  final: Linear({IN_DIM}, {VOCAB}, bias=True)')
    print(f'  init_std={args.init_std}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    for g in optimizer.param_groups:
        g['initial_lr'] = g['lr']

    # Estimate total steps for cosine schedule
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    def lr_scale_at(step):
        # Linear warmup
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        # Post-warmup
        if args.lr_schedule == 'cosine':
            progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    print(f'\nTraining: epochs={args.epochs} batch_size={args.batch_size} '
          f'lr={args.lr} schedule={args.lr_schedule} warmup={args.warmup_steps}\n')
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss_acc = 0.0
        train_n = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            scale = lr_scale_at(step)
            for g in optimizer.param_groups:
                g['lr'] = g['initial_lr'] * scale
            student = model(inputs)
            loss = kl_loss(student, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            train_loss_acc += loss.item() * inputs.size(0)
            train_n += inputs.size(0)
            if step % args.log_every == 0:
                avg = train_loss_acc / train_n
                print(f'  step {step:6d}  ep {epoch+1}/{args.epochs}  '
                      f'train KL={avg:.4f}  ({time.time()-t0:.1f}s)')

        train_avg = train_loss_acc / train_n
        print(f'== ep {epoch+1}: train KL={train_avg:.4f}  ({time.time()-t0:.1f}s)')

    print(f'\nDone. Total time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
