"""Fit a LUT-based unembedder on the dumped (concat, logits) pairs.

Same architecture as exp175's unembedder: LayerNorm(384) → TinyMultiHeadLut
(384→n_sparse, n_heads=1, tph, nap) with built-in sparse_scatter to vocab,
plus trainable per-vocab bias. KL distillation against exp174's logits.

Compare against fit_baseline_mlp.py (KL ≈ 0.012 at 20 epochs).

Usage:
    python nanochat_exps/distill_unembedder/fit_lut.py
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


class LutUnembedder(nn.Module):
    def __init__(self, in_dim, vocab, tph, nap, n_sparse, init_std, seed=42):
        super().__init__()
        self.in_dim = in_dim
        self.vocab = vocab
        self.ln = nn.LayerNorm(in_dim)
        self.lut = TinyMultiHeadLut(
            input_dim=in_dim,
            n_heads=1,
            n_outputs=n_sparse,
            n_anchor_pairs=nap,
            tables_per_head=tph,
            anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
            random_seed=seed,
            initial_weights_noise=init_std,
            sparse_scatter_n_outputs=vocab,
            sparse_scatter_seed=seed + 1,
            sparse_scatter_balanced=True,
            weight_dtype=torch.float32,
            device=torch.device(DEVICE),
        )
        self.bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, x):
        # x: [B, in_dim]
        x = self.ln(x)
        out = self.lut(x)              # [B, 1, vocab]
        return out.squeeze(1) + self.bias


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
    p.add_argument('--tph', type=int, default=4096)
    p.add_argument('--nap', type=int, default=8)
    p.add_argument('--n_sparse', type=int, default=128)
    p.add_argument('--init_std', type=float, default=0.1)
    p.add_argument('--lr_schedule', type=str, default='constant',
                   choices=['constant', 'cosine'])
    p.add_argument('--warmup_steps', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ds = ParquetPairsDataset(args.pairs)
    n = len(ds)
    print(f'\nUsing all {n} samples for training (capacity test).')
    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    model = LutUnembedder(
        in_dim=IN_DIM, vocab=VOCAB,
        tph=args.tph, nap=args.nap, n_sparse=args.n_sparse,
        init_std=args.init_std, seed=args.seed,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: LutUnembedder — {n_params/1e6:.2f}M params')
    print(f'  LN({IN_DIM}) → TinyMultiHeadLut({IN_DIM}→{args.n_sparse}, '
          f'n_heads=1, tph={args.tph}, nap={args.nap}) + sparse_scatter→{VOCAB}'
          f' + bias({VOCAB}). init_std={args.init_std}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    for g in optimizer.param_groups:
        g['initial_lr'] = g['lr']

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    def lr_scale_at(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
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
