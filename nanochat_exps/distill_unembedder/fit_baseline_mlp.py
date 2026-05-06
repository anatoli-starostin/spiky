"""Sanity-check the distillation pipeline.

Trains the same unembedder MLP that exp174 uses (LN(384) → Linear(384, 3072) →
GELU → Linear(3072, vocab=32768) with bias=False) on the dumped pairs from
extract_pairs.py. KL-divergence loss between predicted softmax and target
softmax (target derived from exp174's logits in the parquet).

Should reach near-zero KL trivially — confirms the dataset, loss, and
optimizer pipeline are correct before we try LUT-based heads.

Usage:
    python nanochat_exps/distill_unembedder/fit_baseline_mlp.py
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Architecture (matches exp174 unembedder)
IN_DIM = 384
HIDDEN = 3072
VOCAB = 32768


class MlpUnembedder(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, vocab=VOCAB):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.ln(x))))


class ParquetPairsDataset(torch.utils.data.Dataset):
    """Memory-maps the parquet file, decodes rows on-demand."""
    def __init__(self, path):
        self.tbl = pq.read_table(path, memory_map=True)
        self.n = self.tbl.num_rows
        # Materialize as numpy arrays once — faster than per-row parquet decode.
        # ~5.5 GB for fp16 logits; fits in RAM.
        print(f'Materializing {self.n} rows from parquet...')
        t0 = time.time()
        # FixedSizeList<float32, 384> -> ndarray [N, 384]
        inputs_chunks = self.tbl.column('input').to_numpy(zero_copy_only=False)
        # to_numpy on list-of-lists yields np.array of np.array; stack:
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
            torch.from_numpy(self.inputs[i]),                    # [384] fp32
            torch.from_numpy(self.logits[i]).to(torch.float32),  # [vocab] fp32
        )


def kl_loss(student_logits, teacher_logits):
    """KL(teacher || student) averaged over batch.
    student_logits, teacher_logits: [B, vocab]
    Lower = student matches teacher's softmax better.
    """
    log_p_student = F.log_softmax(student_logits, dim=-1)
    p_teacher     = F.softmax(teacher_logits, dim=-1)
    # KL(teacher || student) = sum_v p_teacher * (log p_teacher - log p_student)
    # F.kl_div expects (log_input, target) and computes sum target*(log target - log input)
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
    p.add_argument('--lr_schedule', type=str, default='constant',
                   choices=['constant', 'cosine'])
    p.add_argument('--warmup_steps', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ds = ParquetPairsDataset(args.pairs)
    n = len(ds)
    print(f'\nUsing all {n} samples for training (capacity test, no val split).')
    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    model = MlpUnembedder().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: MlpUnembedder — {n_params/1e6:.2f}M params')
    print(f'  fc1: Linear({IN_DIM}, {HIDDEN}, bias=False)')
    print(f'  fc2: Linear({HIDDEN}, {VOCAB}, bias=False)')

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
