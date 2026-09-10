"""FVU floor set by the student's OUTPUT RANK, per layer.

A CompressionMHL student ends in `decompress: Linear(H * inner_out -> 384)`, so its output lives
in an affine subspace of dimension at most H * inner_out. No matter how good the tables are, its
held-out FVU is therefore at least the fraction of teacher-output variance OUTSIDE the top-k
principal directions, k = H * inner_out (the bias absorbs the mean). With the exp_n_0238
architecture k = 4 * 48 = 192 of 384. Same val slab as distill_ffn.py (32 rows after skipping
12). CPU is fine.

    CUDA_VISIBLE_DEVICES= python rank_floor.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_ffn as D   # noqa: E402

tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
teacher, tcfg = D.load_teacher(D.DEF_TEACHER, tok.get_vocab_size(), 'cpu')
val_loader = D.tokenizing_distributed_data_loader_bos_bestfit(tok, 48, tcfg['seq_len'],
                                                              split='val', device='cpu')
val_idx = D.take_rows(val_loader, 32, skip_rows=12)
layers = list(range(tcfg['depth']))
outs = {}
for i in range(0, 32, 8):
    for li, (_, o) in D.teacher_ffn_io(teacher, val_idx[i:i + 8], layers).items():
        outs.setdefault(li, []).append(o)
C = tcfg['n_embd']
ks = (96, 192, 288, C)
print('FVU floor at output rank k (variance outside the top-k PCs of the teacher FFN output)')
print('layer   out var  ' + ''.join(f'  k={k:<5}' for k in ks))
for li in layers:
    o = torch.cat(outs[li]).double()
    o = o - o.mean(0)
    ev = torch.linalg.eigvalsh(o.T @ o / o.shape[0]).flip(0)
    tot = ev.sum()
    print(f'  L{li}   {tot.item() / C:.5f}  ' +
          ''.join(f'  {(ev[k:].sum() / tot).item():.4f} ' for k in ks))
