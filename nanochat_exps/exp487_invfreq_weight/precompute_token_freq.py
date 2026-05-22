"""Precompute training-token frequency counts -> token_freq.pt ([V] int64).

Used by exp487 for inverse-frequency loss weighting. Samples N_BATCHES of the
TRAIN split (bs/context from exp475 config) and counts target-token occurrences.
"""
import os, sys
import torch

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

DEVICE = 'cuda'
HERE = os.path.dirname(os.path.abspath(__file__))
N_BATCHES = 2000          # ~2000*16*512 = 16.4M tokens
BS, CTX = 16, 512
VOCAB = 32768

tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
loader = tokenizing_distributed_data_loader_bos_bestfit(tok, BS, CTX, split='train', device=DEVICE)
counts = torch.zeros(VOCAB, dtype=torch.long, device=DEVICE)
seen = 0
for i in range(N_BATCHES):
    _, y = next(loader)
    yv = y.view(-1)
    yv = yv[yv != -1]
    counts += torch.bincount(yv, minlength=VOCAB)
    seen += yv.numel()
    if (i + 1) % 200 == 0:
        print(f'batch {i+1}/{N_BATCHES}  tokens={seen:,}  nonzero_vocab={(counts>0).sum().item()}')

counts = counts.cpu()
torch.save(counts, os.path.join(HERE, 'token_freq.pt'))
print(f'saved token_freq.pt  total_tokens={seen:,}  '
      f'covered_vocab={(counts>0).sum().item()}/{VOCAB}  '
      f'max_count={counts.max().item()}  min_nonzero={counts[counts>0].min().item()}')
