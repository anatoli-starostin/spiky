"""Build & cache a unigram log-frequency table over the training corpus for the MDN head's
per-token bias init (b = log unigram freq). Cached to mdn/unigram_logfreq.npy (len V)."""
import os, sys
import numpy as np, torch
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

N_BATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 400
BS, SEQ = 32, 512
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unigram_logfreq.npy')

tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
V = tok.get_vocab_size()
loader = tokenizing_distributed_data_loader_bos_bestfit(tok, BS, SEQ, split='train', device='cpu')
counts = np.zeros(V, dtype=np.int64)
tot = 0
for i in range(N_BATCHES):
    x, y = next(loader)
    ids = x.reshape(-1).numpy()
    np.add.at(counts, ids, 1)
    tot += ids.size
print(f"counted {tot:,} tokens over {N_BATCHES} batches; vocab seen = {(counts>0).sum()}/{V}")
# add-1 smoothing -> probabilities -> log
p = (counts + 1.0) / (tot + V)
logfreq = np.log(p).astype(np.float32)
np.save(OUT, logfreq)
print(f"saved {OUT}  (min={logfreq.min():.2f} max={logfreq.max():.2f} mean={logfreq.mean():.2f})")
