"""MNIST / Fashion-MNIST as flat float tensors on the GPU (they fit: 60k x 784 fp32 = 188 MB).

Uses torchvision's raw files from DATA_ROOT (default ~/.cache/vision_datasets). Downloading needs network, so
`fetch.py` does that once, outside the cage; this module is offline-only.
"""
import os

import torch
from torchvision import datasets

DATA_ROOT = os.environ.get('PCALM_DATA_ROOT', os.path.expanduser('~/.cache/vision_datasets'))
_SETS = {'mnist': datasets.MNIST, 'fashion': datasets.FashionMNIST}


def load(name, train=True, device='cuda', download=False):
    ds = _SETS[name](DATA_ROOT, train=train, download=download)
    x = ds.data.to(device).float().div_(255.0).reshape(len(ds), -1)
    x = (x - 0.1307) / 0.3081
    y = torch.nn.functional.one_hot(ds.targets.to(device), 10).float()
    return x, y


class TensorLoader:
    """Deterministic shuffled minibatches over GPU-resident tensors (same order for every trainer/seed)."""

    def __init__(self, x, y, batch_size=64, seed=0, shuffle=True, drop_last=True):
        self.x, self.y, self.bs, self.seed, self.shuffle, self.drop_last = x, y, batch_size, seed, shuffle, drop_last

    def __iter__(self):
        n = self.x.shape[0]
        idx = (torch.randperm(n, generator=torch.Generator(device='cpu').manual_seed(self.seed)).to(self.x.device)
               if self.shuffle else torch.arange(n, device=self.x.device))
        last = n - (n % self.bs if self.drop_last else 0)
        for i in range(0, last, self.bs):
            j = idx[i:i + self.bs]
            yield self.x[j], self.y[j]

    def __len__(self):
        return self.x.shape[0] // self.bs
