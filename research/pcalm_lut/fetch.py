"""One-off dataset download (needs network, so it runs OUTSIDE the cage). Everything else is offline.

    python fetch.py        # -> ~/.cache/vision_datasets/{MNIST,FashionMNIST}
"""
from data import DATA_ROOT, load

for name in ('mnist', 'fashion'):
    x, y = load(name, train=True, device='cpu', download=True)
    xt, yt = load(name, train=False, device='cpu', download=True)
    print(f'{name}: train {tuple(x.shape)} {tuple(y.shape)} | test {tuple(xt.shape)} -> {DATA_ROOT}')
