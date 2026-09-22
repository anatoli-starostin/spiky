"""Sample 3 alone, large: does the printed graphic on the pullover survive the bottleneck?

One image (column 3 of every grid in this line of work -- the pullover, test index 1, the first pullover
in the test split), rendered big across the originals and every model, so the printed region can be
judged directly instead of squinting at a 28x28 thumbnail.

RULES THIS FILE FOLLOWS, because the figure exists to answer an honesty question:
  - nearest-neighbour upscaling only. No smoothing, which would manufacture legibility.
  - no sharpening, no contrast stretch, no per-image normalisation. Every panel is the raw decoder
    output put through the same inverse of the loader's standardisation as the original
    (x * 0.3081 + 0.1307, clamped to [0, 1]) and drawn on a fixed vmin=0, vmax=1 greyscale.
  - the same deterministic sample selection as every other figure here (first occurrence of each label
    in the test split), so sample 3 is the same image it has always been.
  - full frame, no crop.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torchvision import datasets  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from data import DATA_ROOT, load  # noqa: E402
from make_vit_dropout_figures import LINEAR, MEAN, OUT, R, STD, build  # noqa: E402

LINEAR128 = 'linear-full-lat128-s10000'
D64 = 'vit-p2-k8-e4d4-nowarm-full-s60000'
D128 = 'vit-p2-k8-e4d4-d128h4-lat128-nowarm-full-aug-s60000'
SAMPLE = 2          # zero-based column index: the third sample, the pullover


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xte, _ = load('fashion', train=False, device=dev)
    labels = datasets.FashionMNIST(DATA_ROOT, train=False, download=False).targets.to(dev)
    idx = [int((labels == c).nonzero()[0]) for c in range(10)]
    print(f'selection {idx}; sample 3 is test index {idx[SAMPLE]}, label '
          f'{int(labels[idx[SAMPLE]])} (pullover)')
    x = xte[idx]
    one = x[SAMPLE:SAMPLE + 1]

    panels = [('original', one)]
    for name, tag in [(LINEAR, 'linear 784-64-784\n12.25x'), (LINEAR128, 'linear 784-128-784\n6.125x')]:
        f, d = build(name, dev)
        panels.append((f'{tag}\ntest {d["summary"]["test_mse"]:.5f}', f(one)))
    f, d = build(D64, dev)
    panels.append((f'ViT d64 lat64 no aug\n12.25x  60K\ntest {d["summary"]["test_mse"]:.5f}', f(one)))

    dd = json.load(open(os.path.join(R, D128, 'run.json')))
    for ck in dd['summary']['checkpoints']:
        f, _ = build(D128, dev, ck['file'])
        panels.append((f'ViT d128 lat128 + aug\n6.125x  {ck["step"]//1000}K\n'
                       f'test {ck["test_mse"]:.5f}', f(one)))

    n = len(panels)
    fig, ax = plt.subplots(1, n, figsize=(1.85 * n, 3.1))
    for a, (label, t) in zip(ax, panels):
        a.imshow((t[0] * STD + MEAN).clamp(0, 1).reshape(28, 28).cpu(), cmap='gray', vmin=0, vmax=1,
                 interpolation='nearest')
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_color('#CCCCCC')
        a.set_title(label, fontsize=7.5, pad=5)
        if t is not one:
            a.set_xlabel(f'MSE {float((t[0] - one[0]).pow(2).mean()):.3f}', fontsize=7.5, labelpad=3)
    fig.suptitle('Sample 3 (pullover with a printed graphic), full frame, nearest-neighbour upscaling, '
                 'no sharpening or contrast adjustment of any kind', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = os.path.join(OUT, 'zoom_pullover_print.png')
    fig.savefig(path, dpi=260)
    print('wrote', path)


if __name__ == '__main__':
    main()
