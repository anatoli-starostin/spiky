"""Energy per LUT-Core evaluation for the gather baseline and the best cell-stationary
kernel, so the measurement can be put next to the white paper's energy table.

Power is read from nvidia-smi while the kernel runs in a tight loop, which measures
whole-board draw (memory, fans, VRMs included) -- the same basis as the paper's 1,000 W
figure for a B200 and its 150 W assumption for the Alveo card, and NOT a per-kernel
counter. Idle draw is subtracted in a second column so both readings are available.
"""
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lac  # noqa: E402
from bench import make_shape_B  # noqa: E402

B = 24576
SECONDS = 6.0


def watts(n=12, gap=0.25):
    vals = []
    for _ in range(n):
        try:
            o = subprocess.run(['nvidia-smi', '--query-gpu=power.draw',
                                '--format=csv,noheader,nounits'],
                               capture_output=True, text=True).stdout.strip()
            vals.append(float(o.splitlines()[0]))
        except Exception:
            pass
        time.sleep(gap)
    vals.sort()
    return vals[len(vals) // 2] if vals else float('nan')


def measure(fn, label, evals):
    lac.burn_in(fn, seconds=2.0)
    t0 = time.perf_counter()
    n = 0
    # keep the GPU saturated while power is sampled
    deadline = t0 + SECONDS
    import threading
    stop = [False]
    pw = [float('nan')]

    def sampler():
        pw[0] = watts()
    th = threading.Thread(target=sampler)
    th.start()
    while time.perf_counter() < deadline:
        for _ in range(4):
            fn()
            n += 1
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    th.join()
    per_iter_ms = el / n * 1e3
    uj = pw[0] * (el / n) / evals * 1e6
    print(f'{label:<38}{per_iter_ms:>10.3f} ms{pw[0]:>9.1f} W{uj:>12.2f} uJ/eval'
          f'{evals/(el/n)/1e6:>12.2f} M eval/s')
    return {'label': label, 'ms': per_iter_ms, 'watts': pw[0], 'uj_per_eval': uj,
            'mevals_per_s': evals / (el / n) / 1e6}


def main():
    lac.mod()
    idle = watts(n=6)
    print(f'idle board power {idle:.1f} W\n')
    print(f'{"kernel":<38}{"per launch":>13}{"board":>9}{"energy":>12}{"rate":>12}')
    T, J, C = make_shape_B(B)
    Tt = lac.transpose_tables(T)
    y = torch.zeros(B, T.shape[2], device='cuda')
    out = [measure(lambda: lac.run_gather(T, J, C, tsplit=1, use_coef=True, y=y,
                                          tok_per_blk=1), 'gather tsplit=1', B)]
    best = json.load(open(os.path.join(HERE, 'artifacts', 'bench.json')))['ladder']
    cand = [r for r in best if r['shape'] == 'B' and r['use_coef'] and r['B'] == B
            and r['family'] == 'v2']
    if cand:
        c = min(cand, key=lambda r: r['median_ms'])['cfg']
        out.append(measure(lambda: lac.run_v2(Tt if c['trans'] else T, J, C, K=c['K'],
                                              M=c['M'], TB=c['TB'], trans=c['trans'],
                                              use_coef=True, y=y),
                           f'cs_v2 K={c["K"]} M={c["M"]} TB={c["TB"]} trans={c["trans"]}', B))
    print(f'\nidle {idle:.1f} W; subtract it for the marginal figure if preferred.')
    print('For scale, the white paper puts one evaluation of this LUT Core at 0.165 uJ on')
    print('a 5 nm LAC ASIC and 4 uJ on the Alveo V80 FPGA, and one evaluation of the FFN')
    print('it replaces at 9.3 uJ on a fully batched B200 and 1,230 uJ at batch 1.')
    p = os.path.join(HERE, 'artifacts', 'power.json')
    json.dump({'idle_w': idle, 'rows': out}, open(p, 'w'), indent=1)
    print('wrote', p)


if __name__ == '__main__':
    main()
