"""Training trace for the L=4 CompressionMHL arm: is the gain growth contained, and has it converged?"""
import json
import math
import os
import sys

S = 0.3081
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1] if len(sys.argv) > 1 else \
    'cmhl-L4-w128-tph128-din128-dout-1-prenorm-finalnorm-noresid-s10000'
MARKS = (1, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)


def main():
    j = json.load(open(os.path.join(HERE, 'runs_ae', RUN, 'run.json')))
    s, H = j['summary'], j['hist']
    nb = s['n_blocks']
    m01 = s['test_mse'] * S * S
    print(f'{RUN}\n  test std {s["test_mse"]:.5f}   MSE[0,1] {m01:.6f}   PSNR {10*math.log10(1/m01):.2f} dB'
          f'\n  train {s["train_mse"]:.5f}   train/test gap '
          f'{100*(s["test_mse"]-s["train_mse"])/s["train_mse"]:.1f}%'
          f'   params {s["params"]/1e6:.2f}M   wall {s["wall_s"]:.0f}s'
          f'   improving {s["still_improving_pct"]:.2f}%/75 at the stop')
    floor = 0.06877
    print(f'  vs the linear 784-128-784 floor {floor:.5f}: '
          f'{100*(s["test_mse"]-floor)/floor:+.1f}%')

    print(f'\n{"step":>6}{"train":>9}{"test":>9}   per-block gain' + ' ' * (6 * nb - 8)
          + f'{"|out| b" + str(nb-1):>10}{"score b0":>10}{"score b" + str(nb-1):>10}'
          f'{"m_min":>8}{"tau":>7}')
    for r in H:
        if r['step'] not in MARKS:
            continue
        g = ' '.join(f'{r[f"branch/ratio_b{b}"]:5.2f}' for b in range(nb))
        print(f'{r["step"]:>6}{r["eval/train_mse"]:>9.5f}{r["eval/test_mse"]:>9.5f}   {g}'
              f'{r[f"lut/norm_out_b{nb-1}"]:>10.1f}{r["lut/score_sum_b0"]:>10.0f}'
              f'{r[f"lut/score_sum_b{nb-1}"]:>10.0f}{r["lut/m_min_mean"]:>8.3f}'
              f'{r["lut/tau_mean"]:>7.3f}')

    last = H[-1]
    print('\nfinal per-block profile')
    for name, key, fmt in [('gain ', 'branch/ratio_b{}', '7.3f'), ('|in| ', 'lut/norm_in_b{}', '7.2f'),
                           ('|out|', 'lut/norm_out_b{}', '7.2f'), ('score', 'lut/score_sum_b{}', '7.1f'),
                           ('m_min', 'lut/m_min_b{}', '7.3f'), ('tau  ', 'lut/tau_b{}', '7.3f')]:
        print(f'  {name} ' + ' '.join(format(last[key.format(b)], fmt) for b in range(nb)))

    # convergence: mean test MSE over the last two 1000-step blocks
    import statistics as st
    for lo, hi in ((7000, 8500), (8500, 10001)):
        v = [r['eval/test_mse'] for r in H if lo < r['step'] <= hi]
        print(f'  mean test over ({lo}, {hi}]: {st.mean(v):.5f}')


if __name__ == '__main__':
    main()
