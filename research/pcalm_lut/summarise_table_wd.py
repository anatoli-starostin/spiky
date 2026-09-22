"""Does decaying the LUT table values stop the growth? The sweep, against the no-decay baseline."""
import json
import math
import os

S = 0.3081
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs_ae')
ARMS = [('cmhl-L4-w128-tph128-din128-dout-1-prenorm-finalnorm-noresid-s10000', 'wd = 0 (baseline)'),
        ('cmhl-L4-din128-finalnorm-noresid-tablewd1e-5-s10000', 'table wd = 1e-5'),
        ('cmhl-L4-din128-finalnorm-noresid-tablewd1e-4-s10000', 'table wd = 1e-4'),
        ('cmhl-L4-din128-finalnorm-noresid-tablewd1e-3-s10000', 'table wd = 1e-3')]
FLOOR = 0.06877


def main():
    print(f'{"arm":<20}{"test std":>10}{"MSE[0,1]":>10}{"PSNR":>7}{"train":>9}{"gap":>7}'
          f'{"vs floor":>10}{"gain b0":>9}{"|out| b3":>10}{"score b0":>10}')
    rows = {}
    for d, lab in ARMS:
        j = json.load(open(os.path.join(R, d, 'run.json')))
        s, h, H = j['summary'], j['hist'][-1], j['hist']
        m01 = s['test_mse'] * S * S
        print(f'{lab:<20}{s["test_mse"]:>10.5f}{m01:>10.6f}{10*math.log10(1/m01):>7.2f}'
              f'{s["train_mse"]:>9.5f}'
              f'{100*(s["test_mse"]-s["train_mse"])/s["train_mse"]:>6.1f}%'
              f'{100*(s["test_mse"]-FLOOR)/FLOOR:>+9.1f}%'
              f'{h["branch/ratio_b0"]:>9.3f}{h["lut/norm_out_b3"]:>10.1f}{h["lut/score_sum_b0"]:>10.0f}')
        rows[lab] = H

    print('\nblock-0 gain over training -- the question is whether decay flattens it')
    marks = (500, 1000, 2000, 3000, 5000, 7000, 9000, 10000)
    print(f'{"arm":<20}' + ''.join(f'{m:>8}' for m in marks))
    for lab, H in rows.items():
        by = {r['step']: r for r in H}
        print(f'{lab:<20}' + ''.join(f'{by[m]["branch/ratio_b0"]:>8.2f}' for m in marks if m in by))

    print('\nblock-0 summed score over training')
    print(f'{"arm":<20}' + ''.join(f'{m:>8}' for m in marks))
    for lab, H in rows.items():
        by = {r['step']: r for r in H}
        print(f'{lab:<20}' + ''.join(f'{by[m]["lut/score_sum_b0"]:>8.0f}' for m in marks if m in by))

    print('\nRMS of the decayed tables, and the stream, at the stop')
    for d, lab in ARMS:
        j = json.load(open(os.path.join(R, d, 'run.json')))
        h, s = j['hist'][-1], j['summary']
        nb = s['n_blocks']
        print(f'  {lab:<20} |in| ' + ' '.join(f'{h[f"lut/norm_in_b{b}"]:7.1f}' for b in range(nb))
              + '   m_min ' + ' '.join(f'{h[f"lut/m_min_b{b}"]:5.3f}' for b in range(nb))
              + '   tau ' + ' '.join(f'{h[f"lut/tau_b{b}"]:5.3f}' for b in range(nb)))


if __name__ == '__main__':
    main()
