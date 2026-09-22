"""Tables of readout-norm dynamics per block and per checkpoint."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs_ae')
MARKS = (500, 1000, 2000, 3000, 5000, 7000, 9000, 10000)
TAPS = ('t0_in', 't1_postnorm', 't2_postcompress', 't3_lut_out', 't5_block_out')


def load(name):
    return json.load(open(os.path.join(R, name, 'run.json')))


def table(H, nb, label):
    by = {r['step']: r for r in H}
    steps = [m for m in MARKS if m in by]

    print(f'\n### {label}')
    print('\nREADOUT (t5 = block output) mean per-sample L2 norm')
    print(f'{"block":<8}' + ''.join(f'{m:>9}' for m in steps))
    for b in range(nb):
        print(f'{b:<8}' + ''.join(f'{by[m][f"norm/t5_block_out_mean_b{b}"]:>9.1f}' for m in steps))

    print('\nPER-BLOCK GAIN  ||t5 out|| / ||t0 in||   (t0 is the PRE-NORM block input)')
    print(f'{"block":<8}' + ''.join(f'{m:>9}' for m in steps))
    for b in range(nb):
        print(f'{b:<8}' + ''.join(f'{by[m][f"norm/gain_b{b}"]:>9.3f}' for m in steps))
    print(f'{"product":<8}' + ''.join(
        f'{__import__("math").prod([by[m][f"norm/gain_b{b}"] for b in range(nb)]):>9.2f}'
        for m in steps))
    print(f'{"t5b3/t0b0":<8}' + ''.join(
        f'{by[m][f"norm/t5_block_out_mean_b{nb-1}"] / by[m]["norm/t0_in_mean_b0"]:>9.2f}'
        for m in steps))

    print('\nSTAGE-BY-STAGE inside block 0, mean norm -- which stage is doing the amplifying')
    print(f'{"tap":<18}' + ''.join(f'{m:>9}' for m in steps))
    for t in TAPS:
        print(f'{t:<18}' + ''.join(f'{by[m][f"norm/{t}_mean_b0"]:>9.1f}' for m in steps))
    print(f'{"t2/t1 compress":<18}'
          + ''.join(f'{by[m]["norm/t2_postcompress_mean_b0"] / by[m]["norm/t1_postnorm_mean_b0"]:>9.3f}'
                    for m in steps))
    print(f'{"t3/t2 the LUT":<18}'
          + ''.join(f'{by[m]["norm/t3_lut_out_mean_b0"] / by[m]["norm/t2_postcompress_mean_b0"]:>9.2f}'
                    for m in steps))

    print('\nSPREAD of the readout across samples: std / mean')
    print(f'{"block":<8}' + ''.join(f'{m:>9}' for m in steps))
    for b in range(nb):
        print(f'{b:<8}' + ''.join(
            f'{by[m][f"norm/t5_block_out_std_b{b}"] / by[m][f"norm/t5_block_out_mean_b{b}"]:>9.3f}'
            for m in steps))

    last = by[steps[-1]]
    print(f'\nFINAL CHECKPOINT ({steps[-1]}): per-sample readout norm distribution, and every tap')
    print(f'{"block":<7}{"min":>9}{"median":>9}{"mean":>9}{"max":>9}{"max/med":>9}   '
          + ''.join(f'{t.split("_")[0]:>10}' for t in TAPS))
    for b in range(nb):
        k = f'norm/t5_block_out_%s_b{b}'
        med = last[k % 'median']
        print(f'{b:<7}{last[k % "min"]:>9.1f}{med:>9.1f}{last[k % "mean"]:>9.1f}'
              f'{last[k % "max"]:>9.1f}{last[k % "max"] / max(med, 1e-9):>9.2f}   '
              + ''.join(f'{last[f"norm/{t}_mean_b{b}"]:>10.1f}' for t in TAPS))


def main():
    for run, label in ((sys.argv[1] if len(sys.argv) > 1 else 'cmhl-L4-taps-wd0-s10000',
                        'wd = 0 (baseline)'),
                       ('cmhl-L4-taps-wd1e-3-s10000', 'table wd = 1e-3')):
        try:
            j = load(run if label.startswith('wd = 0') else 'cmhl-L4-taps-wd1e-3-s10000')
        except FileNotFoundError:
            print(f'\n### {label}: not present')
            continue
        table(j['hist'], j['summary']['n_blocks'], label)
        print(f'\n  final test MSE {j["summary"]["test_mse"]:.5f}, '
              f'train {j["summary"]["train_mse"]:.5f}')


if __name__ == '__main__':
    main()
