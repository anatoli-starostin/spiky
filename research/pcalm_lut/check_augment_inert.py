"""With --augment off, the harness must reproduce the committed run step for step.

The augmentation was added by importing vit_autoencoder.augment and calling it inside the training loop.
That touches the loop, so "off is unchanged" is a claim that has to be checked against the record rather
than asserted -- this compares a fresh short flag-off run to the committed 10000-step run's first probes.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ae_smoke/noaug_check/run.json'
OLD = os.path.join(HERE, 'runs_ae',
                   'cmhl-L4-w128-tph128-din128-dout-1-prenorm-finalnorm-noresid-s10000', 'run.json')


def main():
    a = json.load(open(NEW))['hist']
    old = {r['step']: r for r in json.load(open(OLD))['hist']}
    same = 0
    for r in a:
        o = old.get(r['step'])
        if o is None:
            continue
        hit = r['eval/train_mse'] == o['eval/train_mse'] and r['eval/test_mse'] == o['eval/test_mse']
        same += hit
        print(f'  step {r["step"]:>4}  train {r["eval/train_mse"]:.6f} vs {o["eval/train_mse"]:.6f}'
              f'   test {r["eval/test_mse"]:.6f} vs {o["eval/test_mse"]:.6f}   identical={hit}')
    print(f'{same} of the compared probes are bit-identical to the committed run')


if __name__ == '__main__':
    main()
