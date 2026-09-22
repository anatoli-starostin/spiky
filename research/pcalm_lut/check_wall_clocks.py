"""Cross-check every CompressionMHL run's wall clock against its step count and block count."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    print(f'{"run":<52}{"steps":>6}{"blk":>4}{"wall_s":>9}{"probes":>7}{"last":>7}'
          f'{"ms/step":>9}{"params":>9}')
    for d in sorted(glob.glob(os.path.join(HERE, 'runs_ae', 'cmhl-*', 'run.json'))):
        j = json.load(open(d))
        s, c = j['summary'], j['cfg']
        n = os.path.basename(os.path.dirname(d))
        n = n.replace('cmhl-', '').replace('-w128-tph128', '').replace('-prenorm', '')
        print(f'{n:<52}{c["steps"]:>6}{s["n_blocks"]:>4}{s["wall_s"]:>9.1f}{len(j["hist"]):>7}'
              f'{j["hist"][-1]["step"]:>7}{1000 * s["wall_s"] / c["steps"]:>9.2f}'
              f'{s["params"] / 1e6:>8.2f}M')


if __name__ == '__main__':
    main()
