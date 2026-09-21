"""Gate-1 analysis: depth profile of gradient alignment, zero-gradient layers, steps/time-to-target."""
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

p = sys.argv[1] if len(sys.argv) > 1 else 'runs/gate1_fast.json'
r = json.load(open(p))
N, L = r['args']['width'], r['args']['depth']
print(f"cell N={N} L={L} | T={r['T']} | budget {r['args']['steps']}/{r['epoch_steps']} steps "
      f"({r['args']['steps'] / r['epoch_steps']:.3f} epoch)")
for arm, d in r['arms'].items():
    print(f"  {arm:5s} final loss {d['final_loss']:.4f} | test acc {d['test_acc']:.4f} | "
          f"{d['wall_s']:.1f}s ({d['ms_per_step']:.0f} ms/step)")

# reachable targets from the observed curves
losses = {arm: [h['loss'] for h in d['hist']] for arm, d in r['arms'].items()}
best = max(min(v) for v in losses.values())
targets = [round(best + x, 3) for x in (0.10, 0.05, 0.02, 0.0)]
print('  steps-to-target / time-to-target (targets chosen reachable by every arm):')
for t in targets:
    line = []
    for arm, d in r['arms'].items():
        hit = next((h for h in d['hist'] if h['loss'] <= t), None)
        line.append(f"{arm} " + ('never' if hit is None else f"step {hit['step']:4d} / {hit['time']:6.1f}s"))
    print(f'    loss <= {t:.3f}: ' + ' | '.join(line))

print('  gradient alignment to BP, by checkpoint (interior layers Wi.*):')
prof = {}
for c, al in r['alignment'].items():
    for mode, rows in al.items():
        inter = [v for n, v in rows if n.startswith('Wi.')]
        dead = sum(1 for v in inter if v is None or (isinstance(v, float) and math.isnan(v)))
        live = [v for v in inter if v is not None and not math.isnan(v)]
        prof[(c, mode)] = inter
        print(f'    step {c:>4} {mode:5s}: {dead:3d}/{len(inter)} layers with EXACTLY ZERO gradient | '
              f'mean cos over the rest {sum(live) / max(len(live), 1):+.4f} | '
              f'deepest-5 mean {sum(live[-5:]) / max(len(live[-5:]), 1):+.4f}')

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for (c, mode), inter in prof.items():
    if str(c) != str(max(int(k) for k in r['alignment'])):
        continue
    xs = list(range(1, len(inter) + 1))
    ys = [0.0 if (v is None or math.isnan(v)) else v for v in inter]
    axs[0].plot(xs, ys, marker='.', label=f'{mode} (0 = zero gradient)')
axs[0].set_xlabel('interior layer index (1 = closest to input)')
axs[0].set_ylabel('cosine to BP gradient')
axs[0].set_title(f'Per-layer gradient alignment, N={N} L={L}, T={r["T"]}')
axs[0].legend(); axs[0].grid(ls=':', alpha=0.5)
for arm, d in r['arms'].items():
    axs[1].plot([h['step'] for h in d['hist']], [h['loss'] for h in d['hist']], label=arm)
axs[1].set_xlabel('step'); axs[1].set_ylabel('train loss (batch)'); axs[1].legend(); axs[1].grid(ls=':', alpha=0.5)
axs[1].set_title('Truncated budget: loss vs step')
fig.tight_layout()
out = p.replace('.json', '.png')
fig.savefig(out, dpi=130)
print('wrote', out)
