"""Speed figure: Light trains ~2x faster and infers ~5x slower."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

C_GREEN, C_BLUE, C_ORANGE, C_GREY = '#2f6f4f', '#3b5b8c', '#b5793a', '#999999'

# measured: bench_light_vs_fast.py (layer) and bench_model_step.py (model), fp32, 6,144 tokens
LAYER = {  # name: (fwd_eval, fwd_train, fwd+bwd, peak MiB)
    'fast, gate off': (0.957, 1.012, 33.470, 19948.6),
    "fast + bnorm": (1.140, 1.162, 44.103, 21367.7),
    'light + bnorm': (8.507, 8.405, 16.147, 4036.6),
}
MODEL = {  # name: (step ms, eval ms, peak GiB)
    'dense FFN': (19.8, 6.8, 2.68),
    'fast, gate off': (218.3, 10.3, 21.41),
    'fast + bnorm': (282.5, 11.7, 22.79),
    'light + bnorm': (127.4, 56.4, 13.48),
}
RUNS = {  # run: (measured hours, bpb)
    'baseline S5\n(fast, gate off)': (0.503, 1.434572),
    "arm A'\n(fast + bnorm)": (0.656, 1.441122),
    'arm C\n(bounded x12.61)': (0.657, 1.434988),
    'arm D\n(margin x2.99)': (0.670, 1.432430),
    'arm B\n(light + bnorm)': (0.326, 1.477708),
}
COLS = [C_GREEN, C_BLUE, '#7a4f9c', '#3f8f8f', C_ORANGE]

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.4))

# ---- 1. model step: forward vs backward, and the dense floor -------------------------
a = ax[0]
names = list(MODEL)
step = [MODEL[n][0] for n in names]
x = np.arange(len(names))
bars = a.bar(x, step, .58, color=[C_GREY, C_GREEN, C_BLUE, C_ORANGE])
a.axhline(MODEL['dense FFN'][0], color='#333', ls=':', lw=1.4)
a.annotate('dense-FFN floor (19.8 ms) — everything that is NOT the LUT layer',
           (0.02, MODEL['dense FFN'][0]), xycoords=('axes fraction', 'data'),
           xytext=(0, 6), textcoords='offset points', fontsize=8, color='#333')
for i, v in enumerate(step):
    a.annotate(f'{v:.0f} ms', (i, v), xytext=(0, 4), textcoords='offset points',
               ha='center', fontsize=9)
a.set_xticks(x), a.set_xticklabels(names, fontsize=8.5)
a.set_ylabel('training step, ms (6,144 tokens)')
a.set_title('TRAINING: Light is 2.2x faster than the gated arms\n'
            'and 1.7x faster than gate-off Fast', fontsize=11)
a.grid(axis='y', alpha=.25)

# ---- 2. eval forward: the opposite story ---------------------------------------------
b = ax[1]
ev = [MODEL[n][1] for n in names]
b.bar(x, ev, .58, color=[C_GREY, C_GREEN, C_BLUE, C_ORANGE])
for i, v in enumerate(ev):
    b.annotate(f'{v:.1f} ms', (i, v), xytext=(0, 4), textcoords='offset points',
               ha='center', fontsize=9)
b.annotate('native CUDA kernel\nDISABLED by the gate\n(+14%)', (2, ev[2]),
           xytext=(-6, 26), textcoords='offset points', fontsize=7.5, ha='center',
           arrowprops=dict(arrowstyle='->', lw=.9, color='#555'))
b.set_xticks(x), b.set_xticklabels(names, fontsize=8.5)
b.set_ylabel('eval forward, ms (6,144 tokens)')
b.set_title('INFERENCE: Light is 5.5x SLOWER\n(it materialises rows instead of fusing '
            'gather+sum)', fontsize=11)
b.grid(axis='y', alpha=.25)

# ---- 3. the real trade: wall-clock vs quality ----------------------------------------
c = ax[2]
hrs = [RUNS[k][0] for k in RUNS]
bpb = [RUNS[k][1] for k in RUNS]
# A', C and D sit almost on top of each other (0.656/0.657/0.670 h, all ~baseline bpb),
# so their labels are fanned out rather than stacked.
OFF = {0: (0, 15), 1: (-52, 20), 2: (46, 6), 3: (30, -26), 4: (0, -26)}
for i, ((k, (h, q)), col) in enumerate(zip(RUNS.items(), COLS)):
    c.plot(h, q, 'o', ms=13, color=col)
    c.annotate(k, (h, q), xytext=OFF[i], textcoords='offset points', ha='center',
               fontsize=7.5,
               arrowprops=(dict(arrowstyle='-', lw=.7, color='#888')
                           if abs(OFF[i][0]) > 20 else None))
c.axhline(1.434572, color='#333', ls='--', lw=1.2)
c.axhspan(1.434572 - 0.0096, 1.434572 + 0.0096, color=C_GREEN, alpha=.13,
          label='±1 seed sd of baseline')
c.axhline(1.474749, color=C_GREY, ls=':', lw=1.4)
c.annotate("vanilla dense 1.4747 — Light gives up the LUT FFN's entire margin",
           (0.52, 1.474749), xytext=(0, 5), textcoords='offset points', fontsize=7.5,
           color='#555')
c.set_xlim(0.24, 0.78), c.set_ylim(1.428, 1.487)
c.set_xlabel('measured training wall-clock, hours (4,000 steps)')
c.set_ylabel('final proxy val bpb')
c.set_title('The trade: 2x faster training costs +0.043 bpb —\n'
            'which is the whole LUT-vs-dense advantage', fontsize=11)
c.grid(alpha=.25), c.legend(fontsize=8, loc='upper right')

fig.suptitle('Is LightMultiHeadLUT faster? Yes at training, no at inference — '
             'RTX 5090, fp32, anchor sizing', fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig('/tmp/light_speed.png', dpi=130, bbox_inches='tight')
print('wrote /tmp/light_speed.png')
