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
    'dense FFN': (19.9, 6.8, 2.68),
    'fast, gate off': (217.8, 10.3, 21.41),
    'fast + bnorm': (282.2, 11.6, 22.79),
    'light\n(pre-fusion)': (127.4, 56.4, 13.48),
    'light\nOPTIMISED': (68.3, 21.3, 5.51),
}
RUNS = {  # run: (measured hours, bpb)
    'baseline S5\n(fast, gate off)': (0.503, 1.434572),
    "arm A'\n(fast + bnorm)": (0.656, 1.441122),
    'arm C\n(bounded x12.61)': (0.657, 1.434988),
    'arm D\n(margin x2.99)': (0.670, 1.432430),
    'arm B\n(light, as run)': (0.326, 1.477708),
    'arm B at optimised\ncost (projected)': (0.175, 1.477708),
}
COLS = [C_GREEN, C_BLUE, '#7a4f9c', '#3f8f8f', C_ORANGE, '#d8b48a']

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.4))

# ---- 1. model step: forward vs backward, and the dense floor -------------------------
a = ax[0]
names = list(MODEL)
step = [MODEL[n][0] for n in names]
x = np.arange(len(names))
bars = a.bar(x, step, .58, color=[C_GREY, C_GREEN, C_BLUE, '#d8b48a', C_ORANGE])
a.axhline(MODEL['dense FFN'][0], color='#333', ls=':', lw=1.4)
a.annotate('dense-FFN floor (19.8 ms) — everything that is NOT the LUT layer',
           (0.02, MODEL['dense FFN'][0]), xycoords=('axes fraction', 'data'),
           xytext=(0, 6), textcoords='offset points', fontsize=8, color='#333')
for i, v in enumerate(step):
    a.annotate(f'{v:.0f} ms', (i, v), xytext=(0, 4), textcoords='offset points',
               ha='center', fontsize=9)
a.set_xticks(x), a.set_xticklabels(names, fontsize=8.5)
a.set_ylabel('training step, ms (6,144 tokens)')
a.set_title('TRAINING: optimised Light is 4.1x faster than the gated\n'
            'arms, 3.2x faster than gate-off Fast, 3.9x less memory', fontsize=11)
a.grid(axis='y', alpha=.25)

# ---- 2. eval forward: the opposite story ---------------------------------------------
b = ax[1]
ev = [MODEL[n][1] for n in names]
b.bar(x, ev, .58, color=[C_GREY, C_GREEN, C_BLUE, '#d8b48a', C_ORANGE])
for i, v in enumerate(ev):
    b.annotate(f'{v:.1f} ms', (i, v), xytext=(0, 4), textcoords='offset points',
               ha='center', fontsize=9)
b.annotate('native CUDA kernel\nDISABLED by the gate (+12%)', (2, ev[2]),
           xytext=(0, 34), textcoords='offset points', fontsize=7.5, ha='center',
           arrowprops=dict(arrowstyle='->', lw=.9, color='#555'))
b.set_xticks(x), b.set_xticklabels(names, fontsize=8.5)
b.set_ylabel('eval forward, ms (6,144 tokens)')
b.set_title('INFERENCE: was 5.5x slower, now 2.1x — fusing\n'
            'gather+sum and reusing the native bit-pack', fontsize=11)
b.grid(axis='y', alpha=.25)

# ---- 3. the real trade: wall-clock vs quality ----------------------------------------
c = ax[2]
hrs = [RUNS[k][0] for k in RUNS]
bpb = [RUNS[k][1] for k in RUNS]
# A', C and D sit almost on top of each other (0.656/0.657/0.670 h, all ~baseline bpb),
# so their labels are fanned out rather than stacked.
OFF = {0: (0, 15), 1: (-52, 20), 2: (46, 6), 3: (30, -26), 4: (40, 18), 5: (0, -30)}
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
c.annotate("vanilla dense 1.4747 — Light sits ON this line, +0.003 (inside noise)",
           (0.52, 1.474749), xytext=(0, 5), textcoords='offset points', fontsize=7.5,
           color='#555')
c.set_xlim(0.09, 0.82), c.set_ylim(1.428, 1.487)
c.set_xlabel('measured training wall-clock, hours (4,000 steps)')
c.set_ylabel('final proxy val bpb')
c.set_title('Light matches DENSE quality (+0.003, inside noise)\n'
            'at 3.2x faster training than gate-off Fast', fontsize=11)
c.grid(alpha=.25), c.legend(fontsize=8, loc='upper right')

fig.suptitle('LightMultiHeadLUT after the embedding_bag fusion + native bit-pack — '
             'RTX 5090, fp32, anchor sizing', fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig('/tmp/light_speed.png', dpi=130, bbox_inches='tight')
print('wrote /tmp/light_speed.png')
