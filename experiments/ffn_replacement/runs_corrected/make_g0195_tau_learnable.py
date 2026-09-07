"""Build exp_g_0195: exp_g_0193 + top-2 blend with LEARNABLE tau, flat init 0.5.

The sibling of exp_g_0194. 0194 froze tau at the measured per-layer Delta_m
(0.033-0.108); this one starts flat at 0.5 -- 5x to 15x ABOVE those margins -- and lets
gradient descent move it. The question is where tau migrates to: toward the measured
Delta_m, or somewhere else entirely.

WHAT 0.5 MEANS AT INIT. With margins of 0.033-0.108, 2m/tau is only 0.13-0.43, so the
neighbour weight w1 = sigmoid(-2m/tau) starts at 0.394-0.467 -- near-even blending between
the winning cell and its nearest neighbour. That is a MUCH softer start than 0194, whose
matched tau puts w1 at sigmoid(-1) = 0.269 by construction. Per-layer numbers are printed
by this script and recorded in the arch note.

PARAMETER COUNT SHIFTS BY +6. Learnable tau is an nn.Parameter (one scalar per layer), not
a buffer, so total_params is 67,351,686 against exp_g_0193's and exp_g_0194's 67,351,680.
The comparison against those runs is therefore NOT param-matched. Six parameters out of 67
million is not a capacity story, but it is recorded here so nobody later reads the
leaderboard as exactly matched.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from fork_trainer import fork_trainer                      # noqa: E402
from model_build import MEASURED_TAU_G0193                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1')
NAME = 'exp_g_0195_B16k_light_margin_blend_n2_tau_learn0p5_seed1'
DST = os.path.join(HERE, NAME)
TAU0 = 0.5

w1 = [1.0 / (1.0 + math.exp(2.0 * m / TAU0)) for m in MEASURED_TAU_G0193]

NOTE = (
    "TOP-2 BLENDED READ-OUT with LEARNABLE tau, flat init 0.5. Sibling of exp_g_0194 "
    "(same blend, tau FROZEN at the measured per-layer Delta_m 0.03309-0.10786). Forked "
    "from exp_g_0193_B16k_light_margin_tph128_noznorm_seed1 (margin, no z_norm, nap8/K256, "
    "tph128, H4, d_in=d_out=48, 6L d384 6 heads seq512, device_batch 12 x grad_accum 4 = "
    "24,576 tokens, lr 3e-4, wd 0.1, warmup 0.1, seed 1, lut_base_seed 1000, "
    "tables_no_decay, canonical_full_coverage, untied unembedder, corrected eval bs48x100 "
    "skip-12, final 1.172852). CHANGES, exactly three: lut_read_top_n 1->2, lut_read_tau "
    "->0.5 (flat scalar, NOT 'auto' and NOT the per-layer measured table), "
    "lut_read_tau_learnable ->true. lut_read_tau_measured is deliberately ABSENT: tau here "
    "is neither auto-resolved nor frozen. "
    "THE QUESTION: does tau migrate toward the measured Delta_m (0.033-0.108) from an init "
    "5-15x above them, or somewhere else? exp(log_tau) per layer is logged to metrics.csv "
    "at every eval step so the trajectory can be plotted. "
    "SOFTNESS AT INIT: with tau=0.5 and margins 0.033-0.108, 2m/tau = 0.13-0.43, so the "
    "neighbour weight w1 = sigmoid(-2m/tau) starts at " +
    "/".join(f"{v:.4f}" for v in w1) +
    " by layer -- near-even blending, a much softer start than exp_g_0194 (whose matched "
    "tau gives w1 = sigmoid(-1) = 0.2689 at every layer by construction). "
    "PARAM COUNT: 67,351,686 = exp_g_0193's 67,351,680 + 6, because a learnable tau is an "
    "nn.Parameter per layer rather than a buffer. THE COMPARISON AGAINST exp_g_0193 AND "
    "exp_g_0194 IS THEREFORE NOT PARAM-MATCHED -- 6 params in 67M is not a capacity story, "
    "but it is not exactly matched either and should not be quoted as if it were. "
    "tau sits in the NO-DECAY optimiser group (it is 0-dim, so p.ndim < 2, and it is also a "
    "direct parameter of LightMultiHeadLUT which lut_tables_no_decay exempts) -- verified, "
    "not assumed; decaying a temperature toward zero would be a silent bug. "
    "REFERENCES (corrected protocol): exp_g_0193 1.172852 (the shared control); exp_g_0194 "
    "(frozen tau, running); vanilla@16K seed1 1.165147 / seed2 1.161798, spread 0.00335. "
    "Also carries the step-tagged checkpoint every 4,000 steps (exp_g_0191 died at "
    "15,100/16,000 here on an Xid 8 watchdog timeout)."
)

os.makedirs(DST, exist_ok=True)
cfg = json.load(open(os.path.join(SRC, 'config.json')))
before = dict(cfg)

cfg['lut_read_top_n'] = 2
cfg['lut_read_tau'] = TAU0
cfg['lut_read_tau_learnable'] = True
cfg.pop('lut_read_tau_measured', None)      # must NOT be present
cfg['exp_name'] = NAME
cfg['_arch_note'] = NOTE

json.dump(cfg, open(os.path.join(DST, 'config.json'), 'w'), indent=2)

META = {'exp_name', '_arch_note'}
EXPECTED = {'lut_read_top_n', 'lut_read_tau', 'lut_read_tau_learnable'}
keys = sorted(set(before) | set(cfg))
diffs = [(k, before.get(k, '<absent>'), cfg.get(k, '<absent>'))
         for k in keys if before.get(k) != cfg.get(k)]
sub = [d for d in diffs if d[0] not in META]
print(f'source : {os.path.basename(SRC)}')
print(f'new    : {NAME}\n')
print('FULL key-by-key diff (excluding the two metadata fields):')
for k, a, b in sub:
    print(f'  {k}: {a!r}  ->  {b!r}')
got = {d[0] for d in sub}
print(f'\n  substantive keys differing: {len(sub)}   expected {sorted(EXPECTED)}')
if got != EXPECTED:
    raise SystemExit(f'STOP: diff mismatch. extra={got - EXPECTED} missing={EXPECTED - got}')
print('  OK - exactly three')
print(f'  lut_read_tau_measured present? {"lut_read_tau_measured" in cfg}  (must be False)')
assert 'lut_read_tau_measured' not in cfg
print(f'  keys compared {len(keys)}   identical {len(keys) - len(diffs)}')

print(f'\nINIT SOFTNESS at tau={TAU0}, from the measured Delta_m table:')
print(f'  {"layer":>5} {"Delta_m":>9} {"2m/tau":>8} {"w1 = sigmoid(-2m/tau)":>23}')
for i, (m, v) in enumerate(zip(MEASURED_TAU_G0193, w1)):
    print(f'  {i:>5} {m:>9.5f} {2 * m / TAU0:>8.4f} {v:>23.4f}')
print(f'  exp_g_0194 for contrast: w1 = sigmoid(-1) = {1/(1+math.e):.4f} at EVERY layer '
      f'(matched tau makes 2m/tau = 1 by construction)')

fork_trainer(os.path.join(HERE, '..', 'train_fixed.py'), os.path.join(DST, 'train.py'))
print(f'\ntrainer forked to {NAME}/train.py via the guarded fork_trainer')
