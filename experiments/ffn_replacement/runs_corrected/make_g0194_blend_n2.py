"""Build exp_g_0194: exp_g_0193 + the top-2 blended read-out, tau frozen at measured Delta_m.

THE FIRST TRAINING RUN WITH A DIRECTIONAL ROUTING GRADIENT. Everything measured for the
blend so far has been eval-only, on checkpoints trained single-winner: the top-2 read-out
recovered -0.006 (16K) to -0.011 (48K) bpb for free, and the random-cell control HURT
(+0.002/+0.005), so the gain is specific to the cell across the nearest decision boundary.
But that only shows a post-hoc read-out change helps a model TRAINED hard. Training with
the blend is a different claim, because the blend weights are differentiable in the margins
and dw/dm compares the table rows of ALTERNATIVE cells -- a directional routing gradient
plain Light does not have at all. That signal has never been exercised.

CONFIG DIFF vs exp_g_0193 -- three behavioural keys:
    lut_read_top_n         1     -> 2
    lut_read_tau           (absent, default 0.1) -> "auto"
    lut_read_tau_learnable (absent, default False) -> False   (explicit, frozen)

plus ONE provenance key, lut_read_tau_measured, which pins what "auto" resolves to so the
run does not depend on a table in model_build.py that could later change. Gate 1 of the
launch order asks for the resolved values to be recorded in config.json; this is that.

TAU = Delta_m, and the factor is deliberate. The blend-weight sensitivity
|dw/dtau| ~ (2c/tau^2) w(1-w) peaks at tau* = 2c/z* with z* = 2.399379 the maximiser of
z^2 sigmoid(z) sigmoid(-z), i.e. tau* = 0.8335 * Delta_m. So tau = Delta_m sits at 95.6% of
peak; tau = 2*Delta_m would be 44.8%, and the old flat 0.1 is 22.4% at layer 0 (rising to
98.5% at layer 5) -- badly mismatched exactly where the routing deficit was measured.

PERIODIC CHECKPOINTS, as for exp_g_0193 and for the same reason, more so: this run is ~2.6x
slower per step, so ~2.4 h on a 5090 that also drives the desktop and already killed
exp_g_0191 at step 15,100 with an Xid 8 watchdog timeout. The stock trainer saves only after
the final step. torch.save reads the state_dict and touches no RNG, optimiser, loader or
eval path.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from fork_trainer import fork_trainer                                  # noqa: E402
from model_build import MEASURED_TAU_G0193, resolved_read_taus          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1')
NAME = 'exp_g_0194_B16k_light_margin_blend_n2_tau_auto_seed1'
DST = os.path.join(HERE, NAME)

NOTE = (
    "TOP-2 BLENDED READ-OUT, tau FROZEN at the measured per-layer Delta_m. Forked from "
    "exp_g_0193_B16k_light_margin_tph128_noznorm_seed1 (the standard config: margin, no "
    "z_norm, nap8/K256, tph128, H4, d_in=d_out=48, 6L d384 6 heads seq512, device_batch 12 "
    "x grad_accum 4 = 24,576 tokens, lr 3e-4, wd 0.1, warmup 0.1, seed 1, lut_base_seed "
    "1000, tables_no_decay, anchor policy canonical_full_coverage, untied unembedder, "
    "corrected eval bs48x100 skip-12, final 1.172852). CHANGES: lut_read_top_n 1->2, "
    "lut_read_tau ->'auto', lut_read_tau_learnable ->false, plus lut_read_tau_measured "
    "pinning what 'auto' resolves to. "
    "WHAT IS NEW HERE: the blend weights are differentiable in the margins, so dw/dm "
    "compares the table rows of ALTERNATIVE cells and pushes the code toward the better "
    "one -- a DIRECTIONAL ROUTING GRADIENT that plain Light lacks entirely (its only path "
    "to x is the confidence score, which sharpens a margin but never says which cell would "
    "have been better). This makes the layer a sparse top-n cousin of FastMultiHeadLut's "
    "full-2^NAP softmax surrogate at 2 gathers instead of 256. "
    "TAU INIT: per-layer Delta_m = median of m_(1), the smallest of the nap anchor margins, "
    "measured by diag_margin_gap.py on exp_g_0193 over 8,192 real val tokens. Sensitivity "
    "|dw/dtau| ~ (2c/tau^2) w(1-w) peaks at 0.8335*Delta_m, so tau=Delta_m is 95.6% of peak "
    "(2*Delta_m would be 44.8%; the old flat 0.1 is 22.4% at layer 0). FROZEN for this run "
    "-- log_tau is a BUFFER not a Parameter, so total_params is unchanged at 67,351,680 and "
    "the leaderboard comparison against 0193 stays exact. "
    "PRIOR (eval-only, on single-winner checkpoints): top-2 at tau=0.1 gave -0.005788 "
    "(exp_n_0192), -0.006154 (exp_n_0196), -0.009528 (exp_n_0199), -0.010692 (exp_n_0200), "
    "-0.010407 (exp_n_0202); random-cell control HURT (+0.001986 / +0.005183). Training "
    "with the blend is a DIFFERENT claim and may land either side of that. "
    "REFERENCES (corrected protocol): exp_g_0193 1.172852 (the direct control); vanilla@16K "
    "seed1 1.165147 / seed2 1.161798, seed spread 0.00335. "
    "Cost: measured 2.57x fwd+bwd at nap8/tph128 vs n=1, so ~2.4 h expected. "
    "This fork also adds a step-tagged checkpoint every 4,000 steps (exp_g_0191 died at "
    "15,100/16,000 here on an Xid 8 RC watchdog timeout and the stock trainer saves only "
    "after the final step); the save touches no RNG, optimiser, loader or eval path."
)

os.makedirs(DST, exist_ok=True)
cfg = json.load(open(os.path.join(SRC, 'config.json')))
before = dict(cfg)

cfg['lut_read_top_n'] = 2
cfg['lut_read_tau'] = 'auto'
cfg['lut_read_tau_learnable'] = False
cfg['lut_read_tau_measured'] = list(MEASURED_TAU_G0193)   # pins the 'auto' resolution
cfg['exp_name'] = NAME
cfg['_arch_note'] = NOTE

json.dump(cfg, open(os.path.join(DST, 'config.json'), 'w'), indent=2)

# ---- assert the diff is EXACTLY the intended keys -----------------------------------------
META = {'exp_name', '_arch_note'}
EXPECTED = {'lut_read_top_n', 'lut_read_tau', 'lut_read_tau_learnable',
            'lut_read_tau_measured'}
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
print(f'\n  substantive keys differing: {len(sub)}')
print(f'  expected                  : {sorted(EXPECTED)}')
print(f'  {"OK - exactly the intended keys" if got == EXPECTED else "*** UNEXPECTED ***"}')
if got != EXPECTED:
    raise SystemExit(f'STOP: diff mismatch. extra={got - EXPECTED} missing={EXPECTED - got}')
print(f'  metadata keys differing   : {sorted(d[0] for d in diffs if d[0] in META)}')
print(f'  keys compared             : {len(keys)}   identical: {len(keys) - len(diffs)}')

# ---- GATE 1: resolved tau ------------------------------------------------------------------
taus = resolved_read_taus(cfg)
REQUIRED = [0.03309, 0.07241, 0.07801, 0.08158, 0.09190, 0.10786]
print(f'\nGATE 1  resolved per-layer tau: {taus}')
print(f'        required               : {REQUIRED}')
assert [round(t, 5) for t in taus] == REQUIRED, 'STOP: resolved tau != measured Delta_m'
print('        OK - matches the measured Delta_m table, and pinned in config.json')

fork_trainer(os.path.join(HERE, '..', 'train_fixed.py'), os.path.join(DST, 'train.py'))
print(f'\ntrainer forked to {NAME}/train.py via the guarded fork_trainer '
      '(periodic checkpointing patched separately, as for exp_g_0193)')
