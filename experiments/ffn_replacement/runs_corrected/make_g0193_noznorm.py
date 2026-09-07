"""Build exp_g_0193: exp_n_0192 with lut_z_norm flipped to false, and NOTHING else.

THE QUESTION. Section 5 of LIGHTMHL_SURVEY.md credits the `margin` confidence form with
-0.026855 bpb, 8x the seed spread and the largest single lever on this line. But that number
is the difference between exp_g_0190 (bounded_norm + z_norm) and exp_n_0192 (margin +
z_norm) -- BOTH with z_norm on. `margin` has never been run without it. Section II.6(5)
flags this as the one place where the mechanism behind this line's only real result is
inferred rather than observed.

This run is the missing arm. It forks exp_n_0192's config LITERALLY -- read from disk,
one key changed -- so the comparison is a single variable.

WHAT CHANGES IN THE MODEL. compression_mhl.py:168 builds `nn.LayerNorm(eff_in)` only when
z_norm is true and stores None otherwise, and the forward at :302-304 / :311-312 is guarded
by `if self.z_norm is not None`. So false removes the module AND skips the call. The
parameter count drops by 6 layers x 2 x 48 = 576, from 67,352,256 to 67,351,680 -- the same
count as exp_g_0189 / exp_n_0185, the other two no-z_norm runs. That drop is an objective
check that the flag actually bit, independent of any bpb.

PERIODIC CHECKPOINTS. exp_g_0191 died on this box at step 15,100/16,000 (Xid 8,
cudaErrorLaunchTimeout from the RC watchdog -- the 5090 also drives the desktop) and the
stock trainer only writes checkpoint.pt AFTER the final step, so that run left nothing at
all. This fork adds a save every 4,000 steps to a step-tagged file. It cannot touch the
training math: `torch.save(model.state_dict(), ...)` reads parameters, consumes no RNG,
allocates no graph and never touches the optimiser, the data loader or the eval path. Its
only measurable effect is on the recorded wall clock (3 extra saves of ~270 MB, ~2 s each,
~0.2% of a 1-hour run), noted here so `training_time_hours` is not over-read.
"""
import json
import os
import sys
import shutil

# --- guard: a forked trainer must use the shared corrected eval ---------------------------
# runs_corrected/ still contains one legacy trainer (exp_n_0138, deliberately) whose eval was
# coupled to the training batch size. fork_trainer() refuses to fork any such file, so a new
# run cannot inherit that bug. It fires ONLY at fork time, on the source file -- merely
# having a legacy trainer on disk is fine.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from fork_trainer import fork_trainer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'exp_n_0192_repro0191_seed1')
NAME = 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1'
DST = os.path.join(HERE, NAME)

NOTE = (
    "MARGIN WITHOUT Z_NORM -- the missing arm of the LookupFFN line. Forked LITERALLY from "
    "exp_n_0192_repro0191_seed1's config.json with exactly ONE substantive key changed: "
    "lut_z_norm true -> false. Everything else is byte-identical to 0192 (seed 1, "
    "lut_base_seed 1000, lr 3e-4, wd 0.1, warmup 0.1, n_steps 16000, device_batch 12 x "
    "grad_accum 4 = 24,576 tokens, 6L d384 6 heads seq512, H4 nap8/K256 tph128, "
    "d_in=d_out=48, lut_impl=light, confidence_form=margin, tables_no_decay=true, "
    "anchor policy canonical_full_coverage (default), untied unembedder, corrected eval "
    "bs48x100 skip-12). "
    "PURPOSE: section 5 of LIGHTMHL_SURVEY.md credits `margin` with -0.026855 bpb, but that "
    "is exp_g_0190 (bounded_norm+z_norm) vs exp_n_0192 (margin+z_norm) -- both with z_norm "
    "ON. margin has never been run without it, so the -0.026855 is really "
    "margin-GIVEN-z_norm. This run deconfounds it. "
    "PREDICTION RECORDED BEFORE THE RESULT (see LIGHTMHL_SURVEY.md section 6): lands in "
    "1.185-1.195, i.e. most of margin's gain survives but a real part of it does not. "
    "REFERENCES (corrected protocol): exp_n_0192 margin+z_norm 1.177081; exp_g_0190 "
    "bounded_norm+z_norm 1.203936; exp_g_0189 bounded_norm no-z_norm 1.207493; exp_n_0185 "
    "bounded_norm no-z_norm decay-on 1.206222; vanilla@16K seed1 1.165147 / seed2 1.161798 "
    "(seed spread 0.00335). "
    "Note z_norm=false removes the LayerNorm module entirely (compression_mhl.py:168) and "
    "skips the call (:302-304), so total_params must read 67,351,680 rather than 0192's "
    "67,352,256 -- a 576-parameter drop that objectively proves the flag bit. "
    "This fork also adds a step-tagged checkpoint every 4,000 steps, because exp_g_0191 "
    "died at 15,100/16,000 on this box (Xid 8 / RC watchdog, the 5090 drives the desktop) "
    "and the stock trainer only saves after the final step. The save reads the state_dict "
    "and touches nothing else -- no RNG, no optimiser, no loader, no eval -- so the training "
    "math and the eval path are unchanged; only training_time_hours is inflated ~0.2%."
)

os.makedirs(DST, exist_ok=True)
cfg = json.load(open(os.path.join(SRC, 'config.json')))
before = dict(cfg)

cfg['lut_z_norm'] = False              # <-- THE ONLY SUBSTANTIVE CHANGE
cfg['exp_name'] = NAME
cfg['_arch_note'] = NOTE

json.dump(cfg, open(os.path.join(DST, 'config.json'), 'w'), indent=2)

# ---- prove exactly one substantive key moved --------------------------------------------
META = {'exp_name', '_arch_note'}
keys = sorted(set(before) | set(cfg))
diffs = [(k, before.get(k, '<absent>'), cfg.get(k, '<absent>'))
         for k in keys if before.get(k) != cfg.get(k)]
print(f'source : {os.path.basename(SRC)}')
print(f'new    : {NAME}\n')
print('FULL key-by-key diff (excluding the two metadata fields):')
sub = [d for d in diffs if d[0] not in META]
for k, a, b in sub:
    print(f'  {k}: {a!r}  ->  {b!r}')
print(f'\n  substantive keys differing: {len(sub)}   '
      f'{"OK - exactly one" if len(sub) == 1 else "*** UNEXPECTED ***"}')
print(f'  metadata keys differing    : {sorted(d[0] for d in diffs if d[0] in META)}')
print(f'  keys compared              : {len(keys)}')
print(f'  keys identical             : {len(keys) - len(diffs)}')

fork_trainer(os.path.join(HERE, '..', 'train_fixed.py'), os.path.join(DST, 'train.py'))
print(f'\ntrainer forked to {NAME}/train.py (periodic checkpointing patched separately)')
