"""Autonomous sweep of (nap, n_sparse) per LUT module, anchored on exp115.

Runs short (3K-step) variants sequentially while user is asleep. After each
stage, decides next stage based on observed deltas vs the exp115 baseline.

Stage 1 — Endpoint scan, 6 runs:
    For each module ∈ {qk, v, out}, vary nap ∈ {6, 10} and adjust n_sparse to
    keep total LUT weight constant; other modules at exp115 baseline.

Stage 2 — Mid-point fill-in (gated by Stage 1 deltas):
    For each module, if the endpoint shows a clear winning direction
    (improvement >= 0.005 bpb at step 3000 vs baseline), run the intermediate
    nap (7 if nap=6 won, 9 if nap=10 won).

Stage 3 — Joint best:
    Combine the best-performing nap per module into a single config.

All variants use exp115's SparseLut wrapper (MultiHeadLut + smooth_mode=True
+ n_alternatives=1 + i.i.d. external SparseScatter), to match exp115 exactly.
The only differences are nap, n_sparse, and (for joint best) the combined
choices.

Outputs:
    sweep115_3k/run_<name>/{config.json, train.py, stdout.log, metrics.csv,
                             summary.json, loss.png}
    sweep115_3k/master.log         — orchestrator log (timing, decisions)
    sweep115_3k/results.json       — collected best_val_bpb per run
    sweep115_3k/SUMMARY.md         — final tabulated report
"""
import json, os, shutil, subprocess, time, csv

ROOT     = '/home/starost/spiky/nanochat_exps/sweep115_3k'
EXP115   = '/home/starost/spiky/nanochat_exps/exp115_na1'
PYBIN    = '/home/starost/spiky/.venv/bin/python'
N_STEPS  = 3000
BASELINE_NAP    = 8
BASELINE_NSP    = 8
# Constant-weight pairs: (nap, n_sparse). With baseline tph fixed, each pair
# yields the same total LUT weight as the baseline (table_dim*n_sparse=2048).
NAP_NSP = {
    6:  32,
    7:  16,
    8:  8,
    9:  4,
    10: 2,
}
MODULES = ['qk', 'v', 'out']

os.makedirs(ROOT, exist_ok=True)
master_log_path = os.path.join(ROOT, 'master.log')
results_path    = os.path.join(ROOT, 'results.json')
summary_path    = os.path.join(ROOT, 'SUMMARY.md')


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(master_log_path, 'a') as f:
        f.write(line + '\n')


def baseline_cfg():
    cfg = json.load(open(os.path.join(EXP115, 'config.json')))
    cfg['n_steps'] = N_STEPS
    return cfg


def overrides_for(module: str, nap: int, n_sparse: int) -> dict:
    """Apply (nap, n_sparse) override to a single module while keeping others
    at baseline. Module is one of: 'qk', 'v', 'out'."""
    if module == 'qk':
        return {'qk_input_nap': nap, 'qk_n_sparse_outputs': n_sparse}
    if module == 'v':
        return {'v_input_nap': nap, 'v_n_sparse_outputs': n_sparse}
    if module == 'out':
        return {'out_input_nap': nap, 'out_n_sparse_outputs': n_sparse}
    raise ValueError(module)


def joint_overrides(per_module_nap: dict) -> dict:
    """Combine winning nap per module into a single override dict."""
    ovr = {}
    for m, nap in per_module_nap.items():
        nsp = NAP_NSP[nap]
        ovr.update(overrides_for(m, nap, nsp))
    return ovr


def run_variant(name: str, overrides: dict) -> float | None:
    folder = os.path.join(ROOT, name)
    if os.path.exists(os.path.join(folder, 'summary.json')):
        # Idempotency: skip if already finished.
        prev = json.load(open(os.path.join(folder, 'summary.json')))
        log(f"  [skip already-done] {name}: best_val_bpb={prev.get('best_val_bpb'):.4f}")
        return prev.get('best_val_bpb')
    os.makedirs(folder, exist_ok=True)
    shutil.copy(os.path.join(EXP115, 'train.py'), os.path.join(folder, 'train.py'))
    cfg = baseline_cfg()
    cfg.update(overrides)
    cfg['exp_name'] = name
    cfg['description'] = (
        f"sweep115_3k variant: overrides={overrides}, n_steps={N_STEPS}. "
        f"Baseline = exp115 (nap=8, n_sparse=8 for all three LUT modules)."
    )
    json.dump(cfg, open(os.path.join(folder, 'config.json'), 'w'), indent=2)
    log_path = os.path.join(folder, 'stdout.log')
    log(f"  [start] {name} overrides={overrides}")
    t0 = time.time()
    with open(log_path, 'w') as logfile:
        rc = subprocess.run(
            [PYBIN, '-u', 'train.py'],
            cwd=folder, stdout=logfile, stderr=subprocess.STDOUT,
        ).returncode
    dt = time.time() - t0
    if rc != 0:
        log(f"  [FAIL] {name} rc={rc} dt={dt:.0f}s — see {log_path}")
        return None
    summary = os.path.join(folder, 'summary.json')
    if not os.path.exists(summary):
        log(f"  [no summary] {name} dt={dt:.0f}s")
        return None
    bpb = json.load(open(summary)).get('best_val_bpb')
    log(f"  [done] {name} best_val_bpb={bpb:.4f} dt={dt:.0f}s")
    return bpb


def baseline_at_step_3k() -> float:
    """exp115's val bpb at step 3000 — read from its metrics.csv."""
    path = os.path.join(EXP115, 'metrics.csv')
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row['step']) == 3000:
                return float(row['val_bpb'])
    return float('nan')


def save_results(results: dict):
    json.dump(results, open(results_path, 'w'), indent=2)


def write_summary(results: dict, baseline_3k: float, decisions: dict):
    lines = ['# sweep115_3k — final tabulated results', '']
    lines.append(f"Baseline exp115 @ step 3000: **{baseline_3k:.4f}**")
    lines.append('')
    lines.append('## Per-run results (best_val_bpb)')
    lines.append('| run | best_val_bpb | Δ vs baseline |')
    lines.append('|---|---|---|')
    for name, bpb in sorted(results.items()):
        if bpb is None:
            lines.append(f'| {name} | FAIL | — |')
            continue
        delta = bpb - baseline_3k
        lines.append(f'| {name} | {bpb:.4f} | {delta:+.4f} |')
    lines.append('')
    lines.append('## Decisions')
    for k, v in decisions.items():
        lines.append(f'- {k}: {v}')
    open(summary_path, 'w').write('\n'.join(lines) + '\n')


# ============================================================================
# Main
# ============================================================================

baseline_3k = baseline_at_step_3k()
log(f"=== sweep115_3k starting === exp115 step-3000 baseline = {baseline_3k:.4f}")

results = {}
decisions = {}

# --- Stage 1: endpoint scan (6 runs)
log("--- Stage 1: endpoint scan ---")
for module in MODULES:
    for nap in (6, 10):
        nsp = NAP_NSP[nap]
        name = f'run_{module}_nap{nap}'
        bpb = run_variant(name, overrides_for(module, nap, nsp))
        results[name] = bpb
        save_results(results)

# --- Stage 2: gated mid-point fill-in
log("--- Stage 2: mid-point fill-in (gated) ---")
THRESH = 0.005  # require this improvement vs baseline_3k to fill in
# For each module, compare endpoint best_val_bpb to baseline_3k. Note that
# our runs report best_val_bpb (over the whole 3K run), while baseline_3k is
# the bpb AT step 3000 of exp115 (which is best for monotone curves anyway).
per_module_winner = {}
for module in MODULES:
    n6  = results.get(f'run_{module}_nap6')
    n10 = results.get(f'run_{module}_nap10')
    candidates = {6: n6, 8: baseline_3k, 10: n10}
    # Drop None
    candidates = {k: v for k, v in candidates.items() if v is not None}
    best_nap = min(candidates, key=candidates.get)
    per_module_winner[module] = best_nap
    decisions[f'stage1_{module}_winner_endpoint'] = (
        f'nap={best_nap} (n6={n6}, base={baseline_3k:.4f}, n10={n10})'
    )
    # Fill in mid-point if endpoint clearly beats baseline.
    if best_nap == 6 and (baseline_3k - n6) >= THRESH:
        name = f'run_{module}_nap7'
        bpb = run_variant(name, overrides_for(module, 7, NAP_NSP[7]))
        results[name] = bpb
        save_results(results)
        if bpb is not None and bpb < candidates[6]:
            per_module_winner[module] = 7
            candidates[7] = bpb
        decisions[f'stage2_{module}_nap7'] = bpb
    elif best_nap == 10 and (baseline_3k - n10) >= THRESH:
        name = f'run_{module}_nap9'
        bpb = run_variant(name, overrides_for(module, 9, NAP_NSP[9]))
        results[name] = bpb
        save_results(results)
        if bpb is not None and bpb < candidates[10]:
            per_module_winner[module] = 9
            candidates[10] = bpb
        decisions[f'stage2_{module}_nap9'] = bpb

# --- Stage 3: joint best
log("--- Stage 3: joint best ---")
log(f"  per_module_winner = {per_module_winner}")
decisions['joint_per_module'] = per_module_winner
# Only run joint if at least one module deviates from baseline.
if any(nap != 8 for nap in per_module_winner.values()):
    name = 'run_joint_best'
    bpb = run_variant(name, joint_overrides(per_module_winner))
    results[name] = bpb
    save_results(results)
else:
    decisions['stage3'] = 'skipped — baseline (nap=8) wins all modules'

write_summary(results, baseline_3k, decisions)
log("=== sweep115_3k DONE ===")
log(f"Summary written to {summary_path}")
