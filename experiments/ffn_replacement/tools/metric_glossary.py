"""THE metric glossary for the ffn_replacement trainers: what every key we send to wandb measures.

One entry per logged key (or per-layer pattern with {i}), with a unit and a description written from the code
that computes it, not from its name. It is surfaced three ways (claude/wandb.md):
  * a project report "Metric glossary"          -- wandb_glossary.py publish
  * a per-run `metric_glossary` artifact          -- Tracker.start (log_artifact only, never run history)
  * a link to both in every run's notes           -- wandb_tracking.run_notes

Drift is detectable, never fatal: the tracker flags a logged key with no entry (tag glossary:undocumented,
summary glossary/undocumented), `wandb_glossary.py audit` lists undocumented and stale entries, and
test_metric_glossary.py checks every metrics.csv header in the repo and the tracker's fixed keys.

EDIT THIS FILE when a trainer starts logging a new key or changes how one is computed; then re-run
`wandb_glossary.py publish`. Pure data + helpers: no wandb, no torch.
"""
import hashlib
import json
import re

# section -> order in the report
SECTIONS = ('train step', 'eval', 'eval, per layer', 'summary', 'metrics.csv')

METRICS = {
    # ---- logged by Tracker.train_step at step 1 and every 10th optimiser step --------------------------------
    'train/loss': dict(section='train step', unit='nats/token', desc=(
        'Training cross-entropy of ONE optimiser step: the mean over the grad_accum micro-batches of '
        'F.cross_entropy(logits, targets, ignore_index=-1) with mean reduction over every target position of the '
        'micro-batch (device_batch_size x seq_len). The bos_bestfit loader never pads, so every position counts, '
        'BOS/special-token targets included. It does NOT include any regulariser (e.g. the lut_cell_smoothness TV '
        'term). The value of that single step, logged at step 1 and every 10th step -- not an average over the 10.')),
    'train/loss_ema': dict(section='train step', unit='nats/token', desc=(
        'Exponential moving average of train/loss with decay 0.99, updated at EVERY optimiser step although logged '
        'only at step 1 and every 10th step. Seeded with the step-1 loss, no bias correction (so it lags early in '
        'training). Equal to train_loss at eval steps.')),
    'train/lr': dict(section='train step', unit='learning rate', desc=(
        'Learning rate applied at this step, identical for the decay and no-decay AdamW groups: lr_scale x config lr. '
        'lr_scale = step / w for step < w = int(lr_warmup_fraction x n_steps) (linear warmup from lr/w), then '
        '0.1 + 0.9 x 0.5 x (1 + cos(pi x (step - w) / (n_steps - w))), a cosine down to 0.1 x lr at the last step.')),
    'time/sec_per_step': dict(section='train step', unit='s/step', desc=(
        'Wall-clock seconds per optimiser step over the window since the previous logged row (normally 10 steps): '
        '(now - time of the previous logged call) / steps elapsed. The window right after an eval or a checkpoint save '
        '(e.g. steps 501-510 after the step-500 eval) includes that eval/save time, so it spikes there. The step-1 '
        'value covers everything from Tracker.start to the end of step 1 (first forward, compilation). Measured by the '
        'tracker; not in metrics.csv.')),
    'train/grad_norm': dict(section='train step', unit='L2 norm', desc=(
        'Global L2 norm of all parameter gradients BEFORE clipping: the return value of '
        'torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0), taken after the micro-batch backwards (each of '
        'loss / grad_accum) and after any regulariser backward the trainer runs before clipping -- in trainers with '
        'lut_cell_smoothness > 0 it therefore INCLUDES the TV-penalty gradient. Clipping then multiplies every '
        'gradient by min(1, 1 / (norm + 1e-6)). Logged at step 1 and every 10th step, only by trainers that pass '
        'it: the train_fixed.py template and run folders forked from it after 2026-09-12 (exp_g_0249 and earlier '
        'runs do not have it).')),
    # ---- logged by Tracker.eval_step at every eval (every eval_every steps and the last step) ----------------
    'val_bpb': dict(section='eval', unit='bits/byte', desc=(
        'Validation bits per byte, from tools/fixed_eval.evaluate_bpb_fixed -- the same function '
        'score_checkpoint.py uses. model.eval() and no_grad. Data: the val split read from token 0 with the same '
        'bos_bestfit packing, 100 batches x 48 rows x seq_len tokens, independent of the training '
        'device_batch_size; the first 12 rows of that stream are dropped, leaving 4,788 scored rows. '
        'bpb = (sum of per-token cross-entropy in nats over target tokens whose UTF-8 byte length is > 0, so '
        'BOS/special tokens are excluded) / (ln 2 x total byte length of those targets). It is byte-weighted, not '
        'a per-token mean, and covers different tokens than train/loss, so it is NOT a unit conversion of the '
        'training loss. An optional fixed_eval config sub-dict can override batch size, batches and skipped rows; '
        'the legacy top-level eval_steps is ignored.')),
    'train_loss': dict(section='eval', unit='nats/token', desc=(
        'train/loss_ema at the eval step (the metrics.csv column name, kept for backward compatibility). '
        'Backfilled runs have only this training series, sampled at eval steps (no train/* or time/* keys).')),
    'ln2_norm_L{i}': dict(section='eval, per layer', unit='L2 norm', desc=(
        "L2 norm of the weight (gain) vector of block i's ln2 LayerNorm, the one in front of the FFN/LUT branch "
        '(n_embd entries; bias not included). Read off the parameters at each eval.')),
    'ln2_mean_L{i}': dict(section='eval, per layer', unit='gain', desc=(
        "Mean of block i's ln2 LayerNorm gain vector (a gain can drift or change sign pattern at constant norm).")),
    'ln1_norm_L{i}': dict(section='eval, per layer', unit='L2 norm', desc=(
        "L2 norm of the weight (gain) vector of block i's ln1 LayerNorm, the one in front of attention (bias not "
        'included).')),
    'lm_g_L{i}': dict(section='eval, per layer', unit='log-gain', desc=(
        'learned_margin confidence parameter g of LUT layer i, raw value. i counts the learned_margin '
        'LightMultiHeadLUT modules in module order (= block index when every block has one). Score over the nap '
        'margins m_j = |d_j|: s = exp(g) x sum_j m_j x prod_j sigmoid(beta m_j) ^ gamma. With '
        'lut_learned_margin_freeze_g, g is a buffer held at its init (0.0) and this series is constant. Only '
        'learned_margin runs.')),
    'lm_beta_L{i}': dict(section='eval, per layer', unit='1/margin', desc=(
        'beta = exp(confidence_log_beta) of learned_margin LUT layer i (init 2.0; g = 0, beta = 2, gamma = 1 is '
        'exactly the margin form).')),
    'lm_gamma_L{i}': dict(section='eval, per layer', unit='exponent', desc=(
        'gamma = exp(confidence_log_gamma) of learned_margin LUT layer i (init 1.0). Unrelated to the top-level '
        'config key gamma.')),
    'lut_tv_L{i}': dict(section='eval, per layer', unit='squared table-value distance', desc=(
        "Hamming-1 cell total variation of block i's LightMultiHeadLUT tables (LightMultiHeadLUT.cell_tv). The "
        '2^nap cells of each table are the corners of an nap-cube; for each of the nap axes take the squared '
        "difference ||v_c - v_c'||^2 between the two cells across that axis, SUMMED over the D value dimensions "
        '(not averaged per dimension), add them all up, and divide by n_tables x nap x 2^(nap-1) -- the number of '
        '(table, Hamming-1 pair) combinations. So: the per-(table, pair) mean of the full-vector squared '
        "difference. Measured at the eval step, no_grad, on the tables after that step's optimiser update. It scales "
        'with the magnitude of the tables: it falls when the tables shrink even if neighbouring cells are no more '
        'alike (compare with the mean ||v_c||^2).')),
    'lut_tv': dict(section='eval', unit='squared table-value distance', desc=(
        'Unweighted mean of lut_tv_L{i} over the LUT layers = model.lut_tv_penalty(), exactly the quantity the '
        'trainer multiplies by lut_cell_smoothness and backpropagates once per optimiser step before clipping. The '
        "logged value is measured at eval time after the update; the penalty applied in training is computed before "
        "each step's update. Never included in train/loss. TV-capable trainers log it even when lut_cell_smoothness "
        'is 0.')),
    'tau_L{i}': dict(section='eval, per layer', unit='temperature', desc=(
        'Read-out blend temperature read_tau of LightMultiHeadLUT layer i for the top-n blended read-out '
        '(lut_read_top_n > 1), learnable in exp_g_0195; read at each eval. Only in runs that log it.')),
    # ---- run.summary, set by Tracker.finish from summary.json (scalars only) --------------------------------
    'final_val_bpb': dict(section='summary', unit='bits/byte', desc=(
        "val_bpb of the last eval (at n_steps) -- the run's reported number.")),
    'best_val_bpb': dict(section='summary', unit='bits/byte', desc=(
        'Minimum val_bpb over all in-run evals, the last one included. It is selected on the same validation window '
        'it is measured on, so it is optimistically biased; report final_val_bpb.')),
    'total_params': dict(section='summary', unit='count', desc=(
        'Sum of numel over model.parameters(): every nn.Parameter, embeddings and the unembedding head included; '
        'buffers (e.g. a frozen learned_margin g) excluded.')),
    'training_time_hours': dict(section='summary', unit='h', desc=(
        'Wall-clock hours of the training loop only, rounded to 0.001: from just before step 1 to after the last '
        "step's eval, so it includes every eval and in-loop checkpoint save and excludes model build, data-loader "
        'start-up, the final plot and the final checkpoint write.')),
    'exp_name': dict(section='summary', unit='-', desc=(
        "The run folder's config exp_name; also the wandb run name and id.")),
    'glossary/undocumented': dict(section='summary', unit='-', desc=(
        'Comma-separated keys this run logged that had no glossary entry at the time (set by the tracker drift '
        'check). Absent when every key was documented.')),
    # ---- metrics.csv columns that are not wandb keys --------------------------------------------------------
    'step': dict(section='metrics.csv', unit='step', desc=(
        'metrics.csv only: the optimiser step of the row (1-based). In wandb it is the x-axis (_step), not a key.')),
}

# Config keys whose meaning is not what the name suggests. Shown in the report under the metrics.
CONFIG_NOTES = {
    'eval_steps': (
        "LEGACY and IGNORED: the pre-fix trainer's number of val batches at device_batch_size. tools/fixed_eval never "
        'reads it -- the eval is 100 batches x 48 rows unless fixed_eval overrides it. The tracker sends it as '
        'eval_steps_legacy_ignored since 2026-09-12; runs uploaded before that show it as eval_steps.'),
    'eval_steps_legacy_ignored': 'The legacy eval_steps renamed so it cannot be mistaken for the eval protocol (see eval_steps).',
    'fixed_eval': 'Optional override {eval_batch_size, eval_steps, skip_rows} of the val window; null or absent = 48 x 100, skip 12.',
    'commit': (
        'Live runs: short sha of the checkout HEAD at launch (config and train.py are committed before launch; the '
        "artefacts land in a later commit). Backfilled runs: the commit that recorded the run's metrics.csv. On "
        "backfilled runs wandb's own Git state shows the HEAD at backfill time instead -- use the notes' links."),
    'commit_dirty': 'True if tracked files differed from HEAD at launch (git status --porcelain --untracked-files=no); untracked files are ignored.',
    'host': (
        "socket.gethostname() with a leading 'pasta-' removed (the name seen inside `sbox --net tailnet`'s network "
        'namespace), so caged and uncaged runs share a host; backfilled runs use the recorded machine. exp_g_0249 '
        'predates this and shows pasta-gpustar.'),
    'grad_accum': 'total_batch_size // (device_batch_size x seq_len).',
    'batch_rows_per_step': 'device_batch_size x grad_accum rows per optimiser step.',
    'tokens_per_step': 'device_batch_size x grad_accum x seq_len tokens per optimiser step.',
    'lut_cell_smoothness': 'TV weight: the trainer backpropagates lut_cell_smoothness x lut_tv_penalty() once per optimiser step before clipping; 0 or absent = off.',
    'lut_learned_margin_freeze_g': 'True: the learned_margin g is a buffer held at its init, so lm_g_L{i} is constant.',
    'gamma': 'Top-level model_build switch (parallel Linear when gamma == 1). Unrelated to the learned_margin exponent lm_gamma_L{i}.',
    'description': "Optional 2-4 sentence run description. It goes into the run's notes, not into the wandb config.",
    'backfilled': 'True for runs uploaded after the fact from committed metrics.csv / summary.json: no train/*, time/* or system series.',
}

_PATTERNS = [(re.compile('^' + re.escape(k).replace(re.escape('{i}'), r'(\d+)') + '$'), k)
             for k in METRICS if '{i}' in k]


def entry_key(key):
    """The glossary entry (exact key or {i} pattern) that documents `key`, or None."""
    if key in METRICS:
        return key
    for rx, k in _PATTERNS:
        if rx.match(key):
            return k
    return None


def lookup(key):
    k = entry_key(key)
    return None if k is None else METRICS[k]


def is_documented(key):
    """wandb's own keys (leading underscore, system/*) are not ours to document."""
    return key.startswith('_') or key.startswith('system/') or entry_key(key) is not None


def undocumented(keys):
    return sorted({k for k in keys if not is_documented(k)})


def stale(seen_keys):
    """Entries that match none of `seen_keys`."""
    hit = {entry_key(k) for k in seen_keys}
    return sorted(k for k in METRICS if k not in hit)


def glossary_hash():
    """12 hex chars of the glossary content: changes exactly when an entry or config note changes."""
    blob = json.dumps({'metrics': METRICS, 'config_notes': CONFIG_NOTES}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def table_rows():
    """[key, description, unit] rows for the metric_glossary artifact table (metrics, then config notes)."""
    rows = [[k, v['desc'], v['unit']] for k, v in sorted(METRICS.items(), key=lambda kv: (SECTIONS.index(kv[1]['section']), kv[0]))]
    rows += [[f'config: {k}', v, '-'] for k, v in CONFIG_NOTES.items()]
    return rows


_MD_CELL = re.compile(r'([\\`*_\[\]|])')                # not < >: the report renderer shows "\<" literally


def _cell(s):
    """Literal text inside a markdown table cell (underscores in names like clip_grad_norm_ would italicise)."""
    return _MD_CELL.sub(r'\\\1', s).replace('\n', ' ')


def report_markdown(source_path, commit=None):
    """The report body: one table per section, then the config notes."""
    out = [f'Generated from `{source_path}`' + (f' at commit `{commit}`' if commit else '')
           + f' -- glossary hash `{glossary_hash()}`. Every run that logs these keys carries a `metric_glossary` '
           'artifact with the same hash, and its notes link here. Per-layer keys: `{i}` is the layer index.', '']
    for sec in SECTIONS:
        items = sorted((k, v) for k, v in METRICS.items() if v['section'] == sec)
        out += [f'## {sec}', '', '| key | unit | what it measures |', '|---|---|---|']
        out += [f'| `{k}` | {_cell(v["unit"])} | {_cell(v["desc"])} |' for k, v in items]
        out.append('')
    out += ['## config keys that are not what they look like', '', '| config key | note |', '|---|---|']
    out += [f'| `{k}` | {_cell(v)} |' for k, v in CONFIG_NOTES.items()]
    return '\n'.join(out) + '\n'
