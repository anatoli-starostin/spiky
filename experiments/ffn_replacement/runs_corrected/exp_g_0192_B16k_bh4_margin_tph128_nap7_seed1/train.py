"""ffn_replacement trainer with the FIXED validation protocol (issue: eval decoupled from
training batch size). Fork this into a run folder as `train.py`; it reads `config.json`
from its own directory and writes metrics.csv / summary.json / loss.png / checkpoint.pt
beside it — same as the historical trainer.

WHAT CHANGED vs the historical runs/*/train.py (and ONLY this):
  The val curve is now measured by `tools/fixed_eval.evaluate_bpb_fixed`, which ALWAYS uses
  batch size 48 x 100 eval steps with the leading 12 rows dropped — independent of the
  training `device_batch_size`. The old code built the val loader at device_batch_size, so
  the number of val tokens scored scaled with the training batch (the protocol confound).
  Final scoring (`tools/score_checkpoint.py`) uses the SAME function, so the training-time
  curve and the final number are the identical measurement.

Everything else — model, LR schedule, optimizer grouping, seeds, outputs — is unchanged, so
configs and checkpoints remain compatible with the historical runs.
"""
import csv
import json
import math
import os
import sys
import time

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _find_tools():
    """Locate experiments/ffn_replacement/tools/ whether this file sits at the experiment
    root or has been copied into runs/<exp>/ as train.py."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, 'tools', 'fixed_eval.py')):
            return os.path.join(d, 'tools')
        d = os.path.dirname(d)
    raise RuntimeError('could not locate ffn_replacement/tools/ (fixed_eval.py)')


sys.path.insert(0, _find_tools())
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
from spiky.lutorch.bh4_multi_head_lut import BH4MultiHeadLUT

from model_build import build_model                       # shared config-driven model
from fixed_eval import evaluate_bpb_fixed, eval_config    # THE fixed eval set

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

SEQ_LEN = cfg['seq_len']
DEVICE_BS, TOTAL_BS, N_STEPS = cfg['device_batch_size'], cfg['total_batch_size'], cfg['n_steps']
LR, WD, WARMUP_FRAC = cfg['lr'], cfg['weight_decay'], cfg['lr_warmup_fraction']
EVAL_EVERY = cfg['eval_every']
EVAL = eval_config(cfg)   # fixed eval: bs48 x100 skip12 (NOT a function of DEVICE_BS)

BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
assert VOCAB_SIZE == cfg['tokenizer_vocab_size']
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='train', device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE)
print(f"FIXED EVAL: bs{EVAL['eval_batch_size']} x {EVAL['eval_steps']} steps, skip {EVAL['skip_rows']} rows "
      f"(val window independent of device_batch_size={DEVICE_BS})")


def setup_optimizer(model, lr, weight_decay, tables_no_decay=False):
    # LUT table parameters are exempt from weight decay. That exemption was written when
    # FastMultiHeadLut was the only implementation, and it matches by CLASS -- so
    # LightMultiHeadLUT's `tables` (3-D, hence not caught by the ndim<2 rule either) has been
    # falling into the DECAY group while Fast's identical tables were exempt. Light was
    # therefore training 37.7-75.5M table parameters at weight_decay=0.1 where Fast trained
    # them at 0.0, an unintended asymmetry in every Light-vs-Fast comparison.
    #
    # Fixing it silently would change what every existing Light config does, so it is opt-in
    # via `lut_tables_no_decay` (default False = today's behaviour, bit-identical grouping).
    # BH4MultiHeadLUT joins the exemption for exactly the same reason Light did: its
    # tables are 3-D, so the ndim<2 rule does not catch them, and without a class match
    # they would be decayed at weight_decay while Fast's and Light's identical tables
    # are exempt -- the asymmetry that confounded every Light-vs-Fast comparison until
    # exp_g_0189. Its bh4.blocks are deliberately NOT exempt: they replace compress,
    # which has always been decayed.
    exempt = ((FastMultiHeadLut, LightMultiHeadLUT, BH4MultiHeadLUT)
              if tables_no_decay else (FastMultiHeadLut,))
    lut_ids = {id(p) for m in model.modules() if isinstance(m, exempt)
               for p in m.parameters(recurse=False)}
    decay, nodecay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (nodecay if (id(p) in lut_ids or p.ndim < 2) else decay).append(p)
    groups = [dict(params=decay, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay),
              dict(params=nodecay, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)]
    opt = torch.optim.AdamW(groups)
    for g in opt.param_groups:
        g['initial_lr'] = g['lr']
    return opt


def get_lr_scale(step, n_steps, warmup_frac):
    w = int(warmup_frac * n_steps)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n_steps - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


model = build_model(cfg, VOCAB_SIZE, device=DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"MinimalGPT depth={cfg['depth']} dim={cfg['n_embd']} heads={cfg['n_head']} seq={SEQ_LEN} "
      f"| ffn={cfg.get('ffn_type')} tie={bool(cfg.get('tie_unembedder', False))} | params={total_params:,}")

if os.environ.get('SMOKE'):
    print('SMOKE OK'); sys.exit(0)

optimizer = setup_optimizer(model, lr=LR, weight_decay=WD,
                            tables_no_decay=cfg.get("lut_tables_no_decay", False))
tokens_per_step = DEVICE_BS * SEQ_LEN
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'Tokens/micro-batch: {tokens_per_step:,} | grad_accum: {grad_accum} | effective batch: {grad_accum * tokens_per_step:,} tokens')


# --- per-layer LayerNorm health, logged alongside the val curve -------------------------
# Why: exp_n_0184's layer-0 ln2 gain was found collapsed to ~0 (mean 0.000000, norm 0.00386)
# against Fast's healthy 0.894887 / 17.53908, and with only a FINAL checkpoint saved there
# was no way to tell whether it never bootstrapped or was alive and died. These columns make
# that answerable from the run's own metrics.csv.
#
# STRICTLY ADDITIVE: 'step', 'train_loss', 'val_bpb' keep their names, order and meaning, so
# anything reading an older metrics.csv still works; the new columns simply follow them.
# The gain MEAN is logged as well as the norm, because a gain can drift or change sign
# pattern while keeping its norm, and mean is the quantity compared across runs.
N_LAYERS = len(model.blocks)
LN_COLS = ([f'ln2_norm_L{i}' for i in range(N_LAYERS)]
           + [f'ln2_mean_L{i}' for i in range(N_LAYERS)]
           + [f'ln1_norm_L{i}' for i in range(N_LAYERS)])


@torch.no_grad()
def ln_stats():
    """Read-only scalars off the parameters. No RNG, no graph, no optimiser interaction."""
    out = [b.ln2.weight.norm().item() for b in model.blocks]
    out += [b.ln2.weight.mean().item() for b in model.blocks]
    out += [b.ln1.weight.norm().item() for b in model.blocks]
    return [f'{v:.6f}' for v in out]


csv_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f); csv_w.writerow(['step', 'train_loss', 'val_bpb'] + LN_COLS)
train_losses_logged, val_bpbs, val_steps = [], [], []
ema, best_bpb, t0 = None, float('inf'), time.time()


def eval_now():
    model.eval()
    bpb = evaluate_bpb_fixed(model, tokenizer, token_bytes, SEQ_LEN, DEVICE, **EVAL)
    model.train()
    return bpb


model.train()
for step in range(1, N_STEPS + 1):
    lr_scale = get_lr_scale(step, N_STEPS, WARMUP_FRAC)
    for g in optimizer.param_groups:
        g['lr'] = g['initial_lr'] * lr_scale
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    ema = accum_loss if ema is None else 0.99 * ema + 0.01 * accum_loss
    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | loss={ema:.4f} | lr={lr_scale * LR:.2e}')
    if step % EVAL_EVERY == 0 or step == N_STEPS:
        bpb = eval_now()
        best_bpb = min(best_bpb, bpb)
        print(f'[VAL] step {step}: bpb={bpb:.4f}')
        train_losses_logged.append(ema); val_bpbs.append(bpb); val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}'] + ln_stats()); csv_f.flush()

csv_f.close()
elapsed = time.time() - t0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, train_losses_logged, label='train (ema)'); ax1.set(xlabel='step', ylabel='ce loss', title='Training Loss'); ax1.legend(); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, 'o-', color='tab:orange', label='val bpb'); ax2.set(xlabel='step', ylabel='bpb', title='Validation BPB (fixed protocol)'); ax2.legend(); ax2.grid(True)
plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120); plt.close()

summary = {'exp_name': cfg['exp_name'], 'best_val_bpb': best_bpb,
           'final_val_bpb': val_bpbs[-1] if val_bpbs else None,
           'total_params': total_params, 'training_time_hours': round(elapsed / 3600, 3),
           'eval_protocol': {'eval_batch_size': EVAL['eval_batch_size'], 'eval_steps': EVAL['eval_steps'],
                             'skip_rows': EVAL['skip_rows'], 'batch_size_independent': True}}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ==='); print(json.dumps(summary, indent=2))
