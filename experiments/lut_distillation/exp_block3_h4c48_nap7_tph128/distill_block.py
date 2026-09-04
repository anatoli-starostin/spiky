"""Per-block FFN -> CompressionMHL distillation (issue #111, Goal 1 feasibility probe).

Config-driven, in the style of the run dirs' train.py: reads ./config.json, writes
metrics.csv, summary.json, loss.png and (gitignored) student.pt alongside.

    python distill_block.py            # reads ./config.json
    python distill_block.py --smoke 3  # 3 steps into /tmp, for timing/sanity only

What it does: ONE block of the frozen teacher is distilled. A forward hook on that block's
`mlp` captures the FFN sub-layer's (input, output) pair -- the same [B,T,C]->[B,T,C] map
`Block.ffn_slot()` computes -- and a CompressionMHL student is trained to regress it while
every other teacher weight stays frozen. Then the student is swapped in for that one block
and val bpb is measured end-to-end, which is the number that decides the probe.

Two things this deliberately does NOT do:
  * it does not freeze `compress`. Routing is learned jointly with the tables through the
    existing soft surrogate -- so this is normal gradient training, and no closed-form or
    least-squares shortcut applies (that shortcut only exists if routing is frozen).
  * it does not cache a dataset. The frozen teacher is a generator, so every step streams a
    fresh batch and supervision is effectively unlimited.
"""
import csv
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(EXP_DIR)))
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
for p in (os.path.join(REPO, 'experiments', 'ffn_replacement', 'benchmark'),
          os.path.join(REPO, 'src'), NANOCHAT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as M                                                    # noqa: E402
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT     # noqa: E402
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut        # noqa: E402
from nanochat.common import get_base_dir                              # noqa: E402
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes      # noqa: E402
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit  # noqa: E402
from nanochat.loss_eval import evaluate_bpb                           # noqa: E402

with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

SMOKE = 0
if '--smoke' in sys.argv:
    # Sanity/timing only: a few steps and a token-sized eval, written to /tmp so a smoke
    # run can never be mistaken for -- or overwrite -- a real run's outputs.
    SMOKE = int(sys.argv[sys.argv.index('--smoke') + 1])
    cfg['log_every'] = 1
    cfg['probe_batches'] = 2
    cfg['eval_steps'] = 2

OUT_DIR = '/tmp/distill_smoke' if SMOKE else EXP_DIR
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

BLOCK = cfg['block']
DEVICE_BS, ACCUM = cfg['device_batch_size'], cfg['grad_accum']
SEQ_LEN, N_STEPS = cfg['seq_len'], (SMOKE or cfg['n_steps'])
LR, WD = cfg['lr'], cfg['weight_decay']
WARMUP_FRAC, FINAL_FRAC = cfg['lr_warmup_fraction'], cfg['lr_final_fraction']
TEACHER_DIR = os.path.join(REPO, cfg['teacher_exp_dir'])
TOKENS_PER_STEP = DEVICE_BS * SEQ_LEN * ACCUM
assert TOKENS_PER_STEP == cfg['total_batch_size'], (
    f'device_bs*seq*accum = {TOKENS_PER_STEP} != total_batch_size {cfg["total_batch_size"]}')


class EvalAdapter(nn.Module):
    """`evaluate_bpb` wants get_device() and model(x, y, loss_reduction=...) -> (B,T);
    benchmark/model.py's GPT takes idx only and returns logits. Thin shim, no state."""

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt

    def get_device(self):
        return self.gpt.tok_emb.weight.device

    def forward(self, idx, targets=None, loss_reduction='mean'):
        logits = self.gpt(idx)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                               targets.reshape(-1), ignore_index=-1,
                               reduction=loss_reduction)
        return loss.view(targets.shape) if loss_reduction == 'none' else loss


def setup_optimizer(student, lr, weight_decay):
    """Same grouping as the run dirs' train.py: LUT tables, temperatures and every 1-D
    parameter get no weight decay; 2-D projection weights do."""
    lut_ids = {id(p) for m in student.modules() if isinstance(m, FastMultiHeadLut)
               for p in m.parameters(recurse=False)}
    decay, nodecay = [], []
    for p in student.parameters():
        if p.requires_grad:
            (nodecay if (id(p) in lut_ids or p.ndim < 2) else decay).append(p)
    opt = torch.optim.AdamW(
        [dict(params=decay, weight_decay=weight_decay),
         dict(params=nodecay, weight_decay=0.0)],
        lr=lr, betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'])
    return opt, sum(p.numel() for p in decay), sum(p.numel() for p in nodecay)


def lr_scale(step):
    w = max(1, int(WARMUP_FRAC * N_STEPS))
    if step < w:
        return (step + 1) / w
    t = (step - w) / max(1, N_STEPS - w)
    return FINAL_FRAC + (1 - FINAL_FRAC) * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def cell_occupancy(student, xs):
    """Visit counts per (table, cell), recomputing the index exactly as FastMultiHeadLut
    does: anchor differences -> sign bits -> MSB-first bit-pack."""
    lut = student.lut_batched if hasattr(student, 'lut_batched') else student.lut
    a, b, powers = lut.soft_anchor_a_long, lut.soft_anchor_b_long, lut.soft_powers
    R, K = a.shape[0], lut.weights.shape[1]
    counts = torch.zeros(R, K, dtype=torch.long, device=lut.weights.device)
    for x in xs:
        z = student.compress(x)
        idx = (((z[:, a] - z[:, b]) > 0).to(torch.int64) * powers.view(1, 1, -1)).sum(-1)
        it = idx.t().contiguous()
        counts.scatter_add_(1, it, torch.ones_like(it))
    return counts.cpu()


# ---- teacher: frozen, eval -----------------------------------------------------------
tcfg, teacher = M.build(TEACHER_DIR, load_checkpoint=True, dev=DEVICE)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
C = tcfg['n_embd']
assert tcfg['seq_len'] == SEQ_LEN and tcfg['tokenizer_vocab_size'] == cfg['tokenizer_vocab_size']
assert teacher.blocks[BLOCK].kind == 'dense', 'teacher block must be a dense FFN'
print(f'teacher {cfg["teacher_exp_dir"]}  depth={tcfg["depth"]} n_embd={C} '
      f'params={sum(p.numel() for p in teacher.parameters()):,}')

# ---- data (same path the run dirs use) -----------------------------------------------
base = get_base_dir()
tok = RustBPETokenizer.from_directory(os.path.join(base, 'tokenizer'))
assert tok.get_vocab_size() == cfg['tokenizer_vocab_size']
train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tok, DEVICE_BS, SEQ_LEN, split='train', device=DEVICE)
val_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(   # noqa: E731
    tok, DEVICE_BS, SEQ_LEN, split='val', device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE)

# ---- capture the FFN sub-layer's (input, output) --------------------------------------
# blocks[N].mlp IS the ffn_slot for a dense block: its input is ln2(x) and its output is
# what gets added back to the residual, so hooking it gives exactly the pair to regress.
cap = {}
hook = teacher.blocks[BLOCK].mlp.register_forward_hook(
    lambda m, i, o: cap.update(x=i[0].detach(), y=o.detach()))

# ---- student ---------------------------------------------------------------------------
student = CompressionMultiHeadLUT(
    input_dim=C, output_dim=C,
    inner_in_dim=cfg['lut_inner_in_dim'], inner_out_dim=cfg['lut_inner_out_dim'],
    nap=cfg['lut_n_anchor_pairs'], tph=cfg['lut_tables_per_head'],
    n_heads=cfg['lut_n_heads'],
    joint_head_compression=cfg['lut_joint_head_compression'],
    forward_mode=cfg['lut_forward_mode'], use_bf16=cfg['lut_use_bf16'],
    initial_weights_noise=cfg['lut_init_weights_noise'],
    learnable_temps=cfg['lut_learnable_temps'],
    random_seed=cfg['lut_base_seed'] + BLOCK, device=DEVICE).to(DEVICE)
pc = CompressionMultiHeadLUT.param_count(
    C, C, inner_in_dim=cfg['lut_inner_in_dim'], inner_out_dim=cfg['lut_inner_out_dim'],
    nap=cfg['lut_n_anchor_pairs'], tph=cfg['lut_tables_per_head'],
    n_heads=cfg['lut_n_heads'])
teacher_ffn_params = sum(p.numel() for p in teacher.blocks[BLOCK].mlp.parameters())
R = cfg['lut_n_heads'] * cfg['lut_tables_per_head']
K = 2 ** cfg['lut_n_anchor_pairs']
print(f'student H{cfg["lut_n_heads"]}/c{cfg["lut_inner_in_dim"]} '
      f'nap{cfg["lut_n_anchor_pairs"]} tph{cfg["lut_tables_per_head"]} -> {pc["total"]:,} params '
      f'({pc["total"]/teacher_ffn_params:.2f}x the {teacher_ffn_params:,}-param FFN it replaces)')
print(f'  {R} tables x {K} cells; {TOKENS_PER_STEP:,} tokens/step '
      f'({DEVICE_BS}x{SEQ_LEN}x{ACCUM} accum) -> ~{TOKENS_PER_STEP//K} tokens/row/table')

opt, n_decay, n_nodecay = setup_optimizer(student, LR, WD)
print(f'  AdamW decay={n_decay:,} (wd={WD}) nodecay={n_nodecay:,} (wd=0)')

# ---- train -----------------------------------------------------------------------------
csv_f = open(os.path.join(OUT_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'mse', 'ema', 'lr', 'elapsed_s'])
student.train()
hist, ema, t0 = [], None, time.time()
for step in range(N_STEPS):
    for g in opt.param_groups:
        g['lr'] = LR * lr_scale(step)
    opt.zero_grad(set_to_none=True)
    step_mse = 0.0
    for _ in range(ACCUM):
        x_tok, _ = next(train_loader)
        with torch.no_grad():
            teacher(x_tok)                       # hook fills cap
        xin = cap['x'].reshape(-1, C).float()
        yref = cap['y'].reshape(-1, C).float()
        loss = F.mse_loss(student(xin), yref)
        (loss / ACCUM).backward()
        step_mse += loss.item() / ACCUM
    opt.step()

    ema = step_mse if ema is None else 0.98 * ema + 0.02 * step_mse
    if step % cfg['log_every'] == 0 or step == N_STEPS - 1:
        el = time.time() - t0
        print(f'  step {step:>5}/{N_STEPS}  mse {step_mse:.6f}  ema {ema:.6f}  '
              f'lr {opt.param_groups[0]["lr"]:.2e}  {el:.0f}s', flush=True)
        csv_w.writerow([step, f'{step_mse:.8f}', f'{ema:.8f}',
                        f'{opt.param_groups[0]["lr"]:.6e}', f'{el:.1f}'])
        csv_f.flush()
        hist.append((step, step_mse))
train_s = time.time() - t0
csv_f.close()
print(f'trained {N_STEPS} steps in {train_s/60:.1f} min '
      f'({train_s/max(1,N_STEPS):.2f} s/step)')

# ---- held-out MSE, relative error, cell occupancy ---------------------------------------
student.eval()
val_iter, sse, sst, n_el, xs = val_factory(), 0.0, 0.0, 0.0, []
with torch.no_grad():
    for _ in range(cfg['probe_batches']):
        x_tok, _ = next(val_iter)
        teacher(x_tok)
        xin = cap['x'].reshape(-1, C).float()
        yref = cap['y'].reshape(-1, C).float()
        sse += (student(xin) - yref).pow(2).sum().item()
        sst += (yref - yref.mean()).pow(2).sum().item()
        n_el += yref.numel()
        xs.append(xin)
mse, var = sse / n_el, sst / n_el
rel = mse / var
print(f'held-out MSE {mse:.6f}  teacher-output var {var:.6f}  normalized {rel:.4f}')

counts = cell_occupancy(student, xs)
visited, tot = (counts > 0).sum(1), counts.sum().item()
occ = dict(n_tables=int(R), cells_per_table=int(K),
           tokens_probed=int(tot // R),
           visited_mean=float(visited.float().mean()),
           visited_min=int(visited.min()), visited_max=int(visited.max()),
           visited_frac_mean=float(visited.float().mean() / K),
           cells_never_visited=int((counts == 0).sum()), cells_total=int(R * K),
           frac_cells_never_visited=float((counts == 0).sum().item() / (R * K)),
           top10pct_visit_share=float(
               counts.flatten().sort(descending=True).values[:max(1, (R * K) // 10)]
               .sum().item() / max(1, tot)))
print(f'cell occupancy: {occ["visited_frac_mean"]*100:.1f}% of {K} cells/table visited '
      f'({occ["visited_min"]}-{occ["visited_max"]}); '
      f'{occ["frac_cells_never_visited"]*100:.1f}% of cells never visited; '
      f'top-10% hold {occ["top10pct_visit_share"]*100:.1f}% of visits')

# ---- end-to-end: swap the student in for this block only, measure val bpb ---------------
# Two protocols; see config's _eval_note. The anchored one (bs48 x 10) is the baseline's own
# eval setting and reproduces its published 1.15144 exactly -- any other setting measures a
# different quantity, so the published number can only be compared against under it.
hook.remove()
blk = teacher.blocks[BLOCK]
adapter = EvalAdapter(teacher).to(DEVICE)
EVAL_BS = cfg['eval_device_batch_size']
eval_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(   # noqa: E731
    tok, EVAL_BS, SEQ_LEN, split='val', device=DEVICE)


def bpb_at(steps):
    with torch.no_grad():
        return float(evaluate_bpb(adapter, eval_factory(), steps, token_bytes))


blk.ffn, blk.kind = student, 'compression'      # ffn_slot() now routes through the student
bpb, bpb_long = bpb_at(cfg['eval_steps']), bpb_at(cfg['eval_steps_long'])
blk.ffn, blk.kind = None, 'dense'               # restore, then the same evals as reference
bpb_ref, bpb_ref_long = bpb_at(cfg['eval_steps']), bpb_at(cfg['eval_steps_long'])
print(f'\nval bpb  ANCHORED (bs{EVAL_BS} x {cfg["eval_steps"]}, the baseline protocol):')
print(f'  swapped(block {BLOCK}) {bpb:.5f} | teacher {bpb_ref:.5f} | delta {bpb-bpb_ref:+.5f}'
      f'   (published baseline {cfg["teacher_val_bpb"]:.5f})')
print(f'val bpb  LONG (bs{EVAL_BS} x {cfg["eval_steps_long"]}, 5x data):')
print(f'  swapped(block {BLOCK}) {bpb_long:.5f} | teacher {bpb_ref_long:.5f} | '
      f'delta {bpb_long-bpb_ref_long:+.5f}')

# ---- outputs ----------------------------------------------------------------------------
if hist:
    plt.figure(figsize=(7, 4))
    plt.plot([h[0] for h in hist], [h[1] for h in hist])
    plt.yscale('log'); plt.xlabel('step'); plt.ylabel('MSE vs teacher FFN')
    plt.title(f'block {BLOCK} distillation ({cfg["exp_name"]})')
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'loss.png'), dpi=120); plt.close()

summary = dict(
    exp_name=cfg['exp_name'], block=BLOCK, n_steps=N_STEPS,
    tokens_per_step=TOKENS_PER_STEP, tokens_per_row=TOKENS_PER_STEP // K,
    student_params=pc, teacher_ffn_params=int(teacher_ffn_params),
    student_vs_teacher_ffn=pc['total'] / teacher_ffn_params,
    train_minutes=train_s / 60.0, s_per_step=train_s / max(1, N_STEPS),
    heldout_mse=mse, teacher_output_var=var, normalized_mse=rel,
    cell_occupancy=occ,
    eval_device_batch_size=EVAL_BS,
    anchored=dict(eval_steps=cfg['eval_steps'], swapped=bpb, teacher=bpb_ref,
                  delta=bpb - bpb_ref, published_baseline=cfg['teacher_val_bpb']),
    long=dict(eval_steps=cfg['eval_steps_long'], swapped=bpb_long, teacher=bpb_ref_long,
              delta=bpb_long - bpb_ref_long))
with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
torch.save(student.state_dict(), os.path.join(OUT_DIR, 'student.pt'))
print(f'wrote {OUT_DIR}/metrics.csv, summary.json, loss.png, student.pt')
