"""Train the MDN head (E1 frozen-backbone / E2 joint). Config-driven, mirrors our standard protocol
(AdamW, warmup+cosine, grad-clip 1.0, bpb eval). Outputs metrics.csv / summary.json / loss.png.

config.json fields: baseline_exp, n_maps, n_mix, x_init(cold|warm), freeze_backbone(bool), gamma_dec,
n_steps, device_batch_size, total_batch_size, lr, weight_decay, lr_warmup_fraction, eval_every,
eval_steps, seq_len, random_seed, (SMOKE=1 env -> build+one-step then exit).
"""
import os, sys, json, math, time, csv
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import backbone as bb
from mdn_head import MDNHead
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.common import get_base_dir
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

EXP_DIR = os.path.dirname(os.path.abspath(__file__)) if len(sys.argv) < 2 else os.path.abspath(sys.argv[1])
cfg = json.load(open(os.path.join(EXP_DIR, 'config.json')))
SMOKE = os.environ.get('SMOKE', '0') == '1'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed']); np.random.seed(cfg['random_seed'])

BASELINE = os.path.expanduser(cfg['baseline_exp'])
N, M = cfg['n_maps'], cfg['n_mix']
X_INIT = cfg.get('x_init', 'cold'); FREEZE = bool(cfg.get('freeze_backbone', True))
GAMMA_DEC = cfg.get('gamma_dec', 1e-2)
DBS, TBS, NSTEPS = cfg['device_batch_size'], cfg['total_batch_size'], cfg['n_steps']
SEQ = cfg['seq_len']; LR, WD = cfg['lr'], cfg['weight_decay']; WARM = cfg['lr_warmup_fraction']
EVAL_EVERY, EVAL_STEPS = cfg['eval_every'], cfg['eval_steps']
GRAD_ACCUM = max(1, TBS // (DBS * SEQ))

print(f"[MDN] {cfg['exp_name']}  baseline={os.path.basename(BASELINE)} N={N} M={M} init={X_INIT} "
      f"freeze={FREEZE} steps={NSTEPS} dbs={DBS} accum={GRAD_ACCUM}")

model, bcfg, (miss, unexp) = bb.load_pretrained(BASELINE, device=DEVICE)
D, V = bcfg['n_embd'], bcfg['tokenizer_vocab_size']
Wdense = model.head.weight.detach().float()
DENSE_HEAD_PARAMS = V * D                                   # 12,582,912 (bias-free)
if not FREEZE:
    for p in model.parameters():
        p.requires_grad_(True)

# ---- X init ----
x_init = None
if X_INIT == 'warm':
    Wc = (Wdense - Wdense.mean(0)).cpu().numpy()
    _, S, Vt = np.linalg.svd(Wc, full_matrices=False)
    Xp = Wc @ Vt.T[:, :3 * N]                               # [V,3N] PCA coords
    Xp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-8)             # standardize columns (spec)
    # Overall init scale: standardized (std 1) X with unit-precision init gives a huge initial
    # Mahalanobis (Σ_n||x||^2 ~ 33 -> CE ~31). Scale to the cold regime (std ~0.02) so warm starts
    # near-unigram like cold but keeps the PCA *directions*; L learns the precision/scale from there.
    Xp = Xp * float(cfg.get('warm_x_scale', 0.02))
    x_init = Xp.reshape(V, N, 3)
b_init = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unigram_logfreq.npy'))
head = MDNHead(D, V, n_maps=N, n_mix=M, gamma_dec=GAMMA_DEC, x_init=x_init, b_init=b_init, device=DEVICE)
hp = MDNHead.param_count(D, V, N, M)
print(f"[MDN] head params = {hp['total']:,} (X={hp['X']:,} P={hp['P']:,} b={hp['b']:,}) | "
      f"dense head {DENSE_HEAD_PARAMS:,} -> reduction {DENSE_HEAD_PARAMS/hp['total']:.2f}x")

# ---- optimizer (AdamW two-group: X,b,P.bias no-wd; P.weight wd; + backbone if joint) ----
pg = head.param_groups()
decay = list(pg['decay']); no_decay = list(pg['no_decay'])
if not FREEZE:
    for n_, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
opt = torch.optim.AdamW([
    dict(params=decay, weight_decay=WD),
    dict(params=no_decay, weight_decay=0.0),
], lr=LR, betas=(0.9, 0.95))


def lr_at(step):
    w = int(WARM * NSTEPS)
    if step < w:
        return LR * step / max(1, w)
    prog = (step - w) / max(1, NSTEPS - w)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tok, DBS, SEQ, split='train', device=DEVICE)
val_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(tok, DBS, SEQ, split='val', device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE).float()


def hidden(x):
    if FREEZE:
        with torch.no_grad():
            return model.hidden(x)
    return model.hidden(x)


@torch.no_grad()
def eval_bpb(steps):
    vl = val_factory(); tot_nats = 0.0; tot_bytes = 0.0
    for _ in range(steps):
        x, y = next(vl)
        h = model.hidden(x)
        logits = head(h)
        ce = F.cross_entropy(logits.view(-1, V), y.view(-1), reduction='none', ignore_index=-1)
        mask = (y.view(-1) != -1)
        tot_nats += ce[mask].sum().item()
        tot_bytes += token_bytes[y.view(-1)[mask]].sum().item()
    return tot_nats / tot_bytes / math.log(2)


if SMOKE:
    x, y = next(train_loader)
    h = hidden(x); logits = head(h)
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-1) + head.decorrelation()
    loss.backward()
    print(f"[MDN] SMOKE OK  loss={loss.item():.4f}  logits={tuple(logits.shape)}")
    sys.exit(0)

csv_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f); csv_w.writerow(['step', 'train_loss', 'val_bpb']); csv_f.flush()
val_steps, val_bpbs, tr_losses = [], [], []
ema = None; t0 = time.time(); best = 1e9
for step in range(1, NSTEPS + 1):
    lr = lr_at(step)
    for g in opt.param_groups:
        g['lr'] = lr
    opt.zero_grad(set_to_none=True)
    acc_loss = 0.0
    for _ in range(GRAD_ACCUM):
        x, y = next(train_loader)
        h = hidden(x)
        logits = head(h)
        ce = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-1)
        loss = ce + head.decorrelation()
        (loss / GRAD_ACCUM).backward()
        acc_loss += ce.item() / GRAD_ACCUM
    torch.nn.utils.clip_grad_norm_(
        [p for grp in opt.param_groups for p in grp['params'] if p.requires_grad], 1.0)
    opt.step()
    if step <= 5 or step % 25 == 0:
        print(f"  [t] step {step} | {(time.time() - t0) / step:.2f}s/step avg | loss={acc_loss:.3f}", flush=True)
    ema = acc_loss if ema is None else 0.9 * ema + 0.1 * acc_loss
    if step % EVAL_EVERY == 0 or step == NSTEPS:
        bpb = eval_bpb(EVAL_STEPS)
        best = min(best, bpb)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}']); csv_f.flush()
        val_steps.append(step); val_bpbs.append(bpb); tr_losses.append(ema)
        print(f'step {step:6d} | loss={ema:.4f} | [VAL] bpb={bpb:.4f} | lr={lr:.2e}')

summary = dict(exp_name=cfg['exp_name'], baseline=os.path.basename(BASELINE), n_maps=N, n_mix=M,
               x_init=X_INIT, freeze_backbone=FREEZE, final_val_bpb=val_bpbs[-1], best_val_bpb=best,
               head_params=hp['total'], dense_head_params=DENSE_HEAD_PARAMS,
               head_reduction=DENSE_HEAD_PARAMS / hp['total'], training_time_hours=(time.time() - t0) / 3600)
json.dump(summary, open(os.path.join(EXP_DIR, 'summary.json'), 'w'), indent=2)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(val_steps, tr_losses); ax[0].set(title='train ce', xlabel='step'); ax[0].grid(True)
ax[1].plot(val_steps, val_bpbs); ax[1].set(title='val bpb', xlabel='step'); ax[1].grid(True)
fig.tight_layout(); fig.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=110)
print(f"\n=== DONE === final_val_bpb={val_bpbs[-1]:.5f} best={best:.5f} "
      f"head_params={hp['total']:,} reduction={DENSE_HEAD_PARAMS/hp['total']:.2f}x")
