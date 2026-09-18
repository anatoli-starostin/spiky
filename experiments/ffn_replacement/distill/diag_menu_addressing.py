"""P1 diagnostic: is matrix-menu addressing frozen by Adam's eps, and which remedy wakes it?

One layer (default L2), a matrix-menu student trained on the SAME data stream / schedule as the 4K harness runs
(the first --run-steps of a --steps cosine schedule), once per arm. Every --log-every steps it records, for the
menu_logits / menu_log_tau:
  * gradient rms (over nonzero entries) and the fraction of nonzero logit-gradient entries,
  * the ACTUAL Adam update rms (param delta across the optimizer step),
  * the fraction of cells whose argmax differs from the init argmax (did addressing move?),
  * tau per head,
and held-out FVU on the harness's eval slab every --eval-every steps.

    python diag_menu_addressing.py --out runs/diag_menu_addressing [--layer 2 --tph 32 --run-steps 800]
Writes <out>/diag.json and <out>/diag.png.
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import distill_ffn as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--layer', type=int, default=2)
ap.add_argument('--tph', type=int, default=32)
ap.add_argument('--menu-size', type=int, default=64)
ap.add_argument('--steps', type=int, default=4000, help='schedule length (as the harness runs)')
ap.add_argument('--run-steps', type=int, default=800, help='how many of those steps to actually run')
ap.add_argument('--log-every', type=int, default=25)
ap.add_argument('--eval-every', type=int, default=200)
ap.add_argument('--device', default='cuda')
ap.add_argument('--arms', default=None, help='comma-separated arm indices (default all)')
a = ap.parse_args()

ARMS = [  # name, menu_eps, menu_lr_mult, extra student overrides
    ('eps 1e-8 (control)', None, 1.0, {}),
    ('eps 1e-12', 1e-12, 1.0, {}),
    ('eps 1e-16', 1e-16, 1.0, {}),
    ('eps 1e-16, lr x10', 1e-16, 10.0, {}),
    ('large init 1/sqrt(d_in), eps 1e-8', None, 1.0, {'lut_menu_init_scale': 1 / math.sqrt(48)}),
]

out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
os.makedirs(out, exist_ok=True)
dev = a.device
if a.arms:
    ARMS = [ARMS[int(i)] for i in a.arms.split(',')]
tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
teacher, tcfg = D.load_teacher(D.DEF_TEACHER, tok.get_vocab_size(), dev)
T = tcfg['seq_len']
li = a.layer
val_idx = D.take_rows(D.tokenizing_distributed_data_loader_bos_bestfit(tok, 48, T, split='val', device=dev), 32,
                      skip_rows=12)
vh, vo = D.teacher_ffn_io(teacher, val_idx, [li])[li]
vvar = vo.double().var(dim=0, unbiased=False).mean().item()
base = json.load(open(D.DEF_STUDENT))
ov0 = {'lut_n_heads': 8, 'lut_tables_per_head': a.tph, 'lut_cell_mode': 'matrix_menu',
       'lut_menu_size': a.menu_size, 'lut_menu_forward': 'hard'}


def val_fvu(s):
    s.eval()
    with torch.no_grad():
        pred = torch.cat([D.student_forward(s, vh[i:i + 8192]) for i in range(0, vh.shape[0], 8192)])
    s.train()
    return D.err_stats(pred, vo, vvar)['fvu']


results = {}
for name, eps, mult, extra in ARMS:
    torch.manual_seed(1)
    s = D.build_student({**base, **ov0, **extra}, tcfg['n_embd'], tcfg['n_head'], li, dev)
    lut = s.lut_light
    opt = D.setup_optimizer(s, 3e-4, 0.1, bool(base.get('lut_tables_no_decay')), menu_eps=eps, menu_lr_mult=mult)
    am0 = lut.menu_logits.detach().argmax(-1).clone()
    loader = D.tokenizing_distributed_data_loader_bos_bestfit(tok, 48, T, split='train', device=dev)
    # the harness's train loader: 96 linfit rows are drawn first -- reproduce that so batches match its runs
    D.take_rows(loader, 96)
    log = []
    t0 = time.time()
    for step in range(1, a.run_steps + 1):
        scale = D.lr_scale(step, a.steps, 0.1)
        for g in opt.param_groups:
            g['lr'] = g['initial_lr'] * scale
        idx, _ = next(loader)
        h, o = D.teacher_ffn_io(teacher, idx, [li])[li]
        opt.zero_grad(set_to_none=True)
        loss = D.train_step(s, h, o, 1)
        torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0)
        rec = step % a.log_every == 0 or step == 1
        if rec:
            gl = lut.menu_logits.grad
            gt = lut.menu_log_tau.grad
            gm = lut.menu.grad
            nzl = gl != 0
            L_before = lut.menu_logits.detach().clone()
            tau_before = lut.menu_log_tau.detach().clone()
        opt.step()
        if rec:
            dL = lut.menu_logits.detach() - L_before
            log.append(dict(
                step=step, train_mse=loss,
                logit_grad_rms=(gl[nzl].pow(2).mean().sqrt().item() if nzl.any() else 0.0),
                logit_grad_nonzero=nzl.float().mean().item(),
                tau_grad_rms=gt.pow(2).mean().sqrt().item(),
                menu_grad_rms=gm.pow(2).mean().sqrt().item(),
                logit_update_rms=(dL[nzl].pow(2).mean().sqrt().item() if nzl.any() else 0.0),
                tau_update_abs=(lut.menu_log_tau.detach() - tau_before).abs().max().item(),
                argmax_changed=(lut.menu_logits.detach().argmax(-1) != am0).float().mean().item(),
                tau=lut.menu_tau.detach().tolist(),
                menu_std=lut.menu.detach().std().item(),
                lr=opt.param_groups[0]['lr']))
        if step % a.eval_every == 0:
            log[-1]['val_fvu'] = val_fvu(s)
    last = log[-1]
    print(f'{name:36s} {time.time() - t0:5.1f}s | step {last["step"]} | FVU {last.get("val_fvu", float("nan")):.4f} | '
          f'argmax changed {last["argmax_changed"]:.4f} | logit grad rms {last["logit_grad_rms"]:.2e} | '
          f'logit update rms {last["logit_update_rms"]:.2e} | tau {min(last["tau"]):.4f}-{max(last["tau"]):.4f}',
          flush=True)
    results[name] = log
    del s, opt

json.dump({'args': vars(a), 'arms': results}, open(os.path.join(out, 'diag.json'), 'w'), indent=1)
fig, axs = plt.subplots(2, 3, figsize=(17, 8.5))
panels = [('logit_grad_rms', 'menu_logits grad rms (nonzero entries)', True),
          ('logit_update_rms', 'menu_logits ACTUAL Adam update rms', True),
          ('argmax_changed', 'fraction of cells with argmax != init', False),
          ('tau_update_abs', 'max |d log tau| per step', True),
          ('menu_std', 'menu matrix std', True),
          ('val_fvu', f'held-out FVU (L{li})', False)]
for ax, (k, title, logy) in zip(axs.flat, panels):
    for name, log in results.items():
        pts = [(r['step'], r[k]) for r in log if k in r and (r[k] > 0 or not logy)]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='.' if k == 'val_fvu' else None, label=name)
    ax.set_title(title, fontsize=10)
    if logy:
        ax.set_yscale('log')
    ax.axhline(1e-8, color='grey', ls=':', lw=0.8) if k == 'logit_grad_rms' else None
    ax.grid(ls=':', alpha=0.5)
    ax.set_xlabel('step')
axs.flat[0].legend(fontsize=8)
fig.suptitle(f'Matrix-menu addressing diagnostic: L{li}, H8, tph{a.tph}, M{a.menu_size}, hard/STE, first '
             f'{a.run_steps} steps of the {a.steps}-step schedule (dotted: Adam eps 1e-8)', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(out, 'diag.png'), dpi=120)
print('wrote', out)
