"""Per-layer FFN distillation harness: how hard is each layer's FFN for a fixed LUT?

A frozen, fully trained vanilla transformer is the TEACHER. For every requested layer we
train an independent CompressionMHL(LightMHL) STUDENT to reproduce that layer's FFN map
    h = ln2(x)  ->  o = mlp(h)
on real data, then measure how much of the teacher's FFN output the student fails to
explain. Holding the student architecture and budget fixed across layers turns that error
into a per-layer "FFN complexity" measure.

DESIGN, stated so the numbers can be read correctly
---------------------------------------------------
* EXACT activations. Forward hooks on each block's `mlp` capture its true input (the output
  of ln2) and its true output (before the residual add). The teacher forward runs
  embeddings + blocks only, never the 32,768-way unembedding, and stops after the deepest
  requested layer.
* FRESH DATA every step: real train-split tokens are pushed through the teacher on the fly,
  so a student cannot memorise a fixed activation set. Error is measured on a FIXED held-out
  val-split slab, never on training samples.
* ONE PROCESS, ALL LAYERS, SAME BATCHES. Every student has its own optimiser and is
  otherwise independent, but every layer sees byte-identical batches, steps and LR schedule.
  That is the cleanest same-budget comparison, and ~N-layers cheaper than separate runs.
* STUDENT = the LM code path. Each student is the `ffn` of a `MinimalBlock` built from a run
  config (default: exp_n_0238), then initialised exactly as `MinimalGPT` initialises it
  (normal std 0.02 on Linear weights, decompress weight zeroed). So the LUT kwargs, seeds and
  init are byte-identical to what an LM training run of that config builds -- nothing here is
  a re-implementation of the architecture. A zeroed decompress means every student starts at
  output 0, i.e. relative error 1.0 and FVU ~1.0 at step 0.
* METRICS. Raw MSE is NOT comparable across layers, because FFN output scale differs by
  depth. So the headline is FVU = MSE / Var(teacher output), the fraction of variance left
  unexplained, alongside MSE and relative error ||e||^2 / ||o_t||^2.
* LINEAR BASELINE. A closed-form least-squares affine map h -> o is fit per layer on train
  data and scored on the same val slab. Without it an FVU cannot be read as easy or hard;
  with it you see how much of each FFN is nonlinear at all, and whether the LUT beats a
  plain linear layer on it.

REUSE FOR GOAL 2 (per-layer LUT hyperparameters): pass `--student-overrides` as a JSON list,
one dict per layer (or a single dict applied to all). The architecture is otherwise taken
from `--student-config`, so varying it per layer needs no code change.

    python distill_ffn.py --out runs/sweep_base --steps 6000
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(FR, 'tools'))
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir                                  # noqa: E402
from nanochat.tokenizer import RustBPETokenizer                           # noqa: E402
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit  # noqa: E402
from model_build import build_model, MinimalBlock                         # noqa: E402
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut            # noqa: E402
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT          # noqa: E402
from spiky.lutorch.bh4_multi_head_lut import BH4MultiHeadLUT              # noqa: E402

DEF_TEACHER = os.path.join(FR, 'runs', 'exp_n_0151_long48k_untied_vanilla')
DEF_STUDENT = os.path.join(FR, 'runs_corrected', 'exp_n_0238_pureLUT0193_TVlam10_48k_seed1',
                           'config.json')


# ------------------------------------------------------------------------------------------
# teacher
# ------------------------------------------------------------------------------------------
def load_teacher(run_dir, vocab, device):
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    assert cfg.get('ffn_type', 'compression') == 'dense', 'teacher must be a dense-FFN model'
    model = build_model(cfg, vocab, device=device)
    sd = torch.load(os.path.join(run_dir, 'checkpoint.pt'), map_location=device)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        raise RuntimeError(f'teacher checkpoint mismatch: missing={miss} unexpected={unexp}')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


@torch.no_grad()
def teacher_ffn_io(model, idx, layers):
    """Run embeddings + blocks up to max(layers); return {layer: (h, o)} with h,o [N, C]."""
    got = {}
    hooks = []
    for li in layers:
        def hook(mod, inp, out, _li=li):
            got[_li] = (inp[0].detach(), out.detach())
        hooks.append(model.blocks[li].mlp.register_forward_hook(hook))
    try:
        x = model.tok_emb(idx)
        for bi, block in enumerate(model.blocks):
            x = block(x, model.rope.cos, model.rope.sin)
            if bi >= max(layers):
                break
    finally:
        for h in hooks:
            h.remove()
    C = x.shape[-1]
    return {li: (h.reshape(-1, C), o.reshape(-1, C)) for li, (h, o) in got.items()}


# ------------------------------------------------------------------------------------------
# student
# ------------------------------------------------------------------------------------------
def build_student(student_cfg, n_embd, n_head, layer_idx, device):
    """The `ffn` of a MinimalBlock built from `student_cfg`, initialised as MinimalGPT does."""
    blk = MinimalBlock(n_embd, n_head, layer_idx, student_cfg)
    if blk.ffn_type == 'dense':
        raise ValueError('student config builds a dense FFN; expected a LUT FFN')
    ffn = blk.ffn
    for m in ffn.modules():                              # MinimalGPT._init_weights
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
    if getattr(ffn, 'has_decompress', False) and hasattr(ffn.decompress, 'weight'):
        nn.init.zeros_(ffn.decompress.weight)            # MinimalGPT zeroes it
        if student_cfg.get('lut_impl', 'fast') == 'bh4' and ffn.decompress.bias is not None:
            nn.init.zeros_(ffn.decompress.bias)
    return ffn.to(device)


def setup_optimizer(module, lr, weight_decay, tables_no_decay):
    """Exactly train_fixed.py's grouping (tables exempt by class when tables_no_decay)."""
    exempt = ((FastMultiHeadLut, LightMultiHeadLUT, BH4MultiHeadLUT)
              if tables_no_decay else (FastMultiHeadLut,))
    lut_ids = {id(p) for m in module.modules() if isinstance(m, exempt)
               for p in m.parameters(recurse=False)}
    decay, nodecay = [], []
    for p in module.parameters():
        if not p.requires_grad:
            continue
        (nodecay if (id(p) in lut_ids or p.ndim < 2) else decay).append(p)
    opt = torch.optim.AdamW([
        dict(params=decay, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay),
        dict(params=nodecay, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)])
    for g in opt.param_groups:
        g['initial_lr'] = g['lr']
    return opt


def lr_scale(step, n_steps, warmup_frac):
    """Exactly train_fixed.py's get_lr_scale: linear warmup, cosine to a 0.1x floor."""
    w = int(warmup_frac * n_steps)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n_steps - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def student_forward(ffn, h):
    o = ffn(h)
    if o.dim() == 3:                                     # raw FastMHL: [N, heads, C]
        o = o.sum(dim=1)
    return o.to(h.dtype)


INT32_LIMIT = 2**31 - 1


def auto_chunks(ffn, n_tokens):
    """Chunks needed so no LightMHL kernel sees >= 2^31 elements in one call.

    The fused read is F.embedding_bag with per_sample_weights; its CUDA backward handles
    n_tokens * n_tables * output_dim elements with 32-bit indexing. Past 2^31 it faults with
    "illegal memory access" (hit at H=32: 24,576 * 4,096 * 48 = 4.8e9). Chunking the student
    step keeps every call under the limit and is gradient-identical (see train_step).
    """
    worst = 0
    for m in ffn.modules():
        if isinstance(m, LightMultiHeadLUT):
            worst = max(worst, n_tokens * m.n_tables * m.output_dim)
    return max(1, math.ceil(worst / INT32_LIMIT))


def train_step(ffn, h, o_t, n_chunks):
    """Forward + backward of mean-squared error over ALL tokens, in `n_chunks` pieces.

    Each chunk's MSE is weighted by its share of tokens, so the summed loss IS the full-batch
    MSE and the accumulated gradient is the full-batch gradient (up to float reordering).
    n_chunks == 1 is exactly the unchunked step. Returns the full-batch loss as a float.
    """
    if n_chunks == 1:
        loss = F.mse_loss(student_forward(ffn, h), o_t)
        loss.backward()
        return loss.item()
    N, total = h.shape[0], 0.0
    for hc, oc in zip(h.chunk(n_chunks), o_t.chunk(n_chunks)):
        loss = F.mse_loss(student_forward(ffn, hc), oc) * (hc.shape[0] / N)
        loss.backward()
        total += loss.item()
    return total


# ------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------
def err_stats(pred, tgt, tgt_var):
    """MSE, relative error, FVU over all tokens and dims."""
    e = (pred - tgt).double()
    mse = e.pow(2).mean().item()
    rel = e.pow(2).sum().item() / max(tgt.double().pow(2).sum().item(), 1e-30)
    return {'mse': mse, 'rel_err': rel, 'fvu': mse / max(tgt_var, 1e-30)}


@torch.no_grad()
def fit_linear(h, o):
    """Least-squares affine map h -> o (with bias), float64. Returns (W, b)."""
    X = torch.cat([h.double(), torch.ones(h.shape[0], 1, dtype=torch.float64,
                                          device=h.device)], dim=1)
    sol = torch.linalg.lstsq(X, o.double()).solution            # [C+1, C]
    return sol[:-1], sol[-1]


def take_rows(loader, n_rows, skip_rows=0):
    rows, seen = [], 0
    while sum(r.shape[0] for r in rows) < n_rows:
        x, _ = next(loader)
        x = x.clone()
        if seen < skip_rows:
            drop = min(x.shape[0], skip_rows - seen)
            seen += drop
            x = x[drop:]
        if x.shape[0]:
            rows.append(x)
    return torch.cat(rows)[:n_rows]


# ------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output dir (created; must not already hold results)')
    ap.add_argument('--teacher', default=DEF_TEACHER)
    ap.add_argument('--student-config', default=DEF_STUDENT,
                    help='run config.json whose lut_* keys define the student architecture')
    ap.add_argument('--student-overrides', default=None,
                    help='JSON: one dict for all layers, or a list with one dict per layer')
    ap.add_argument('--layers', default='all')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch-rows', type=int, default=48, help='rows x seq_len tokens per step')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=0.1)
    ap.add_argument('--warmup-frac', type=float, default=0.1)
    ap.add_argument('--cell-smoothness', type=float, default=0.0,
                    help='TV regularizer lambda on the student tables (0 = off)')
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--eval-rows', type=int, default=32)
    ap.add_argument('--linfit-rows', type=int, default=96)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--student-chunks', default='auto',
                    help="split each student's step into N gradient-identical chunks; 'auto' "
                         "= fewest that keep LightMHL kernels under 2^31 elements")
    ap.add_argument('--max-steps-smoke', type=int, default=0,
                    help='if >0, stop after this many steps (timing / smoke only)')
    a = ap.parse_args()

    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, 'results.json')) and not a.max_steps_smoke:
        raise SystemExit(f'{out} already holds results.json -- use a fresh --out dir')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)

    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    vocab = tok.get_vocab_size()
    teacher, tcfg = load_teacher(a.teacher, vocab, dev)
    C, NH, depth, T = tcfg['n_embd'], tcfg['n_head'], tcfg['depth'], tcfg['seq_len']
    layers = list(range(depth)) if a.layers == 'all' else [int(v) for v in a.layers.split(',')]

    base = json.load(open(a.student_config))
    ov = json.loads(a.student_overrides) if a.student_overrides else {}
    per_layer_cfg = {}
    for li in layers:
        o = ov[layers.index(li)] if isinstance(ov, list) else ov
        per_layer_cfg[li] = {**base, **o}

    students, opts = {}, {}
    for li in layers:
        students[li] = build_student(per_layer_cfg[li], C, NH, li, dev)
        opts[li] = setup_optimizer(students[li], a.lr, a.weight_decay,
                                   bool(per_layer_cfg[li].get('lut_tables_no_decay', False)))
    nparam = {li: sum(p.numel() for p in s.parameters()) for li, s in students.items()}

    # ---- fixed held-out val slab + train slab for the linear baseline -----------------------
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(tok, 48, T, split='val', device=dev)
    val_idx = take_rows(val_loader, a.eval_rows, skip_rows=12)     # skip the 12 anomalous rows
    val_io = teacher_ffn_io(teacher, val_idx, layers)
    val_var = {li: val_io[li][1].double().var(dim=0, unbiased=False).mean().item()
               for li in layers}
    val_norm = {li: val_io[li][1].double().norm(dim=-1).mean().item() for li in layers}

    train_loader = tokenizing_distributed_data_loader_bos_bestfit(tok, a.batch_rows, T,
                                                                  split='train', device=dev)
    lin_idx = take_rows(train_loader, a.linfit_rows)
    lin = {}
    for li in layers:
        hs, os_ = [], []
        for i in range(0, a.linfit_rows, 16):
            io = teacher_ffn_io(teacher, lin_idx[i:i + 16], [li])
            hs.append(io[li][0]); os_.append(io[li][1])
        W, b = fit_linear(torch.cat(hs), torch.cat(os_))
        pred = (val_io[li][0].double() @ W + b).float()
        lin[li] = err_stats(pred, val_io[li][1], val_var[li])
        del hs, os_, W, b, pred

    manifest = {
        'teacher': os.path.relpath(a.teacher, FR), 'teacher_cfg_name': tcfg.get('exp_name'),
        'student_config': os.path.relpath(a.student_config, FR),
        'student_overrides': ov, 'layers': layers, 'steps': a.steps,
        'batch_rows': a.batch_rows, 'tokens_per_step': a.batch_rows * T, 'lr': a.lr,
        'weight_decay': a.weight_decay, 'warmup_frac': a.warmup_frac,
        'cell_smoothness': a.cell_smoothness, 'eval_rows': a.eval_rows,
        'eval_tokens': val_idx.numel(), 'linfit_rows': a.linfit_rows, 'seed': a.seed,
        'student_params_per_layer': nparam,
        'teacher_ffn_params_per_layer': 2 * C * 4 * C,
        'teacher_output_var': val_var, 'teacher_output_mean_norm': val_norm,
        'linear_baseline': lin,
    }
    chunks = {li: (auto_chunks(students[li], a.batch_rows * T) if a.student_chunks == 'auto'
                   else int(a.student_chunks)) for li in layers}
    manifest['student_chunks'] = chunks
    json.dump(manifest, open(os.path.join(out, 'manifest.json'), 'w'), indent=2)
    print(f'teacher {tcfg.get("exp_name")} | layers {layers} | student params/layer '
          f'{nparam[layers[0]]:,} | chunks/step {chunks[layers[0]]} | eval tokens '
          f'{val_idx.numel():,}', flush=True)
    for li in layers:
        print(f'  L{li}: teacher out var {val_var[li]:.5f}  mean|o| {val_norm[li]:.4f}  '
              f'linear FVU {lin[li]["fvu"]:.4f}', flush=True)

    # ---- training ------------------------------------------------------------------------
    curve_f = open(os.path.join(out, 'curves.csv'), 'w', newline='')
    cw = csv.writer(curve_f)
    cw.writerow(['step', 'layer', 'train_mse', 'val_mse', 'val_rel_err', 'val_fvu', 'elapsed_s'])
    last_train = {li: float('nan') for li in layers}

    @torch.no_grad()
    def evaluate(step, elapsed):
        res = {}
        for li in layers:
            s = students[li]
            s.eval()
            preds = [student_forward(s, val_io[li][0][i:i + 8192])
                     for i in range(0, val_io[li][0].shape[0], 8192)]
            s.train()
            st = err_stats(torch.cat(preds), val_io[li][1], val_var[li])
            res[li] = st
            cw.writerow([step, li, f'{last_train[li]:.6e}', f'{st["mse"]:.6e}',
                         f'{st["rel_err"]:.6f}', f'{st["fvu"]:.6f}', f'{elapsed:.1f}'])
        curve_f.flush()
        return res

    n_run = a.max_steps_smoke or a.steps
    t0 = time.time()
    evaluate(0, 0.0)
    final = None
    for step in range(1, n_run + 1):
        scale = lr_scale(step, a.steps, a.warmup_frac)
        idx, _ = next(train_loader)
        io = teacher_ffn_io(teacher, idx, layers)
        for li in layers:
            s, opt = students[li], opts[li]
            for g in opt.param_groups:
                g['lr'] = g['initial_lr'] * scale
            h, o_t = io[li]
            opt.zero_grad(set_to_none=True)
            last_train[li] = train_step(s, h, o_t, chunks[li])   # backward of the MSE
            if a.cell_smoothness > 0:                              # grads accumulate onto it
                tv = [m.cell_tv() for m in s.modules() if isinstance(m, LightMultiHeadLUT)]
                if tv:
                    (a.cell_smoothness * torch.stack(tv).mean()).backward()
            torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0)
            opt.step()
        del io
        if step % 100 == 0 or step == 1:
            el = time.time() - t0
            print(f'step {step:6d}/{a.steps} | {el / step:.3f} s/step | train mse ' +
                  ' '.join(f'L{li}={last_train[li]:.3e}' for li in layers), flush=True)
        if step % a.eval_every == 0 or step == n_run:
            final = evaluate(step, time.time() - t0)
            print(f'[EVAL] step {step}: FVU ' +
                  ' '.join(f'L{li}={final[li]["fvu"]:.4f}' for li in layers), flush=True)
    curve_f.close()

    peak_gib = torch.cuda.max_memory_allocated() / 2**30 if dev == 'cuda' else 0.0
    if a.max_steps_smoke:
        print(f'SMOKE DONE: {n_run} steps in {time.time() - t0:.1f} s | peak GPU mem '
              f'{peak_gib:.2f} GiB', flush=True)
        return

    results = {
        **manifest,
        'wall_clock_s': round(time.time() - t0, 1),
        'peak_gpu_mem_gib': round(peak_gib, 2),
        'final': {li: {**final[li], 'linear_fvu': lin[li]['fvu'],
                       'fvu_ratio_vs_linear': final[li]['fvu'] / max(lin[li]['fvu'], 1e-30)}
                  for li in layers},
    }
    json.dump(results, open(os.path.join(out, 'results.json'), 'w'), indent=2)
    print('\nFINAL (held-out val slab)')
    print(f'{"layer":>5} {"MSE":>11} {"rel err":>9} {"FVU":>8} {"linear FVU":>11} {"LUT/linear":>11}')
    for li in layers:
        r = results['final'][li]
        print(f'{li:>5} {r["mse"]:>11.4e} {r["rel_err"]:>9.4f} {r["fvu"]:>8.4f} '
              f'{r["linear_fvu"]:>11.4f} {r["fvu_ratio_vs_linear"]:>11.3f}')


if __name__ == '__main__':
    main()
