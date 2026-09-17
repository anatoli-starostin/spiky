#!/usr/bin/env python3
"""Adversarial re-verification of the p2int8 5090 FFN-slot benchmark (task 0341bbbc). Independent of p2int8_slot.py.

    python paper_timings/p2int8_verify.py --mode timing --tag run1     # one fresh-process launch (run >= 3 times)
    python paper_timings/p2int8_verify.py --mode glue                  # the old harness's "other" glue, decomposed
    python paper_timings/p2int8_verify.py --mode tune                  # block_n x load16 sweep
    python paper_timings/p2int8_verify.py --mode sweep                 # memory- vs compute-bound
    python paper_timings/p2int8_verify.py --mode verify                # table bytes, correctness gate, rounding ties, path

Every timed region is ONE call of block 0's FFN slot on an input already in the row's own dtype and shape
[48, 512, 384]: reshape to [N, 384], the FFN module, reshape back. No dtype casts inside any timed region. Inputs are
REAL block-0 FFN inputs (val text through each row's own trained model), captured once, outside timing.

Three timing paths per row, interleaved in the same rounds:
  P1 harness  bench.timeit: 30 calls, one synchronize before and after (the paper's clock)
  P2 events   torch.cuda.Event per call (enable_timing), synchronize once after the 30 calls
  P3 wall     time.perf_counter around every call with synchronize before AND after it
"""
import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
FR = os.path.dirname(BENCH)
REPO = os.path.dirname(os.path.dirname(FR))
for p in (BENCH, os.path.join(REPO, 'src'), os.path.join(FR, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

import bench                                            # noqa: E402
from model_build import build_model                     # noqa: E402
from spiky.lutorch import pow2_int8, pow2_read          # noqa: E402

LA = os.path.join(FR, 'lut_ablation')
VAN = 'exp_g_abl_10_B16k_vanilla_dense_untied_seed1'
UNQ = 'exp_g_abl_46_B16k_light_lmfrozeng_n2_tau0p5_tv10_tph128_seed1_postmerge126'
QNT = 'exp_g_abl_48_B16k_light_lmfrozeng_n2_tau0p5_tv10_tph128_seed1_p2int8_postmerge126'
B, SEQ, C = 48, 512, 384
N = B * SEQ
OUT_DIR = os.path.join(HERE, 'verify_out')


# ---------------------------------------------------------------------------------------------------------------- utils
def log(*a):
    print(*a, flush=True)


def smi(fields):
    try:
        return subprocess.run(['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f'n/a ({type(e).__name__})'


def compute_apps():
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return [f'n/a ({type(e).__name__})']
    me = str(os.getpid())
    return [l.strip() for l in out.splitlines() if l.strip() and l.split(',')[0].strip() != me]


class Telemetry:
    """nvidia-smi sampled every second in a subprocess while timing runs."""
    FIELDS = 'timestamp,clocks.sm,clocks_event_reasons.active,temperature.gpu,power.draw,utilization.gpu'

    def __init__(self):
        self.rows = []
        self.p = subprocess.Popen(['nvidia-smi', f'--query-gpu={self.FIELDS}', '--format=csv,noheader', '-lms', '1000'],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.p.stdout:
            self.rows.append([x.strip() for x in line.split(',')])

    def stop(self):
        self.p.terminate()
        self.t.join(timeout=5)
        reasons = sorted({r[2] for r in self.rows if len(r) > 2})
        temps = [float(r[3]) for r in self.rows if len(r) > 3 and r[3].replace('.', '').isdigit()]
        clocks = [r[1] for r in self.rows if len(r) > 1]
        return dict(samples=len(self.rows), event_reasons_seen=reasons,
                    temp_C_min_max=[min(temps), max(temps)] if temps else None,
                    clocks_first_last=[clocks[0], clocks[-1]] if clocks else None,
                    power=[r[4] for r in self.rows[:: max(1, len(self.rows) // 8)] if len(r) > 4],
                    util=[r[5] for r in self.rows[:: max(1, len(self.rows) // 8)] if len(r) > 5])


def mapped_p2_ext():
    with open('/proc/self/maps') as fh:
        return sorted({os.path.basename(l.split()[-1]) for l in fh if 'pow2_int8' in l and l.split()[-1].endswith('.so')})


def load(exp):
    d = os.path.join(LA, exp)
    cfg = json.load(open(os.path.join(d, 'config.json')))
    m = build_model(cfg, cfg['tokenizer_vocab_size'])
    sd = torch.load(os.path.join(d, 'checkpoint.pt'), map_location='cuda')
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    crit = [k for k in missing if not any(s in k for s in ('rope', 'cos', 'sin'))]
    if crit or unexpected:
        raise RuntimeError(f'{exp}: state_dict mismatch missing={crit[:4]} unexpected={list(unexpected)[:4]}')
    return cfg, m.eval()


_VAL = None


def val_tokens(rows=B):
    """Real val text: rows 48..48+rows of the deterministic val stream (past the 12 contaminated leading rows)."""
    global _VAL
    if _VAL is None:
        from nanochat.common import get_base_dir
        from nanochat.tokenizer import RustBPETokenizer
        from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
        tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
        it = iter(tokenizing_distributed_data_loader_bos_bestfit(tok, 48, SEQ, split='val', device='cuda'))
        batches = [next(it)[0].clone() for _ in range(3)]
        _VAL = torch.cat(batches, 0)[48:]
    return _VAL[:rows]


def capture_ffn_inputs(model, toks, layers=None):
    """Block FFN inputs as [B, T, C] fp32, via pre-hooks (the block reshapes to [N, C] for LUT layers)."""
    got = {}
    L = len(model.blocks)
    layers = range(L) if layers is None else layers
    hs = []
    for i in layers:
        blk = model.blocks[i]
        mod = blk.mlp if blk.ffn_type == 'dense' else blk.ffn
        hs.append(mod.register_forward_pre_hook(
            lambda m, a, _i=i: got.__setitem__(_i, a[0].detach().float().reshape(toks.shape[0], toks.shape[1], -1).clone())))
    with torch.no_grad():
        logits = model(toks)
    for h in hs:
        h.remove()
    return got, logits


def timeit_events(fn, iters=30):
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return statistics.mean(s.elapsed_time(e) for s, e in ev)


def timeit_wall_sync(fn, iters=30):
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(ts)


def fgeom_model(table_dtype):
    """The paper's fused float CUDA kernel (FastMultiHeadLut single-cell hard read) at THIS geometry, random init."""
    import gather_fused
    import hybrid
    import model as M
    d = os.path.join(OUT_DIR, 'fgeom_H4d48_nap8_tph128')
    os.makedirs(d, exist_ok=True)
    json.dump(dict(ffn_type='compression', tie_unembedder=False, depth=1, n_embd=C, n_head=6, seq_len=SEQ,
                   tokenizer_vocab_size=32768, random_seed=1, lut_inner_in_dim=48, lut_inner_out_dim=48,
                   lut_n_anchor_pairs=8, lut_tables_per_head=128, lut_n_heads=4, lut_joint_head_compression=False,
                   lut_forward_mode='hard', lut_use_bf16=False, lut_init_weights_noise=1e-3, lut_base_seed=1000,
                   lut_learnable_temps=False), open(os.path.join(d, 'config.json'), 'w'))
    _, ref = M.build(d)
    m = M.build(d)[1]
    m.load_state_dict(ref.state_dict())
    if table_dtype == 'bf16':
        m = hybrid.apply(m)                               # the paper's hybrid-v2: bf16 dense, fp32 LUT input
    ok, why = gather_fused.supported(M.lut_modules(m)[0])
    if not ok:
        raise RuntimeError(f'fused kernel does not support this geometry: {why}')
    assert gather_fused.patch(m, table_dtype=table_dtype) == len(M.lut_modules(m))
    return ref, m


# ---------------------------------------------------------------------------------------------------------------- timing
def mode_timing(args):
    res = {'mode': 'timing', 'tag': args.tag, 'pid': os.getpid(), 'gpu': bench.gpu_name(), 'torch': torch.__version__}
    # idle check BEFORE this process touches the GPU heavily
    res['other_compute_apps_start'] = compute_apps()
    idle = [smi('utilization.gpu,clocks.sm,power.draw') for _ in range(5)]
    res['idle_samples_before'] = idle
    log(f'[{args.tag}] other compute apps at start: {res["other_compute_apps_start"]} | idle samples {idle}')
    ok_ext, msg = pow2_int8.available()
    registered = pow2_int8.ensure_registered()
    res['kernel'] = dict(available=ok_ext, message=msg, registered=registered, mapped_before=mapped_p2_ext())
    if not (ok_ext and res['kernel']['mapped_before']):
        log('kernel not serving; abort'); return res
    torch.set_float32_matmul_precision('high')
    toks = val_tokens()

    _, van32 = load(VAN)
    x_van, _ = capture_ffn_inputs(van32, toks, [0])
    x_van = x_van[0]
    _, m46 = load(UNQ)
    x46, _ = capture_ffn_inputs(m46, toks, [0])
    x46 = x46[0]
    _, m46h = load(UNQ)
    m46h = m46h.to(torch.bfloat16)                        # a SEPARATE instance, converted before any forward
    _, m48 = load(QNT)
    x48, _ = capture_ffn_inputs(m48, toks, [0])
    x48 = x48[0]
    _, van16 = load(VAN)
    van16 = van16.to(torch.bfloat16)

    q_k = m48.blocks[0].ffn.export_quantised().cuda()     # kernel row
    q_t32 = m48.blocks[0].ffn.export_quantised().cuda()   # torch read, fp32
    q_t16 = m48.blocks[0].ffn.export_quantised().cuda()   # torch read, bf16 input (the only bf16 path the module has)
    q_t32._uses_kernel = lambda x: False

    def _forbidden(x):
        raise RuntimeError('torch fallback served in the kernel row')
    q_k._forward_torch = _forbidden

    f32_ref, f32 = fgeom_model('fp32')
    f16_ref, f16 = fgeom_model('bf16')

    ffn46, ffn46h, mlp32, mlp16 = m46.blocks[0].ffn, m46h.blocks[0].ffn, van32.blocks[0].mlp, van16.blocks[0].mlp
    xv32, xv16 = x_van.contiguous(), x_van.to(torch.bfloat16).contiguous()
    xu32, xu16 = x46.contiguous(), x46.to(torch.bfloat16).contiguous()
    xq32, xq16 = x48.contiguous(), x48.to(torch.bfloat16).contiguous()

    def lut_slot(mod, x):
        return lambda: mod(x.reshape(N, C)).reshape(B, SEQ, C)

    rows = {
        'V32  vanilla dense fp32': lambda: mlp32(xv32),
        'V16  vanilla dense bf16': lambda: mlp16(xv16),
        'U32  3.2+TV float read fp32 (torch.compile)': lut_slot(ffn46, xu32),
        'U16  3.2+TV float read bf16 (torch.compile)': lut_slot(ffn46h, xu16),
        'Q32  3.2+TV int8 CUDA kernel fp32': lut_slot(q_k, xq32),
        'Q32t 3.2+TV int8 torch read fp32 (torch.compile)': lut_slot(q_t32, xq32),
        'Q16t 3.2+TV int8 torch read bf16 (torch.compile)': lut_slot(q_t16, xq16),
        'F32  paper fused float kernel, this geometry, fp32 (1 cell, random init)': (lambda: f32.blocks[0].ffn_slot(xu32)),
        'F16  paper fused float kernel, this geometry, hybrid bf16 (1 cell, random init)': (lambda: f16.blocks[0].ffn_slot(xu16)),
    }
    # the torch reads must be pure torch (no p2_scalars CUDA op): disable AFTER every export (construction re-enables)
    pow2_int8.set_enabled(False)

    row_errors = {}
    for k in list(rows):
        try:
            with torch.no_grad():
                rows[k]()
            torch.cuda.synchronize()
        except Exception as e:
            row_errors[k] = f'{type(e).__name__}: {str(e).strip().splitlines()[0][:300] if str(e).strip() else ""}'
            del rows[k]
    res['row_errors'] = row_errors
    log(f'[{args.tag}] rows that could not run: {row_errors}')

    def rel(a, b, key):
        if a not in rows or b not in rows:
            return
        with torch.no_grad():
            ya, yb = rows[a]().float(), rows[b]().float()
        chk[key] = dict(max_abs=(ya - yb).abs().max().item(), rel=((ya - yb).abs().max() / yb.abs().max()).item())

    # ---- sanity / correctness of each row, before timing
    chk = {}
    with torch.no_grad():
        yq = rows['Q32  3.2+TV int8 CUDA kernel fp32']()
    rel('Q32t 3.2+TV int8 torch read fp32 (torch.compile)', 'Q32  3.2+TV int8 CUDA kernel fp32', 'Q32t_vs_Q32')
    rel('Q16t 3.2+TV int8 torch read bf16 (torch.compile)', 'Q32  3.2+TV int8 CUDA kernel fp32', 'Q16t_vs_Q32')
    rel('U16  3.2+TV float read bf16 (torch.compile)', 'U32  3.2+TV float read fp32 (torch.compile)', 'U16_vs_U32')
    rel('V16  vanilla dense bf16', 'V32  vanilla dense fp32', 'V16_vs_V32')
    with torch.no_grad():
        yf = f32_ref.blocks[0].ffn_slot(xu32).float()
        for k, key in (('F32  paper fused float kernel, this geometry, fp32 (1 cell, random init)', 'F32_vs_unpatched_fp32'),
                       ('F16  paper fused float kernel, this geometry, hybrid bf16 (1 cell, random init)', 'F16_vs_unpatched_fp32')):
            if k in rows:
                yk = rows[k]().float()
                chk[key] = dict(max_abs=(yk - yf).abs().max().item(), rel=((yk - yf).abs().max() / yf.abs().max()).item())
    # the kernel row equals the training read? (training read needs the op: re-enable for this check only)
    pow2_int8.set_enabled(True)
    with torch.no_grad():
        ytr = m48.blocks[0].ffn(xq32.reshape(N, C)).reshape(B, SEQ, C)
    pow2_int8.set_enabled(False)
    chk['Q32_vs_training_read_max_abs'] = (yq - ytr).abs().max().item()
    # fused-hit counters for the F rows (wrapper installed for one call, then removed)
    for name, mdl in (('F32', f32), ('F16', f16)):
        from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
        mods = [mm for mm in mdl.modules() if isinstance(mm, FastMultiHeadLut)]
        cnt = {'n': 0}
        saved = [mm._hard_eval_native for mm in mods]
        for mm, o in zip(mods, saved):
            mm._hard_eval_native = (lambda x, w, _o=o: (cnt.__setitem__('n', cnt['n'] + 1), _o(x, w))[1])
        with torch.no_grad():
            mdl.blocks[0].ffn_slot(xu32 if name == 'F32' else xu16)
        for mm, o in zip(mods, saved):
            mm._hard_eval_native = o
        chk[f'{name}_fused_hits_per_slot_call'] = cnt['n']
    res['row_checks'] = chk
    log(f'[{args.tag}] row checks: {json.dumps(chk)}')

    # ---- warm-up: the paper's 300-call global warm-up on V16, then 60 per row (includes torch.compile)
    with torch.no_grad():
        for _ in range(300):
            rows['V16  vanilla dense bf16']()
        for f in rows.values():
            for _ in range(60):
                f()
    torch.cuda.synchronize()
    tele = Telemetry()
    t_start = time.time()

    # ---- protocol A (phase_split.py): per row, 60-call burn-in then median of 7 x timeit(30)
    A = {}
    with torch.no_grad():
        for k, f in rows.items():
            for _ in range(60):
                f()
            torch.cuda.synchronize()
            A[k] = statistics.median([bench.timeit(f, iters=30) for _ in range(7)])
    # ---- interleaved rounds, three timing paths
    R = args.rounds
    acc = {k: {'P1': [], 'P2': [], 'P3': []} for k in rows}
    with torch.no_grad():
        for _ in range(R):
            for k, f in rows.items():
                acc[k]['P1'].append(bench.timeit(f, iters=30, warmup=2))
                acc[k]['P2'].append(timeit_events(f, 30))
                acc[k]['P3'].append(timeit_wall_sync(f, 30))
    res['timing_seconds'] = time.time() - t_start
    res['telemetry'] = tele.stop()
    res['other_compute_apps_end'] = compute_apps()
    res['kernel']['mapped_after'] = mapped_p2_ext()
    res['kernel']['torch_fallback_compiled_in_kernel_row'] = q_k._compiled is not None
    res['protocol_A'] = A
    res['interleaved'] = {k: {p: dict(median=statistics.median(v), min=min(v), max=max(v)) for p, v in d.items()}
                          for k, d in acc.items()}
    log(f'\n[{args.tag}] {"row":<82}{"A":>9}{"P1":>9}{"P2 ev":>9}{"P3 wall":>9}')
    for k in rows:
        i = res['interleaved'][k]
        log(f'[{args.tag}] {k:<82}{A[k]:>9.4f}{i["P1"]["median"]:>9.4f}{i["P2"]["median"]:>9.4f}{i["P3"]["median"]:>9.4f}')
    log(f'[{args.tag}] telemetry {json.dumps(res["telemetry"])}')
    log(f'[{args.tag}] other compute apps at end: {res["other_compute_apps_end"]} | mapped {res["kernel"]["mapped_after"]}'
        f' | fallback compiled in kernel row: {res["kernel"]["torch_fallback_compiled_in_kernel_row"]}')
    return res


# ---------------------------------------------------------------------------------------------------------------- glue
def mode_glue(args):
    """Rebuild the OLD harness's slot and phase definitions verbatim and decompose its 'other'."""
    res = {'mode': 'glue'}
    pow2_int8.available()
    pow2_int8.ensure_registered()
    torch.set_float32_matmul_precision('high')
    # one model family per PROCESS (--part): compiling the float light read in a process that also holds the quant
    # model aborts inside dynamo ("failed to add version watcher to dict") on this torch build
    torch.manual_seed(0)
    xb = torch.randn(B, SEQ, C, device='cuda', dtype=torch.bfloat16)   # the OLD input: randn bf16
    xf = xb.reshape(N, C).float()
    y32 = torch.randn(B, SEQ, C, device='cuda')
    res['part'] = args.part
    if args.part == 'u':
        _, m46 = load(UNQ)
        ffn46 = m46.blocks[0].ffn
        lut = ffn46.lut_light
        with torch.no_grad():
            z46 = ffn46.compress(xf).view(N, lut.n_heads, lut.input_dim)
            y46 = ffn46.lut_light(z46).to(z46.dtype).reshape(N, -1)
        items = {
            'OLD slot_unq (cast in + ffn46 + reshape + cast out)': lambda: ffn46(xb.reshape(N, C).float()).reshape(B, SEQ, C).to(xb.dtype),
            'cast in  bf16->fp32 [N,C]': lambda: xb.reshape(N, C).float(),
            'cast out fp32->bf16 [B,T,C]': lambda: y32.to(torch.bfloat16),
            'ffn46(xf) module, no casts': lambda: ffn46(xf),
            'OLD phase u compress module': lambda: ffn46.compress(xf),
            'OLD phase u read lut_light(z)': lambda: ffn46.lut_light(z46),
            'OLD phase u decompress module': lambda: ffn46.decompress(y46),
        }
        return _glue_time(res, items, lambda med: med['OLD slot_unq (cast in + ffn46 + reshape + cast out)'] - (
            med['OLD phase u compress module'] + med['OLD phase u read lut_light(z)'] + med['OLD phase u decompress module']))
    toks = val_tokens()
    _, m48 = load(QNT)
    q0 = m48.blocks[0].ffn.export_quantised().cuda()
    xreal = capture_ffn_inputs(m48, toks, [0])[0][0].reshape(N, C).contiguous()
    mt = q0.meta
    kc = q0._kernel_cache(xf.device)
    ext = pow2_int8.load()
    with torch.no_grad():
        zq = F.linear(xf, q0.compress_weight, q0.compress_bias).view(N, mt['n_heads'], mt['input_dim']).contiguous()
        accq = q0._forward_fused(xf)                      # warm
        accq = pow2_int8.read_fused(zq, kc['anchor_a'], kc['anchor_b'], kc['tables'], kc['scalars'], mt['n_anchor_pairs'],
                                    mt['output_dim'], q0.cfg['lo'], q0.cfg['hi'], q0.cfg['Q']).reshape(N, -1)
    e_u8 = torch.empty(0, device='cuda', dtype=torch.uint8)
    H, T, NAP, Din, D = mt['n_heads'], mt['tables_per_head'], mt['n_anchor_pairs'], mt['input_dim'], mt['output_dim']
    sc = kc['scalars']

    def raw_ext():
        return ext.read(zq, kc['anchor_a'], kc['anchor_b'], e_u8, kc['tables'], N, H, T, NAP, 1 << NAP, Din, D,
                        q0.cfg['lo'], q0.cfg['hi'], q0.cfg['Q'], *sc, pow2_int8.DEFAULT_BLOCK_N, True, True)

    def q_inline():
        z = F.linear(xf, q0.compress_weight, q0.compress_bias).view(N, H, Din)
        a = ext.read(z, kc['anchor_a'], kc['anchor_b'], e_u8, kc['tables'], N, H, T, NAP, 1 << NAP, Din, D,
                     q0.cfg['lo'], q0.cfg['hi'], q0.cfg['Q'], *sc, pow2_int8.DEFAULT_BLOCK_N, True, True)
        return F.linear(a.reshape(N, H * D), q0.decompress_weight, q0.decompress_bias)

    items = {
        'OLD slot_q  (cast in + q0 + reshape + cast out)': lambda: q0(xb.reshape(N, C).float()).reshape(B, SEQ, C).to(xb.dtype),
        'cast in  bf16->fp32 [N,C]': lambda: xb.reshape(N, C).float(),
        'cast out fp32->bf16 [B,T,C]': lambda: y32.to(torch.bfloat16),
        'q0(xf) module, no casts': lambda: q0(xf),
        'q inline: linear + ext.read + linear (no module wrapper)': q_inline,
        'OLD phase q compress F.linear': lambda: F.linear(xf, q0.compress_weight, q0.compress_bias),
        'OLD phase q read (pow2_int8.read_fused wrapper)': lambda: pow2_int8.read_fused(
            zq, kc['anchor_a'], kc['anchor_b'], kc['tables'], kc['scalars'], NAP, D, q0.cfg['lo'], q0.cfg['hi'], q0.cfg['Q']),
        'q read raw ext.read (no python wrapper)': raw_ext,
        'OLD phase q decompress F.linear': lambda: F.linear(accq, q0.decompress_weight, q0.decompress_bias),
        'q0(xreal) module on REAL input': lambda: q0(xreal),
        'q0 wrapper checks only: _uses_kernel + _kernel_cache': lambda: (q0._uses_kernel(xf), q0._kernel_cache(xf.device)),
    }
    return _glue_time(res, items, lambda med: med['OLD slot_q  (cast in + q0 + reshape + cast out)'] - (
        med['OLD phase q compress F.linear'] + med['OLD phase q read (pow2_int8.read_fused wrapper)'] + med['OLD phase q decompress F.linear']),
        raw_ext=raw_ext, lin=lambda: F.linear(accq, q0.decompress_weight, q0.decompress_bias))


def _glue_time(res, items, other_fn, raw_ext=None, lin=None):
    with torch.no_grad():
        for f in items.values():
            for _ in range(60):
                f()
    torch.cuda.synchronize()
    if raw_ext is not None:                           # is ext.read host-synchronous? (returns only after the device finished)
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter(); raw_ext(); t1 = time.perf_counter(); torch.cuda.synchronize(); t2 = time.perf_counter()
            t3 = time.perf_counter(); lin(); t4 = time.perf_counter()
            torch.cuda.synchronize(); t5 = time.perf_counter()
        res['ext_read_host_return_ms'] = (t1 - t0) * 1e3
        res['ext_read_residual_after_sync_ms'] = (t2 - t1) * 1e3
        res['linear_host_return_ms'] = (t4 - t3) * 1e3
        res['linear_residual_after_sync_ms'] = (t5 - t4) * 1e3
    acc = {k: [] for k in items}
    with torch.no_grad():
        for _ in range(11):
            for k, f in items.items():
                acc[k].append(statistics.median([bench.timeit(f, iters=30, warmup=2) for _ in range(3)]))
    med = {k: statistics.median(v) for k, v in acc.items()}
    res['median_ms'] = med
    for k, v in med.items():
        log(f'  {k:<62} {v:.4f} ms')
    res['old_other'] = other_fn(med)
    log(f'  OLD other (re-measured, part {res["part"]}): {res["old_other"]:+.4f}')
    if raw_ext is not None:
        log(f'  ext.read host return {res["ext_read_host_return_ms"]:.4f} ms, residual after sync {res["ext_read_residual_after_sync_ms"]:.4f} ms;'
            f' F.linear host return {res["linear_host_return_ms"]:.4f} ms, residual {res["linear_residual_after_sync_ms"]:.4f} ms')
    return res


# ---------------------------------------------------------------------------------------------------------------- tune
def mode_tune(args):
    res = {'mode': 'tune'}
    pow2_int8.available()
    pow2_int8.ensure_registered()
    torch.set_float32_matmul_precision('high')
    _, m48 = load(QNT)
    q0 = m48.blocks[0].ffn.export_quantised().cuda()
    toks = val_tokens()
    xf = capture_ffn_inputs(m48, toks, [0])[0][0].reshape(N, C).contiguous()
    mt = q0.meta
    kc = q0._kernel_cache(xf.device)
    H, NAP, Din, D = mt['n_heads'], mt['n_anchor_pairs'], mt['input_dim'], mt['output_dim']
    with torch.no_grad():
        z = F.linear(xf, q0.compress_weight, q0.compress_bias).view(N, H, Din).contiguous()
        ref = pow2_int8.read_fused(z, kc['anchor_a'], kc['anchor_b'], kc['tables'], kc['scalars'], NAP, D,
                                   q0.cfg['lo'], q0.cfg['hi'], q0.cfg['Q'])

    def mk(bn, l16, slot):
        def read():
            return pow2_int8.read_fused(z, kc['anchor_a'], kc['anchor_b'], kc['tables'], kc['scalars'], NAP, D,
                                        q0.cfg['lo'], q0.cfg['hi'], q0.cfg['Q'], block_n=bn, load16=l16)

        def full():
            zz = F.linear(xf, q0.compress_weight, q0.compress_bias).view(N, H, Din)
            a = pow2_int8.read_fused(zz, kc['anchor_a'], kc['anchor_b'], kc['tables'], kc['scalars'], NAP, D,
                                     q0.cfg['lo'], q0.cfg['hi'], q0.cfg['Q'], block_n=bn, load16=l16)
            return F.linear(a.reshape(N, H * D), q0.decompress_weight, q0.decompress_bias)
        return full if slot else read

    cfgs = [(bn, l16) for bn in pow2_int8.BLOCK_NS for l16 in (True, False)]
    exact = {}
    with torch.no_grad():
        for bn, l16 in cfgs:
            try:
                exact[f'{bn}/{l16}'] = bool(torch.equal(mk(bn, l16, False)(), ref))
            except Exception as e:
                exact[f'{bn}/{l16}'] = f'error: {e}'
    res['bit_exact_vs_default'] = exact
    log(f'bit-exact vs default: {exact}')
    good = [(bn, l16) for bn, l16 in cfgs if exact[f'{bn}/{l16}'] is True]
    fns = {f'read {bn}/{l16}': mk(bn, l16, False) for bn, l16 in good}
    fns.update({f'slot {bn}/{l16}': mk(bn, l16, True) for bn, l16 in good})
    fns['slot module q0 (defaults)'] = lambda: q0(xf)
    with torch.no_grad():
        for f in fns.values():
            for _ in range(60):
                f()
    torch.cuda.synchronize()
    acc = {k: {'P1': [], 'P2': []} for k in fns}
    with torch.no_grad():
        for _ in range(11):
            for k, f in fns.items():
                acc[k]['P1'].append(bench.timeit(f, iters=30, warmup=2))
                acc[k]['P2'].append(timeit_events(f, 30))
    res['median_ms'] = {k: {p: statistics.median(v) for p, v in d.items()} for k, d in acc.items()}
    for k, d in res['median_ms'].items():
        log(f'  {k:<30} P1 {d["P1"]:.4f}  P2 {d["P2"]:.4f}')
    return res


# ---------------------------------------------------------------------------------------------------------------- sweep
def mode_sweep(args):
    """Memory- vs compute-bound: vary table bytes at FIXED reads (NAP), reads at fixed bytes-per-table (T), tokens (N).
    Real layer-0 codes and trained scalars, so skip / drop rates are the real ones; int8 table VALUES are random (the read
    does the same work whatever the bytes are). Rows actually read are counted from the kernel's own cells."""
    res = {'mode': 'sweep'}
    pow2_int8.available()
    pow2_int8.ensure_registered()
    torch.set_float32_matmul_precision('high')
    _, m48 = load(QNT)
    q0 = m48.blocks[0].ffn.export_quantised().cuda()
    toks = val_tokens()
    xf = capture_ffn_inputs(m48, toks, [0])[0][0].reshape(N, C).contiguous()
    mt = q0.meta
    H, T0, NAP0, Din, D = mt['n_heads'], mt['tables_per_head'], mt['n_anchor_pairs'], mt['input_dim'], mt['output_dim']
    kc = q0._kernel_cache(xf.device)
    with torch.no_grad():
        z_all = F.linear(xf, q0.compress_weight, q0.compress_bias).view(N, H, Din).contiguous()
    A0 = kc['anchor_a'].view(H, T0, NAP0)
    B0 = kc['anchor_b'].view(H, T0, NAP0)
    g = torch.Generator('cuda').manual_seed(0)

    def setup(nap, T, ntok):
        if T <= T0:
            aa, ab = A0[:, :T, :nap], B0[:, :T, :nap]
        else:
            rep = (T + T0 - 1) // T0
            aa, ab = A0.repeat(1, rep, 1)[:, :T, :nap], B0.repeat(1, rep, 1)[:, :T, :nap]
        aa, ab = aa.contiguous(), ab.contiguous()
        tables = torch.randint(-128, 128, (H * T * (1 << nap), D), dtype=torch.int8, device='cuda', generator=g)
        tables = pow2_int8.stride_tables(tables, D)
        reps = (ntok + N - 1) // N
        z = z_all.repeat(reps, 1, 1)[:ntok].contiguous()
        cells = torch.empty(ntok, H, T, 3, dtype=torch.uint8, device='cuda')
        with torch.no_grad():
            pow2_int8.read_fused(z, aa, ab, tables, kc['scalars'], nap, D, q0.cfg['lo'], q0.cfg['hi'], q0.cfg['Q'], cells_out=cells)
        sh = cells[..., 2]
        rows_read = int(((sh & 15) != pow2_int8.DISCARD).sum()) + int(((sh >> 4) != pow2_int8.DISCARD).sum())
        del cells
        fn = lambda: pow2_int8.read_fused(z, aa, ab, tables, kc['scalars'], nap, D, q0.cfg['lo'], q0.cfg['hi'], q0.cfg['Q'])
        return fn, rows_read, tables.numel()

    points = [('NAP', nap, 128, N) for nap in (4, 5, 6, 7, 8)] + \
             [('T', 8, T, N) for T in (16, 32, 64, 96, 128, 192, 256)] + \
             [('N', 8, 128, n) for n in (3072, 6144, 12288, 24576, 49152, 98304)]
    out = []
    for kind, nap, T, ntok in points:
        fn, rows_read, tbytes = setup(nap, T, ntok)
        with torch.no_grad():
            for _ in range(40):
                fn()
            torch.cuda.synchronize()
            t = statistics.median([bench.timeit(fn, iters=20, warmup=2) for _ in range(7)])
            te = statistics.median([timeit_events(fn, 20) for _ in range(3)])
        reads = ntok * H * T
        r = dict(kind=kind, nap=nap, T=T, tokens=ntok, table_MB=tbytes / 1e6, ms=t, ms_events=te,
                 table_reads=reads, rows_read=rows_read, ns_per_row_read=t * 1e6 / max(rows_read, 1),
                 ns_per_table_read=t * 1e6 / reads, GB_per_s_rows=rows_read * D / 1e9 / (t / 1e3))
        out.append(r)
        log(f'  {kind:<3} nap {nap} T {T:>3} tokens {ntok:>6} table {r["table_MB"]:>7.2f} MB | {t:.4f} ms (ev {te:.4f}) | '
            f'rows read {rows_read:>10} ({rows_read / reads / 2 * 100:4.1f}% of 2/table) | {r["ns_per_row_read"]:.3f} ns/row | '
            f'{r["GB_per_s_rows"]:.1f} GB/s row bytes')
        del fn
        torch.cuda.empty_cache()
    res['points'] = out
    # reference bandwidths measured on this device
    refs = {}
    with torch.no_grad():
        src = torch.randint(-128, 128, (1 << 30,), dtype=torch.int8, device='cuda', generator=g)
        dst = torch.empty_like(src)
        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize()
        t = statistics.median([bench.timeit(lambda: dst.copy_(src), iters=5, warmup=1) for _ in range(5)])
        refs['sequential_copy_1GiB_GB_per_s'] = (1 << 30) / 1e9 / (t / 1e3)
        del src, dst
        for tbl_mb in (6, 256, 4096):
            rows = tbl_mb * 10**6 // D
            W = torch.randint(-128, 128, (rows, D), dtype=torch.int8, device='cuda', generator=g)
            idx = torch.randint(0, rows, (N * H * 64,), device='cuda', generator=g)
            for _ in range(3):
                W.index_select(0, idx)
            torch.cuda.synchronize()
            t = statistics.median([bench.timeit(lambda: W.index_select(0, idx), iters=10, warmup=2) for _ in range(5)])
            refs[f'random_row_gather_from_{tbl_mb}MB_GB_per_s'] = idx.numel() * D / 1e9 / (t / 1e3)
            del W, idx
            torch.cuda.empty_cache()
    res['reference_bandwidth'] = refs
    log(f'  reference bandwidth: {json.dumps(refs)}')
    return res


# ---------------------------------------------------------------------------------------------------------------- verify
def mode_verify(args):
    res = {'mode': 'verify'}
    ok_ext, msg = pow2_int8.available()
    registered = pow2_int8.ensure_registered()
    res['kernel'] = dict(available=ok_ext, message=msg, registered=registered, mapped=mapped_p2_ext())
    log(f'kernel: {msg} | registered {registered} | mapped {res["kernel"]["mapped"]}')
    torch.set_float32_matmul_precision('high')

    # ---- G10: table memory, measured on the actual tensors and on the allocator
    _, m46 = load(UNQ)
    _, m48 = load(QNT)
    L = len(m48.blocks)
    un = [b.ffn.lut_light.tables for b in m46.blocks]
    res['unquantised_tables'] = dict(dtype=str(un[0].dtype), shape=list(un[0].shape),
                                     bytes_numel_x_elsize=sum(t.numel() * t.element_size() for t in un),
                                     bytes_untyped_storage=sum(t.untyped_storage().nbytes() for t in un))
    torch.cuda.synchronize()
    a0 = torch.cuda.memory_allocated()
    qs = [b.ffn.export_quantised().cuda() for b in m48.blocks]
    torch.cuda.synchronize()
    a1 = torch.cuda.memory_allocated()
    qt = [q.tables for q in qs]
    kt = [q._kernel_cache(torch.device('cuda'))['tables'] for q in qs]
    res['int8_tables'] = dict(dtype=str(qt[0].dtype), shape=list(qt[0].shape),
                              bytes_numel_x_elsize=sum(t.numel() * t.element_size() for t in qt),
                              bytes_untyped_storage=sum(t.untyped_storage().nbytes() for t in qt),
                              kernel_view_shares_storage=all(a.data_ptr() == b.data_ptr() for a, b in zip(qt, kt)),
                              all_artefact_buffers_bytes=sum(sum(getattr(q, k).untyped_storage().nbytes() for k in
                                                                 ('compress_weight', 'compress_bias', 'anchor_a', 'anchor_b', 'powers', 'table_offset',
                                                                  'tables', 'tau', 'g', 'beta', 'gamma', 'decompress_weight', 'decompress_bias')) for q in qs),
                              allocator_delta_for_6_exports=a1 - a0)
    t0 = torch.cuda.memory_allocated()
    clone_un = [t.detach().clone() for t in un]
    torch.cuda.synchronize()
    t1 = torch.cuda.memory_allocated()
    clone_q = [t.detach().clone() for t in qt]
    torch.cuda.synchronize()
    t2 = torch.cuda.memory_allocated()
    res['allocator_delta_clone_unquantised_tables'] = t1 - t0
    res['allocator_delta_clone_int8_tables'] = t2 - t1
    del clone_un, clone_q
    log(f'tables: {json.dumps({k: res[k] for k in ("unquantised_tables", "int8_tables", "allocator_delta_clone_unquantised_tables", "allocator_delta_clone_int8_tables")})}')

    # ---- G11: correctness gate on REAL val text, every layer; independent fp64 re-derivation of the integers
    toks = val_tokens()
    xs, logits_train = capture_ffn_inputs(m48, toks)
    per = []
    LN2 = torch.log(torch.tensor(2.0, dtype=torch.float64, device='cuda'))
    for i in range(L):
        q = qs[i]
        x = xs[i].reshape(N, C).contiguous()
        mt = q.meta
        H, T, NAP, Din, D = mt['n_heads'], mt['tables_per_head'], mt['n_anchor_pairs'], mt['input_dim'], mt['output_dim']
        with torch.no_grad():
            assert q._uses_kernel(x)
            y_k = q(x)
            y_tr = m48.blocks[i].ffn(x)
            kc = q._kernel_cache(x.device)
            z = F.linear(x, q.compress_weight, q.compress_bias).view(N, H, Din).contiguous()
            cells_k = torch.empty(N, H, T, 3, dtype=torch.uint8, device='cuda')
            pow2_int8.read_fused(z, kc['anchor_a'], kc['anchor_b'], kc['tables'], kc['scalars'], NAP, D, q.cfg['lo'],
                                 q.cfg['hi'], q.cfg['Q'], cells_out=cells_k)
            pow2_int8.set_enabled(False)
            cells_t = q._reference_cells(x)
            pow2_int8.set_enabled(True)
            # fp64, from the formula in csrc/pow2_scalars.cuh, written here without pow2_read
            tau, g_, beta, gamma = (t.double() for t in (q.tau, q.g, q.beta, q.gamma))
            n64 = dict(fields=[0, 0, 0], k_or_q_or_skip=0)
            mism_kt = (cells_k != cells_t).any(-1)
            dist_worst_kt = 0.0
            mism_k64 = 0
            mism_t64 = 0
            dist_worst_k64 = 0.0
            chunk = 4096
            aa = q.anchor_a.view(H, T, NAP).long()
            ab = q.anchor_b.view(H, T, NAP).long()
            for s in range(0, N, chunk):
                zc = z[s:s + chunk].double()
                n = zc.shape[0]
                ia = aa.reshape(1, H, T * NAP).expand(n, H, T * NAP)
                ib = ab.reshape(1, H, T * NAP).expand(n, H, T * NAP)
                d = (torch.gather(zc, 2, ia) - torch.gather(zc, 2, ib)).view(n, H, T, NAP)
                bits = (d > 0).long()
                pw = (2 ** torch.arange(NAP - 1, -1, -1, device='cuda')).long()
                c1 = (bits * pw).sum(-1)
                m = d.abs()
                mv, mj = m.min(-1)
                c2 = c1 + pw[mj] * (1 - 2 * torch.gather(bits, -1, mj.unsqueeze(-1)).squeeze(-1))
                qpre = mv * 2 / (tau * LN2) + 0.5
                qq = torch.clamp(torch.floor(qpre), 0, 64)
                cq = torch.where(qq < 8, torch.log2(1 + torch.pow(2.0, -qq)), torch.zeros_like(qq))
                S = m.sum(-1)
                ls = F.logsigmoid(beta * m).sum(-1)
                kpre = torch.log2(S) + (g_ + gamma * ls) / LN2 - cq + 0.5
                kr = torch.floor(kpre)
                skip = ~(kr >= q.cfg['lo'])
                kcl = torch.clamp(kr, q.cfg['lo'], q.cfg['hi'])
                drop = qq > q.cfg['Q']
                # shift codes as pack_cells builds them
                grp = pow2_read.shift_groups(qq.float(), kcl.float(), skip, drop)
                sh = torch.where(grp == pow2_read.N_SHIFTS, torch.full_like(grp, pow2_int8.DISCARD), grp)
                c64 = torch.stack([c1, c2, sh[..., 0] | (sh[..., 1] << 4)], -1).to(torch.uint8)
                ck, ct = cells_k[s:s + n], cells_t[s:s + n]
                mk64 = (ck != c64).any(-1)
                mt64 = (ct != c64).any(-1)
                mism_k64 += int(mk64.sum())
                mism_t64 += int(mt64.sum())
                fk = kpre - torch.floor(kpre)
                fq = qpre - torch.floor(qpre)
                dist = torch.minimum(torch.minimum(fk, 1 - fk), torch.minimum(fq, 1 - fq))
                mkt = mism_kt[s:s + n]
                if mkt.any():
                    dist_worst_kt = max(dist_worst_kt, dist[mkt].max().item())
                if mk64.any():
                    dist_worst_k64 = max(dist_worst_k64, dist[mk64].max().item())
                    for a_, h_, t_ in (mk64 & (dist > 1e-5)).nonzero().tolist()[:5]:
                        dd = d[a_, h_, t_]
                        srt = dd.abs().sort().values
                        log(f'    NON-TIE layer {i} tok {s + a_} head {h_} table {t_}: kernel {ck[a_, h_, t_].tolist()} '
                            f'torch {ct[a_, h_, t_].tolist()} fp64 {c64[a_, h_, t_].tolist()} | d64 {[f"{v:.3e}" for v in dd.tolist()]} '
                            f'| min|d| {srt[0].item():.3e} 2nd {srt[1].item():.3e} | S {S[a_, h_, t_].item():.6e} '
                            f'kpre {kpre[a_, h_, t_].item():.9f} qpre {qpre[a_, h_, t_].item():.9f} | fp32 z entries equal-anchor zero-margins '
                            f'{int((dd == 0).sum())}')
        rec = dict(layer=i, table_reads=mism_kt.numel(), kernel_vs_training_read_max_abs=(y_k - y_tr).abs().max().item(),
                   kernel_vs_pure_torch_integer_mismatches=int(mism_kt.sum()), kernel_vs_fp64_mismatches=mism_k64,
                   pure_torch_vs_fp64_mismatches=mism_t64,
                   worst_fp64_distance_from_rounding_point_kernel_vs_torch=dist_worst_kt,
                   worst_fp64_distance_from_rounding_point_kernel_vs_fp64=dist_worst_k64)
        per.append(rec)
        log(f'  layer {i}: {json.dumps(rec)}')
        del cells_k, cells_t
        torch.cuda.empty_cache()
    res['correctness_real_text'] = per

    class QSlot(nn.Module):
        def __init__(self, q):
            super().__init__()
            self.q = q

        def forward(self, x):
            return self.q(x)
    saved = [b.ffn for b in m48.blocks]
    for i in range(L):
        m48.blocks[i].ffn = QSlot(qs[i])
    with torch.no_grad():
        logits_q = m48(toks)
    for i in range(L):
        m48.blocks[i].ffn = saved[i]
    tgt = torch.cat([toks[:, 1:], toks[:, :1]], 1)[:, :-1]
    res['full_model_real_text'] = dict(max_abs_logit_diff=(logits_q - logits_train).abs().max().item(),
                                       ce_training_read=F.cross_entropy(logits_train[:, :-1].reshape(-1, logits_train.shape[-1]), toks[:, 1:].reshape(-1)).item(),
                                       ce_artefact=F.cross_entropy(logits_q[:, :-1].reshape(-1, logits_q.shape[-1]), toks[:, 1:].reshape(-1)).item())
    log(f'  full model on real text: {json.dumps(res["full_model_real_text"])}')

    # ---- G13: which path serves: profiler over kernel-row calls + counters
    q = qs[0]
    x = xs[0].reshape(N, C).contiguous()
    calls = {'fused': 0, 'torch': 0}
    of, ot = q._forward_fused, q._forward_torch
    q._forward_fused = lambda xx: (calls.__setitem__('fused', calls['fused'] + 1), of(xx))[1]
    q._forward_torch = lambda xx: (calls.__setitem__('torch', calls['torch'] + 1), ot(xx))[1]
    with torch.no_grad():
        for _ in range(20):
            q(x)
    q._forward_fused, q._forward_torch = of, ot
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            q(x)
        torch.cuda.synchronize()
    names = sorted({e.key for e in prof.key_averages()})
    res['path'] = dict(calls_20=calls, profiler_op_names=names, mapped_after=mapped_p2_ext(),
                       torch_fallback_compiled=q._compiled is not None)
    log(f'  path: {json.dumps(res["path"])}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('timing', 'glue', 'tune', 'sweep', 'verify'))
    ap.add_argument('--tag', default='run')
    ap.add_argument('--rounds', type=int, default=11)
    ap.add_argument('--part', default='q', choices=('q', 'u'))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    log(f'GPU {bench.gpu_name()} | torch {torch.__version__} | mode {args.mode} | pid {os.getpid()}')
    res = globals()[f'mode_{args.mode}'](args)
    path = os.path.join(OUT_DIR, f'{args.mode}_{args.tag}.json')
    json.dump(res, open(path, 'w'), indent=2, default=str)
    log(f'wrote verify_out/{os.path.basename(path)}')


if __name__ == '__main__':
    main()
