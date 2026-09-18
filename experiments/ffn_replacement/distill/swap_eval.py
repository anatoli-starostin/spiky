"""Swap distilled FFN students into the frozen teacher and measure val bpb (fixed protocol).

Teacher: exp_n_0151 (dense, model_build MinimalGPT, via distill_ffn.load_teacher). Students: the per-layer
CompressionMultiHeadLUT students a distill_ffn run saved (--save-students), rebuilt with build_student from the
run's base config + manifest student_overrides. Swapping block i replaces its FFN sub-layer mlp(ln2(x)) -- the
exact map the student was distilled on -- while attention, norms and the head stay the teacher's:

    blocks[i].ffn = student; blocks[i].ffn_type = 'compression'; blocks[i].lin = None

(dense blocks have no `lin`, which the compression branch of MinimalBlock.forward reads). Scoring is
tools/fixed_eval.evaluate_bpb_fixed: bs48 x 100, skip 12, held-out val shard, fp32, no autocast, no_grad.
Every eval restores the teacher afterwards; the teacher baseline is re-measured in the same process.

    python swap_eval.py --noop-check               # teacher mlp wrapped as the "student" -> must equal baseline
    python swap_eval.py <run> [<run> ...]          # singles, cumulative ladder 1..6, all-6 -> runs/<run>/swap_eval.json
"""
import argparse, json, os, sys, time
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import distill_ffn as D  # noqa: E402  (puts tools/ and nanochat on sys.path)
from nanochat.tokenizer import get_token_bytes  # noqa: E402
from fixed_eval import evaluate_bpb_fixed, eval_config  # noqa: E402

RECORDED_TEACHER_BPB = 1.1154196032137111   # runs_corrected/exp_n_0151_long48k_untied_vanilla/corrected_score.json


class MlpAsStudent(nn.Module):
    """No-op 'student': the teacher block's own mlp, called the way a student is ([N, C] -> [N, C])."""

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, h):
        return self.mlp(h)


def swap(model, students):
    """Swap {layer: module} into the model; return a restore() closure."""
    saved = {}
    for li, s in students.items():
        b = model.blocks[li]
        saved[li] = (b.ffn_type, getattr(b, 'ffn', None), 'ffn' in b._modules, getattr(b, 'lin', '__absent__'))
        b.ffn, b.ffn_type, b.lin = s, 'compression', None

    def restore():
        for li, (ft, ffn, had_ffn, lin) in saved.items():
            b = model.blocks[li]
            b.ffn_type = ft
            if had_ffn:
                b.ffn = ffn
            else:
                del b._modules['ffn']
            if lin == '__absent__':
                del b.lin
            else:
                b.lin = lin
    return restore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--noop-check', action='store_true')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = a.device
    tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
    teacher, tcfg = D.load_teacher(D.DEF_TEACHER, tok.get_vocab_size(), dev)
    teacher.eval()
    token_bytes = get_token_bytes(device=dev)
    ec = eval_config(tcfg)

    def bpb():
        t0 = time.time()
        with torch.no_grad():
            v = evaluate_bpb_fixed(teacher, tok, token_bytes, tcfg['seq_len'], dev, **ec)
        return v, time.time() - t0

    base, dt = bpb()
    print(f'teacher bpb {base!r} ({dt:.1f}s) | recorded {RECORDED_TEACHER_BPB!r} | '
          f'{"EXACT" if base == RECORDED_TEACHER_BPB else "DIFF %+.3e" % (base - RECORDED_TEACHER_BPB)}', flush=True)
    if a.noop_check:
        restore = swap(teacher, {li: MlpAsStudent(teacher.blocks[li].mlp) for li in range(6)})
        v, dt = bpb()
        restore()
        again, _ = bpb()
        print(f'no-op swap of all 6 blocks: bpb {v!r} ({dt:.1f}s) | '
              f'{"EXACT" if v == RECORDED_TEACHER_BPB else "DIFF %+.3e" % (v - RECORDED_TEACHER_BPB)} vs recorded, '
              f'{"EXACT" if v == base else "DIFF %+.3e" % (v - base)} vs in-process | after restore {again!r}',
              flush=True)
        return
    base_cfg = json.load(open(D.DEF_STUDENT))
    for run in a.runs:
        rd = os.path.join(HERE, 'runs', run)
        ov = json.load(open(os.path.join(rd, 'manifest.json')))['student_overrides']
        fvu = json.load(open(os.path.join(rd, 'results.json')))['final']
        students = {}
        for li in range(6):
            s = D.build_student({**base_cfg, **ov}, tcfg['n_embd'], tcfg['n_head'], li, dev)
            s.load_state_dict(torch.load(os.path.join(rd, f'student_L{li}.pt'), map_location=dev))
            students[li] = s.eval()
        out = {'run': run, 'teacher_bpb': base, 'recorded_teacher_bpb': RECORDED_TEACHER_BPB,
               'protocol': ec, 'single': {}, 'cumulative': {}}
        for li in range(6):
            restore = swap(teacher, {li: students[li]})
            v, dt = bpb(); restore()
            out['single'][li] = {'bpb': v, 'delta': v - base, 'fvu': fvu[str(li)]['fvu']}
            print(f'{run} single L{li}: bpb {v:.6f} delta {v - base:+.6f} (FVU {fvu[str(li)]["fvu"]:.4f}, {dt:.1f}s)',
                  flush=True)
        for k in range(1, 7):
            restore = swap(teacher, {li: students[li] for li in range(k)})
            v, dt = bpb(); restore()
            out['cumulative'][k] = {'layers': list(range(k)), 'bpb': v, 'delta': v - base}
            print(f'{run} cumulative L0..L{k - 1}: bpb {v:.6f} delta {v - base:+.6f} ({dt:.1f}s)', flush=True)
        out['all6'] = out['cumulative'][6]
        check, _ = bpb()
        out['teacher_bpb_after'] = check
        json.dump(out, open(os.path.join(rd, 'swap_eval.json'), 'w'), indent=1)
        print(f'{run}: all-6 delta {out["all6"]["delta"]:+.6f} | teacher after restore {check!r}', flush=True)


if __name__ == '__main__':
    main()
