"""Why did exp_g_0244 (tanh_margin) take 1.721 h vs exp_g_0193 (margin) 0.92 h and exp_g_0243 (min_margin)
0.955 h? Diagnose only. One variant per process:   python bench_forms.py <variant> [--profile]

Builds the exact model of exp_g_0193's config with the variant's form/gain/gamma on CUDA (same
build_model as train.py) and times, with cuda synchronize:
  * the TRAINING step as train.py runs it: 4 micro-batches of 12x512 fwd+bwd, grad clip, AdamW step
    (random tokens -- the data path is form-independent);
  * the no-grad EVAL forward at bs48 (evaluate_bpb_fixed's batch), i.e. the native-kernel-vs-torch path;
  * the isolated score op on d [6144, 4, 128, 8]: _confidence_score forward, and forward+backward
    through autograd (what LightMHL training uses).
--profile adds a torch.profiler table of the top CUDA ops over 2 training steps.
"""
import json
import os
import sys
import time

import torch

FR = '/home/astarostin/projects/spiky/experiments/ffn_replacement'
sys.path.insert(0, os.path.join(FR, 'tools'))
from model_build import build_model                                     # noqa: E402
from spiky.lutorch.fast_multi_head_lut import _confidence_score         # noqa: E402

VARIANTS = {
    'margin': dict(lut_confidence_form='margin', lut_confidence_gain=1.0),
    'min_margin': dict(lut_confidence_form='min_margin', lut_confidence_gain=37.4),
    'tanh_margin': dict(lut_confidence_form='tanh_margin', lut_confidence_gain=1.0),
    'sharp_g1.75': dict(lut_confidence_form='sharp_margin', lut_confidence_gain=3.9, lut_sharp_margin_gamma=1.75),
    'sharp_g3': dict(lut_confidence_form='sharp_margin', lut_confidence_gain=25.4, lut_sharp_margin_gamma=3.0),
}


def sync_time(fn, n):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main():
    name = sys.argv[1]
    cfg = json.load(open(os.path.join(FR, 'runs_corrected', 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1',
                                      'config.json')))
    cfg.update(VARIANTS[name])
    torch.manual_seed(0)
    model = build_model(cfg, cfg['tokenizer_vocab_size'], device='cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    V, S, B = cfg['tokenizer_vocab_size'], cfg['seq_len'], cfg['device_batch_size']
    accum = cfg['total_batch_size'] // (B * S)
    g = torch.Generator(device='cuda').manual_seed(0)
    xs = [torch.randint(0, V, (B, S), device='cuda', generator=g) for _ in range(accum)]
    ys = [torch.randint(0, V, (B, S), device='cuda', generator=g) for _ in range(accum)]
    xe = torch.randint(0, V, (48, S), device='cuda', generator=g)

    def train_step():
        opt.zero_grad(set_to_none=True)
        for x, y in zip(xs, ys):
            (model(x, y) / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.train()
    sync_time(train_step, 5)                                            # warm-up (NVRTC JIT, allocator)
    t_train = sync_time(train_step, 30)

    model.eval()
    with torch.no_grad():
        sync_time(lambda: model(xe, xe), 3)
        t_eval = sync_time(lambda: model(xe, xe), 20)
    model.train()

    form, gain = cfg['lut_confidence_form'], cfg['lut_confidence_gain']
    gamma = cfg.get('lut_sharp_margin_gamma')
    d = torch.randn(B * S, 4, 128, 8, device='cuda', generator=g) * 0.6
    with torch.no_grad():
        sync_time(lambda: _confidence_score(d, form, gain, gamma), 3)
        t_score_f = sync_time(lambda: _confidence_score(d, form, gain, gamma), 50)
    dg = d.clone().requires_grad_(True)
    go = torch.randn(B * S, 4, 128, device='cuda', generator=g)

    def score_fb():
        (_confidence_score(dg, form, gain, gamma) * go).sum().backward()
        dg.grad = None

    sync_time(score_fb, 3)
    t_score_fb = sync_time(score_fb, 50)
    # 16K steps + 32 evals x 100 batches, no data loading / logging / checkpoints
    est_h = (16000 * t_train + 32 * 100 * t_eval) / 3600
    print(f'RESULT {name:<12} train_step {t_train * 1e3:8.1f} ms | eval_fwd_bs48 {t_eval * 1e3:7.1f} ms | '
          f'score fwd {t_score_f * 1e3:6.2f} ms  fwd+bwd {t_score_fb * 1e3:6.2f} ms (per call, x24 per step) | '
          f'est 16K run {est_h:.3f} h')

    if '--profile' in sys.argv:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            train_step()
            train_step()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=18, max_name_column_width=60))


if __name__ == '__main__':
    main()
