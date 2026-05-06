"""Check whether learnable attn_scale compensated the annealing ramp.

Init: attn_scale = 0.4 (config: learnable_attn_scale_init).
Ramp: 1.0x at step 0 -> 10.0x at step 8000 (config: attn_scale_target_ramp).
At end of training, effective_scale_final = attn_scale_final * 10.0 * (1/sqrt(P)).

If compensation is FULL: attn_scale collapses to 0.04 -> effective same as exp154.
If compensation is NONE: attn_scale stays ~0.4 -> effective 10x sharper.
Partial compensation in between.

Run from repo root:
    PYTHONPATH=/home/starost/nanochat .venv/bin/python \
        nanochat_exps/exp156_attn_anneal/analyze_attn_scale.py
"""
import os, json, math, torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

ckpt_path = os.path.join(EXP_DIR, 'checkpoint.pt')
if not os.path.exists(ckpt_path):
    raise SystemExit(f'No checkpoint at {ckpt_path}')

sd = torch.load(ckpt_path, map_location='cpu')

init = float(cfg.get('learnable_attn_scale_init', 0.25))
target_ramp = float(cfg.get('attn_scale_target_ramp', 1.0))
n_layers = cfg['num_layers']
d_qk = cfg['d_qk']
P = d_qk * (d_qk - 1) // 2
inv_sqrt_p = 1.0 / math.sqrt(P)

# Reference scale: what exp154 ended up with (no ramp).
# Without observed value, compare to init * inv_sqrt_p as the "no-compensation,
# no-learning" anchor; the ratio attn_scale/init * ramp is the key signal.
print(f'init                 = {init}')
print(f'attn_scale_target_ramp = {target_ramp}')
print(f'1/sqrt(P)            = {inv_sqrt_p:.5f}  (P = {d_qk}*({d_qk}-1)/2 = {P})')
print(f'\nAt end of training, effective_scale = attn_scale * {target_ramp} / sqrt({P})\n')

print(f'{"layer":>5} {"attn_scale":>12} {"x_init":>8} {"x_ramp":>8} '
      f'{"effective":>10} {"vs_baseline":>12}  {"interpretation"}')
print('-' * 95)
# Baseline (no ramp): attn_scale_baseline ~= init * inv_sqrt_p.
baseline_eff = init * inv_sqrt_p

scales = []
for i in range(n_layers):
    key = f'layers.{i}.attn_scale'
    if key not in sd:
        # try alternatives
        candidates = [k for k in sd.keys() if k.endswith('attn_scale')]
        print(f'  Missing {key}. Candidates: {candidates[:5]}')
        continue
    s = float(sd[key])
    scales.append(s)
    eff = s * target_ramp * inv_sqrt_p
    ratio_init = s / init
    ratio_baseline = eff / baseline_eff
    if ratio_init < 0.15:
        interp = 'FULL compensation (~rejected ramp)'
    elif ratio_init < 0.5:
        interp = 'partial compensation'
    elif ratio_init < 0.9:
        interp = 'mild compensation'
    else:
        interp = 'NO compensation (kept full ramp)'
    print(f'{i:>5} {s:>12.5f} {ratio_init:>8.3f} {ratio_init*target_ramp:>8.3f} '
          f'{eff:>10.5f} {ratio_baseline:>12.3f}x  {interp}')

if scales:
    mean = sum(scales) / len(scales)
    print(f'\nmean attn_scale = {mean:.5f}  ({mean/init:.2%} of init)')
    print(f'mean effective_scale = {mean*target_ramp*inv_sqrt_p:.5f}  '
          f'(baseline {baseline_eff:.5f}, ratio {mean*target_ramp*inv_sqrt_p/baseline_eff:.2f}x)')
