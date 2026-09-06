"""Does the fused scored kernel match the torch score, and is it faster?"""
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/native/lutorch'))
from spiky.lutorch.fast_multi_head_lut import _confidence_score, _get_native_lutorch_manager

DEV = torch.device('cuda:0')
B, H, T, NAP, DIN = 6144, 4, 256, 8, 32
FORMS = {0: 'bounded_norm', 1: 'bounded', 2: 'margin'}


def timed(fn, reps=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


mgr = _get_native_lutorch_manager()
scored = getattr(mgr, 'anchor_pairs_lookup_eval_forward_msb_scored', None)
plain = getattr(mgr, 'anchor_pairs_lookup_eval_forward_msb', None)
print('scored kernel present:', scored is not None)
if scored is None:
    raise SystemExit(1)

for dtype in (torch.float64, torch.float32):
    torch.manual_seed(0)
    x = (torch.randn(B, H * DIN, device=DEV, dtype=dtype) * 0.6)
    a = torch.randint(0, DIN, (H * T, NAP), device=DEV)
    b = (a + 1 + torch.randint(0, DIN - 1, (H * T, NAP), device=DEV)) % DIN
    head_off = (torch.arange(H, device=DEV).repeat_interleave(T).view(-1, 1) * DIN)
    a = (a + head_off).contiguous()
    b = (b + head_off).contiguous()
    powers = (2 ** torch.arange(NAP - 1, -1, -1, device=DEV)).to(torch.int64)

    d = x[:, a] - x[:, b]                                    # [B, n_tables, NAP]
    idx_ref = ((d > 0).to(torch.int64) * powers.view(1, 1, -1)).sum(dim=-1)

    print(f'\n=== {str(dtype).split(".")[-1]} ===')
    for form_id, form in FORMS.items():
        for gain in (1.0, 12.61):
            s_ref = _confidence_score(d, form, gain)
            idx_k, s_k = scored(x, a, b, 0.0, form_id, gain, 256)
            ok_idx = torch.equal(idx_k, idx_ref)
            ae = (s_k - s_ref).abs().max().item()
            re = ((s_k - s_ref).abs() / s_ref.abs().clamp_min(1e-30)).max().item()
            print(f'  {form:<13} gain {gain:<6} index_equal={ok_idx}  '
                  f'max|abs|={ae:.3e}  max|rel|={re:.3e}')

# ---- speed, fp32 ----
torch.manual_seed(0)
x = (torch.randn(B, H * DIN, device=DEV) * 0.6)
a = torch.randint(0, DIN, (H * T, NAP), device=DEV)
b = (a + 1 + torch.randint(0, DIN - 1, (H * T, NAP), device=DEV)) % DIN
head_off = (torch.arange(H, device=DEV).repeat_interleave(T).view(-1, 1) * DIN)
a = (a + head_off).contiguous()
b = (b + head_off).contiguous()
powers = (2 ** torch.arange(NAP - 1, -1, -1, device=DEV)).to(torch.int64)
d0 = x[:, a] - x[:, b]

t_plain = timed(lambda: plain(x, a, b, 0.0, 256))
t_scored = timed(lambda: scored(x, a, b, 0.0, 0, 1.0, 256))
t_gather = timed(lambda: x[:, a] - x[:, b])
t_score_torch = timed(lambda: _confidence_score(d0, 'bounded_norm', 1.0))
print(f'\n=== speed (fp32, {B:,} x {H*T} tables x {NAP} anchors) ===')
print(f'  native plain  (index only)          {t_plain:7.3f} ms')
print(f'  native SCORED (index + score)       {t_scored:7.3f} ms   '
      f'(+{t_scored - t_plain:.3f} ms for the score)')
print(f'  torch margin gather                 {t_gather:7.3f} ms')
print(f'  torch score from gathered margins   {t_score_torch:7.3f} ms')
print(f'  -> torch route (gather+score)       {t_gather + t_score_torch:7.3f} ms')
print(f'  -> fused route                      {t_scored:7.3f} ms   '
      f'= {(t_gather + t_score_torch) / t_scored:.2f}x')
