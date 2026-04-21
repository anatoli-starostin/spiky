import sys, torch
A = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
B = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
a_sd, b_sd = A['sd'], B['sd']
print(f"final_loss A={A['final_loss']:.10f} B={B['final_loss']:.10f} diff={A['final_loss']-B['final_loss']:+.3e}")
assert set(a_sd) == set(b_sd)
TOL = 1e-8
diffs = []
for k in a_sd:
    a, b = a_sd[k], b_sd[k]
    if a.dtype.is_floating_point or a.dtype == torch.bfloat16:
        md = (a.to(torch.float64) - b.to(torch.float64)).abs().max().item()
    else:
        md = float((a != b).sum().item())
    diffs.append((k, md, tuple(a.shape), str(a.dtype)))
bad = [d for d in diffs if d[1] > TOL]
print(f'total tensors: {len(diffs)}')
print(f'max_abs_diff > {TOL:.0e}: {len(bad)} tensors')
bad.sort(key=lambda x: -x[1])
for k, md, sh, dt in bad[:25]:
    print(f'  max_diff={md:.3e}  {dt:12s} {str(sh):20s} {k}')
