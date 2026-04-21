import sys, torch
A = torch.load(sys.argv[1], map_location='cpu')
B = torch.load(sys.argv[2], map_location='cpu')
assert set(A) == set(B), (set(A) ^ set(B))
TOL = 1e-8
n = 0
diff_keys = []
for k in A:
    a, b = A[k], B[k]
    if a.dtype.is_floating_point or a.dtype == torch.bfloat16:
        af = a.to(torch.float64)
        bf = b.to(torch.float64)
        md = (af - bf).abs().max().item()
    else:
        md = (a != b).sum().item()
        md = float(md)
    if md > TOL:
        diff_keys.append((k, md, tuple(a.shape), str(a.dtype)))
    n += 1
print(f'total tensors: {n}')
print(f'max_abs_diff > {TOL:.0e}: {len(diff_keys)} tensors')
for k, md, sh, dt in diff_keys[:20]:
    print(f'  {k:50s} dtype={dt:15s} shape={sh} max_abs_diff={md:.3e}')
if not diff_keys:
    print('ALL TENSORS IDENTICAL WITHIN 1e-08')
