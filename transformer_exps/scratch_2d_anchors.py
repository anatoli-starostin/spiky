"""
Explore 2D convolution-style hierarchical anchor sampling for embedding_dim=64 (8x8 grid).

Each table: 3x3 dilated kernel → 9 points sorted by 1D index → 8 sequential pairs (nap=8).
Groups: dilation d=1,2,3,... sliding with stride=1 over all valid top-left positions.
"""

H, W = 8, 8

def idx(r, c):
    return r * W + c

def grid_pos(i):
    return i // W, i % W

def compute_groups(H, W):
    groups = []
    d = 1
    while True:
        max_r = H - 1 - 2 * d
        max_c = W - 1 - 2 * d
        if max_r < 0 or max_c < 0:
            break
        tables = []
        for r in range(max_r + 1):
            for c in range(max_c + 1):
                pts = sorted(idx(r + dr * d, c + dc * d)
                             for dr in range(3) for dc in range(3))
                anchor_a = pts[:8]
                anchor_b = pts[1:]
                tables.append((anchor_a, anchor_b))
        groups.append(dict(d=d, n=len(tables), max_r=max_r, max_c=max_c,
                           footprint_span=2*d+1, tables=tables))
        d += 1
    return groups

groups = compute_groups(H, W)

print(f"Grid: {H}x{W}  (embedding_dim={H*W})")
print(f"nap=8 (9 points per kernel → 8 sequential pairs)")
print("=" * 60)

total = 0
for g in groups:
    d = g['d']
    print(f"\n--- Group d={d} ---")
    print(f"  Kernel samples at offsets {{0,{d},{2*d}}} x {{0,{d},{2*d}}}")
    print(f"  Footprint span: {g['footprint_span']}x{g['footprint_span']} pixels")
    nr = g['max_r'] + 1
    nc = g['max_c'] + 1
    print(f"  Valid top-left positions: rows 0..{g['max_r']}, "
          f"cols 0..{g['max_c']} → {nr}x{nc} = {g['n']} tables")

    # Show first 3 tables
    for ti, (a, b) in enumerate(g['tables'][:3]):
        pts = [a[0]] + b  # all 9 points
        grid = [['.' for _ in range(W)] for _ in range(H)]
        for i, p in enumerate(pts):
            r, c = grid_pos(p)
            grid[r][c] = str(i)
        tl_r = pts[0] // W
        tl_c = pts[0] % W
        print(f"  Table {ti} (top-left=({tl_r},{tl_c})):")
        print(f"    grid (indices 0-8 mark the 9 kernel points):")
        for row in grid:
            print(f"      {'  '.join(row)}")
        print(f"    a={a}")
        print(f"    b={b}")
    if g['n'] > 3:
        print(f"  ... ({g['n']-3} more tables)")
    total += g['n']

print(f"\n{'='*60}")
print(f"Total tables: {total}")
print(f"\nPer-group summary:")
for g in groups:
    nr = g['max_r'] + 1
    nc = g['max_c'] + 1
    print(f"  d={g['d']}: {g['n']:3d} tables  "
          f"(footprint {g['footprint_span']}x{g['footprint_span']}, "
          f"{nr}x{nc} positions)")
