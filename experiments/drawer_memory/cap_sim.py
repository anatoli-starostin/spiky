"""Capacity without training (minutes, CPU): store n random key->value pairs, query with noisy keys, count how often
the read is closest to the right value. Equal cost per token: drawers touch tph * 2**flip cells whatever their size;
a dense memory with a feat-wide state touches feat rows. Keys/values are random (ideal addresses), so this is an
upper bound on what training could reach -- it says whether the direction can work at all.
Usage: python cap_sim.py  -> runs/cap_sim/results.csv"""
import csv, os, torch
torch.manual_seed(0)
DK, DH = 32, 32
HERE = os.path.dirname(os.path.abspath(__file__))


def drawer_keys(z, tph, nap, flip, tau, g):
    """Fixed random anchor-pair sign bits per table + flip-b soft weights -> dense unit-norm key [n, tph*2**nap]."""
    pairs = torch.combinations(torch.arange(DK), 2)
    idx = torch.stack([torch.randperm(len(pairs), generator=g)[:nap] for _ in range(tph)])   # [tph, nap]
    a, b = pairs[idx, 0], pairs[idx, 1]
    zn = z / z.abs().mean(-1, keepdim=True)
    u = zn[:, a] - zn[:, b]                                                                  # [n, tph, nap]
    pow2 = 2 ** torch.arange(nap - 1, -1, -1)
    c = ((u > 0).long() * pow2).sum(-1)
    K = torch.zeros(len(z), tph, 2 ** nap)
    if flip == 0:
        K.scatter_(-1, c[..., None], 1.0)
    else:
        mb, jb = (-u.abs()).topk(flip, dim=-1)
        sel = ((torch.arange(2 ** flip)[:, None] >> torch.arange(flip)[None]) & 1)
        cells = c[..., None] ^ (sel[None, None] * pow2[jb][..., None, :]).sum(-1)
        w = torch.exp(-(sel.float()[None, None] * (-mb)[..., None, :]).sum(-1) / tau)
        K.scatter_add_(-1, cells, w / w.sum(-1, keepdim=True))
    K = K.flatten(1)
    return K / K.norm(dim=-1, keepdim=True)


def make_keys(kind, z, g, **kw):
    if kind == "drawers":
        return drawer_keys(z, g=g, **kw)
    if kind == "dense_ideal":  # best case for a dense memory: independent random unit keys (noise added by caller)
        return None
    # dense random Fourier features of the same 32-dim key (nonlinear, full rank, similarity-preserving)
    R = torch.randn(kw["feat"], DK, generator=g) * kw.get("scale", 0.3)
    b = torch.rand(kw["feat"], generator=g) * 6.2832
    K = torch.cos(z @ R.T + b)
    return K / K.norm(dim=-1, keepdim=True)


def store_and_read(Kw, Kr, V, delta):
    """Kw/Kr: write/read keys [n, F]; V [n, DH]. delta: S += k (v - S^T k)^T (beta=1), else S += k v^T."""
    S = torch.zeros(Kw.shape[1], DH)
    for k, v in zip(Kw, V):
        S += torch.outer(k, v - (S.T @ k if delta else 0))
    return Kr @ S


def run(kind, n, noise, delta, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, DK, generator=g)
    q = z + noise * torch.randn(n, DK, generator=g)                                          # noisy query
    V = torch.nn.functional.normalize(torch.randn(n, DH, generator=g), dim=-1)
    gk = torch.Generator().manual_seed(seed + 100)
    Kw = make_keys(kind, z, gk, **kw)
    gk = torch.Generator().manual_seed(seed + 100)                                           # same hash/features
    Kr = make_keys(kind, q, gk, **kw)
    if kind == "dense_ideal":
        Kw = torch.nn.functional.normalize(torch.randn(n, kw["feat"], generator=g), dim=-1)
        Kr = torch.nn.functional.normalize(Kw + noise * torch.randn(n, kw["feat"], generator=g) / kw["feat"] ** 0.5, dim=-1)
    R = store_and_read(Kw, Kr, V, delta)
    return ((R @ V.T).argmax(-1) == torch.arange(n)).float().mean().item()


CONFIGS = [  # name, kind, touched rows per token, kwargs
    ("dense 16, random features (equal cost)", "dense", 16, dict(feat=16)),
    ("dense 256, random features (16x cost)", "dense", 256, dict(feat=256)),
    ("dense 16, ideal keys (equal cost)", "dense_ideal", 16, dict(feat=16)),
    ("dense 256, ideal keys (16x cost)", "dense_ideal", 256, dict(feat=256)),
    ("drawers 4x6 bits, flip-2 (256 cells)", "drawers", 16, dict(tph=4, nap=6, flip=2, tau=0.3)),
    ("drawers 4x10 bits, flip-2 (4096 cells)", "drawers", 16, dict(tph=4, nap=10, flip=2, tau=0.3)),
    ("drawers 16x8 bits, hard (4096 cells)", "drawers", 16, dict(tph=16, nap=8, flip=0, tau=0.3)),
]
if __name__ == "__main__":
    out = os.path.join(HERE, "runs", "cap_sim")
    os.makedirs(out, exist_ok=True)
    rows = []
    for noise in (0.0, 0.3, 0.6):
        for delta in (True,):
            for name, kind, cost, kw in CONFIGS:
                accs = [run(kind, n, noise, delta, **kw) for n in (16, 64, 256, 1024)]
                rows.append(dict(noise=noise, write="delta" if delta else "add", model=name, touched=cost,
                                 **{f"n{n}": f"{a:.3f}" for n, a in zip((16, 64, 256, 1024), accs)}))
                print(rows[-1], flush=True)
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
