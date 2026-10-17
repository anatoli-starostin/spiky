"""CPU-only spectral summary of the LUT projection weights in a nanochat checkpoint (mmap: only the 48
compress/decompress weight tensors are materialised). Usage: proj_spectra.py <model_XXXXXX.pt> [tag]"""
import os, sys, re
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_num_threads(2)

path, tag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
rows = {"compress": [], "decompress": []}
for k, v in sd.items():
    m = re.search(r"h\.(\d+)\..*mhl\.(compress|decompress)\.weight$", k)
    if not m:
        continue
    W = v.float()
    s = torch.linalg.svdvals(W)
    p = s / s.sum()
    erank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()      # entropy effective rank
    rows[m.group(2)].append(dict(layer=int(m.group(1)), fro=W.norm().item(), top=s[0].item(),
                                 med=s[len(s) // 2].item(), cond=(s[0] / s[-1].clamp_min(1e-12)).item(),
                                 erank=erank, n=len(s), rms=W.pow(2).mean().sqrt().item()))
print(f"== {tag} {os.path.basename(path)} ==")
for kind, rs in rows.items():
    if not rs:
        print(f"  {kind}: none found"); continue
    rs.sort(key=lambda r: r["layer"])
    def stat(key):
        xs = [r[key] for r in rs]
        return f"{min(xs):.3g}/{sum(xs)/len(xs):.3g}/{max(xs):.3g}"
    print(f"  {kind:10s} n_layers={len(rs)} rank_max={rs[0]['n']} | min/mean/max  fro {stat('fro')}  rms {stat('rms')}  "
          f"sigma_max {stat('top')}  sigma_med {stat('med')}  sigma_max/sigma_min {stat('cond')}  eff_rank {stat('erank')}")
