"""GATE (a) for projmuonfp8-lrrs: CPU-only, real model geometry (meta GPT d24 + real LUT blocks on CPU), real
GPT.setup_optimizer called with the LRs base_train passes when the batch LR rescale is ON (x sqrt(2^20/2^19)).
Compares proj_fp8_muon OFF (v2) vs ON: per-group param count / numel, where every projection weight AND bias
lands, the Muon effective LR per shape (optim.py: lr * max(1, rows/cols)**0.5), and the coverage check."""
import os, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.lut_ffn import LUTFFNConfig, apply_lut_ffn

S = math.sqrt(2**20 / 2**19)                      # batch_lr_scale with the rescale ON
WD = 0.062990                                     # weight_decay_scaled at d24 (v2 log; independent of the LR rescale)
print(f"batch_lr_scale = {S:.6f}")

def build(proj_fp8_muon):
    depth, aspect, hd = 24, 64, 128
    dim = ((depth * aspect + hd - 1) // hd) * hd
    cfg = GPTConfig(sequence_len=2048, vocab_size=32768, n_layer=depth, n_head=dim // hd, n_kv_head=dim // hd,
                    n_embd=dim, window_pattern="SSSL")
    with torch.device("meta"):
        m = GPT(cfg)
    apply_lut_ffn(m, LUTFFNConfig(enabled=True, proj_fp8_muon=proj_fp8_muon), device="cpu")
    opt = m.setup_optimizer(unembedding_lr=0.008 * S, embedding_lr=0.3 * S, scalar_lr=0.5 * S,
                            matrix_lr=0.02 * S, weight_decay=WD, lut_lr=3e-3 * S)
    return m, opt

def report(tag, m, opt):
    names = {id(p): n for n, p in m.named_parameters()}
    print(f"\n==================== {tag} ====================")
    where = {}
    for gi, g in enumerate(opt.param_groups):
        ps = g["params"]
        shapes = sorted({tuple(p.shape) for p in ps})
        eff = ""
        if g["kind"] == "muon":
            sh = shapes[0]
            eff = f" eff_lr={g['lr'] * max(1.0, sh[-2] / sh[-1]) ** 0.5:.6f}"
        nproj = sum(1 for p in ps if ".mhl.compress." in names[id(p)] or ".mhl.decompress." in names[id(p)])
        print(f"  g{gi:<2d} {g['kind']:5s} lr={g['lr']:.6f}{eff} wd={g.get('weight_decay')} n={len(ps):3d} "
              f"numel={sum(p.numel() for p in ps):>12,d} proj_params={nproj:3d} shapes={shapes if len(shapes) <= 3 else str(len(shapes)) + ' shapes'}")
        for p in ps:
            where.setdefault(id(p), []).append((gi, g["kind"]))
    proj = [(n, p) for n, p in m.named_parameters() if ".mhl.compress." in n or ".mhl.decompress." in n]
    kinds = {}
    for n, p in proj:
        key = n.split(".mhl.")[1]                      # compress.weight / compress.bias / decompress.*
        kinds.setdefault(key, set()).update(where.get(id(p), [("NONE", "NONE")]))
    print("  projection params (24 layers each) ->", {k: sorted(v) for k, v in sorted(kinds.items())})
    trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    orph = [n for n, p in trainable if id(p) not in where]
    dup = [n for n, p in trainable if len(where.get(id(p), [])) > 1]
    print(f"  COVERAGE: requires_grad={len(trainable)} / in groups={sum(1 for _, p in trainable if id(p) in where)} "
          f"| numel trainable={sum(p.numel() for _, p in trainable):,} / grouped={sum(p.numel() for g in opt.param_groups for p in g['params']):,} "
          f"| orphans={orph or 'none'} | double={dup or 'none'}")
    lut = [g for g in opt.param_groups if g["kind"] == "adamw" and abs(g["lr"] - 3e-3 * S * (1536 / 768) ** -0.5) < 1e-12]
    muon = [g for g in opt.param_groups if g["kind"] == "muon"]
    return dict(lut_n=sum(len(g["params"]) for g in lut), lut_numel=sum(p.numel() for g in lut for p in g["params"]),
                muon_n=sum(len(g["params"]) for g in muon), muon_numel=sum(p.numel() for g in muon for p in g["params"]))

off = report("proj_fp8_muon OFF (v2 routing), LR rescale ON", *build(False))
on = report("proj_fp8_muon ON (new run), LR rescale ON", *build(True))
print("\n==================== MOVE (OFF -> ON) ====================")
print(f"  LUT AdamW group: n {off['lut_n']} -> {on['lut_n']} ({on['lut_n'] - off['lut_n']:+d}); "
      f"numel {off['lut_numel']:,} -> {on['lut_numel']:,} ({on['lut_numel'] - off['lut_numel']:+,d})")
print(f"  Muon groups:     n {off['muon_n']} -> {on['muon_n']} ({on['muon_n'] - off['muon_n']:+d}); "
      f"numel {off['muon_numel']:,} -> {on['muon_numel']:,} ({on['muon_numel'] - off['muon_numel']:+,d})")
