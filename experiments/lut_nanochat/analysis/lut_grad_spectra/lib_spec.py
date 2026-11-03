"""Shared helpers for the LUT-table gradient-geometry analysis (no training).

Model build = nanochat.checkpoint_manager.build_model (the same LUT rebuild-from-meta path base_train/eval use:
GPT on meta -> to_empty -> init_weights -> apply_lut_ffn -> load_state_dict) + the SAME fp8 conversion/filter as
scripts/base_train.py (tensorwise, LUT projections tagged _lut_no_fp8 stay fp32).  Not torch.compile'd (eager).
"""
import json
import math
import os

import torch

BASE = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/work/nanochat_base"))

# tag -> (checkpoint dir under base_checkpoints, step, kind, run folder under results/)
CKPTS = {
    "lut1500":       ("d24_lutffn_confn1_1xh100_nodropout1500", 1500, "lut", "d24-lutffn-confn1-1xh100-s0-nodropout1500"),
    "lut500":        ("d24_lutffn_confn1_1xh100_v2repro500", 500, "lut", "d24-lutffn-confn1-1xh100-s0-v2repro500"),
    "lut500_nodrop": ("d24_lutffn_confn1_1xh100_v2repro500", 500, "lut", "d24-lutffn-confn1-1xh100-s0-v2repro500"),
    "dense2000":     ("d24_dense_adamw_lr3e-4_2k", 2000, "dense", "d24-dense-adamw-lr3e-4-1xh100-s0-2k"),
    "dense5568":     ("d24_dense_adamw_lr3e-4_from2k", 5568, "dense", "d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k"),
}
# the data position every model is analysed at (dense step-2000 loader state): later than anything the LUT
# checkpoints (steps 500/1500) have seen, and "next" data for the dense 2000 checkpoint
DATA_POS_META = ("d24_dense_adamw_lr3e-4_2k", 2000)


def get_batches(n, device, cache_path):
    """n micro-batches (x, y) of 16x2048 tokens from the ClimbMix train loader, starting at the saved loader state of
    the dense step-2000 checkpoint. Cached on disk so every model sees byte-identical batches."""
    if os.path.exists(cache_path):
        d = torch.load(cache_path)
        if len(d["x"]) >= n:
            return [(d["x"][i].to(device), d["y"][i].to(device)) for i in range(n)], d["state"]
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
    from nanochat.tokenizer import get_tokenizer
    with open(os.path.join(BASE, "base_checkpoints", DATA_POS_META[0], f"meta_{DATA_POS_META[1]:06d}.json")) as f:
        st = json.load(f)["dataloader_state_dict"]
    tok = get_tokenizer()
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tok, 16, 2048, split="train", device="cuda",
                                                                       resume_state_dict=st)
    xs, ys, last = [], [], None
    for _ in range(n):
        x, y, last = next(loader)
        xs.append(x.cpu()); ys.append(y.cpu())
    torch.save({"x": xs, "y": ys, "state": {"start": st, "end": last}}, cache_path)
    return [(x.to(device), y.to(device)) for x, y in zip(xs, ys)], {"start": st, "end": last}


def build(tag, device, fp8=True):
    from nanochat.checkpoint_manager import build_model
    ckdir, step, kind, _ = CKPTS[tag]
    model, _tok, meta = build_model(os.path.join(BASE, "base_checkpoints", ckdir), step, device, "train")
    if tag == "lut500_nodrop":
        for c in cartridges(model):
            c.table_dropout_rate = 0.0
    if fp8:
        import torch.nn as nn
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training

        def filt(mod, fqn):  # identical to base_train.fp8_module_filter
            if not isinstance(mod, nn.Linear):
                return False
            if getattr(mod, "_lut_no_fp8", False):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True
        convert_to_float8_training(model, config=Float8LinearConfig.from_recipe_name("tensorwise"), module_filter_fn=filt)
    return model, meta


def cartridges(model):
    return [b.mlp.mhl.cartridge for b in model.transformer.h if hasattr(b.mlp, "mhl")]


def group_of(name):
    """(layer or -1, group) of a parameter name, for the E (gradient-norm share) table."""
    layer = -1
    if name.startswith("transformer.h."):
        layer = int(name.split(".")[2])
    if ".mlp.mhl.cartridge.weights" in name:
        return layer, "lut_table"
    if ".mlp.mhl.cartridge." in name:
        return layer, "lut_conf_scalars"
    if ".mlp.mhl.compress" in name:
        return layer, "lut_compress"
    if ".mlp.mhl.decompress" in name:
        return layer, "lut_decompress"
    if ".attn." in name:
        return layer, "attn"
    if ".mlp." in name:
        return layer, "dense_mlp"
    return layer, "other"


# ---------------------------------------------------------------- spectra
def spec_stats_from_gram(gram):
    """gram [..., n, n] (fp64, = M^T M). Returns dict of [...] tensors; NaN where the matrix is all-zero."""
    ev = torch.linalg.eigvalsh(gram).clamp_min(0).flip(-1)          # sigma^2, descending
    tot = ev.sum(-1)
    ok = tot > 0
    tot_s = torch.where(ok, tot, torch.ones_like(tot))
    p = ev / tot_s[..., None]
    top1 = p[..., 0]
    ent = -(p * torch.log(p.clamp_min(1e-300))).sum(-1)
    sig = ev.sqrt()
    med = torch.quantile(sig, 0.5, dim=-1)
    nan = torch.full_like(top1, float("nan"))
    return {
        "top1": torch.where(ok, top1, nan),
        "stable_rank": torch.where(ok, 1.0 / top1.clamp_min(1e-300), nan),
        "eff_rank": torch.where(ok, ent.exp(), nan),
        "cond_med": torch.where(ok & (med > 0), sig[..., 0] / med.clamp_min(1e-300), nan),
        "n": torch.full_like(top1, float(gram.shape[-1])),
    }


def gram_of(M):
    """M [..., m, n] -> gram of the narrow side in fp64 (same nonzero singular values either way)."""
    M = M.double()
    return M.transpose(-1, -2) @ M if M.shape[-1] <= M.shape[-2] else M @ M.transpose(-1, -2)


def spec_stats(M):
    return spec_stats_from_gram(gram_of(M))


def tolist(d):
    return {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in d.items()}


def nan_summary(x):
    """x: tensor -> dict(median, p10, p90, min, max, mean) ignoring NaN."""
    x = x.flatten().double()
    x = x[~torch.isnan(x)]
    if x.numel() == 0:
        return {k: float("nan") for k in ("median", "p10", "p90", "min", "max", "mean")}
    q = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, device=x.device))
    return {"median": q[1].item(), "p10": q[0].item(), "p90": q[2].item(), "min": x.min().item(),
            "max": x.max().item(), "mean": x.mean().item()}


def avg_ranks(v):
    """tie-averaged ranks along the last dim (v [..., n])."""
    s = v.sort(-1).values.contiguous()
    v = v.contiguous()
    lo = torch.searchsorted(s, v, right=False)
    hi = torch.searchsorted(s, v, right=True)
    return (lo + hi - 1).double() / 2.0


def spearman_rows(a, b):
    """row-wise Spearman between a and b [R, n]; NaN where a row is constant."""
    ra, rb = avg_ranks(a), avg_ranks(b)
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    den = ra.norm(dim=-1) * rb.norm(dim=-1)
    out = (ra * rb).sum(-1) / den.clamp_min(1e-300)
    return torch.where(den > 0, out, torch.full_like(out, float("nan")))


def gini_rows(c):
    """Gini coefficient along the last dim of non-negative c [R, n]."""
    c = c.double().sort(-1).values
    n = c.shape[-1]
    idx = torch.arange(1, n + 1, device=c.device, dtype=torch.float64)
    tot = c.sum(-1)
    g = (2 * (idx * c).sum(-1) / (n * tot.clamp_min(1e-300))) - (n + 1) / n
    return torch.where(tot > 0, g, torch.full_like(g, float("nan")))


def hit_stats(counts):
    """counts [T, K] (hits per cell for each of T tables) -> dict of [T] tensors."""
    T, K = counts.shape
    c = counts.double()
    tot = c.sum(-1, keepdim=True)
    p = c / tot.clamp_min(1)
    ent = -(p * torch.log(p.clamp_min(1e-300))).sum(-1) / math.log(K)
    srt = c.sort(-1, descending=True).values
    k1, k10 = max(1, math.ceil(0.01 * K)), max(1, math.ceil(0.10 * K))
    lc = torch.where(c > 0, torch.log(c.clamp_min(1)), torch.full_like(c, float("nan")))
    m = ~torch.isnan(lc)
    cnt = m.sum(-1).clamp_min(1)
    mu = torch.where(m, lc, torch.zeros_like(lc)).sum(-1) / cnt
    var = (torch.where(m, (lc - mu[:, None]) ** 2, torch.zeros_like(lc))).sum(-1) / cnt
    return {
        "frac_zero": (c == 0).double().mean(-1),
        "gini": gini_rows(c),
        "top1pct_share": srt[:, :k1].sum(-1) / tot[:, 0].clamp_min(1),
        "top10pct_share": srt[:, :k10].sum(-1) / tot[:, 0].clamp_min(1),
        "entropy_norm": ent,
        "min": c.min(-1).values,
        "max": c.max(-1).values,
        "median": c.median(-1).values,
        "std_log_hits": var.sqrt(),
    }
