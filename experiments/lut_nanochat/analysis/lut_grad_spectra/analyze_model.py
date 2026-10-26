"""Gradient geometry of one checkpoint (no training). See README.md in this folder.

usage (from experiments/lut_nanochat/nanochat, env as in the task):
  python ../analysis/lut_grad_spectra/analyze_model.py --tag lut1500 --out OUT_DIR [--n-grad 8 --n-held 4]
tags: lut1500 lut500 lut500_nodrop dense2000 dense5568   (see lib_spec.CKPTS)
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("LUTORCH_EX_NO_COMPILE", "1")   # reference ConfidenceLUT (used only for the FD cross-check) in eager mode
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_spec import (CKPTS, build, cartridges, get_batches, group_of, hit_stats, nan_summary, spearman_rows,
                      spec_stats, spec_stats_from_gram, gram_of, tolist)

ADAM_EPS = 1e-10          # AdamW eps of the LUT group (gpt.py setup_optimizer)
TV_LAMBDA = 10.0          # --lut-ffn-tv-lambda


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tv_loss(model, lam=TV_LAMBDA):
    from nanochat.lut_ffn import lut_cell_tv_loss
    return lut_cell_tv_loss(model, lam)


@torch.no_grad()
def addr_and_score(cart, z):
    """cell index c [N,G,tph] (the module's own addressing: ManifestoLUT._addresses) and confidence score s [N,G,tph]
    (ConfidenceLUT._score formula, learned beta/gamma) for cartridge inputs z [N,h,d]. Chunked."""
    from spiky.lutorch_ex import ConfidenceLUT
    cs, ss = [], []
    for i in range(0, z.shape[0], 8192):
        _, u, c, _, _, _ = cart._addresses(z[i:i + 8192])
        cs.append(c.long()); ss.append(ConfidenceLUT._score(cart, u))
    return torch.cat(cs), torch.cat(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(CKPTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-grad", type=int, default=8)
    ap.add_argument("--n-held", type=int, default=4)
    ap.add_argument("--batch-cache", default=os.path.expanduser("~/work/.cache/lut_grad_spectra_batches.pt"))
    ap.add_argument("--skip", default="", help="comma list of parts to skip: B,C,D,E")
    args = ap.parse_args()
    skip = set(args.skip.split(","))
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(1234)
    ckdir, step, kind, folder = CKPTS[args.tag]
    is_lut = kind == "lut"

    batches, dstate = get_batches(args.n_grad + args.n_held, dev, args.batch_cache)
    gb, hb = batches[:args.n_grad], batches[args.n_grad:]
    model, meta = build(args.tag, dev)
    model.train()
    uc = meta["user_config"]
    res = {"tag": args.tag, "ckpt": f"{ckdir}@{step}", "kind": kind, "fp8": True, "n_grad": args.n_grad,
           "n_held": args.n_held, "tokens_per_grad": args.n_grad * 16 * 2048, "data_state": dstate,
           "table_dropout_in_grad_pass": (0.0 if args.tag == "lut500_nodrop" else uc.get("lut_ffn_table_dropout"))
           if is_lut else None, "val_bpb_ckpt": meta.get("val_bpb")}
    log("built", args.tag, "dropout", res["table_dropout_in_grad_pass"])
    params = dict(model.named_parameters())
    carts = cartridges(model) if is_lut else []
    L = len(model.transformer.h)

    # ------------------------------------------------------------ gradient pass (n_grad micro-batches)
    names = list(params)
    tab_names = [n for n in names if n.endswith("cartridge.weights")]
    gsum = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items() if n not in tab_names}
    S1 = [torch.zeros_like(params[n]) for n in tab_names]        # sum of per-mb table grads
    S2 = [torch.zeros_like(params[n]) for n in tab_names]        # sum of squares
    Qg = [torch.zeros(48, 48, dtype=torch.float64, device=dev) for _ in tab_names]  # sum_i g_i^T g_i
    G0 = [None] * len(tab_names)                                 # mb0 grads
    sq_norm_mb = [[] for _ in tab_names]                         # per-mb ||g_i||^2
    losses = []
    counts = [torch.zeros(16 * 64 * 256, dtype=torch.int64, device=dev) for _ in carts]
    score_mass = [torch.zeros(16 * 64 * 256, dtype=torch.float64, device=dev) for _ in carts]
    z0 = [None] * len(carts)
    hooks = []
    if is_lut and "B" not in skip:
        def mk(li):
            def pre(mod, inp):
                z = inp[0].detach()
                if mod._grad_i == 0 and z0[li] is None:
                    z0[li] = z.clone()
                with torch.no_grad():
                    c, s = addr_and_score(mod, z)
                    G_, tph, K = mod.weights.shape[:3]
                    flat = ((torch.arange(G_, device=dev)[None, :, None] * tph +
                             torch.arange(tph, device=dev)[None, None, :]) * K + c).reshape(-1)
                    counts[li] += torch.bincount(flat, minlength=G_ * tph * K)
                    score_mass[li].index_add_(0, flat, s.reshape(-1).double())
            return pre
        for li, c in enumerate(carts):
            c._grad_i = 0
            hooks.append(c.register_forward_pre_hook(mk(li)))
    elif is_lut:
        for c in carts:
            c._grad_i = 0

    for i, (x, y) in enumerate(gb):
        for c in carts:
            c._grad_i = i
        model.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        losses.append(loss.item())
        for n, p in params.items():
            if n in gsum:
                if p.grad is not None:
                    gsum[n] += p.grad.float()
        for k, n in enumerate(tab_names):
            g = params[n].grad
            S1[k] += g; S2[k] += g * g
            g2 = g.reshape(-1, 48).double()
            Qg[k] += g2.T @ g2
            sq_norm_mb[k].append((g.double() ** 2).sum().item())
            if i == 0:
                G0[k] = g.clone()
        log(f"mb {i} loss {loss.item():.4f}")
    for h in hooks:
        h.remove()
    model.zero_grad(set_to_none=True)
    res["train_loss_mbs"] = losses
    res["train_loss_mean"] = sum(losses) / len(losses)
    n_mb = args.n_grad
    Gtab = [s / n_mb for s in S1]                                  # token-loss batch gradient of the tables

    # TV gradient (deterministic)
    Gtv = [None] * len(tab_names)
    if is_lut:
        tvl = tv_loss(model)
        tv_g = torch.autograd.grad(tvl, [params[n] for n in tab_names])
        Gtv = list(tv_g)
        res["tv_loss_value"] = tvl.item()
        res["tv_per_layer_mean_cell_tv"] = [c.cell_tv().item() for c in cartridges(model)]

    # ------------------------------------------------------------ A: spectra
    log("A: spectra")
    A = {}
    if is_lut:
        def table_block(G, key):
            out = {"layer": [], "head": [], "table": []}
            for k in range(L):
                g = G[k].reshape(16, 64, 256, 48)
                out["layer"].append(tolist(spec_stats(g.reshape(-1, 48))))
                out["head"].append(tolist(spec_stats(g.reshape(16, 64 * 256, 48))))
                out["table"].append(tolist(spec_stats(g.reshape(16 * 64, 256, 48))))
            return out
        A["tok8"] = table_block(Gtab, "tok8")
        # alignment of the dominant 48-wide direction: across tables of one head, across heads (random 48-d baseline 0.115)
        al_t, al_h = [], []
        off_t = ~torch.eye(64, dtype=torch.bool, device=dev)
        off_h = ~torch.eye(16, dtype=torch.bool, device=dev)
        for k in range(L):
            g = Gtab[k].reshape(16, 64, 256, 48)
            vt = torch.linalg.eigh(gram_of(g.reshape(1024, 256, 48)))[1][..., -1].reshape(16, 64, 48)
            vh = torch.linalg.eigh(gram_of(g.reshape(16, 64 * 256, 48)))[1][..., -1]
            al_t.append((vt @ vt.transpose(1, 2)).abs()[:, off_t].mean().item())
            al_h.append((vh @ vh.T).abs()[off_h].mean().item())
        A["top_dir_alignment"] = {"within_head_across_tables_mean_abs_cos": al_t, "across_heads_mean_abs_cos": al_h,
                                  "random_baseline": math.sqrt(2 / (math.pi * 48))}
        A["tok8_plus_tv"] = {"layer": [tolist(spec_stats((Gtab[k] + Gtv[k]).reshape(-1, 48))) for k in range(L)]}
        A["mb0"] = {"layer": [tolist(spec_stats(G0[k].reshape(-1, 48))) for k in range(L)]}
        A["tv_only"] = {"layer": [tolist(spec_stats(Gtv[k].reshape(-1, 48))) for k in range(L)]}
        # signal vs noise (token grads): signal = ||mean||^2, noise = mean_i ||g_i - mean||^2
        sig, noi, noise_top1, signal_top1 = [], [], [], []
        for k in range(L):
            m2 = (Gtab[k].double() ** 2).sum().item()
            mean_sq = sum(sq_norm_mb[k]) / n_mb
            sig.append(m2); noi.append(mean_sq - m2)
            Gm = Gtab[k].reshape(-1, 48).double()
            Cn = Qg[k] - n_mb * (Gm.T @ Gm)                          # sum_i N_i^T N_i
            noise_top1.append(spec_stats_from_gram(Cn)["top1"].item())
            signal_top1.append(spec_stats_from_gram(Gm.T @ Gm)["top1"].item())
        A["signal_noise"] = {"signal_sq": sig, "noise_sq_per_mb": noi,
                             "snr_per_mb": [a / max(b, 1e-300) for a, b in zip(sig, noi)],
                             "snr_batch": [a / max(b / n_mb, 1e-300) for a, b in zip(sig, noi)],
                             "noise_cov_top1_share": noise_top1, "signal_gram_top1_share": signal_top1}
        # also the element-wise per-mb rms (Adam second-moment proxy) fraction explained by the mean
    # dense matrices (both models: the dense ones in the dense reference; in the LUT model only attention)
    mat_names = {"attn.c_q": "attn", "attn.c_k": "attn", "attn.c_v": "attn", "attn.c_proj": "attn"}
    if not is_lut:
        mat_names.update({"mlp.c_fc": "mlp", "mlp.c_proj": "mlp"})
    dense = {}
    for mn, grp in mat_names.items():
        rows = []
        for l in range(L):
            nm = f"transformer.h.{l}.{mn}.weight"
            g = gsum[nm] / n_mb
            rows.append(tolist(spec_stats(g)))
            rows[-1]["shape"] = list(g.shape)
        dense[mn] = {"group": grp, "rows": rows}
    A["dense_mats"] = dense
    if not is_lut or True:
        # mb0 spectra for the attention q matrix only needs a re-run; skip (batch-level result is the one reported)
        pass
    res["A"] = A

    # ------------------------------------------------------------ E: gradient-norm shares
    log("E: gradient norm shares")
    E = {"per_layer": [], "global": {}}
    grp_sq = {}
    for n, g in gsum.items():
        l, gname = group_of(n)
        grp_sq.setdefault((l, gname), 0.0)
        grp_sq[(l, gname)] += ((g / n_mb).double() ** 2).sum().item()
    for k, n in enumerate(tab_names):
        l, _ = group_of(n)
        grp_sq[(l, "lut_table")] = (Gtab[k].double() ** 2).sum().item()
        if is_lut:
            grp_sq[(l, "lut_table_tv")] = (Gtv[k].double() ** 2).sum().item()
            grp_sq[(l, "lut_table_tok_plus_tv")] = ((Gtab[k] + Gtv[k]).double() ** 2).sum().item()
    gnames = sorted({g for _, g in grp_sq})
    for l in range(-1, L):
        E["per_layer"].append({"layer": l, **{g: math.sqrt(grp_sq[(l, g)]) for g in gnames if (l, g) in grp_sq}})
    tot = {}
    for (l, g), v in grp_sq.items():
        if g in ("lut_table_tv", "lut_table_tok_plus_tv"):
            continue
        tot[g] = tot.get(g, 0.0) + v
    ssum = sum(tot.values())
    E["global"] = {"norm_by_group": {g: math.sqrt(v) for g, v in tot.items()},
                   "sq_share_by_group": {g: v / ssum for g, v in tot.items()}, "total_norm": math.sqrt(ssum)}
    res["E"] = E
    del gsum

    # ------------------------------------------------------------ B: hit statistics
    if is_lut and "B" not in skip:
        log("B: hit statistics")
        B = {"verify": [], "layers": []}
        agg = {}
        for l in range(L):
            cnt = counts[l].reshape(16 * 64, 256)
            hs = hit_stats(cnt)
            rown = Gtab[l].reshape(16 * 64, 256, 48).norm(dim=-1)
            sp_hits = spearman_rows(cnt.double(), rown.double())
            sm = score_mass[l].reshape(16 * 64, 256)
            sp_sm = spearman_rows(sm, rown.double())
            msk = cnt > 0
            # the same Spearman restricted to cells that were hit (removes the trivial zero-hit/zero-grad ties)
            row = {k: nan_summary(v) for k, v in hs.items()}
            row["spearman_hits_vs_rownorm_mean"] = torch.nanmean(sp_hits).item()
            row["spearman_scoremass_vs_rownorm_mean"] = torch.nanmean(sp_sm).item()
            srt = cnt.double().sort(-1).values
            lor = (srt.cumsum(-1) / srt.sum(-1, keepdim=True).clamp_min(1)).mean(0)      # mean Lorenz curve over tables
            row["lorenz_cells_frac_0_to_1_step16"] = [0.0] + lor[15::16].tolist()
            lr_ = torch.log2((cnt.double() + 0.5) / cnt.double().mean())
            row["hist_log2_hits_over_mean"] = {"edges": list(range(-12, 5)), "counts": torch.histc(lr_.flatten(), bins=16, min=-12, max=4).tolist()}
            row["mean_hits_per_cell"] = cnt.double().mean().item()
            B["layers"].append(row)
            for k, v in hs.items():
                agg.setdefault(k, []).append(v)
            agg.setdefault("spearman_hits_vs_rownorm", []).append(sp_hits)
            agg.setdefault("spearman_scoremass_vs_rownorm", []).append(sp_sm)
        B["aggregate"] = {k: nan_summary(torch.cat(v)) for k, v in agg.items()}
        B["tokens"] = res["tokens_per_grad"]
        B["hits_per_cell_per_real_step_mean"] = res["tokens_per_grad"] * 4 / 256
        # verification of the addressing/score reproduction against the fused kernel output
        with torch.no_grad():
            for l in range(L):
                cart = carts[l]
                was = cart.training
                cart.eval()
                z = z0[l][:4096]
                y_f = cart(z).double()
                c, s = addr_and_score(cart, z)
                W = cart.weights
                flat = (torch.arange(16, device=dev)[None, :, None] * 64 + torch.arange(64, device=dev)[None, None, :]) * 256 + c
                rows = W.reshape(-1, 48)[flat]                           # [N,16,64,48]
                y_r = (rows.double() * s[..., None].double()).sum(2)
                err = (y_f - y_r).abs().max().item()
                B["verify"].append({"layer": l, "max_abs_err": err, "max_abs_y": y_f.abs().max().item(),
                                    "rel": err / max(y_f.abs().max().item(), 1e-30)})
                cart.train(was)
        res["B"] = B
        res["B_counts_hist"] = None
        torch.save({"counts": torch.stack([c.reshape(16, 64, 256).int() for c in counts]).cpu()},
                   os.path.join(os.environ.get("TMPDIR", "/tmp"), f"counts_{args.tag}.pt"))

    # ------------------------------------------------------------ C: cross-table redundancy
    if is_lut and "C" not in skip:
        log("C: redundancy")
        Cres = {"layers": []}
        for l in range(L):
            cart = carts[l]
            Gf = Gtab[l] + Gtv[l]                                           # the real gradient (token + TV)
            rms = (S2[l] / n_mb + 2 * Gtv[l] * Gtab[l] + Gtv[l] ** 2).clamp_min(0).sqrt()  # per-mb second moment proxy
            # directions
            rn = Gf.reshape(-1, 48).norm(dim=-1, keepdim=True)
            nz = rn[rn > 0]
            med = nz.median() if nz.numel() else torch.tensor(1.0, device=dev)
            D = {"sgd": Gf, "adam_sign": Gf / (Gf.abs() + ADAM_EPS), "adam_proxy": Gf / (rms + ADAM_EPS),
                 "rownorm": (Gf.reshape(-1, 48) / rn.clamp_min(1e-30) * med * (rn > 0)).reshape(Gf.shape)}
            c, s = addr_and_score(cart, z0[l])
            N = c.shape[0]
            flat = (torch.arange(16, device=dev)[None, :, None] * 64 + torch.arange(64, device=dev)[None, None, :]) * 256 + c
            row = {}
            for dn, Dd in D.items():
                Df = Dd.reshape(-1, 48)
                gram = torch.zeros(16, 64, 64, dtype=torch.float64, device=dev)
                Rtok_sum = torch.zeros(16, dtype=torch.float64, device=dev)
                Rtok_n = torch.zeros(16, dtype=torch.float64, device=dev)
                for i in range(0, N, 1024):
                    dy = Df[flat[i:i + 1024]] * s[i:i + 1024, ..., None]      # [n,16,64,48]
                    tot_ = dy.sum(2)
                    num = (tot_ ** 2).sum(-1).double()
                    den = (dy ** 2).sum((2, 3)).double()
                    ok = den > 0
                    Rtok_sum += torch.where(ok, num / den.clamp_min(1e-300), torch.zeros_like(num)).sum(0)
                    Rtok_n += ok.sum(0)
                    gram += torch.einsum("ngtd,ngsd->gts", dy.double(), dy.double())
                tr = torch.diagonal(gram, dim1=1, dim2=2)
                R_tok = (Rtok_sum / Rtok_n.clamp_min(1))
                R_sums = gram.sum((1, 2)) / tr.sum(-1).clamp_min(1e-300)
                dn_ = tr.sqrt()
                cosm = gram / (dn_[:, :, None] * dn_[:, None, :]).clamp_min(1e-300)
                off = ~torch.eye(64, dtype=torch.bool, device=dev)
                row[dn] = {"R_token_mean": R_tok.mean().item(), "R_token_head_min": R_tok.min().item(),
                           "R_token_head_max": R_tok.max().item(), "R_ratio_of_sums": R_sums.mean().item(),
                           "mean_cos_contrib": cosm[:, off].mean().item(),
                           "mean_abs_cos_contrib": cosm[:, off].abs().mean().item()}
            # (ii) cosine between the gradients of different tables of the same head
            gt = Gf.reshape(16, 64, 256 * 48)
            gt = gt / gt.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            cg = gt @ gt.transpose(1, 2)
            off = ~torch.eye(64, dtype=torch.bool, device=dev)
            row["grad_table_cos_mean"] = cg[:, off].mean().item()
            row["grad_table_cos_abs_mean"] = cg[:, off].abs().mean().item()
            Cres["layers"].append(row)
            log(f"C layer {l}: R(sgd)={row['sgd']['R_token_mean']:.2f} R(adam_sign)={row['adam_sign']['R_token_mean']:.2f} "
                f"R(rownorm)={row['rownorm']['R_token_mean']:.2f}")
        res["C"] = Cres

    # ------------------------------------------------------------ E2: finite-difference checks
    if is_lut and "E" not in skip:
        log("E: finite differences")
        FD = {}
        # (1) cartridge in isolation, fp32, real checkpoint tables + real inputs
        from spiky.lutorch_ex import ConfidenceLUT, LUTSpec
        FD["cartridge"] = []
        for l in (0, 12, 23):
            cart = carts[l]
            cart.eval()
            z = z0[l][:8192]
            R = torch.randn(z.shape[0], 16, 48, device=dev, generator=torch.Generator(dev).manual_seed(5))
            W = cart.weights
            W.grad = None
            f = (cart(z).double() * R.double()).sum()
            f.backward()
            ana = W.grad.clone()
            # reference ConfidenceLUT (eager embedding_bag autograd) with the same state
            ref = ConfidenceLUT(cart.spec, read_top_n=1, learnable_score=True, table_dropout_rate=0.0).to(dev)
            ref.load_state_dict(cart.state_dict(), strict=False)
            ref.eval()
            ref.weights.grad = None
            (ref(z).double() * R.double()).sum().backward()
            rel_ref = ((ref.weights.grad - ana).norm() / ana.norm()).item()
            flatg = ana.flatten()
            top = flatg.abs().topk(10).indices.tolist()
            rnd = torch.randint(0, flatg.numel(), (10,), device=dev, generator=torch.Generator(dev).manual_seed(7)).tolist()
            ent = []
            for fi in top + rnd:
                idx = torch.unravel_index(torch.tensor(fi), W.shape)
                idx = tuple(int(t) for t in idx)
                eps = 0.5
                with torch.no_grad():
                    w0 = W[idx].item()
                    W[idx] = w0 + eps; fp = (cart(z).double() * R.double()).sum().item()
                    W[idx] = w0 - eps; fm = (cart(z).double() * R.double()).sum().item()
                    W[idx] = w0
                fd = (fp - fm) / (2 * eps)
                ent.append({"idx": list(idx), "analytic": flatg[fi].item(), "fd": fd,
                            "rel_err": abs(fd - flatg[fi].item()) / max(abs(fd), 1e-12)})
            W.grad = None
            FD["cartridge"].append({"layer": l, "rel_diff_vs_reference_autograd_full_tensor": rel_ref,
                                    "n_entries": len(ent), "max_rel_err_top10": max(e["rel_err"] for e in ent[:10]),
                                    "max_rel_err_random10": max(e["rel_err"] for e in ent[10:]),
                                    "median_rel_err": sorted(e["rel_err"] for e in ent)[len(ent) // 2],
                                    "entries": ent[:4]})
            cart.train()
        # (2) whole model (fp8 backbone): directional derivative along the table gradient + single entries
        model.eval()
        x0_, y0_ = gb[0]
        model.zero_grad(set_to_none=True)
        model(x0_, y0_).backward()
        g0 = [params[n].grad.clone() for n in tab_names]
        model.zero_grad(set_to_none=True)
        gn = math.sqrt(sum((g.double() ** 2).sum().item() for g in g0))
        Wn = math.sqrt(sum((params[n].double() ** 2).sum().item() for n in tab_names))

        @torch.no_grad()
        def loss0():
            return model(x0_, y0_).item()
        FD["model"] = {"grad_norm_tables_mb0": gn, "W_norm_tables": Wn, "directional": [], "entries": []}
        base = loss0()
        for rho in (1e-4, 1e-3, 1e-2):
            eps = rho * Wn
            vals = []
            for sgn in (+1, -1):
                with torch.no_grad():
                    for k, n in enumerate(tab_names):
                        params[n].add_(g0[k], alpha=sgn * eps / gn)
                vals.append(loss0())
                with torch.no_grad():
                    for k, n in enumerate(tab_names):
                        params[n].add_(g0[k], alpha=-sgn * eps / gn)
            fd = (vals[0] - vals[1]) / (2 * eps)
            FD["model"]["directional"].append({"rho": rho, "eps_abs": eps, "fd": fd, "analytic_norm": gn,
                                               "ratio": fd / gn, "loss_plus": vals[0], "loss_minus": vals[1], "loss0": base})
        for l in (12,):
            k = l
            gl = g0[k].flatten()
            for fi in gl.abs().topk(3).indices.tolist():
                idx = tuple(int(t) for t in torch.unravel_index(torch.tensor(fi), params[tab_names[k]].shape))
                for eps in (1e-3, 1e-2, 1e-1):
                    with torch.no_grad():
                        w0 = params[tab_names[k]][idx].item()
                        params[tab_names[k]][idx] = w0 + eps; lp = loss0()
                        params[tab_names[k]][idx] = w0 - eps; lm = loss0()
                        params[tab_names[k]][idx] = w0
                    FD["model"]["entries"].append({"layer": l, "idx": list(idx), "eps": eps, "analytic": gl[fi].item(),
                                                   "fd": (lp - lm) / (2 * eps)})
        res["E_fd"] = FD
        model.train()
        del g0

    # ------------------------------------------------------------ D: one-step lookahead
    if is_lut and "D" not in skip:
        log("D: lookahead")
        model.eval()
        W0 = [params[n].detach().clone() for n in tab_names]
        Wn_l = [w.norm().item() for w in W0]
        Wn = math.sqrt(sum(v * v for v in Wn_l))

        @torch.no_grad()
        def held():
            ls = [model(x, y).item() for x, y in hb]
            tv = tv_loss(model).item()
            return sum(ls) / len(ls), tv
        L0, TV0 = held()
        Gf = [Gtab[k] + Gtv[k] for k in range(L)]
        rms = [(S2[k] / n_mb + 2 * Gtv[k] * Gtab[k] + Gtv[k] ** 2).clamp_min(0).sqrt() for k in range(L)]

        def rownorm(Gk):
            rn = Gk.reshape(-1, 48).norm(dim=-1, keepdim=True)
            nz = rn[rn > 0]
            med = nz.median() if nz.numel() else torch.tensor(1.0, device=dev)
            return (Gk.reshape(-1, 48) / rn.clamp_min(1e-30) * med * (rn > 0)).reshape(Gk.shape)

        def whiten(Gk, level, eps_rel):
            shp = {"layer": (1, -1, 48), "head": (16, -1, 48), "table": (16 * 64, -1, 48)}[level]
            M = Gk.reshape(*shp).double()
            gram = M.transpose(1, 2) @ M
            ev, V = torch.linalg.eigh(gram)
            eps = eps_rel * ev.clamp_min(0).mean(-1, keepdim=True)
            P = (V * (ev.clamp_min(0) + eps).rsqrt()[:, None, :]) @ V.transpose(1, 2)
            return (M @ P).float().reshape(Gk.shape)

        zero_hit = [None] * L
        dirs = {
            "SGD": lambda k: Gf[k],
            "Adam-proxy (G/rms_mb)": lambda k: Gf[k] / (rms[k] + ADAM_EPS),
            "Adam-sign (G/|G|)": lambda k: Gf[k] / (Gf[k].abs() + ADAM_EPS),
            "row-norm (per cell)": lambda k: rownorm(Gf[k]),
            "whiten-layer (48-wide)": lambda k: whiten(Gf[k], "layer", 1e-3),
            "whiten-head": lambda k: whiten(Gf[k], "head", 1e-3),
            "whiten-table": lambda k: whiten(Gf[k], "table", 1e-3),
            "row-norm o whiten-layer": lambda k: rownorm(whiten(Gf[k], "layer", 1e-3)),
            "whiten-layer o Adam-proxy": lambda k: whiten(Gf[k] / (rms[k] + ADAM_EPS), "layer", 1e-3),
            "whiten-table o row-norm": lambda k: whiten(rownorm(Gf[k]), "table", 1e-3),
            "SGD (per-layer norm)": lambda k: Gf[k],      # per-layer-normalised step, see below
            "Adam-proxy (per-layer norm)": lambda k: Gf[k] / (rms[k] + ADAM_EPS),
            "token-only SGD (no TV)": lambda k: Gtab[k],
            "token-only Adam-proxy (no TV)": lambda k: Gtab[k] / ((S2[k] / n_mb).sqrt() + ADAM_EPS),
        }
        per_layer_norm = {"SGD (per-layer norm)", "Adam-proxy (per-layer norm)"}
        rhos = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
        D = {"L0_tok": L0, "TV0": TV0, "total0": L0 + TV0, "rhos": rhos, "W_norm": Wn, "W_rms": Wn / math.sqrt(sum(w.numel() for w in W0)),
             "rho_if_adam_lr_times_unit_rms": 3e-3 / math.sqrt(2) * math.sqrt(sum(w.numel() for w in W0)) / Wn, "dirs": {}}
        # evaluation noise floor: a numerically negligible SGD step (rho 1e-6..1e-5 changes the loss by <1e-7) -> any
        # difference to L0 is fp8/bf16 forward noise caused by perturbing the weights
        U = [Gf[k] for k in range(L)]
        Un = math.sqrt(sum(u.norm().item() ** 2 for u in U))
        nf = []
        for rho in (1e-6, 3e-6, 1e-5):
            with torch.no_grad():
                for k, n in enumerate(tab_names):
                    params[n].copy_(W0[k] - rho * Wn / Un * U[k])
            l_, tv_ = held()
            nf.append(l_ - L0)
        with torch.no_grad():
            for k, n in enumerate(tab_names):
                params[n].copy_(W0[k])
        D["noise_floor_dloss"] = nf
        for dn, fn in dirs.items():
            U = [fn(k) for k in range(L)]
            un = [u.norm().item() for u in U]
            Un = math.sqrt(sum(v * v for v in un))
            Ls, TVs = [], []
            for rho in rhos:
                with torch.no_grad():
                    for k, n in enumerate(tab_names):
                        eta = (rho * Wn_l[k] / max(un[k], 1e-300)) if dn in per_layer_norm else (rho * Wn / Un)
                        params[n].copy_(W0[k] - eta * U[k])
                l_, tv_ = held()
                Ls.append(l_); TVs.append(tv_)
            with torch.no_grad():
                for k, n in enumerate(tab_names):
                    params[n].copy_(W0[k])
            tot = [a + b for a, b in zip(Ls, TVs)]
            bi, bt = min(range(len(rhos)), key=lambda i: tot[i]), min(range(len(rhos)), key=lambda i: Ls[i])
            D["dirs"][dn] = {"loss_tok": Ls, "tv": TVs, "d_tok": [L0 - a for a in Ls], "d_total": [L0 + TV0 - a for a in tot],
                             "best_rho_total": rhos[bi], "best_d_total": L0 + TV0 - tot[bi],
                             "best_rho_tok": rhos[bt], "best_d_tok": L0 - Ls[bt]}
            log(f"D {dn}: best rho {rhos[bt]:g} dtok {L0 - Ls[bt]:.5f} | best total rho {rhos[bi]:g} d {L0 + TV0 - tot[bi]:.5f}")
            del U
        res["D"] = D

    with open(os.path.join(args.out, f"{args.tag}.json"), "w") as f:
        json.dump(res, f)
    log("done", args.tag)


if __name__ == "__main__":
    main()
