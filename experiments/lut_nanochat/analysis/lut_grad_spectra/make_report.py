"""Turn the raw per-model json of analyze_model.py into results.json / results.csv / summary.md / PNG plots.
Runs with plain numpy + matplotlib (python3.12 + envs/extra-matplotlib, see run_all.sh)."""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
R = {t: json.load(open(os.path.join(args.raw, f"{t}.json"))) for t in
     ("lut1500", "lut500", "lut500_nodrop", "dense2000", "dense5568")
     if os.path.exists(os.path.join(args.raw, f"{t}.json"))}
L = 24
# colour-blind-safe categorical (Okabe-Ito subset)
C = {"lut1500": "#0072B2", "lut500": "#D55E00", "lut500_nodrop": "#E69F00", "dense2000": "#009E73", "dense5568": "#555555",
     "a": "#0072B2", "b": "#D55E00", "c": "#009E73", "d": "#CC79A7", "e": "#56B4E9", "f": "#E69F00"}
LAB = {"lut1500": "LUT step 1500 (dropout 0)", "lut500": "LUT step 500 (dropout 0.2)",
       "lut500_nodrop": "LUT step 500 (dropout off)", "dense2000": "dense AdamW step 2000",
       "dense5568": "dense AdamW step 5568"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 9, "figure.dpi": 110})


def med(x):
    return float(np.nanmedian(np.array(x, dtype=float)))


def dense_group(r, grp, key):
    """per-layer mean over the matrices of a group of a spectral stat."""
    out = []
    for l in range(L):
        v = [m["rows"][l][key] for m in r["A"]["dense_mats"].values() if m["group"] == grp]
        out.append(float(np.mean(v)))
    return out


def dense_mat(r, name, key):
    return [r["A"]["dense_mats"][name]["rows"][l][key] for l in range(L)]


def lut_level(r, which, key, src="tok8"):
    rows = r["A"][src][which]
    return [med(x[key]) if isinstance(x[key], list) else x[key] for x in rows]


# ------------------------------------------------------------------ results.json (compact)
S = {}
for t, r in R.items():
    s = {"ckpt": r["ckpt"], "train_loss_mean_8mb": r["train_loss_mean"], "n_grad_mb": r["n_grad"],
         "table_dropout_in_grad_pass": r["table_dropout_in_grad_pass"], "E_global": r["E"]["global"]}
    if r["kind"] == "lut":
        s["A_layer_level"] = {k: lut_level(r, "layer", k) for k in ("top1", "stable_rank", "eff_rank", "cond_med")}
        s["A_layer_level_plus_tv"] = {k: lut_level(r, "layer", k, "tok8_plus_tv") for k in ("top1", "stable_rank", "eff_rank", "cond_med")}
        s["A_mb0_layer_level"] = {k: lut_level(r, "layer", k, "mb0") for k in ("top1", "stable_rank", "eff_rank", "cond_med")}
        s["A_tv_only_layer_level_top1"] = lut_level(r, "layer", "top1", "tv_only")
        s["A_head_median_over_heads"] = {k: lut_level(r, "head", k) for k in ("top1", "stable_rank", "eff_rank", "cond_med")}
        s["A_table_median_over_tables"] = {k: lut_level(r, "table", k) for k in ("top1", "stable_rank", "eff_rank", "cond_med")}
        s["A_signal_noise"] = r["A"]["signal_noise"]
        s["A_top_dir_alignment"] = r["A"]["top_dir_alignment"]
        s["B"] = r["B"]
        s["C"] = r["C"]
        s["E_fd"] = r["E_fd"]
        s["D"] = r["D"]
        s["tv_loss_value"] = r["tv_loss_value"]
    s["A_dense_mats_per_layer"] = r["A"]["dense_mats"]
    s["E_per_layer_grad_norm"] = r["E"]["per_layer"]
    S[t] = s
json.dump(S, open(os.path.join(args.out, "results.json"), "w"), indent=1)

# ------------------------------------------------------------------ results.csv (per layer rows)
rows = []
for t, r in R.items():
    for l in range(L):
        row = {"model": t, "layer": l}
        if r["kind"] == "lut":
            for lv in ("layer", "head", "table"):
                for k in ("top1", "stable_rank", "eff_rank", "cond_med"):
                    v = r["A"]["tok8"][lv][l][k]
                    row[f"tab_{lv}_{k}"] = med(v) if isinstance(v, list) else v
            sn = r["A"]["signal_noise"]
            row["snr_per_mb"] = sn["snr_per_mb"][l]; row["snr_batch8"] = sn["snr_batch"][l]
            row["noise_cov_top1"] = sn["noise_cov_top1_share"][l]
            row["align_within_head"] = r["A"]["top_dir_alignment"]["within_head_across_tables_mean_abs_cos"][l]
            row["align_across_heads"] = r["A"]["top_dir_alignment"]["across_heads_mean_abs_cos"][l]
            b = r["B"]["layers"][l]
            for k in ("frac_zero", "gini", "top1pct_share", "top10pct_share", "entropy_norm", "min", "median", "max", "std_log_hits"):
                row[f"hits_{k}_median_over_tables"] = b[k]["median"]
            row["spearman_hits_vs_rownorm"] = b["spearman_hits_vs_rownorm_mean"]
            row["spearman_scoremass_vs_rownorm"] = b["spearman_scoremass_vs_rownorm_mean"]
            for d in ("sgd", "adam_sign", "adam_proxy", "rownorm"):
                row[f"R_token_{d}"] = r["C"]["layers"][l][d]["R_token_mean"]
                row[f"R_sums_{d}"] = r["C"]["layers"][l][d]["R_ratio_of_sums"]
                row[f"cos_contrib_{d}"] = r["C"]["layers"][l][d]["mean_cos_contrib"]
            row["grad_table_cos_mean"] = r["C"]["layers"][l]["grad_table_cos_mean"]
            row["grad_table_cos_abs_mean"] = r["C"]["layers"][l]["grad_table_cos_abs_mean"]
        for e in r["E"]["per_layer"]:
            if e["layer"] == l:
                for k, v in e.items():
                    if k != "layer":
                        row[f"gradnorm_{k}"] = v
        for mn, m in r["A"]["dense_mats"].items():
            for k in ("top1", "stable_rank", "eff_rank", "cond_med"):
                row[f"mat_{mn}_{k}"] = m["rows"][l][k]
        rows.append(row)
keys = sorted({k for r_ in rows for k in r_}, key=lambda k: (k not in ("model", "layer"), k))
with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)

# ------------------------------------------------------------------ summary.md
md = []


def mm(x):
    x = np.array(x, dtype=float)
    return f"{np.nanmedian(x):.3g} [{np.nanmin(x):.3g}, {np.nanmax(x):.3g}]"


md.append("## A. spectra (median [min, max] over the 24 layers)\n")
md.append("| model / object | top-1 energy share | stable rank | effective rank | sigma1/sigma_median | dim |")
md.append("|---|---|---|---|---|---|")
for t in ("lut1500", "lut500", "lut500_nodrop"):
    if t not in R: continue
    for lv, nm, dim in (("layer", "tables, whole layer [262144x48]", 48), ("head", "tables, per head [16384x48] (median over 16 heads)", 48),
                        ("table", "tables, per table [256x48] (median over 1024 tables)", 48)):
        v = {k: lut_level(R[t], lv, k) for k in ("top1", "stable_rank", "eff_rank", "cond_med")}
        md.append(f"| {LAB[t]}: {nm} | {mm(v['top1'])} | {mm(v['stable_rank'])} | {mm(v['eff_rank'])} | {mm(v['cond_med'])} | {dim} |")
for t in ("dense2000", "dense5568", "lut1500"):
    if t not in R: continue
    for nm, mats in (("attention q/k/v/proj", ["attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj"]), ("MLP c_fc/c_proj", ["mlp.c_fc", "mlp.c_proj"])):
        if mats[0] not in R[t]["A"]["dense_mats"]: continue
        v = {k: [np.mean([R[t]["A"]["dense_mats"][m]["rows"][l][k] for m in mats]) for l in range(L)] for k in ("top1", "stable_rank", "eff_rank", "cond_med")}
        md.append(f"| {LAB[t]}: {nm} (mean over matrices of a layer) | {mm(v['top1'])} | {mm(v['stable_rank'])} | {mm(v['eff_rank'])} | {mm(v['cond_med'])} | 1536 |")
md.append("")
for t in ("lut1500", "lut500", "lut500_nodrop"):
    if t not in R: continue
    r = R[t]
    md.append(f"### {LAB[t]} - per-layer table-gradient numbers\n")
    md.append("| layer | top1 (layer) | top1 (head med) | top1 (table med) | eff rank (layer) | SNR/mb | noise-cov top1 | align within head | align across heads | top1 incl. TV | top1 TV only |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for l in range(L):
        a = r["A"]
        md.append(f"| {l} | {a['tok8']['layer'][l]['top1']:.3f} | {med(a['tok8']['head'][l]['top1']):.3f} | {med(a['tok8']['table'][l]['top1']):.3f} | "
                  f"{a['tok8']['layer'][l]['eff_rank']:.1f} | {a['signal_noise']['snr_per_mb'][l]:.2f} | {a['signal_noise']['noise_cov_top1_share'][l]:.3f} | "
                  f"{a['top_dir_alignment']['within_head_across_tables_mean_abs_cos'][l]:.3f} | {a['top_dir_alignment']['across_heads_mean_abs_cos'][l]:.3f} | "
                  f"{a['tok8_plus_tv']['layer'][l]['top1']:.3f} | {a['tv_only']['layer'][l]['top1']:.3f} |")
    md.append("")

md.append("## B. hit statistics (per table, median over tables, per layer)\n")
for t in ("lut1500", "lut500"):
    if t not in R: continue
    r = R[t]
    md.append(f"### {LAB[t]}\n")
    md.append("| layer | frac zero-hit cells | Gini | top-1% share | top-10% share | entropy/log256 | min | median | max | std log hits | Spearman hits~rownorm | Spearman score-mass~rownorm |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for l in range(L):
        b = r["B"]["layers"][l]
        md.append(f"| {l} | {b['frac_zero']['median']:.4f} | {b['gini']['median']:.3f} | {b['top1pct_share']['median']:.3f} | {b['top10pct_share']['median']:.3f} | "
                  f"{b['entropy_norm']['median']:.3f} | {b['min']['median']:.0f} | {b['median']['median']:.0f} | {b['max']['median']:.0f} | {b['std_log_hits']['median']:.2f} | "
                  f"{b['spearman_hits_vs_rownorm_mean']:.3f} | {b['spearman_scoremass_vs_rownorm_mean']:.3f} |")
    md.append("")
    md.append("aggregate over all 24x1024 tables (median / p10 / p90): " + "; ".join(
        f"{k}: {v['median']:.3g} / {v['p10']:.3g} / {v['p90']:.3g}" for k, v in r["B"]["aggregate"].items()))
    md.append(f"\nmean hits per cell: {r['B']['tokens'] / 256:.0f} per {r['B']['tokens']} tokens, i.e. {r['B']['hits_per_cell_per_real_step_mean']:.0f} per real optimizer step (x4)\n")
    md.append(f"fused-kernel vs reference-read max relative error over layers: {max(v['rel'] for v in r['B']['verify']):.2e}\n")

md.append("## C. cross-table redundancy R (mean over tokens and heads; 1 = independent, 64 = fully aligned)\n")
for t in ("lut1500", "lut500_nodrop"):
    if t not in R: continue
    r = R[t]
    md.append(f"### {LAB[t]}\n")
    md.append("| layer | R sgd | R adam-sign | R adam-proxy | R row-norm | cos(contrib) sgd | cos(contrib) adam-sign | cos of table grads (signed / abs) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for l in range(L):
        c = r["C"]["layers"][l]
        md.append(f"| {l} | {c['sgd']['R_token_mean']:.2f} | {c['adam_sign']['R_token_mean']:.2f} | {c['adam_proxy']['R_token_mean']:.2f} | {c['rownorm']['R_token_mean']:.2f} | "
                  f"{c['sgd']['mean_cos_contrib']:.4f} | {c['adam_sign']['mean_cos_contrib']:.4f} | {c['grad_table_cos_mean']:.4f} / {c['grad_table_cos_abs_mean']:.4f} |")
    md.append("")
    md.append("median over layers: " + ", ".join(f"R {d}: {med([r['C']['layers'][l][d]['R_token_mean'] for l in range(L)]):.2f}" for d in ("sgd", "adam_sign", "adam_proxy", "rownorm")) + "\n")

md.append("## D. one-step lookahead (held-out loss decrease; token loss, and total = token + lambda*TV)\n")
for t in ("lut1500", "lut500", "lut500_nodrop"):
    if t not in R: continue
    D = R[t]["D"]
    md.append(f"### {LAB[t]}: held-out L0 = {D['L0_tok']:.4f}, TV0 = {D['TV0']:.4f}; ||W||={D['W_norm']:.1f}, rms W={D['W_rms']:.2e}; rho of an Adam step with unit-rms update at lr 2.1e-3: {D['rho_if_adam_lr_times_unit_rms']:.2f}; noise floor (dLoss at negligible steps) {D['noise_floor_dloss']}\n")
    md.append("| direction | best rho (token) | best token-loss decrease | best rho (total) | best total decrease | token decrease at rho=" + " / ".join(f"{x:g}" for x in D["rhos"]) + " |")
    md.append("|---|---|---|---|---|---|")
    for dn, v in sorted(D["dirs"].items(), key=lambda kv: -kv[1]["best_d_tok"]):
        md.append(f"| {dn} | {v['best_rho_tok']:g} | {v['best_d_tok']:.5f} | {v['best_rho_total']:g} | {v['best_d_total']:.5f} | " + " / ".join(f"{x:.4f}" for x in v["d_tok"]) + " |")
    md.append("")

md.append("## E. sanity and gradient-norm shares\n")
for t in ("lut1500", "lut500"):
    if t not in R: continue
    fd = R[t]["E_fd"]
    md.append(f"### {LAB[t]}\n")
    md.append("cartridge in isolation (fp32, real tables and inputs, random output weights): " + "; ".join(
        f"layer {c['layer']}: |fused-bwd - reference autograd|/|g| = {c['rel_diff_vs_reference_autograd_full_tensor']:.1e}, max rel err vs central FD (top-10 / random-10 entries) = {c['max_rel_err_top10']:.1e} / {c['max_rel_err_random10']:.1e}" for c in fd["cartridge"]))
    md.append("\nfull model (fp8 backbone), directional derivative along the table gradient, one micro-batch: " + "; ".join(
        f"rho {d['rho']:g}: FD/analytic = {d['ratio']:.3f}" for d in fd["model"]["directional"]))
    md.append("\nsingle entries, full model (FD is noise-dominated: forward noise ~1e-4 per micro-batch loss): " + "; ".join(
        f"eps {e['eps']:g}: analytic {e['analytic']:.2e}, FD {e['fd']:.2e}" for e in fd["model"]["entries"][:6]) + "\n")
for t in ("lut1500", "lut500", "dense2000", "dense5568"):
    if t not in R: continue
    g = R[t]["E"]["global"]
    md.append(f"{LAB[t]}: global grad norm {g['total_norm']:.4f}; share of squared norm: " + ", ".join(f"{k} {v*100:.2f}%" for k, v in sorted(g["sq_share_by_group"].items(), key=lambda kv: -kv[1])))
md.append("\nper-layer table / compress / decompress / attn gradient norms (LUT step 1500):\n")
md.append("| layer | tables (token) | tables TV | compress | decompress | attn |\n|---|---|---|---|---|---|")
for e in R["lut1500"]["E"]["per_layer"][1:]:
    md.append(f"| {e['layer']} | {e.get('lut_table', 0):.4f} | {e.get('lut_table_tv', 0):.4f} | {e.get('lut_compress', 0):.4f} | {e.get('lut_decompress', 0):.4f} | {e.get('attn', 0):.4f} |")
open(os.path.join(args.out, "summary.md"), "w").write("\n".join(md) + "\n")

# ------------------------------------------------------------------ plots
x = np.arange(L)
# 1 spectra LUT vs dense
fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
lt = R["lut1500"]
panels = [("top1", "top-1 energy share  sigma1^2 / sum sigma^2", True), ("eff_rank", "effective rank (exp entropy of sigma^2)", False),
          ("cond_med", "sigma1 / sigma_median", True)]
for j, (k, ttl, logy) in enumerate(panels):
    a = ax[0, j]
    a.plot(x, lut_level(lt, "layer", k), color=C["lut1500"], lw=2, label="LUT tables: whole layer [262144x48]")
    a.plot(x, lut_level(lt, "head", k), color=C["lut1500"], lw=1.5, ls="--", label="LUT tables: per head (median)")
    a.plot(x, lut_level(lt, "table", k), color=C["lut1500"], lw=1.2, ls=":", label="LUT tables: per table (median)")
    a.plot(x, dense_group(R["dense2000"], "mlp", k), color=C["dense2000"], lw=2, label="dense MLP c_fc/c_proj (1536 wide)")
    a.plot(x, dense_group(R["dense2000"], "attn", k), color=C["f"], lw=2, label="dense attention q/k/v/proj")
    if "dense5568" in R:
        a.plot(x, dense_group(R["dense5568"], "mlp", k), color=C["dense5568"], lw=1.2, ls="--", label="dense MLP, step 5568")
    a.set_title(ttl); a.set_xlabel("layer")
    if logy: a.set_yscale("log")
    if k == "eff_rank": a.set_yscale("log")
    if j == 0: a.legend(fontsize=7)
# normalised effective rank, signal/noise, alignment
a = ax[1, 0]
a.plot(x, np.array(lut_level(lt, "layer", "eff_rank")) / 48, color=C["lut1500"], lw=2, label="LUT whole layer (/48)")
a.plot(x, np.array(lut_level(lt, "table", "eff_rank")) / 48, color=C["lut1500"], lw=1.2, ls=":", label="LUT per table (/48)")
a.plot(x, np.array(dense_group(R["dense2000"], "mlp", "eff_rank")) / 1536, color=C["dense2000"], lw=2, label="dense MLP (/1536)")
a.plot(x, np.array(dense_group(R["dense2000"], "attn", "eff_rank")) / 1536, color=C["f"], lw=2, label="dense attention (/1536)")
a.set_title("effective rank / matrix width"); a.set_xlabel("layer"); a.legend(fontsize=7)
a = ax[1, 1]
for t, ls in (("lut1500", "-"), ("lut500", "--")):
    al = R[t]["A"]["top_dir_alignment"]
    a.plot(x, al["within_head_across_tables_mean_abs_cos"], color=C[t], ls=ls, label=f"{LAB[t]}: tables of one head")
    a.plot(x, al["across_heads_mean_abs_cos"], color=C["c"], ls=ls, lw=1, label=f"{LAB[t]}: across heads")
a.axhline(al["random_baseline"], color="k", lw=0.8, ls=":"); a.text(0.3, al["random_baseline"] * 1.05, "random 48-d", fontsize=7)
a.set_title("|cos| of dominant 48-wide gradient direction"); a.set_xlabel("layer"); a.legend(fontsize=6)
a = ax[1, 2]
for t, ls in (("lut1500", "-"), ("lut500", "--")):
    sn = R[t]["A"]["signal_noise"]
    a.plot(x, sn["snr_per_mb"], color=C[t], ls=ls, label=f"{LAB[t]}: |mean|^2 / noise per 32k-token mb")
a.axhline(1, color="k", lw=0.8, ls=":")
a.set_title("table-gradient signal/noise per micro-batch"); a.set_xlabel("layer"); a.legend(fontsize=7)
fig.suptitle("Output-side (48-wide) gradient spectra of the LUT tables vs dense matrices (8 micro-batches, 262k tokens)")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "spectra_lut_vs_dense.png")); plt.close(fig)

# 2 hit skew
fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
for t, ls in (("lut1500", "-"), ("lut500", "--")):
    B = R[t]["B"]["layers"]
    for k, a, ttl in (("gini", ax[0], "Gini of hits over the 256 cells"), ("top10pct_share", ax[1], "share of hits in the top 10% cells"),
                      ("frac_zero", ax[2], "fraction of cells with zero hits")):
        m_ = [b[k]["median"] for b in B]; lo = [b[k]["p10"] for b in B]; hi = [b[k]["p90"] for b in B]
        a.plot(x, m_, color=C[t], ls=ls, lw=2, label=LAB[t]); a.fill_between(x, lo, hi, color=C[t], alpha=0.12)
        a.set_title(ttl + " (median, p10-p90 over tables)"); a.set_xlabel("layer")
ax[0].legend(fontsize=7)
ax[2].set_yscale("symlog", linthresh=1e-3)
for l, col in ((0, C["a"]), (6, C["b"]), (12, C["c"]), (18, C["d"]), (23, C["f"])):
    lor = R["lut1500"]["B"]["layers"][l]["lorenz_cells_frac_0_to_1_step16"]
    ax[3].plot(np.linspace(0, 1, len(lor)), lor, color=col, label=f"layer {l}")
ax[3].plot([0, 1], [0, 1], "k:", lw=0.8); ax[3].set_title("mean Lorenz curve of hits (LUT 1500)"); ax[3].set_xlabel("fraction of cells (ascending hits)"); ax[3].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "hit_skew.png")); plt.close(fig)

# 2b skew vs row norm
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for t, ls in (("lut1500", "-"), ("lut500", "--")):
    B = R[t]["B"]["layers"]
    ax[0].plot(x, [b["spearman_hits_vs_rownorm_mean"] for b in B], color=C[t], ls=ls, label=LAB[t] + " (hits)")
    ax[0].plot(x, [b["spearman_scoremass_vs_rownorm_mean"] for b in B], color=C[t], ls=ls, lw=1, alpha=0.6, label=LAB[t] + " (score mass)")
ax[0].set_title("Spearman(cell hits, ||G_cell||), mean over tables"); ax[0].set_xlabel("layer"); ax[0].legend(fontsize=7)
for l, col in ((0, C["a"]), (12, C["c"]), (23, C["f"])):
    h = R["lut1500"]["B"]["layers"][l]["hist_log2_hits_over_mean"]
    ax[1].step(h["edges"][:-1] if len(h["edges"]) == len(h["counts"]) else h["edges"][:len(h["counts"])], np.array(h["counts"]) / sum(h["counts"]), where="post", color=col, label=f"layer {l}")
ax[1].set_title("histogram of log2(hits / mean hits) over all cells (LUT 1500)"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "hit_vs_gradnorm.png")); plt.close(fig)

# 3 redundancy
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for t, a in (("lut1500", ax[0]), ("lut500_nodrop", ax[1])):
    for d, col, nm in (("sgd", C["a"], "SGD direction G"), ("adam_sign", C["b"], "Adam-normalised G/(|G|+eps)"),
                       ("rownorm", C["c"], "row-normalised"), ("adam_proxy", C["d"], "Adam-proxy G/rms_mb")):
        a.plot(x, [R[t]["C"]["layers"][l][d]["R_token_mean"] for l in range(L)], color=col, lw=2 if d != "adam_proxy" else 1, ls="-" if d != "adam_proxy" else "--", label=nm)
    a.axhline(1, color="k", ls=":", lw=0.8)
    a.set_title(f"{LAB[t]}: R = |sum_t dy_t|^2 / sum_t |dy_t|^2 (1 = independent, 64 = aligned)", fontsize=8); a.set_xlabel("layer"); a.set_ylabel("R (mean over tokens, heads)")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "redundancy_R.png")); plt.close(fig)

# 4 lookahead
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999", "#8B4513", "#7B68EE", "#B22222", "#2E8B57", "#FF69B4", "#4682B4"]
for a, t in zip(ax, ("lut1500", "lut500", "lut500_nodrop")):
    D = R[t]["D"]
    for (dn, v), col in zip(D["dirs"].items(), cols):
        if "no TV" in dn or "per-layer" in dn: continue
        a.plot(D["rhos"], v["d_tok"], marker="o", ms=3, color=col, lw=1.5, label=dn)
    a.axhline(0, color="k", lw=0.8)
    nf = np.std(D["noise_floor_dloss"]) if len(D["noise_floor_dloss"]) else 0
    a.axhspan(-2 * abs(np.mean(D["noise_floor_dloss"])) - nf, 2 * abs(np.mean(D["noise_floor_dloss"])) + nf, color="k", alpha=0.08)
    a.set_xscale("log"); a.set_ylim(-0.012, 0.011)
    a.set_xlabel("relative step  rho = ||dW|| / ||W||  (all tables)"); a.set_ylabel("held-out token-loss decrease")
    a.set_title(f"{LAB[t]} (L0={D['L0_tok']:.3f})")
ax[0].legend(fontsize=6.5, ncol=1)
fig.suptitle("One-step lookahead: held-out loss decrease after W <- W - eta*U on all LUT tables (grey band: fp8 evaluation noise)")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "lookahead_curves.png")); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharey=False)
for a, t in zip(ax, ("lut1500", "lut500", "lut500_nodrop")):
    D = R[t]["D"]
    items = sorted(D["dirs"].items(), key=lambda kv: kv[1]["best_d_tok"])
    a.barh([k for k, _ in items], [v["best_d_tok"] for _, v in items], color=C["a"], label="token loss (best rho)")
    a.barh([k for k, _ in items], [v["best_d_total"] for _, v in items], color=C["b"], height=0.35, label="total incl. TV (best rho)")
    a.axvline(2 * abs(np.mean(D["noise_floor_dloss"])), color="k", ls=":", lw=0.8)
    a.set_title(f"{LAB[t]}: best one-step decrease", fontsize=9); a.tick_params(axis="y", labelsize=7)
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "lookahead_best.png")); plt.close(fig)

# 5 gradient norm shares
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for t, a in (("lut1500", ax[0]), ("dense2000", ax[1])):
    pl = R[t]["E"]["per_layer"][1:]
    groups = [("lut_table", "LUT tables (token)", C["a"]), ("lut_compress", "compress", C["b"]), ("lut_decompress", "decompress", C["c"]),
              ("attn", "attention", C["f"]), ("dense_mlp", "dense MLP", C["dense2000"])]
    for g, nm, col in groups:
        if g in pl[0]:
            a.plot(x, [e[g] for e in pl], color=col, lw=2, label=nm)
    if t == "lut1500":
        a.plot(x, [e["lut_table_tv"] for e in pl], color=C["a"], ls=":", label="LUT tables (TV term)")
    a.set_yscale("log"); a.set_title(f"{LAB[t]}: ||batch grad|| per layer and group"); a.set_xlabel("layer"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "grad_norms.png")); plt.close(fig)
print("report written to", args.out)
