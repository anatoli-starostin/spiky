"""Build the live HTML report from run logs + hand-written notes.

Data (tables, curves, benchmark) are regenerated from runs/*/{config,summary,metrics}; the abstract and
per-batch interpretation come from report_notes.json, which is edited by hand as results come in.
Output: report/drawer_memory_report.html (override with REPORT_OUT).
"""
import csv
import html
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUT = os.environ.get("REPORT_OUT", os.path.join(HERE, "report", "drawer_memory_report.html"))
sys.argv = sys.argv[:1] + ["night"]  # run_grid reads argv at import
import run_grid  # noqa: E402

# one colour per model family, fixed across every chart (reference palette, slots in fixed order)
FAMILY = {
    "softmax": ("softmax attention", 7),
    "softmax_rope": ("softmax attention + RoPE positions", 7),
    "linear": ("linear attention", 2),
    "linear_lean": ("linear attention, F=32 (param-matched)", 6),
    "lut_hard_tied": ("drawers, hard", 1),
    "lut_nbr_tied": ("drawers + neighbour", 3),
    "lut_conf_tied": ("drawers + confidence", 4),
    "lut_nc_tied": ("drawers + neighbour + confidence", 5),
    "lut_hard": ("drawers, hard, untied", 1),
    "lut_nc": ("drawers + nbr + conf, untied", 5),
    "hybrid": ("hybrid: local window 16 + drawers, half width each", 8),
    "local_then_lut_hard_tied": ("layer 1 local window 16, layer 2 drawers", 4),
    "local": ("local softmax window 16 only", 7),
}


def shade(cfg):
    """Lighter shades of the family colour: 6-bit tables, non-additive writes, powered reads, local-only window."""
    if cfg["mixer"] in ("local", "softmax_rope"):
        return 0.45 if cfg["mixer"] == "local" else 1.0
    if not cfg["mixer"].startswith("lut"):
        return 1.0
    op = 0.45 if cfg.get("nap", 6) >= 6 else 1.0
    op *= {"add": 1.0, "decay": 0.6, "ema": 0.35}.get(cfg.get("write", "add"), 1.0)
    return op * (0.6 if cfg.get("read_power", 1) != 1 else 1.0)


def family(cfg):
    m = cfg["mixer"]
    if m == "linear" and cfg.get("feat") == 32:
        return "linear_lean"
    return m if m in FAMILY else m


def label(cfg):
    fam = family(cfg)
    base = FAMILY.get(fam, (fam, 1))[0]
    if cfg["mixer"].startswith(("lut", "hybrid", "local_then_lut")):
        tph = cfg["feat"] // (2 ** cfg["nap"])
        base += f" {tph}×{cfg['nap']}b"
    if cfg.get("write", "add") != "add":
        base += {"decay": ", decay write", "ema": ", overwrite-style (EMA) write"}[cfg["write"]]
    if cfg.get("read_power", 1) != 1:
        base += f", match count^{cfg['read_power']}"
    if cfg.get("learned_addr"):
        base += f", LEARNED addressing (top-{cfg['learned_addr']} by learned cell keys, control)"
    if cfg.get("init_mem"):
        base += ", learned initial memory"
    if cfg.get("out_gate"):
        base += ", normed + gated read"
    if cfg.get("bitflip"):
        base += f", soft read over {2 ** cfg['bitflip']} cells (flip {cfg['bitflip']} least-certain bits)"
    if cfg.get("topk"):
        base += f", soft read of top {cfg['topk']} cells"
    if cfg.get("anneal_frac"):
        base += f", soft→hard over {cfg['anneal_frac']:.0%} of steps"
    if cfg.get("lr", 1e-3) != 1e-3:
        base += f", lr {cfg['lr']:g}"
    return base


def load_runs():
    runs = {}
    for name in os.listdir(RUNS):
        d = os.path.join(RUNS, name)
        if not os.path.isfile(os.path.join(d, "config.json")) and not os.path.isfile(os.path.join(d, "SKIPPED.json")):
            continue
        if not os.path.isfile(os.path.join(d, "config.json")) and os.path.isfile(os.path.join(d, "SKIPPED.json")):
            cfg = json.load(open(os.path.join(d, "SKIPPED.json")))
        else:
            cfg = json.load(open(os.path.join(d, "config.json")))
        skipped = os.path.isfile(os.path.join(d, "SKIPPED.json"))
        summ = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else None
        curve = []
        if os.path.exists(os.path.join(d, "metrics.csv")):
            for r in csv.DictReader(open(os.path.join(d, "metrics.csv"))):
                y = r.get("eval_acc") or r.get("val_bpc")
                curve.append((int(r["step"]), float(y)))
        runs[name] = dict(name=name, cfg=cfg, summary=summ, curve=curve, lm="n_pairs" not in cfg, skipped=skipped)
    return runs


def click(curve, thr=0.5):
    for s, y in curve:
        if y >= thr:
            return s
    return None


def esc(s):
    return html.escape(str(s))


# ---------- SVG charts ----------
def line_panel(title, series, lm=False, w=340, h=210):
    """series: list of (label, slot, dash, points[(step, y)]). MQAR y = recall in [0,1]; LM y = bpc."""
    L, R, T, B = 44, 12, 26, 30
    xs = [p[0] for s in series for p in s[3]] or [1]
    xmax = max(xs)
    if lm:
        ys = [p[1] for s in series for p in s[3]] or [1, 2]
        y0, y1 = math.floor(min(ys) * 10) / 10, min(4.5, math.ceil(max(ys) * 10) / 10)
    else:
        y0, y1 = 0.0, 1.0
    X = lambda s: L + s / xmax * (w - L - R)
    Y = lambda v: T + (1 - (min(max(v, y0), y1) - y0) / (y1 - y0)) * (h - T - B)
    g = [f'<text x="{L}" y="15" class="ptitle">{esc(title)}</text>']
    ticks = [y0 + i * (y1 - y0) / 4 for i in range(5)]
    for t in ticks:
        lab = f"{t:.1f}" if lm else f"{t * 100:.0f}%"
        g.append(f'<line x1="{L}" x2="{w - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                 f'<text x="{L - 6}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    for s, anc in ((0, "start"), (xmax // 2, "middle"), (xmax, "end")):
        g.append(f'<text x="{X(s):.1f}" y="{h - 10}" class="tick" text-anchor="{anc}">{s:,}</text>')
    for lab, slot, dash, pts, op in series:
        if not pts:
            continue
        d = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in [(0, pts[0][1] if lm else 0.004)] + pts)
        last = pts[-1][1]
        val = f"{last:.3f} bpc" if lm else f"{last * 100:.2f}%"
        g.append(f'<polyline points="{d}" fill="none" stroke="var(--s{slot})" stroke-opacity="{op}" stroke-width="2" '
                 f'stroke-dasharray="{dash}" stroke-linejoin="round"><title>{esc(lab)}: final {val}</title></polyline>')
    return f'<svg viewBox="0 0 {w} {h}" class="panel" role="img" aria-label="{esc(title)}">{"".join(g)}</svg>'


def legend(slots_labels):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:var(--s{s[0] if isinstance(s, tuple) else s});opacity:{s[1] if isinstance(s, tuple) else 1}"></i>{esc(l)}</span>'
        for l, s in slots_labels) + \
        '<span><i class="dash"></i>seed 1 (dashed)</span></div>'


def setting_key(cfg, lm):
    if lm:
        if cfg.get("corpus") == "enwik8":
            return (f"enwik8 (byte-level Wikipedia), d={cfg.get('d', 128)}, {cfg.get('layers', 4)} layers, "
                    f"T={cfg.get('T', 256)}, {cfg.get('steps', 3000):,} steps")
        if cfg.get("d", 128) != 128 or cfg.get("T", 256) != 256:
            return f"Tiny Shakespeare, scale-up: d={cfg['d']}, {cfg['layers']} layers, T={cfg['T']}"
        return "Tiny Shakespeare, char-level, 4 layers, T=256"
    k = f"{cfg['n_pairs']} pairs, F={cfg['feat']}"
    if cfg.get("gap"):
        k += f", gap {cfg['gap']}"
    if cfg.get("updates"):
        k += f", {cfg['updates']} keys updated"
    if cfg.get("steps", 3000) != 3000:
        k += f", {cfg['steps']:,} steps"
    return k


def batch_section(gname, notes, runs):
    grid = run_grid.GRIDS[gname]
    names = []
    for c in grid:
        n = run_grid.run_name(c)
        if n not in names:
            names.append(n)
    present = [runs[n] for n in names if n in runs]
    meta = notes.get("batches", {}).get(gname, {})
    out = [f'<section id="{gname}"><h2>{esc(meta.get("title", gname))}</h2>']
    done = sum(1 for r in present if r["summary"] or r.get("skipped"))
    out.append(f'<p class="meta">{done} of {len(names)} runs finished'
               + (f' · {esc(meta["when"])}' if meta.get("when") else "") + "</p>")
    if meta.get("goal"):
        out.append(f'<p><b>Question.</b> {meta["goal"]}</p>')
    if meta.get("conclusion"):
        out.append(f'<div class="concl"><b>What we learned.</b> {meta["conclusion"]}</div>')
    if not present:
        return "".join(out) + "<p class='meta'>Not started yet.</p></section>"
    # small multiples by setting
    groups = {}
    for r in present:
        groups.setdefault(setting_key(r["cfg"], r["lm"]), []).append(r)
    used = {}
    panels = []
    for key, rs in groups.items():
        series = []
        for r in sorted(rs, key=lambda r: (label(r["cfg"]), r["cfg"].get("seed", 0))):
            slot = FAMILY.get(family(r["cfg"]), ("", 1))[1]
            used[label(r["cfg"])] = (slot, shade(r["cfg"]))
            series.append((f'{label(r["cfg"])}, seed {r["cfg"].get("seed", 0)}', slot,
                           "5 4" if r["cfg"].get("seed", 0) else "0", r["curve"], shade(r["cfg"])))
        if not any(r["cfg"]["mixer"] == "linear" for r in rs):  # faint linear reference from other batches
            for r in runs.values():
                if (r["cfg"]["mixer"] == "linear" and r["cfg"].get("feat") != 32 and r["lm"] == rs[0]["lm"]
                        and setting_key(r["cfg"], r["lm"]) == key and r["summary"]):
                    series.append((f"linear attention, seed {r['cfg'].get('seed', 0)} (reference, other batch)", 2,
                                   "2 3", r["curve"], 0.6))
                    used["linear attention (reference, dotted)"] = (2, 0.6)
        panels.append(line_panel(key, series, lm=rs[0]["lm"]))
    out.append(legend(sorted(used.items(), key=lambda kv: kv[1])))
    out.append('<div class="panels">' + "".join(panels) + "</div>")
    # table
    lm = present[0]["lm"]
    upd = any(r["summary"] and r["summary"].get("final_acc_updated") is not None for r in present)
    out.append("<details open><summary>Run table</summary><div class='tw'><table><thead><tr><th>setting</th><th>model</th><th>seed</th>"
               + ("<th class='n'>final val bpc</th><th class='n'>best</th>" if lm else
                  "<th class='n'>final recall</th><th class='n'>step reaching 50%</th>")
               + ("<th class='n'>updated keys</th><th class='n'>unchanged keys</th>" if upd else "")
               + "<th class='n'>params</th><th class='n'>train min</th><th class='n'>max °C</th></tr></thead><tbody>")
    for r in sorted(present, key=lambda r: (setting_key(r["cfg"], r["lm"]), label(r["cfg"]), r["cfg"].get("seed", 0))):
        s = r["summary"]
        if s:
            res = (f"<td class='n'>{s['final_acc']:.3f}</td><td class='n'>{s['best_acc']:.3f}</td>" if lm else
                   f"<td class='n'>{s['final_acc'] * 100:.2f}%</td><td class='n'>{click(r['curve']) or 'never'}</td>")
            tail = f"<td class='n'>{s['params']:,}</td><td class='n'>{s['train_s'] / 60:.1f}</td><td class='n'>{s['max_cpu_temp']}</td>"
            if upd:
                fu, fs = s.get("final_acc_updated"), s.get("final_acc_static")
                res += (f"<td class='n'>{fu * 100:.1f}%</td><td class='n'>{fs * 100:.1f}%</td>" if fu is not None
                        else "<td></td><td></td>")
        elif r.get("skipped"):
            res = "<td class='n' colspan='2'>skipped (seed 0 failed)</td>" + ("<td></td><td></td>" if upd else "")
            tail = "<td></td><td></td><td></td>"
        else:
            res = "<td class='n' colspan='2'>running…</td>" + ("<td></td><td></td>" if upd else "")
            tail = "<td></td><td></td><td></td>"
        out.append(f"<tr><td>{esc(setting_key(r['cfg'], r['lm']))}</td><td>{esc(label(r['cfg']))}</td>"
                   f"<td>{r['cfg'].get('seed', 0)}</td>{res}{tail}</tr>")
    out.append("</tbody></table></div></details>")
    out.append(diag_block(present))
    # LM samples
    samples = [(r, os.path.join(RUNS, r["name"], "sample.txt")) for r in present if r["lm"]]
    samples = [(r, p) for r, p in samples if os.path.exists(p) and r["cfg"].get("seed", 0) == 0]
    if samples:
        out.append("<details><summary>Generated text samples (seed 0, temperature 0.8)</summary>")
        for r, p in samples:
            out.append(f"<h4>{esc(label(r['cfg']))}</h4><pre>{esc(open(p).read())}</pre>")
        out.append("</details>")
    out.append("</section>")
    return "".join(out)


def diag_block(present):
    """Address diagnostics (diag.csv) for runs that have them: last-layer table + match rate over training."""
    rows = []
    for r in present:
        p = os.path.join(RUNS, r["name"], "diag.csv")
        if os.path.exists(p):
            rows += [dict(run=r, **d) for d in csv.DictReader(open(p))]
    if not rows:
        return ""
    out = ["<h3>Address diagnostics</h3><p class='meta'>Measured on 256 fresh sequences. match = query lands in the "
           "same cell as the token that stored its value (share of tables); collide = shares a cell with an unrelated "
           "pair (chance = 1/cells); read share = weight of the correct value in what the read returns.</p>"]
    series = []
    for r in present:
        pts = sorted((int(d["step"]), float(d["match_value"])) for d in rows if d["run"] is r and d["layer"] == "0")
        if len(pts) > 1:
            series.append((f"{label(r['cfg'])}: layer 1 address match", FAMILY.get(family(r["cfg"]), ("", 1))[1],
                           "0", pts, shade(r["cfg"])))
            series.append((f"{label(r['cfg'])}: recall", FAMILY.get(family(r["cfg"]), ("", 1))[1], "2 3", r["curve"], shade(r["cfg"])))
    if series:
        out.append('<div class="panels">' + line_panel("Layer-1 address match (solid) vs recall (dotted)", series) + "</div>")
    out.append("<div class='tw'><table><thead><tr><th>model</th><th>layer</th><th class='n'>match</th><th class='n'>≥1 table</th>"
               "<th class='n'>collide (chance)</th><th class='n'>occupancy</th><th class='n'>entropy</th>"
               "<th class='n'>bit bias</th><th class='n'>bit corr</th><th class='n'>read share</th></tr></thead><tbody>")
    for d in rows:
        if int(d["step"]) != d["run"]["cfg"]["steps"]:
            continue
        f = lambda k: f"{float(d[k]):.3f}"
        out.append(f"<tr><td>{esc(label(d['run']['cfg']))}</td><td>{int(d['layer']) + 1}</td><td class='n'>{f('match_value')}</td>"
                   f"<td class='n'>{f('any_value')}</td><td class='n'>{f('collide')} ({f('chance_collide')})</td>"
                   f"<td class='n'>{f('occupancy')}</td><td class='n'>{f('entropy')}</td><td class='n'>{f('bit_bias')}</td>"
                   f"<td class='n'>{f('bit_corr')}</td><td class='n'>{f('read_share')}</td></tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def bench_section(notes):
    p = os.path.join(RUNS, "bench_decode", "results.csv")
    meta = notes.get("bench", {})
    out = ['<section id="bench"><h2>Decode speed benchmark</h2>']
    if meta.get("goal"):
        out.append(f'<p><b>Question.</b> {meta["goal"]}</p>')
    if meta.get("conclusion"):
        out.append(f'<div class="concl"><b>What we learned.</b> {meta["conclusion"]}</div>')
    if not os.path.exists(p):
        return "".join(out) + "<p class='meta'>Not run yet.</p></section>"
    rows = list(csv.DictReader(open(p)))
    slot = {"softmax (KV cache)": 7, "linear F=256": 2, "linear F=32": 6, "drawers 16x4 (F=256)": 1, "drawers 4x6 (F=256)": 3}
    panels = []
    for B in sorted({r["batch"] for r in rows}, key=int):
        rs = [r for r in rows if r["batch"] == B]
        ctxs = sorted({int(r["context"]) for r in rs})
        vals = [float(r["us_per_token"]) for r in rs]
        w, h, L, R, T, Bm = 340, 220, 52, 12, 26, 30
        lo, hi = math.log10(min(vals)) - 0.1, math.log10(max(vals)) + 0.1
        X = lambda c: L + ctxs.index(c) / (len(ctxs) - 1) * (w - L - R)
        Y = lambda v: T + (1 - (math.log10(v) - lo) / (hi - lo)) * (h - T - Bm)
        g = [f'<text x="{L}" y="15" class="ptitle">µs per token, batch {B} (log scale)</text>']
        for e in range(math.ceil(lo), math.floor(hi) + 1):
            g.append(f'<line x1="{L}" x2="{w - R}" y1="{Y(10 ** e):.1f}" y2="{Y(10 ** e):.1f}" class="grid"/>'
                     f'<text x="{L - 6}" y="{Y(10 ** e) + 4:.1f}" class="tick" text-anchor="end">{10 ** e:g}</text>')
        for c in ctxs:
            g.append(f'<text x="{X(c):.1f}" y="{h - 10}" class="tick" text-anchor="middle">{c:,}</text>')
        for m in slot:
            pts = sorted((int(r["context"]), float(r["us_per_token"])) for r in rs if r["mixer"] == m)
            if pts:
                g.append(f'<polyline points="{" ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts)}" fill="none" '
                         f'stroke="var(--s{slot[m]})" stroke-width="2"><title>{esc(m)}</title></polyline>')
                for a, b in pts:
                    g.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="3.5" fill="var(--s{slot[m]})">'
                             f'<title>{esc(m)}, context {a:,}: {b} µs/token</title></circle>')
        panels.append(f'<svg viewBox="0 0 {w} {h}" class="panel" role="img">{"".join(g)}</svg>')
    out.append(legend([(m, s) for m, s in slot.items()]).replace('<span><i class="dash"></i>seed 1 (dashed)</span>', ""))
    out.append('<div class="panels">' + "".join(panels) + "</div>")
    out.append("<details><summary>Benchmark table</summary><div class='tw'><table><thead><tr><th>batch</th><th>context</th><th>mixer</th>"
               "<th class='n'>µs/step</th><th class='n'>µs/token</th><th class='n'>state KB/seq</th></tr></thead><tbody>")
    for r in rows:
        out.append(f"<tr><td>{r['batch']}</td><td>{int(r['context']):,}</td><td>{esc(r['mixer'])}</td><td class='n'>{r['us_per_step']}</td>"
                   f"<td class='n'>{r['us_per_token']}</td><td class='n'>{r['state_kb_per_seq']}</td></tr>")
    out.append("</tbody></table></div></details>")
    pc = os.path.join(RUNS, "bench_cost", "results.csv")
    if os.path.exists(pc):
        out.append("<h3>Cost by memory size (batch 64, context 1,024)</h3><div class='tw'><table><thead><tr><th>mixer</th>"
                   "<th class='n'>memory F</th><th class='n'>µs/token</th><th class='n'>state KB/seq</th></tr></thead><tbody>")
        for r in csv.DictReader(open(pc)):
            out.append(f"<tr><td>{esc(r['mixer'])}</td><td class='n'>{r['memory_F']}</td><td class='n'>{r['us_per_token']}</td>"
                       f"<td class='n'>{r['state_kb_per_seq']}</td></tr>")
        out.append("</tbody></table></div>")
    out.append("</section>")
    return "".join(out)


CSS = """
:root{color-scheme:light;--bg:#f4f3ef;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--ink3:#7a7974;--line:#d9d8d2;--grid:#e6e5df;--acc:#2a78d6;--accbg:#dbe8f8;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7;--s8:#e34948}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#111110;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;--line:#3a3a37;--grid:#2e2e2b;--acc:#3987e5;--accbg:#1d3150;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 24px 60px}h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:0 0 6px}h4{margin:14px 0 4px}
.sub{color:var(--ink2);margin:0 0 18px}section,.abstract{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin:0 0 20px}
.abstract{border-left:4px solid var(--acc)}.meta{color:var(--ink3);font-size:13px;margin:0 0 10px}
.concl{background:var(--accbg);border-radius:8px;padding:10px 14px;margin:10px 0 14px}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;margin:0 0 18px;font-size:14px}nav a{color:var(--acc);text-decoration:none}
.panels{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(320px,100%),1fr));gap:14px;margin:8px 0}.panels>*{min-width:0}
section,.abstract,details,main{min-width:0;max-width:100%}p,li,td{overflow-wrap:anywhere}
.panel{width:100%;height:auto;background:var(--card)}.panel text{fill:var(--ink2);font-size:11px}.panel .ptitle{fill:var(--ink);font-size:12.5px;font-weight:600}
.panel .grid{stroke:var(--grid)}.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--ink2);margin:6px 0}
.legend i{display:inline-block;width:16px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}.legend i.dash{background:repeating-linear-gradient(90deg,var(--ink3) 0 5px,transparent 5px 9px)}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:6px 0}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}th{color:var(--ink2);font-weight:600}
.n{text-align:right;font-variant-numeric:tabular-nums}.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%}.tw table{min-width:560px}
@media (max-width:640px){main{padding:14px 10px 40px}section,.abstract{padding:14px 12px;border-radius:8px}h1{font-size:21px}h2{font-size:17px}
body{font-size:14.5px}.panels{grid-template-columns:minmax(0,1fr)}.panel text{font-size:12px}.legend{font-size:12px}table{font-size:12.5px}th,td{padding:5px 6px}pre{font-size:11.5px}}summary{cursor:pointer;color:var(--ink2);margin:8px 0}pre{white-space:pre-wrap;background:var(--bg);padding:10px;border-radius:6px;font-size:12.5px}
.panels+table,h3{margin-top:12px}.abstract ul{margin:6px 0 0;padding-left:20px}.abstract li{margin:3px 0}
"""


def main():
    notes = json.load(open(os.path.join(HERE, "report_notes.json")))
    runs = load_runs()
    order = notes.get("order", ["pilot", "tied", "night", "night2", "poc"])
    sections = []
    for g in order:
        if g == "bench":
            sections.append(bench_section(notes))
        elif g in run_grid.GRIDS:
            sections.append(batch_section(g, notes, runs))
    nav = '<a href="#data">1 data and tasks</a> · <a href="#model">2 model and training</a> · <a href="#findings">3 key findings</a> · ' + " · ".join(f'<a href="#{g}">{esc("batch " + notes.get("batches", {}).get(g, {}).get("short", g) if g != "bench" else "speed benchmark")}</a>' for g in order)
    n_done = sum(1 for r in runs.values() if r["summary"])
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drawer memory — live experiment report</title><style>{CSS}</style></head><body><main>
<h1>Drawer memory: live experiment report</h1>
<p class="sub">LUT-addressed memory written at inference time as a replacement for attention. Generated {time.strftime('%Y-%m-%d %H:%M')} from {n_done} finished runs; abstract last revised {esc(notes.get('abstract_updated', ''))}.</p>
<div class="abstract"><h2>Abstract</h2>{notes['abstract']}</div>
<nav>{nav}</nav>
<section id="data"><h2>1. Data and tasks</h2>{notes.get('chapter_data', '')}</section>
<section id="model"><h2>2. Model and training</h2>{notes.get('chapter_model', '')}</section>
<section id="findings"><h2>3. Key findings — current state of knowledge</h2>{notes.get('findings', '')}</section>
<section><h2>4. Experiments, batch by batch</h2><p>Each batch below states its question, setup and conclusion, followed by its learning curves and a table of every run. Batches are in chronological order; later batches supersede earlier conclusions where they conflict (the key findings above are kept current).</p></section>
<nav>{nav}</nav>
{''.join(sections)}
<section><h2>Glossary</h2>{notes.get('glossary', '')}</section>
</main></body></html>"""
    tmp = OUT + ".tmp"
    open(tmp, "w").write(doc)
    shutil.move(tmp, OUT)
    print("report written:", OUT)


if __name__ == "__main__":
    main()
