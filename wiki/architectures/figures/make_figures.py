"""Generate the architecture diagrams (SVG). Run `python make_figures.py && python render.py`.

Every figure is one function; each writes <name>.svg next to this script.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgdraw import Canvas, MUTED, INK  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def out(name):
    return os.path.join(HERE, name + ".svg")


# ---------------------------------------------------------------------------
def lut_sdpa():
    c = Canvas(900, 640, "LUT SDPA: data flow of one attention layer")
    c.text(450, 26, "LUT SDPA: one attention layer", 17, bold=True)
    c.text(450, 46, "table sizes are those of the June 2026 LUTGPT recipe", 11.5, color=MUTED)

    x = c.box(300, 62, 300, 52, ["x  (rank-coded stream, width E)", "a vector whose order is what the tables read"], "stream")
    n = c.box(300, 136, 300, 36, "MeanAbsNorm", "norm")
    c.arrow([x["b"], n["t"]])

    qk = c.box(70, 218, 330, 62, ["qk_lut  (LUT, nap 4)", "emits q and k as real vectors"], "lut")
    v = c.box(560, 218, 270, 76, ["v_lut  (LUT, nap 6)", "emits v, blended by attention scores;", "the blend is read permutationally"], "lut")
    c.line([n["b"], (450, 196)])
    c.line([(235, 196), (695, 196)])
    c.arrow([(235, 196), qk["t"]])
    c.arrow([(695, 196), v["t"]])

    c.group(62, 304, 360, 240, "classic scaled-dot-product attention", bottom=True)
    ln = c.box(70, 330, 330, 42, ["LayerNorm on q and k", "width d_qk each"], "dense")
    rope = c.box(70, 392, 330, 36, "RoPE on q and k", "dense")
    sc = c.box(70, 450, 330, 76, ["scores = q kᵀ / √d_qk", "causal mask, then softmax", "over the earlier tokens"], "dense")
    c.arrow([qk["b"], ln["t"]])
    c.arrow([ln["b"], rope["t"]])
    c.arrow([rope["b"], sc["t"]])

    mix = c.box(560, 456, 270, 64, ["weighted sum of v", "weights = the softmax of the scores"], "dense")
    c.arrow([sc["r"], mix["l"]])
    c.arrow([v["b"], (695, 456)])

    outp = c.box(560, 552, 270, 56, ["out_proj  (LUT, nap 7)", "back to width E"], "lut")
    c.arrow([mix["b"], outp["t"]])
    add = c.box(300, 557, 190, 46, ["x + out_proj(...)", "next block"], "stream")
    c.arrow([outp["l"], add["r"]])

    # skip connection, routed outside the boxes
    g = "#15803d"
    c.line([x["l"], (24, x["l"][1]), (24, 580)], color=g, dash="5 4")
    c.arrow([(24, 580), add["l"]], color=g, dash="5 4")
    c.text(32, 340, "skip", 11.5, anchor="start", color=g)

    c.legend(725, 340, [("lut", "lookup table"), ("dense", "classic operation"), ("stream", "residual stream")])
    c.save(out("lut_sdpa"))


def history_timeline():
    import datetime as dt
    d0 = dt.date(2025, 10, 1)
    d1 = dt.date(2026, 11, 30)
    X0, X1 = 215, 1070
    W, H = 1120, 600
    c = Canvas(W, H, "Timeline of the LUT architectures, October 2025 to November 2026")

    def X(y, m, d):
        return X0 + (dt.date(y, m, d) - d0).days / (d1 - d0).days * (X1 - X0)

    c.text(W / 2, 28, "How the lines moved (bars are approximate, dots are commits or documents)", 16, bold=True)
    # month grid
    months = []
    y, m = 2025, 10
    while dt.date(y, m, 1) <= d1:
        months.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    top, bottom = 70, 548
    for (y, m) in months:
        x = X(y, m, 1)
        c.line([(x, top), (x, bottom)], color="#e2e8f0", sw=1)
        lab = dt.date(y, m, 1).strftime("%b")
        c.text(x + 3, top - 6, lab if m != 1 else lab + " " + str(y), 10.5, anchor="start", color=MUTED)
    c.text(X(2025, 10, 1) + 3, top - 20, "2025", 10.5, anchor="start", color=MUTED, bold=True)
    c.text(X(2026, 1, 1) + 3, top - 20, "2026", 10.5, anchor="start", color=MUTED, bold=True)

    lanes = [
        ("Convolutional networks", "dense"),
        ("Byte-level LUT attention", "lut"),
        ("Permutational (rank-only)", "rank"),
        ("LUT SDPA and dual stream", "lut"),
        ("FFN replacement", "stream"),
    ]
    from svgdraw import PALETTE
    lh = 88
    ly = {}
    for i, (name, kind) in enumerate(lanes):
        y0 = top + 6 + i * lh
        ly[name] = y0
        c.text(12, y0 + 38, name, 13, anchor="start", bold=True)
    mid = lambda name: ly[name] + 34

    def bar(name, a, b, kind, label=None, faded=False):
        fill, stroke = PALETTE[kind]
        y = ly[name] + 22
        c.rect(X(*a), y, X(*b) - X(*a), 24, fill, stroke, 1.4, rx=6, dash="4 3" if faded else None)
        if label:
            c.text(X(*a) + 6, y + 16, label, 11, anchor="start", color=INK)

    def dot(name, d, label, dy=-4, anchor="middle", dx=0, kind="plain", above=True):
        fill, stroke = PALETTE[kind]
        x = X(*d)
        y = mid(name)
        c.items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#1f2933"/>')
        ty = y - 18 + dy if above else y + 26 + dy
        for k, line in enumerate(label.split("|")):
            c.text(x + dx, ty + k * 12.5 - (12.5 * (len(label.split("|")) - 1) if above else 0), line, 10.5, anchor=anchor, color=INK)

    n = "Convolutional networks"
    bar(n, (2025, 11, 14), (2026, 3, 21), "dense")
    dot(n, (2025, 11, 14), "first MNIST workbook|14 Nov", above=False, anchor="start", dx=-6)
    dot(n, (2026, 3, 21), "last convolution commit|21 Mar", above=False, anchor="end", dx=6)

    n = "Byte-level LUT attention"
    bar(n, (2025, 12, 4), (2026, 4, 16), "lut")
    dot(n, (2025, 12, 4), "first LUTTransformer|4 Dec", above=False, anchor="start", dx=-6)
    dot(n, (2026, 4, 9), "attention V3|9 Apr", above=False, anchor="middle")

    n = "Permutational (rank-only)"
    bar(n, (2026, 3, 29), (2026, 5, 12), "rank")
    bar(n, (2026, 10, 12), (2026, 10, 31), "rank", faded=True)
    dot(n, (2026, 3, 29), "ranking_tools.py|29 Mar", above=False, anchor="end", dx=6)
    dot(n, (2026, 4, 29), "BPE port|29 Apr", above=False, anchor="middle", dx=6)
    dot(n, (2026, 10, 12), "dominance stack|in the FFN (Oct)", above=False, anchor="end", dx=6)

    n = "LUT SDPA and dual stream"
    bar(n, (2026, 4, 16), (2026, 7, 26), "lut")
    dot(n, (2026, 4, 16), "byte-level exp247 to exp250|16 Apr", above=False, anchor="end", dx=6)
    dot(n, (2026, 5, 12), "dual stream on BPE|12 May", above=True, anchor="end", dx=4)
    dot(n, (2026, 5, 14), "RoPE runs|14 May", above=False, anchor="start", dx=-4)
    dot(n, (2026, 6, 13), "LUTGPT report|12-13 Jun", above=True, anchor="start", dx=-4)
    dot(n, (2026, 7, 26), "single stream replaces|the dual stream, 26 Jul", above=False, anchor="start", dx=-6)

    n = "FFN replacement"
    bar(n, (2026, 8, 15), (2026, 11, 9), "stream")
    dot(n, (2026, 5, 28), "first try (exp606)|late May", above=False, anchor="end", dx=6)
    dot(n, (2026, 8, 15), "main line|15 Aug", above=False, anchor="end", dx=6)
    dot(n, (2026, 9, 3), "FFN paper 3 Sep,|fixed evaluation 5 Sep", above=True, anchor="end", dx=6)
    dot(n, (2026, 10, 1), "lutorch_ex|1 Oct", above=False, anchor="start", dx=-4)
    dot(n, (2026, 10, 15), "d24 NanoChat|15 Oct to Nov", above=True, anchor="start", dx=-4)

    c.text(W / 2, 580, "Dates are commit dates in anatoli-starostin/spiky (see the pages for pinned links); the lines overlap on purpose.", 11, color=MUTED)
    c.save(out("history_timeline"))


def dual_vs_single():
    c = Canvas(980, 640, "Dual stream against single stream")
    c.text(490, 26, "Dual stream (April to July 2026) against single stream (from July 2026)", 16, bold=True)
    g = "#15803d"
    # ---------------- left: dual stream
    c.text(235, 56, "Dual stream: two residual streams", 14, bold=True)
    tok = c.box(150, 70, 170, 36, "tokens", "plain")
    emb = c.box(150, 124, 170, 40, ["embedding", "width E"], "dense")
    c.arrow([tok["b"], emb["t"]])
    c.group(40, 186, 270, 232, "block, repeated N times", bottom=True)
    blk = c.box(60, 210, 230, 78, ["attention block", "rank-coded stream x, width E", "qk_lut, v_lut, SDPA, out_proj"], "lut", fs=12.5)
    res = c.box(60, 312, 230, 46, ["residual_lut", "E to D, a table read"], "lut", fs=12.5)
    c.arrow([emb["b"], (175, 186), blk["t"]]) if False else None
    c.arrow([(235, 164), (235, 210)])
    c.arrow([blk["b"], res["t"]])
    # stream labels
    c.text(40, 436, "E stream (rank-coded): x + out_proj, as in the LUT SDPA figure", 11.5, anchor="start", color="#6d28d9")
    # D stream on the right side of the group
    d0 = c.box(350, 124, 120, 40, ["emb_resid_lut", "E to D"], "lut", fs=12)
    c.arrow([(320, 144), (350, 144)])
    c.line([(290, 335), (410, 335)], color=g)
    c.arrow([(410, 335), (410, 470)], color=g)
    c.line([(410, 164), (410, 335)], color=g)
    c.items.append('<circle cx="410" cy="335" r="9" fill="#ffffff" stroke="#15803d" stroke-width="1.6"/>')
    c.text(410, 339.5, "+", 14, color="#15803d", bold=True)
    c.text(420, 260, "D stream:", 11.5, anchor="start", color=g, bold=True)
    c.text(420, 275, "Euclidean sum,", 11.5, anchor="start", color=g)
    c.text(420, 290, "width D", 11.5, anchor="start", color=g)
    lnf = c.box(300, 470, 220, 36, "LayerNorm", "norm")
    un = c.box(300, 524, 220, 40, ["Linear unembedder", "D to vocabulary"], "dense")
    c.arrow([(410, 506), (410, 524)]) if False else c.arrow([lnf["b"], un["t"]])
    c.arrow([(410, 470), (410, 470)], color=g) if False else None

    # ---------------- right: single stream
    c.text(755, 56, "Single stream", 14, bold=True)
    c.line([(560, 66), (560, 600)], color="#cbd5e1", sw=1.2, dash="4 4")
    tok2 = c.box(670, 70, 170, 36, "tokens", "plain")
    emb2 = c.box(670, 124, 170, 40, ["embedding", "width D"], "dense")
    c.arrow([tok2["b"], emb2["t"]])
    c.group(600, 186, 310, 232, "block, repeated N times", bottom=True)
    b2 = c.box(620, 210, 270, 78, ["attention block (the FFN line adds an FFN)", "one residual stream x, width D", "each sub-block adds its output to x"], "lut", fs=12.5)
    c.arrow([(755, 164), (755, 210)])
    c.line([b2["r"], (906, b2["r"][1]), (906, 330), (755, 330)], color=g, dash="5 4")
    add = c.box(620, 312, 270, 42, "x + sub-block output", "stream")
    c.arrow([b2["b"], add["t"]]) if False else None
    c.arrow([(755, 288), (755, 312)])
    lnf2 = c.box(645, 470, 220, 36, "LayerNorm", "norm")
    un2 = c.box(645, 524, 220, 40, ["Linear unembedder", "D to vocabulary"], "dense")
    c.arrow([lnf2["b"], un2["t"]])
    c.arrow([(755, 354), (755, 470)])
    c.text(490, 616, "Colour: blue = lookup table, grey = classic operation, green = Euclidean stream. Layout is schematic.", 11, color=MUTED)
    c.save(out("dual_vs_single"))


def conv_lut():
    c = Canvas(980, 560, "Wiring of the convolutional LUT layers")
    c.text(490, 26, "Two ways to wire a LUT layer to an image", 16, bold=True)
    c.line([(490, 44), (490, 520)], color="#cbd5e1", sw=1.2, dash="4 4")
    c.text(245, 58, "Conv2DLUT: one table set shared by every patch", 13.5, bold=True)
    a = c.box(70, 76, 350, 46, ["input  [B, C, H, W]", "image or feature map"], "plain")
    b = c.box(70, 148, 350, 52, ["unfold  (kernel k, stride s)", "P = Hp x Wp patches, each a vector of C k k numbers"], "dense", fs=12.5)
    c.arrow([a["b"], b["t"]])
    d = c.box(70, 226, 350, 86, ["MultiHeadLut, applied to every patch", "same anchor pairs and tables at every position", "(weights shared, as in a convolution)"], "lut", fs=12.5)
    c.arrow([b["b"], d["t"]])
    e = c.box(70, 338, 350, 52, ["output per patch  [B P, out_channels]", "sum of the rows read in each head"], "dense", fs=12.5)
    c.arrow([d["b"], e["t"]])
    f = c.box(70, 416, 350, 46, ["reshape  [B, out_channels, Hp, Wp]", "next layer or pooling"], "plain")
    c.arrow([e["b"], f["t"]])
    c.text(735, 58, "ProjectionLUT: one head per patch", 13.5, bold=True)
    a2 = c.box(560, 76, 350, 46, ["input  [B, H, W]", "one flat grid"], "plain")
    b2 = c.box(560, 148, 350, 52, ["patch windows (kernel k, stride s)", "head p may use only the cells of window p as anchors"], "dense", fs=12.5)
    c.arrow([a2["b"], b2["t"]])
    d2 = c.box(560, 226, 350, 86, ["MultiHeadLut with P heads", "each patch has its own tables", "(no sharing across positions)"], "lut", fs=12.5)
    c.arrow([b2["b"], d2["t"]])
    e2 = c.box(560, 338, 350, 52, ["per-patch outputs  [B, P, O]", "O numbers per patch"], "dense", fs=12.5)
    c.arrow([d2["b"], e2["t"]])
    f2 = c.box(560, 416, 350, 62, ["optional fold: scatter-add", "each patch writes its O numbers to O of the", "cells of its window in an output grid [B, H, W]"], "plain", fs=12.5)
    c.arrow([e2["b"], f2["t"]])
    c.text(490, 548, "Colour: blue = lookup table, grey = reshaping. Layout is schematic; code in the page.", 11, color=MUTED)
    c.save(out("conv_lut"))


def ffn_block():
    c = Canvas(980, 660, "FFN replacement: one transformer block")
    c.text(490, 26, "FFN replacement: only the feed-forward slot changes", 16, bold=True)
    c.text(235, 56, "Dense block", 14, bold=True)
    x = c.box(95, 70, 280, 40, "x  (residual stream, width D)", "stream", fs=12.5)
    n1 = c.box(95, 130, 280, 32, "LayerNorm", "norm")
    at = c.box(95, 180, 280, 56, ["attention", "Q, K, V linear, RoPE, SDPA, out"], "dense", fs=12.5)
    n2 = c.box(95, 262, 280, 32, "LayerNorm", "norm")
    ff = c.box(95, 312, 280, 92, ["feed-forward", "Linear D to 4D", "GELU", "Linear 4D to D"], "dense", fs=12.5)
    o = c.box(95, 440, 280, 40, "x + attention + feed-forward", "stream", fs=12.5)
    c.arrow([x["b"], n1["t"]]); c.arrow([n1["b"], at["t"]]); c.arrow([at["b"], n2["t"]]); c.arrow([n2["b"], ff["t"]]); c.arrow([ff["b"], o["t"]])
    g = "#15803d"
    c.line([x["l"], (60, x["l"][1]), (60, 460)], color=g, dash="5 4")
    c.arrow([(60, 460), o["l"]], color=g, dash="5 4")
    c.line([(490, 66), (490, 625)], color="#cbd5e1", sw=1.2, dash="4 4")
    c.text(735, 56, "Block with a LUT feed-forward", 14, bold=True)
    x2 = c.box(595, 70, 280, 40, "x  (residual stream, width D)", "stream", fs=12.5)
    n1b = c.box(595, 130, 280, 32, "LayerNorm", "norm")
    atb = c.box(595, 180, 280, 56, ["attention (unchanged)", "dense, RoPE, SDPA"], "dense", fs=12.5)
    n2b = c.box(595, 262, 280, 32, "LayerNorm", "norm")
    c.arrow([x2["b"], n1b["t"]]); c.arrow([n1b["b"], atb["t"]]); c.arrow([atb["b"], n2b["t"]])
    c.group(570, 308, 330, 264, "")
    c.text(582, 566, "LUT feed-forward", 11.5, anchor="start", color=MUTED, italic=True)
    cp = c.box(595, 340, 280, 44, ["compress", "Linear D to H d"], "dense", fs=12.5)
    lu = c.box(595, 404, 280, 74, ["LUT cartridge", "H heads: sign bits of anchor pairs ->", "address -> read and add table rows"], "lut", fs=12)
    dc = c.box(595, 506, 280, 44, ["decompress   Linear H d to D"], "dense", fs=12.5)
    c.arrow([n2b["b"], cp["t"]]); c.arrow([cp["b"], lu["t"]]); c.arrow([lu["b"], dc["t"]])
    o2 = c.box(595, 592, 280, 40, "x + attention + LUT output", "stream", fs=12.5)
    c.arrow([dc["b"], o2["t"]])
    c.line([x2["l"], (560, x2["l"][1]), (560, 612)], color=g, dash="5 4")
    c.arrow([(560, 612), o2["l"]], color=g, dash="5 4")
    c.text(490, 650, "Blue = lookup table, grey = classic operation, yellow = norm, green = residual stream. Schematic.", 11, color=MUTED)
    c.save(out("ffn_block"))


FIGS = {"lut_sdpa": lut_sdpa, "history_timeline": history_timeline, "dual_vs_single": dual_vs_single, "conv_lut": conv_lut, "ffn_block": ffn_block}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()
        print("wrote", n)
