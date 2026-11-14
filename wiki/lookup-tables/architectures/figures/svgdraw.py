"""Tiny SVG drawing helper for the architecture diagrams (no dependencies).

Arrowheads are drawn as polygons, not SVG markers, so that every renderer
(GitHub, browsers, MuPDF) shows the same picture. Text width is estimated;
`render.py` converts the SVGs to PNG so that the layout can be checked by eye.
"""
from xml.sax.saxutils import escape

FONT = "Helvetica, Arial, sans-serif"
INK = "#1f2933"
MUTED = "#52606d"
PALETTE = {
    # name: (fill, stroke)
    "lut": ("#dbeafe", "#2563eb"),      # lookup-table blocks
    "dense": ("#f1f5f9", "#64748b"),    # ordinary dense / classic blocks
    "norm": ("#fef9c3", "#a16207"),     # normalisations
    "stream": ("#dcfce7", "#15803d"),   # residual streams / data
    "rank": ("#ede9fe", "#6d28d9"),     # rank-coded data
    "warn": ("#fee2e2", "#b91c1c"),
    "plain": ("#ffffff", "#334155"),
}


def text_width(s, fs, bold=False):
    try:
        import pymupdf
        return pymupdf.get_text_length(s, "hebo" if bold else "helv", fs)
    except Exception:
        return 0.56 * fs * len(s) * (1.05 if bold else 1.0)


class Canvas:
    def __init__(self, w, h, title=None):
        self.w, self.h = w, h
        self.items = []
        self.title = title

    # ---- primitives -------------------------------------------------
    def text(self, x, y, s, fs=13, anchor="middle", color=INK, bold=False, italic=False):
        weight = ' font-weight="bold"' if bold else ""
        style = ' font-style="italic"' if italic else ""
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{fs}" '
            f'text-anchor="{anchor}" fill="{color}"{weight}{style}>{escape(s)}</text>'
        )

    def rect(self, x, y, w, h, fill="#fff", stroke=INK, sw=1.5, rx=8, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def line(self, pts, color=INK, sw=1.6, dash=None):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linejoin="round"{d}/>'
        )

    def head(self, x, y, direction, color=INK, size=8):
        dx, dy = {"r": (1, 0), "l": (-1, 0), "d": (0, 1), "u": (0, -1)}[direction]
        px, py = -dy, dx
        a = (x, y)
        b = (x - dx * size * 1.4 + px * size * 0.6, y - dy * size * 1.4 + py * size * 0.6)
        c = (x - dx * size * 1.4 - px * size * 0.6, y - dy * size * 1.4 - py * size * 0.6)
        self.items.append(
            f'<polygon points="{a[0]:.1f},{a[1]:.1f} {b[0]:.1f},{b[1]:.1f} {c[0]:.1f},{c[1]:.1f}" fill="{color}"/>'
        )

    def arrow(self, pts, color=INK, sw=1.6, dash=None):
        """Polyline with an arrowhead at the last point."""
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        if abs(x1 - x0) >= abs(y1 - y0):
            d = "r" if x1 > x0 else "l"
        else:
            d = "d" if y1 > y0 else "u"
        shorten = 8 * 1.4 * 0.9
        dx, dy = {"r": (1, 0), "l": (-1, 0), "d": (0, 1), "u": (0, -1)}[d]
        pts = list(pts[:-1]) + [(x1 - dx * shorten, y1 - dy * shorten)]
        self.line(pts, color, sw, dash)
        self.head(x1, y1, d, color)

    # ---- composites ---------------------------------------------------
    def box(self, x, y, w, h, lines, kind="plain", fs=13, bold_first=True, sw=1.5, dash=None, sub_fs=None):
        """Centered box with one or more text lines; returns the geometry dict."""
        fill, stroke = PALETTE[kind]
        self.rect(x, y, w, h, fill, stroke, sw, dash=dash)
        if isinstance(lines, str):
            lines = [lines]
        sub_fs = sub_fs or fs - 1.5
        n = len(lines)
        lh = fs * 1.3
        total = lh * n
        y0 = y + (h - total) / 2 + fs * 0.95
        for i, s in enumerate(lines):
            first = i == 0 and bold_first and n > 1
            self.text(x + w / 2, y0 + i * lh, s, fs if i == 0 else sub_fs,
                      bold=first, color=INK if i == 0 else MUTED)
        return dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2,
                    t=(x + w / 2, y), b=(x + w / 2, y + h), l=(x, y + h / 2), r=(x + w, y + h / 2))

    def group(self, x, y, w, h, label, color="#94a3b8", fs=12, bottom=False):
        self.rect(x, y, w, h, "none", color, 1.4, rx=10, dash="6 4")
        ty = y + h - 8 if bottom else y + 16
        self.text(x + 10, ty, label, fs, anchor="start", color=MUTED, italic=True)

    def legend(self, x, y, entries):
        """entries: list of (kind, label); stacked downwards."""
        from svgdraw import PALETTE
        for i, (kind, label) in enumerate(entries):
            fill, stroke = PALETTE[kind]
            self.rect(x, y + i * 20, 12, 12, fill, stroke, 1.2, 2)
            self.text(x + 18, y + i * 20 + 10.5, label, 11, anchor="start", color=MUTED)

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img">')
        t = f"<title>{escape(self.title)}</title>" if self.title else ""
        bg = f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>'
        with open(path, "w") as f:
            f.write(head + t + bg + "\n" + "\n".join(self.items) + "\n</svg>\n")
