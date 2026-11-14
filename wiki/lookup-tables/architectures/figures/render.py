"""Render every figures/*.svg to figures/*.png (needs PyMuPDF: `pip install pymupdf`).

Usage: python render.py [dpi]   -- the PNGs are for checking the layout and for readers
who open the files outside GitHub.
"""
import glob
import os
import sys

import pymupdf

dpi = int(sys.argv[1]) if len(sys.argv) > 1 else 144
here = os.path.dirname(os.path.abspath(__file__))
for svg in sorted(glob.glob(os.path.join(here, "*.svg"))):
    doc = pymupdf.open(svg)
    pix = doc[0].get_pixmap(dpi=dpi)
    png = svg[:-4] + ".png"
    pix.save(png)
    print(os.path.basename(png), pix.width, "x", pix.height)
