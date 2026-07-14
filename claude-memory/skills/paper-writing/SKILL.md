---
name: paper-writing
description: >
  Universal, toolchain-agnostic workflow for authoring LaTeX papers / technical
  writeups and compiling them to a real PDF on any machine. Detects the LaTeX
  engine (latexmk → pdflatex → tectonic, never assume), uses an arXiv-style
  single-column article template with the recurring gotchas baked in
  (\texorpdfstring in headings, no math in captions, pgfplots compat, hyperref
  last), and two figure paths (matplotlib from a reproducible venv → vector PDF,
  or inline TikZ/pgfplots). Self-contained, with no version-control assumptions.
  Trigger on: writing/editing a paper or writeup, compiling a .tex to PDF, making
  a figure for a paper.
---

# Paper Writing

A portable workflow to author LaTeX and compile it to a real PDF on any host.
Detect the toolchain first, then apply the conventions. Nothing here is tied to a
machine, user, or specific paper.

## 1. Detect the LaTeX engine (do this first)

Never assume `pdflatex` exists. Pick the first engine that's present:

```sh
if command -v latexmk   >/dev/null; then ENGINE=latexmk
elif command -v pdflatex >/dev/null; then ENGINE=pdflatex
elif command -v tectonic >/dev/null; then ENGINE=tectonic
else echo "no LaTeX engine found"; fi
echo "using: $ENGINE"
```

Compile with whichever you found (all produce `<name>.pdf`):

- **latexmk** (best — handles reruns, bibtex, pgfplots itself):
  ```sh
  latexmk -pdf -interaction=nonstopmode -halt-on-error article.tex
  latexmk -c        # clean aux files, keep the PDF
  ```
- **pdflatex** (run twice so refs/citations resolve; add bibtex if using `.bib`):
  ```sh
  pdflatex -interaction=nonstopmode article.tex >/dev/null 2>&1
  pdflatex -interaction=nonstopmode article.tex 2>&1 \
    | grep -E "^!|Error|Warning|Overfull|Underfull"
  ```
- **tectonic** (self-contained, no system TeX Live; fetches+caches packages, does
  the reruns for you):
  ```sh
  tectonic -X compile article.tex     # newer CLI
  # or older single-file form:  tectonic article.tex
  ```
  tectonic supports TikZ/pgfplots and `\includegraphics` (PDF/PNG) out of the box.
  It does NOT do pgfplots *external* mode — keep `\usepgfplotslibrary{external}`
  OFF (the default). First compile needs network to fetch packages; then offline.

If a host has only a network and no TeX Live, installing `tectonic` (a single
static binary, e.g. to `~/.local/bin`) is the lightest way to get a full engine.

## 2. The acceptance bar: a real, non-empty PDF

A document is not done until the engine produces a **real, non-empty PDF with no
`^!`/Error lines**. Always verify the artifact, not just the exit code:

```sh
ls -l article.pdf                              # must be non-zero
pdfinfo article.pdf 2>/dev/null | grep Pages   # page count, if pdfinfo present
```

If the PDF is missing or 0 bytes, the build FAILED regardless of exit code — read
the `.log` (or the engine's stderr) and fix it before doing anything else. Never
report success without a compiled PDF in hand.

## 3. LaTeX template (arXiv single-column)

Single-column `article` (arXiv preprint style), 11pt, 1in margins, `lmodern`.
Minimal preamble:

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[numbers]{natbib}
\usepackage{tikz}
\usepackage{pgfplots}\pgfplotsset{compat=1.18}
\usepackage{hyperref}      % load hyperref LAST
```

Citations: `natbib` with `[numbers]`; `\citep`/`\citet` for parenthetical/textual,
with either a `thebibliography` environment or a `.bib` + `\bibliographystyle`.

## 4. Gotchas that keep biting

- **Math in section titles** breaks hyperref bookmarks — wrap it:
  `\section{The \texorpdfstring{$\tau$}{tau} operator}`.
- **Math in `\caption{}`** — avoid; prefer plain-text phrasing (it can break the
  list-of-figures / bookmarks).
- **pgfplots** — always `\pgfplotsset{compat=1.18}` (or your version) or you get
  deprecation warnings and shifted axes.
- **hyperref loads LAST** in the preamble (except a few packages like `cleveref`
  that must come after it).
- **tectonic + external pgfplots** — leave externalisation off (see §1).
- **Overfull/Underfull \hbox** — warnings to tidy, not errors.

## 5. Figures

### (a) matplotlib from a dedicated venv → vector PDF
Use a reproducible, isolated venv — never the system Python. Save as **PDF**
(vector, best for LaTeX) or high-DPI PNG.

```sh
uv venv ~/.venvs/paper-plotting        # fast+reproducible; else python3 -m venv + pip
uv pip install --python ~/.venvs/paper-plotting/bin/python matplotlib numpy pandas
~/.venvs/paper-plotting/bin/python make_fig.py   # writes fig.pdf next to the .tex
```
```python
# make_fig.py
import matplotlib; matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt, numpy as np
x = np.linspace(0, 2*np.pi, 200)
fig, ax = plt.subplots(figsize=(4, 2.6))
ax.plot(x, np.sin(x)); ax.set_xlabel("x"); ax.set_ylabel("sin x")
fig.tight_layout(); fig.savefig("fig.pdf")        # vector PDF for LaTeX
```
```latex
\begin{figure}[t]\centering
  \includegraphics[width=.6\linewidth]{fig.pdf}
  \caption{Generated with matplotlib.}
\end{figure}
```

### (b) inline TikZ / pgfplots
For plots/diagrams that live in the source (no external file):
```latex
\begin{tikzpicture}
  \begin{axis}[xlabel=x, ylabel=y, width=.7\linewidth, height=5cm]
    \addplot[blue, thick, domain=0:6.283, samples=100] {sin(deg(x))};
  \end{axis}
\end{tikzpicture}
```
Diagrams (boxes/arrows/graphs) go in a plain `tikzpicture`. All engines above
render TikZ/pgfplots — test-compile before considering it done.
