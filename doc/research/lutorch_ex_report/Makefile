# Rebuild the lutorch_ex report PDF from its .tex source.
#   make          # build lutorch_ex_report.pdf (latexmk, multi-pass)
#   make clean    # remove LaTeX aux + the PDF
# The compiled PDF is committed alongside the source and MUST be rebuilt + re-committed on every .tex change.
TEX := lutorch_ex_report.tex
PDF := lutorch_ex_report.pdf

all: $(PDF)

$(PDF): $(TEX)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	latexmk -C

.PHONY: all clean
