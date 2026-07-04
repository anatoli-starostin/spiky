---
name: paper-writing
description: Workflow for editing the LaTeX paper "Permutations Are All You Need" in the Overleaf-synced repo.
---

# Paper Writing Workflow

## Repository
- **Repo:** `/home/starost/Permutations-is-all-you-need/`
- **Paper source:** `paper/article.tex`
- **Research notes:** `notes/` directory
- **Synced via Git** to GitHub, which syncs with Overleaf

## Git config (already set)
- email: anatoli.starostin@gmail.com
- name: Anatoly Starostin
- remote: git@github.com:anatoli-starostin/Permutations-is-all-you-need.git

## Before every commit
1. **Compile locally** to check for errors:
   ```
   cd /home/starost/Permutations-is-all-you-need/paper
   pdflatex -interaction=nonstopmode article.tex > /dev/null 2>&1
   pdflatex -interaction=nonstopmode article.tex 2>&1 | grep -E "^!|Error|Warning|Overfull|Underfull"
   ```
   Run twice (first pass resolves references, second pass checks for real issues).

2. **Only push if clean** — zero warnings, zero errors.

## Commit and push
Always pull before push (Overleaf may have made changes):
```
cd /home/starost/Permutations-is-all-you-need
git pull
git add -A
git commit -m "message

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
git push
```

## Key conventions
- **Co-authors are: Eugene Izhikevich, Vyacheslav Kluchnikov, Anatoly Starostin** — never refer to them in third person (no "Izhikevich argues", no "proposed by Kluchnikov")
- **Template:** Single-column `article` class (arXiv preprint style), 11pt, 1in margins, lmodern fonts
- **Citations:** `natbib` with `[numbers]` option, `thebibliography` environment
- **Figures:** TikZ diagrams inline. Test compilation before pushing.
- **Math in section titles:** Use `\texorpdfstring{$\tau$}{tau}` to avoid hyperref PDF bookmark warnings
- **Math in captions:** Avoid math symbols in `\caption{}` — use plain text alternatives
- **Build artifacts:** `.gitignore` excludes .aux, .log, .out, .pdf, etc.

## Experiment data sources
- **Experiment folders:** `/home/starost/spiky/transformer_exps/exp001_*` through `exp232_*`
- **Summary:** `/home/starost/spiky/transformer_exps/SUMMARY.md`
- **Detailed journal:** `/home/starost/spiky/transformer_exps/experiments.md`
- **Research report:** `/home/starost/spiky/transformer_exps/research_report_draft.md`
- **Config/results per experiment:** `config.json`, `metrics.csv`, `summary.json`, `loss.png` in each exp folder

## Research notes (in repo)
- `notes/01_spiking_manifesto_summary.md` — Summary of the Spiking Manifesto
- `notes/02_lut_transformer_experiments.md` — Experiment structure and results
- `notes/03_spiky_library_summary.md` — Spiky library architecture overview
