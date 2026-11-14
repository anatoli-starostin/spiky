# Wiki conventions

## Principles

- **Abstract story, concrete links.** Pages explain ideas, models and math at a readable level; code, runs and raw numbers live in the source repos and are linked, not copied.
- **Nothing is left behind.** Every piece of knowledge found in a source we analyse lands somewhere in the wiki unless it is explicitly dropped by an editorial decision. Drops are recorded in [`SOURCES.md`](SOURCES.md).
- **Models and math over particular numbers.** Report headline results and what they show; avoid big result tables unless the comparison is the point.
- **Slow and reviewed.** One topic per pull request, reviewed by a maintainer before merge.

## Page template

Each content page uses these headings where they apply:

1. **Idea / model**: what it is, with the math.
2. **Why it matters / what was found**: findings, with the evidence strength (see below).
3. **Dead ends**: what did not work, so nobody retries it blindly.
4. **Open ideas**: worth trying, untested.
5. **Sources**: links to code, runs, docs, papers.

Mark noteworthy items with **Idea worth keeping** or **Dead end** in bold at the start of the paragraph.

## Evidence labels

State how solid a result is. Use one of: *measured* (with the setup noted), *claimed* (in a note or report, not re-checked), *untested*, *contradicted*.

**Evaluation caveat.** Some historical language-model results were produced with an evaluation that had a very small validation set and a first-rows bias (older `spiky` branches). Such numbers are comparable only within their own era. Always say which evaluation a number comes from. The corrected protocol is bs48 x 100 steps, skipping the first 12 rows, over 170 shards.

## Math

Plain markdown with LaTeX, rendered by GitHub.

- Inline `$...$`; display `$$...$$` on its own lines.
- No custom macros or `\newcommand`.
- Put subscripts only inside math; write code identifiers in backticks.
- Avoid `|` inside math in tables (use `\lvert`, `\rvert`, or move the formula out of the table).
- After each formula, give a one-line plain reading, and define every symbol on first use.

## Sources and links

- Cite sources as links to the repository, **pinned to a branch or commit**. For branches that may be deleted, prefer a commit URL.
- Use relative links between wiki pages.
- Keep [`SOURCES.md`](SOURCES.md) in sync: tick a source when its content has landed in the wiki, and link the pull request.
- Do not put secrets, keys, or private correspondence in the wiki.

## Terms

Use one name per concept across pages. A term used on several pages is defined where it is first used and linked from the later pages.
