---
name: Spiky LUT goal — purely ranking-coded architecture
description: When suggesting LUT-transformer experiments in spiky/, do not propose recipes that drop the inter-block rank canonicalization (out_v2d/out_d2v) or that use raw real-valued Q/K SDPA, even if they score better.
type: feedback
originSessionId: fcdc73bb-26b2-4eb8-b830-7366f678ae01
---
The spiky LUT-transformer research goal is a **purely ranking-coded** architecture: every inter-block residual stream stays projected into ranking space (via `out_v2d / out_d2v` or equivalent), and Q/K go through dominance/canonicalization rather than raw real-valued SDPA. Recipes that drop these (e.g. exp236 `no_rank_canon` with plain LayerNorm between blocks; exp223 line with raw Q/K SDPA + QK-norm) score better empirically — exp237 = 1.480 bpb at 48K vs the ranking-faithful exp260 = 1.464 — but are off-strategy and should not be promoted as the path forward.

**Why:** the project's core thesis (downstream goal: hardware that needs only comparators + table reads, no MAC units) requires the ranking representation to be load-bearing throughout the trunk, not just at the LUT input. Dropping rank-canon between blocks reintroduces real-valued magnitude as a learnable signal — that breaks the eventual hardware mapping and the architectural story.

**How to apply:** when proposing next experiments, candidate ablations, or "what's promising", filter out anything that would (a) replace inter-block V2D/D2V with plain LN, or (b) feed real-valued Q/K to SDPA. Prefer recipes from the exp260 / TinyMHLut-soft lineage and similar pure-ranking variants. If reporting a leaderboard, mark non-ranking-faithful entries as off-strategy rather than treating them as the SOTA to chase.
