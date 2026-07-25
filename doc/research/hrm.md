# Hierarchical Reasoning Model (HRM) — notes, and how it might combine with LUT models

*Research note for the spiky/LUTGPT line. Covers what HRM is and why it drew attention, then
analyzes concretely how our lookup-table primitives (`FastMultiHeadLut`,
`HyperplaneMultiHeadLUT`) might be combined with HRM-style recursive reasoning.*

**Tracking issue:** #72 · **Branch:** `research/recursive-reasoning-models`

> **Provenance caveat.** The HRM summary below is written from prior knowledge of the paper and
> the surrounding discussion, not from a live re-read. Treat exact figures (param counts,
> benchmark %s) as **approximate — verify against the primary sources in §9** before quoting them
> as fact. The load-bearing content for us is the architecture idea and the LUT-combination
> analysis (§8), which do not depend on the exact numbers.

---

## 1. Core idea

**HRM** (Sapient Intelligence, 2025) is a small (~27M-parameter) recurrent network that solves hard
symbolic-reasoning puzzles — extreme Sudoku, hard mazes, ARC-AGI tasks — **from ~1000 examples per
task, with no pretraining and no chain-of-thought**. Its thesis: the depth that reasoning needs
should come from **recursion / iterative latent refinement**, not from stacking more layers or
generating more CoT tokens. A fixed-depth transformer is "shallow" for combinatorial search; HRM
gets **effective depth** by running a small model in a loop and letting a latent state converge.

It is explicitly brain-inspired: a **hierarchy of modules operating at different timescales**
(fast local computation under slow abstract control), analogous to cortical hierarchy and
multi-timescale neural dynamics.

## 2. Architecture — two-timescale hierarchical recursion

Two coupled recurrent modules, each a small Transformer-style block (self-attention + gated MLP,
RMSNorm/RoPE-ish):

- **Low-level module `L` (fast):** many update steps; does detailed, local computation conditioned
  on the current high-level state and the input.
- **High-level module `H` (slow):** updated **once per cycle** of `T` low-level steps; carries the
  abstract "plan" / running summary.

**Hierarchical convergence** is the mechanism: within one H-cycle, `L` iterates `T` steps toward a
local equilibrium (a fixed point, given the current `H` and input). Then `H` takes **one** step
using `L`'s converged state, which **resets** `L`'s dynamics for the next cycle. Nesting a
fast inner convergence inside a slow outer update yields high effective depth while keeping each
step's dynamics well-conditioned (it sidesteps the vanishing/exploding gradients of naively
unrolling a deep recurrence). Reasoning happens in **latent space** in a single "forward pass" —
there are no intermediate natural-language reasoning tokens.

## 3. Recursion instead of scale

The bet is compute-reuse over parameter-count: the same ~27M weights are applied over many
recursion steps, so *depth of computation* is decoupled from *number of parameters*. This is why a
tiny model can match or beat far larger LLMs on tasks that reward search/backtracking (Sudoku,
mazes) — the work is in the loop, not the weights.

## 4. Training approach

- **Deep supervision.** The model runs several **segments** (repeated forward passes); state is
  carried (detached) between segments and a loss + gradient is applied at *each* segment, not only
  at the end. This is a large part of what makes the iterative refinement trainable.
- **One-step / implicit gradient.** Rather than backprop-through-time across all `T×cycles`
  recursion steps (expensive, unstable), HRM approximates the gradient using essentially the **last
  step** at the converged state — a fixed-point / implicit-function (DEQ-style) 1-step gradient,
  giving ~O(1) memory in the recursion depth.
- **Adaptive Computation Time (ACT).** A small **Q-learning halting head** decides, per input, how
  many segments to run (halt vs. continue) — so easy instances get less compute and hard ones get
  more. This variable "thinking time" is part of the reported gains.
- Heavy **task-specific data augmentation** (especially for ARC: many augmentations per puzzle at
  train time, and test-time augmentation + voting).

## 5. Reported results & benchmarks *(approximate — verify, §9)*

- **Sudoku-Extreme (9×9, minimal givens):** near-solved, where direct-prediction transformers and
  CoT LLMs score ≈0%.
- **Maze-Hard (large grids, optimal pathfinding):** high accuracy where baselines fail.
- **ARC-AGI-1:** ~40% on the eval set — striking for a 27M model trained on the task data, and
  competitive with much larger systems; ARC-AGI-2 far lower (low single digits).
- Trained from scratch, ~1000 examples/task, no pretraining, no CoT.

## 6. Efficiency

Tiny parameter count (~27M) + tiny data (~1k examples/task) + no pretraining, with reasoning done
in latent recursion rather than token generation. The efficiency story is **compute reuse**: buy
depth with a loop, not with parameters or with long decoded CoT.

## 7. Known criticisms / caveats

Independent analysis (notably the ARC Prize team) argued the credit assignment is subtler than
"the hierarchy did it":

- **The hierarchical H/L architecture contributed less than the training recipe.** Ablations
  swapping in an ordinary similarly-sized transformer under the *same* outer loop recovered much of
  the performance; the **outer refinement loop (deep supervision + iterative segments)** and the
  **data augmentation** were the dominant drivers.
- **Transductive / per-task.** Per-puzzle embeddings and per-task training make it closer to
  learning each puzzle distribution than to general reasoning; generalization claims need care.
- Practically: it is **specialized per task**, and depends on the ACT/refinement machinery.

Net: the *results are real and interesting*, but when we borrow from HRM we should **ablate the
pieces separately** (hierarchy vs. outer-loop vs. augmentation) rather than assume the hierarchy is
the magic.

---

## 8. Combining LUT models with HRM-style recursion

This is the part that matters for spiky. Our LUT primitives (`FastMultiHeadLut`, and the learned
`HyperplaneMultiHeadLUT` — sign-pack an input into a `2^NAP` row index, gather, reduce) make a
transformer block's compute **(nearly) matmul-free**: a lookup replaces a GEMM. HRM makes depth
**recursive**: the same small module runs many times. These two efficiency axes are orthogonal and
compose.

### 8.1 Why the fit is natural

1. **Compute reuse × matmul-free = doubly efficient.** HRM already amortizes a small module over
   many steps; if each step is a lookup instead of a GEMM, the per-step cost collapses too. A
   recursive reasoner where every inner step is a table lookup is a genuinely new efficiency point
   (few params **and** few FLOPs/step).
2. **Discreteness matches the tasks.** LUT indexing is inherently discrete; HRM's target tasks
   (Sudoku cells, maze grid states, ARC transformations) are discrete/combinatorial. A learned
   lookup is a natural *learned discrete state machine*.
3. **Fixed-point framing meets the soft-backward surrogate.** HRM's 1-step implicit (DEQ-like)
   gradient already treats the recursion as converging to a fixed point and approximates the
   gradient there. LUTGPT's **soft-backward surrogate** is likewise an approximate gradient through
   a non-differentiable (hard argmax) lookup. The two approximations are conceptually compatible —
   the discreteness HRM's fixed-point view tolerates is exactly what the LUT introduces.
4. **Lookup tables *are* associative/attractor memories.** "Hierarchical convergence" is `L`
   settling into an attractor each cycle; LUT rows are a natural attractor set to settle *into*.

### 8.2 Where a LUT plugs in

- **(a) Inside the `L` and `H` blocks (lowest-risk).** Replace the attention projections (q/k/v/out)
  and the MLP with `HyperplaneMultiHeadLUT` / `FastMultiHeadLut`, exactly as LUTGPT does to a plain
  transformer. Because `L` runs `T` steps per cycle, the per-step savings **multiply** across the
  recursion.
- **(b) As the recurrent state-transition itself (most interesting).** Model `L`'s fast update as a
  **LUT-keyed transition**: `state → HyperplaneMHL sign-pack index → table row → state increment`.
  The learned hyperplanes quantize the latent into a discrete code and the table stores the
  learned next-state — i.e. a **learned discrete dynamical system / fast-weight associative
  memory**, in place of an attention update.
- **(c) At the `H` (abstract) level as an explicit discrete latent code.** Expose the sign-pack
  index as a symbolic bottleneck the recursion converges to; the learned hyperplane partition
  becomes an interpretable discrete "reasoning state."

### 8.3 Concrete experiments worth trying

Ordered by signal-to-risk. All reuse the existing `HyperplaneMultiHeadLUT` (anchor-pairs init gives
a strict A/B against fixed anchors / a matmul baseline; random init tests learning from scratch).

- **E1 — LUT-ify HRM's blocks (do this first).** Take a reference HRM (or a minimal reimpl), swap
  the q/k/v/out + MLP linears in **both** `L` and `H` for `HyperplaneMultiHeadLUT`. Train on
  **Sudoku-Extreme** and **Maze-Hard**; report accuracy vs. **params and FLOPs** against baseline
  HRM. Question: does matmul-free recursion preserve HRM's reasoning ability?
- **E2 — LUT as the `L`-module recurrence.** Replace only the `L` update with a LUT-keyed state
  transition (8.2b); keep `H` standard. Test whether a **learned lookup recurrence** still achieves
  hierarchical convergence and solves Sudoku. Probes "LUT as dynamical system."
- **E3 — Discrete-latent probe.** Expose the `H`-level sign-pack index as an explicit code; analyze
  whether the learned hyperplane partition aligns with task structure (e.g. Sudoku subgrid /
  constraint clusters). Measures interpretability alongside accuracy.
- **E4 — Efficiency frontier.** HRM(matmul) vs. HRM(LUT) at matched accuracy on **ARC-AGI-1**; plot
  params × FLOPs. Goal: quantify the compute win from recursion × lookup.
- **E5 — Gradient-compatibility study.** Combine HRM's 1-step implicit gradient with LUTGPT's soft
  surrogate backward; check **training stability** under deep supervision + ACT halting (does the
  LUT surrogate destabilize fixed-point training, or ride along cleanly?).
- **E6 — Tie ACT to the LUT temperature (stretch).** Reuse HRM's ACT halting signal to also anneal
  the LUT selection temperature `T_sel`, linking *how long to think* to *how hard the lookup
  commits* to a row.

### 8.4 Risks / open questions

- Hard-argmax non-differentiability **inside** a fixed-point loop may fight convergence; the soft
  backward's stability across `T` recursion steps is untested.
- Capacity: a discrete LUT bottleneck may be too coarse for ARC's diversity (fine for Sudoku/maze).
- Per §7, **ablate carefully** — isolate the LUT contribution from HRM's outer-loop / augmentation
  effects, or we'll mis-attribute gains the way the original HRM results were argued to.

---

## 9. Sources *(verify before quoting figures)*

- **HRM paper:** Guan Wang et al., "Hierarchical Reasoning Model," arXiv:2506.21734 (2025). —
  <https://arxiv.org/abs/2506.21734>
- **Reference code:** Sapient Intelligence — <https://github.com/sapientinc/HRM>
- **Independent analysis / critique:** ARC Prize team, "HRM analysis" (2025) — <https://arcprize.org>
  (search "HRM"). Key claim: the outer refinement loop + augmentation, more than the H/L hierarchy,
  drive the results.
- **Internal (spiky):** `claude/thesis.md` (the LUT thesis); `doc/lutorch/lutgpt_research_report.pdf`
  (LUTGPT report); `src/spiky/lutorch/hyperplane_multi_head_lut.py` and issues #61 / #64 / #70 (the
  hyperplane-LUT primitive this note proposes to recurse).
