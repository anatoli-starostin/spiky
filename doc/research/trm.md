# Tiny Recursive Models (TRM) — literature review and LUT integration ideas

*Author: gpustar (Claude). Companion to nucstar's HRM write-up. Part of the
"small recursive reasoning models" research thread. Last updated 2026-07-25.*

This note reviews the **Tiny Recursive Model (TRM)** — the ~5–7M-parameter recursive
reasoner from Samsung SAIL that followed HRM — and then analyses, concretely, where
Anatoli's **lookup-table (LUT / hyperplane)** primitive could replace or augment a TRM
component and what that might buy us. The science of spiky/LUTGPT is assumed (see
[`claude/thesis.md`](../../claude/thesis.md)); the point here is the *intersection*.

---

## 1. Context: HRM → TRM in one paragraph

**HRM** (Hierarchical Reasoning Model, Sapient Intelligence, Jun 2025) showed that a
**27M-parameter** recurrent network, trained from scratch on **~1000 examples**, can
near-solve tasks that large CoT LLMs fail outright — Sudoku-Extreme, 30×30 Maze-Hard,
and ARC-AGI — by *recursing* a small network many times instead of scaling it up. HRM
used **two** networks running at different "frequencies" (a low-level module `f_L`
looping fast, a high-level module `f_H` updating slowly), justified by a brain analogy,
plus a 1-step **fixed-point / Implicit-Function-Theorem gradient approximation** (backprop
only through the last 2 of 6 function evaluations) and **ACT** (a Q-learning halting head).

**TRM** (*"Less is More: Recursive Reasoning with Tiny Networks"*, Alexia
Jolicoeur-Martineau, Samsung SAIL Montréal, arXiv:2510.04871, Oct 2025) strips HRM down
and *beats* it: **one** tiny 2-layer network, **~5–7M params**, **full backprop** through
the whole recursion (no IFT, no fixed-point assumption), a simpler single-forward-pass
halting, and EMA for stability. It is smaller, simpler, and generalizes better.

An independent ARC-Prize analysis found that **deep supervision** — not the hierarchy — was
the primary driver of HRM's gains (deep supervision alone lifted accuracy 19%→39%; the
hierarchical recursion added little). TRM is essentially the model you get when you take
that lesson seriously.

---

## 2. TRM architecture and the recursion mechanism

### 2.1 The three tensors

At every step TRM juggles three embedded tensors of shape `[B, L, D]`:

- **`x`** — the embedded **question** (fixed for the sample);
- **`y`** — the current **answer / proposed solution** (in HRM terms this was `z_H`; TRM
  reinterprets it as *literally the current embedded solution* — reverse-embed + argmax `y`
  and you get the current guess);
- **`z`** — a **latent reasoning** feature (HRM's `z_L`), free-form scratch state that does
  *not* directly decode to a solution.

TRM's key reinterpretation (§4.2 of the paper): **there is nothing hierarchical here.** You
keep two features simply because the update `z ← f(x, y, z)` needs the question `x`, the
previous answer `y`, and the running scratch `z`; and the update `y ← f(y, z)` deliberately
*omits* `x` so its job is "refine the solution" rather than "reason". Two features (`y`,`z`)
is optimal — one forces the solution to be stored inside the scratch (worse); three-or-more
`z_i` also hurt (Table 2: `y,z`=87.4% vs single-`z`=71.9% vs multi-scale-`z`=77.6% on
Sudoku-Extreme).

### 2.2 The loop (this is the whole model)

```python
def latent_recursion(x, y, z, n=6):
    for i in range(n):          # n steps of latent reasoning
        z = net(x, y, z)        # update scratch given question, answer, scratch
    y = net(y, z)               # then refine the answer once
    return y, z

def deep_recursion(x, y, z, n=6, T=3):
    with torch.no_grad():       # T-1 recursion passes WITHOUT gradients
        for j in range(T - 1):
            y, z = latent_recursion(x, y, z, n)
    y, z = latent_recursion(x, y, z, n)   # one final pass WITH gradients
    return y.detach(), z.detach(), output_head(y), Q_head(y)

# Deep supervision (outer loop), up to N_sup = 16 steps:
for x_input, y_true in loader:
    y, z = y_init, z_init
    for step in range(N_sup):
        x = input_embedding(x_input)
        (y, z), y_hat, q_hat = deep_recursion(x, y, z)
        loss  = softmax_cross_entropy(y_hat, y_true)
        loss += binary_cross_entropy(q_hat, (y_hat == y_true))  # halting head
        loss.backward(); opt.step(); opt.zero_grad()
        if q_hat > 0: break     # ACT early-stop (learned)
```

- **Default `T=3`, `n=6`** → one supervision step contains `T·(n+1) = 21` function
  evaluations; with `n_layers=2` that is an **effective depth of 42 layers per supervision
  step**, from a 2-layer network. Over up to `N_sup=16` supervision steps the model refines
  the same sample many times.
- **Gradients flow through only the *last* recursion pass** (the `T-1` earlier passes are
  `no_grad`), but through that pass TRM backprops the **full `n+1` evaluations** — this is
  real BPTT over the recursion, *not* HRM's 1-step IFT trick. The `no_grad` warm-up passes
  are cheap "get closer to the fixed region" iterations; the graph you actually differentiate
  is bounded, which is what keeps memory tractable (bigger `n` → OOM, the paper's main
  compute wall).
- **Deep supervision** carries `(y, z)` *detached* across the `N_sup` steps — residual-like
  restart points that emulate very deep networks too expensive to run in one pass.

### 2.3 What TRM removed vs HRM

| Aspect | HRM | TRM |
|---|---|---|
| Networks | two (`f_L`, `f_H`) | **one shared** tiny net |
| Depth | 4-layer transformers | **2 layers** (fewer = better, §4.4) |
| Gradient | 1-step IFT / fixed-point approx | **full backprop** through the recursion |
| Halting (ACT) | Q-learning w/ **2 forward passes** (halt+continue loss) | **1 forward pass** (BCE halt only) |
| Justification | biological hierarchy, complex | none — "no theorem, no hierarchy, no biology" |
| Params | 27M | **~5M** (Sudoku) / 7M (ARC-Att) / 19M (Maze-MLP) |

### 2.4 Attention vs attention-free

TRM tried two backbones for the 2-layer `net`:

- **Self-attention** (RoPE, RMSNorm, SwiGLU, no bias) — better for large fixed grids
  (30×30 Maze, 30×30 ARC) where `L` is large.
- **Attention-free MLP-Mixer** — replace self-attention with an MLP over the *sequence*
  dimension. **Best on Sudoku** (9×9, small `L ≤ D`): 87.4% vs 74.7% for the attention
  variant. When `L` is small a full `[L,L]` attention matrix is wasteful; a linear mixer
  is cheaper and generalizes better.

Two knobs matter a lot: **EMA** on the weights (prevents the sharp overfit-then-diverge
collapse on tiny data: 79.9%→87.4%) and heavy **data augmentation** (Sudoku: 1000 shuffles
per example; Maze: 8 dihedral; ARC: 1000 color+dihedral+translation augmentations).

---

## 3. Results (exact numbers from the paper)

**Puzzle benchmarks** (Table 4 — % test accuracy, small-sample training, 1000 examples):

| Method | # Params | Sudoku-Extreme | Maze-Hard |
|---|---|---|---|
| Deepseek R1 (CoT) | 671B | 0.0 | 0.0 |
| Claude 3.7 8K (CoT) | — | 0.0 | 0.0 |
| o3-mini-high (CoT) | — | 0.0 | 0.0 |
| Direct pred (no recursion) | 27M | 0.0 | 0.0 |
| **HRM** | 27M | 55.0 | 74.5 |
| **TRM-Att** | 7M | 74.7 | **85.3** |
| **TRM-MLP** | 5M / 19M | **87.4** | 0.0 |

**ARC-AGI** (Table 5 — % test accuracy, 2 tries):

| Method | # Params | ARC-AGI-1 | ARC-AGI-2 |
|---|---|---|---|
| Deepseek R1 | 671B | 15.8 | 1.3 |
| Gemini 2.5 Pro (32K TTC) | — | 37.0 | 4.9 |
| o3-mini-high | — | 34.5 | 3.0 |
| Bespoke (Grok-4) | 1.7T | 79.6 | 29.4 |
| **HRM** | 27M | 40.3 | 5.0 |
| **TRM-Att** | **7M** | **44.6** | **7.8** |
| TRM-MLP | 19M | 29.6 | 2.4 |

Headline: **TRM at 7M params (<0.01% of a frontier LLM) beats HRM (27M) on every task**,
and beats Deepseek R1 / Gemini 2.5 Pro / o3-mini on these structured puzzles — while still
far below task-specific frontier ensembles (Grok-4 bespoke) on ARC. The depth-matched study
(Table 3) shows TRM's advantage is not just more compute: at equal effective depth TRM's
2-layer-recursed net beats HRM's 4-layer-recursed net (e.g. depth ~42: 87.4% vs 61.6%).

**Costs / caveats.** These are *per-task, trained-from-scratch, single-puzzle-distribution*
models with heavy augmentation — not general reasoners. Training is not cheap for the size:
Sudoku ~18h on 1×L40S; ARC ~3 days on 4×H100 each. The recursion-depth wall is memory
(`n` too large → OOM). Generalization off the training puzzle distribution is untested.

---

## 4. Follow-up work (as of mid-2026)

- **Tiny Recursive Reasoning with a Mamba-2 / Attention hybrid** (arXiv:2602.12078, 2026) —
  swaps the backbone for a Mamba-2 + attention hybrid to push the recursion's sequence model.
- **"What Survives When You Compress a Recursive Reasoner for the Edge?"** (arXiv:2606.26488)
  — quantization/pruning study of recursive reasoners for edge deployment. **Directly
  adjacent to the LUT thesis** (extreme compression of a recursive reasoner).
- **Tab-TRM** (arXiv:2601.07675) — TRM applied to tabular insurance pricing; shows the
  recursion generalizes beyond grid puzzles.
- **ARC Prize 2025 Technical Report** (arXiv:2601.10904) — situates HRM/TRM in the broader
  ARC-AGI landscape; corroborates that deep supervision, not hierarchy, was the load-bearing
  part.
- The official TRM repo (`SamsungSAILMontreal/TinyRecursiveModels`, MIT) was **archived
  (read-only) 2026-04-01** — the reference implementation is frozen.

---

## 5. Where LUTs meet TRM — concrete integration ideas

TRM and the LUT line share a deep structural sympathy: **both bet that a small,
cheaply-evaluated primitive, *iterated*, beats a large dense one evaluated once.** TRM
iterates a 2-layer net; LUTGPT replaces every dense projection with a rank-coded table read.
The natural question: **make the recursed `net` a LUT network.** Below, from most to least
load-bearing.

### 5.1 Replace the recursed `net` with a LUT network (the headline idea)

TRM's entire model is a single 2-layer `net` called ~21×/supervision. If that `net` is a
**LUTGPT-style block** (`FastMultiHeadLut` + MeanAbsNorm, hard forward), then the cost of one
"reasoning step" collapses from two dense matmuls to **sign-compares + table gathers, zero
multiply-accumulate**.

- **Why it's a good fit:** TRM's cost is *recursion depth × per-step cost*. LUTs attack the
  per-step cost exactly where it hurts, and TRM amortizes any per-step overhead across many
  cheap iterations — so the "LUT is slower per matmul on current GPUs" penalty is diluted,
  while the **HBM-read and sparse-active-set** advantages (~3–3.8× less HBM/token, <5% active
  params) compound over ~21 evaluations.
- **Expected profit:** on **rank-coded / in-memory-compute hardware**, a recursive reasoner
  is the *ideal* LUT workload — a tiny weight table read thousands of times fits on-chip and
  is fetched once, so recursion becomes almost free. This is arguably the most compelling
  concrete use-case for the LUT efficiency thesis to date: **a fixed tiny table, iterated.**
- **Risk:** LUTs need the **full-K soft backward** to train well (a load-bearing spiky
  finding). TRM already does full BPTT through the last recursion pass, so the soft-surrogate
  gradient composes naturally — but memory is TRM's wall, and the dense soft backward adds to
  it. Mitigation: TRM's `no_grad` warm-up passes can use the **hard** forward (free), and only
  the single gradient pass pays the soft-backward cost.

### 5.2 A LUT halting head (cheap, low-risk, high-value first experiment)

TRM's ACT halting is a scalar `Q_head(y) → halt?`. A halting decision is a **classification
on the ordering of `y`** — exactly what a rank-coded LUT does natively (sign-compares → table
lookup → halt probability). Replacing the linear `Q_head` with a small LUT is a **tiny,
isolated, low-risk** change that (a) tests LUT-in-TRM plumbing end-to-end on one component,
and (b) fits the manifesto (a magnitude-blind halt is plausibly *more* robust, since "is this
solution self-consistent?" is largely an ordering question). **Recommended as exp001** of any
LUT×TRM branch: smallest possible surface, clean signal.

### 5.3 LUT as the answer-refinement update `y ← f(y, z)`

The two TRM updates have different characters: `z ← f(x,y,z)` is open-ended *reasoning*;
`y ← f(y,z)` is *"clean up the current solution toward a valid one."* Solution-cleanup on a
grid (Sudoku/ARC) is heavily **combinatorial / constraint-satisfaction** — closer to
table-lookup than to smooth interpolation. Hypothesis: the **answer update is a better LUT
target than the latent update**. Keep `z ← f(x,y,z)` as a small MLP/attention (magnitudes may
matter for scratch reasoning), make **only** `y ← f_LUT(y, z)` a LUT. This mirrors LUTGPT's
own split (LUTs everywhere except where magnitudes are genuinely needed, e.g. SDPA scores).

### 5.4 Rank-coded latent state (the deepest, most speculative idea)

TRM's `z` is a free Euclidean scratch tensor. The Izhikevich premise is that **ordering** is
the robust code. A **rank-coded / permutation latent `z`** — where the recursion transforms
*orderings* rather than magnitudes — would be scale/shift/monotone invariant across the ~21
iterations, which is exactly the regime where magnitude state drifts and needs normalization
(TRM already leans on EMA + MeanAbsNorm-style stabilization). If a recursion can be made to
operate purely on ranks, each step is comparator+gather and the *whole reasoning trajectory*
is matmul-free. This is the boldest bet and the least de-risked — but it is the cleanest
statement of "Permutations Are All You Need" applied to *reasoning* rather than to a single
forward pass, and would be a genuinely novel result.

### 5.5 What LUTs likely should *not* replace (yet)

- **The attention scores** in the TRM-Att variant — same open frontier as in LUTGPT
  (SDPA is the one place magnitudes genuinely carry information; RoPE sits on top).
- **The input/output embedding + unembedder** — LUT-based unembedding over a large token/
  color vocab is still an open problem in the spiky line; ARC's small color vocab (~10
  symbols) might actually make ARC an *easier* place to attempt a LUT unembedder than
  32k-vocab language — worth flagging as a bonus opportunity.

### 5.6 Why this is worth a branch

- **Scientific fit:** TRM is the cleanest published demonstration that *iterate-a-tiny-net*
  beats *scale-a-big-net* on structured reasoning. The LUT thesis is *iterate-a-tiny-table*.
  Combining them tests both theses at once, on a benchmark (Sudoku/ARC) where the LUT's
  ordering prior is plausibly well-matched to the task (constraint puzzles are about
  *relative* placement, not magnitudes).
- **Practical fit:** TRM trains from scratch on 1000 examples on a **single GPU** (Sudoku
  ~18h on one L40S; feasible on the local RTX 5090). Unlike LUTGPT's nanochat runs, a
  LUT×TRM Sudoku experiment is small enough to iterate quickly here.
- **Cheapest first cut:** §5.2 (LUT halting head) or §5.3 (LUT answer-update) on
  Sudoku-Extreme MLP-mixer TRM — one component swapped, hard/soft LUT forward, compare test
  accuracy to the 87.4% baseline. If a single LUT component holds accuracy, escalate to §5.1
  (full LUT net).

---

## 6. Suggested next steps

1. **Coordinate the branch/issue with nucstar** — one umbrella idea "small recursive
   reasoning models (HRM+TRM)" → one GitHub issue → one shared `research/<slug>` branch, with
   this file and nucstar's HRM file both living on it. (Per
   [`claude/experiment-methodology.md`](../../claude/experiment-methodology.md): idea → issue
   → branch.)
2. **Reproduce a TRM baseline** on Sudoku-Extreme locally (MLP-mixer, 5M, from the archived
   MIT repo) to establish the 87.4% anchor on our hardware before changing anything.
3. **First LUT experiment:** §5.2 (LUT halting head) or §5.3 (LUT answer-update `y←f(y,z)`),
   forked into its own `experiments/<idea>/exp_<slug>/` folder, code committed before the run
   per **agree → commit → go**.

---

## Sources

- Alexia Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks*,
  arXiv:2510.04871 (Samsung SAIL Montréal, Oct 2025) — https://arxiv.org/abs/2510.04871
  (PDF: https://arxiv.org/pdf/2510.04871)
- Official code (MIT, archived 2026-04-01):
  https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- Wang et al., *Hierarchical Reasoning Model* (HRM), arXiv:2506.21734 —
  https://arxiv.org/abs/2506.21734 ; code https://github.com/sapientinc/HRM
- ARC Prize Foundation analysis of HRM (deep-supervision-is-the-driver) — see ARC Prize 2025
  Technical Report, arXiv:2601.10904 — https://arxiv.org/abs/2601.10904
- Forbes, *Samsung AI Research Team Builds A Tiny Model With Big Power* (Oct 2025) —
  https://www.forbes.com/sites/ronschmelzer/2025/10/09/samsung-ai-research-team-builds-a-tiny-model-with-big-power/
- Follow-ups: Mamba-2/attention hybrid recursion, arXiv:2602.12078 ; edge-compression of
  recursive reasoners, arXiv:2606.26488 ; Tab-TRM (tabular), arXiv:2601.07675
- spiky internal grounding: [`claude/thesis.md`](../../claude/thesis.md),
  [`claude/experiment-methodology.md`](../../claude/experiment-methodology.md)
