# exp_n_0037 — H16/d24, FIXED-PARTITION FFN slot (no learned compress), AdamW, tph64/nap6, tied, 16k

> **STOPPED EARLY @ step 10000/16000 (last val_bpb = 1.282485).** Verdict: the fixed-partition FFN slot
> (frozen axis-aligned head-partition, no learned compression) tracked a **steady ~+0.025 bpb behind
> exp_n_0033** (learned compression = learnable-hyperplane routing) at matched steps (+0.02556 @9600, +0.02511
> @9800, +0.02507 @10000 — flat, not closing). That's ~6–10× the spread between recipe variants, so it's a real
> architectural effect: **the learned compression matrix is doing real work and can't be deleted** — a fixed
> partition costs ~0.025 bpb. Killed once unambiguous to free the GPU; exp_n_0036 held pending this result.

**NOTE — the dir name `no_attn_outproj` is a MISNOMER kept for queue/waiter stability.** This slot was
re-tasked from option B (drop attention out_proj) to **option A** (per the owner), keeping the same
experiment name/dir so all serial-queue waiter wiring stays intact (0034 → 0037 → 0036). This experiment
is **option A: fixed-partition FFN**, and it KEEPS the attention out_proj.

Clone of **exp_n_0036**'s train.py with **one architectural change** (option A): inside the CompressionMHL
FFN slot, the **learned compression matrix is removed and replaced by a fixed contiguous partition.**
CompressionMHL normally does `compress = Linear(384 → n_heads·inner_in = 16·24 = 384)`, then
`view(N, n_heads, inner_in)`, then per-head FastMHL. Here `compress` is replaced by `nn.Identity()`, so the
raw 384-dim `h` is reshaped straight into 16 heads × 24 dims — **head h reads the fixed slice
`h*24:(h+1)*24`** with no learned compression weight/bias. Verified: `compress` is Identity with **0 learnable
params**; forward (N,384)→(N,384); partition head0←x[:,0:24], head15←x[:,360:384].

**Everything else is vanilla / exp_n_0036 recipe:**
- Attention **out_proj RESTORED** (normal `MinimalAttention`, `self.proj = Linear(384→384, bias=False)` intact);
  residual add and both LayerNorms (ln1, ln2) untouched.
- **AdamW everywhere** (no Lion), **no MeanAbsNorm**, **learnable temperatures ON**.
- The orthogonal per-head compress init from exp_n_0036 **no longer applies** (there is no compress matrix to
  init) — dropped cleanly via `compress_ortho_init=False`. (The `_ortho_init_compress_heads` helper remains in
  the file but is gated off and never called.)
- H16/d24/tph64/nap6, tied, warmup+cosine floor schedule, grad-clip 1.0, 16k steps, all data/backbone settings.

**Param count: 26,456,256** (SMOKE-confirmed) = exp_n_0036's 27,343,296 **− 887,040** = the removed compress
`Linear(384→384)`: 884,736 weight (6×384×384, → was in the AdamW decay group: 17,891,328 → 17,006,592) **+ 2,304
bias** (6×384, → was in nodecay: 9,451,968 → 9,449,664). LUT tables unchanged at 9,437,184. = 1.140× tied dense.
Optimizer print:
`AdamW-everywhere (no Lion) | decay(2-D weights)=17,006,592 wd=0.1 | nodecay(LUT tables+temps+1-D)=9,449,664 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=9,437,184 in nodecay]`.

Runs 16k. **Serial order (unchanged): 0034 (done) → 0037 → 0036.** Waiters unchanged (0037 keys off
exp_n_0034/summary.json; 0036 keys off exp_n_0037/summary.json). 0034 & 0036 untouched. Question: can the
FastMHL routing work on raw fixed head-partitions of the residual stream — i.e. is the learned compression
matrix doing real work, or is a fixed partition (−0.89M params) just as good?
