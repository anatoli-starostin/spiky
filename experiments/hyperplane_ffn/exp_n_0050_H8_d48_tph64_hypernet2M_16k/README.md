# exp_n_0050 — hypernetwork-reparametrized LUT, ~2M-param generator (H8/d48/tph64 hard), 16k steps

> **RESULT: final_val_bpb = 1.2860195 (best=final; 16k, 1.51 h). Bigger generator does NOT rescue it — hypernet
> is a dead-end at any size.** +0.0577 vs the matched free-row control exp_g_0006 (1.228335, same 16k/standard
> batch), +0.0894 vs dense. Scaling the generator 60× (exp_n_0049's 34K → 2.09M here) did NOT recover the free-row
> capacity. It even finished above 0049's 34K generator (1.259109), but that's confounded by 0049's 24k steps vs
> 0050's 16k; netting out the ~0.023 token effect, **2M ≈ 34K at matched tokens — generator size barely matters;
> the reparametrization itself is the ceiling.** (The 2M gen DID fit faster early: step-200 bpb 2.629 vs 0049's
> 2.773 — but converged no better.) Verdict: **free per-row params are irreplaceable; hypernet is a dead-end.**

Clone of **exp_n_0049**'s hypernet plumbing, scaled up: ~2M-param shared generator (vs 0049's 34K), standard
batch, **16k steps** (vs 0049's 24k). Model identical (H8/d48/tph64/nap6 hard, seed 1). All in train.py — no
edits to `fast_multi_head_lut.py`.

**Generator = 2,086,960 params:** MLP `Linear(262→1152)→GELU→Linear(1152→1152)→GELU→Linear(1152→48)` (1,686,576)
+ `Embedding(3072,128)` per-(module,table) (393,216) + `Embedding(48,64)` per-module (3,072) + `Embedding(64,64)`
per-cluster (4,096). Input 262 = code6 + table_emb128 + module_emb64 + cluster_emb64; rows generated vectorized
over (table, cluster).

**Trainable = 19,992,976** (17.9M non-table + 2.09M generator). The 9.44M free-row `original`s frozen
(`requires_grad_(False)`) + excluded from the optimizer (MLP-weights + embeddings → decay; MLP biases → nodecay).
Total params (incl. frozen originals) 29,430,160. Bake-back to plain static LUT before save = EXACT (max
|pre−post| = 0.0).

Completed negative result — see exp_n_0049 (34K generator) for the smaller-generator twin. Both fail: the LUT rows
need their own free parameters.
