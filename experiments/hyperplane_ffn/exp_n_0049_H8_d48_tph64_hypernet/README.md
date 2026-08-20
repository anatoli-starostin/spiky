# exp_n_0049 — hypernetwork-reparametrized LUT (H8/d48/tph64 hard), 24k steps

> **RESULT: final_val_bpb = 1.2591091 (best=final; 24k, 1.96 h). BIG REGRESSION — the rows need their capacity.**
> +0.0535 vs the apples-to-apples free-row control exp_n_0047 (1.205657), +0.0625 vs dense. The hypernet
> hypothesis (shared-gradient row generation fixes rare-cluster gradient starvation) is FALSE at this budget:
> compressing the 9.44M free table rows into a ~34K shared generator UNDERFITS badly. Mechanism was clean
> (register_parametrization drove training, grads reached the hypernet, bake to plain LUT = EXACT, max diff 0.0)
> — but the ~280× capacity trade loses far more than gradient-sharing gains. **Verdict: dead-end; free rows are
> the right representation.**

Clone of **exp_n_0047** (H8/d48/tph64/nap6 hard, standard batch bs48, 24k steps, clean val 245,760) with each
FastMHL table ROW reparametrized as `row = f_theta(cluster_code, table_emb, module_emb)` via ONE shared
generator MLP, so gradient is shared across all rows (intended to fix rare-cluster gradient starvation) with a
Hamming-smoothness prior. **All in train.py — no edits to `src/spiky/lutorch/fast_multi_head_lut.py`.**

**Architecture:** shared `HyperGen` = MLP `Linear(22→64)→GELU→Linear(64→64)→GELU→Linear(64→48)` (input 22 =
code6 + table_emb8 + module_emb8) + `Embedding(3072,8)` per-(module,table) + `Embedding(48,8)` per-module =
**33,712 params**. Attached to each of the 48 FastMHL modules via `torch.nn.utils.parametrize.register_parametrization`
on `weights`; codes reuse each module's existing `soft_bit_matrix.t()` (fixed 6-bit ±1 cluster signatures).

**Optimizer:** the 9.44M free-row `original`s are frozen (`requires_grad_(False)`) and EXCLUDED; the hypernet is
IN (MLP-weights + embeddings → decay, MLP biases → nodecay, 0033 convention). **Trainable = 17,939,728** (17.9M
non-table + 34K hypernet); table rows driven by ~34K params vs the 9.44M free-row baseline. Total params (incl.
frozen originals) 27,376,912.

**Bake-back:** before save, `remove_parametrizations(leave_parametrized=True)` on all 48 modules → plain static
LUT, verified forward-identical (max|pre−post| = 0.0). The saved checkpoint is an ordinary LUT (hypergen deleted).

Not launched again — this is a completed negative result.
