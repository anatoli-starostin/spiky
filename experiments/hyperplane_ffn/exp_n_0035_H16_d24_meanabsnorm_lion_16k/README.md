# exp_n_0035 — H16/d24, MeanAbsNorm + Lion, tph64, learnable_temps, tied, 16k

Clone of **exp_n_0033** (H16/d24, nap6, tph64, tied, vanilla exp073 backbone, std0.02 compress,
learnable_temps=True, 16k) with **two changes that reproduce documented historical best-practice** from
the earlier lutgpt / exp006-017 line, to test whether they recover loss on the modern FFN-slot backbone:

1. **MeanAbsNorm on each head's compressed activation `z`, applied BEFORE FastMHL routing.** Param-free:
   `z_h = z_h / (z_h.abs().mean(-1, keepdim=True) + 1e-6)`. In the independent-per-head path this normalizes
   each head's `z_h` (shape `[N, eff_in]`) just before `lut(z_h)`; in the joint path it normalizes the shared
   `z`. Gated behind `CompressionMHL(pre_lut_meanabsnorm=...)` — **default False** (no behaviour change for any
   other experiment; module tests 19/19 pass with the default). Enabled here via config `lut_pre_meanabsnorm=True`.

2. **Lion optimizer on the LUT table tensors only (hybrid).** Matches the canonical historical recipe
   (exp010 / `examples/lutgpt/train.py`): Lion drives the ndim≥3 table `weights`, AdamW drives everything
   else — **including the learnable log-temperature scalars** (exp010/lutgpt group by ndim, so the 0-dim
   temps fall through to AdamW-nodecay; putting them on Lion's fixed-size sign step was an artifact of an
   earlier "all FastMHL params" grouping and was corrected before launch). `setup_optimizers` builds **two**
   optimizers:
   * **Lion** over each FastMHL's table tensor only: lr=2e-4, betas=(0.9, 0.95), weight_decay=0.
     Covers **9,437,184** params.
   * **AdamW** over the rest (compress/decompress/attn/embeds **+ the 192 log-temp scalars**): 2-D weights →
     decay (wd=0.1), 1-D/0-D → nodecay (wd=0.0); lr=3e-4, betas=(0.9, 0.95), eps=1e-8.
   Both optimizers share the same warmup + cosine LR schedule (`0.1 + 0.9·½(1+cos)` floor, identical to exp010),
   with global grad-clip 1.0.

**Everything else is identical to exp_n_0033/0030:** H16, d24 (H·d=384 fixed throughput, ~4× cheaper FFN than
dense's 8·384²), nap6 (2⁶=64 clusters/table), tph64, tied embeddings, vanilla exp073 backbone, 16k steps.

**Params = 27,343,296 (SMOKE-confirmed)** = 27,343,104 + 192 learnable temp scalars (16 heads × 2 × 6 layers)
= **1.178× tied dense** (23,209,728). Same param count as exp_n_0033 (MeanAbsNorm is param-free; the optimizer
grouping change doesn't touch the model).

**Fidelity vs `examples/lutgpt/train.py` (the canonical reference).** Matched: Lion on ndim≥3 table tensors
only + temps/rest on AdamW; Lion lr 2e-4 betas (0.9,0.95) wd 0; AdamW lr 3e-4 betas (0.9,0.95) eps 1e-8;
warmup+cosine schedule (identical `get_lr_scale`); grad-clip 1.0; MeanAbsNorm formula `x/(x.abs().mean(-1)+1e-6)`
eps 1e-6, applied immediately before the FastMHL router. **One deliberate difference:** the reference has no
compression stage, so its MeanAbsNorm sits on the **full E-width** residual and *is* the block pre-norm; our
CompressionMHL keeps `LayerNorm(E)` as the block pre-norm and applies MeanAbsNorm on the **compressed** router
input (width inner_in) — i.e. same "normalize the router's input" principle, but at compressed width and in
addition to (not replacing) the LayerNorm. Also our tables are fp32 (the reference's SOTA uses bf16 tables +
fp32-master Lion; we run the plain fp32 Lion branch).

Runs 16k, serial **after exp_n_0033, before exp_n_0034** (order 0033 → 0035 → 0034). Compare to exp_n_0033
(nap6/tph64 learnable-temp, **1.228762**), exp_n_0030 (fixed-temp, 1.22936), and tied dense (1.19665). Question:
do MeanAbsNorm + Lion — the historical best-practice pairing — close any of the ~0.03 bpb gap to dense at 16k?
