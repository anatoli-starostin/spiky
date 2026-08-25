# exp_n_0081 — 23M LUT (exp074 architecture) at 1.5× batch, batched path

The 23M-parameter 1.5×-batch LUT run (option (a): exp074's architecture), for an
apples-to-apples comparison with the 23M dense baseline exp073 (val_bpb 1.19665).

**Architecture = exp074_tied_compressionmhl_6heads_inner64_16k** (23,214,336 params):
depth 6, n_embd 384, n_head 6, seq 512; LUT **H6 / inner_in=inner_out=64 / tph36 / nap6**;
tied unembed; hard forward. Built here on the **batched** path
(`lut_batched_multi_head_input=true`, forward-equivalent to exp074's loop path, faster) —
the same batched mechanism exp_n_0052/exp_n_0046/exp_n_0080 use. **Verified param count
23,214,348** (= exp074's 23,214,336 + 12 learnable-temp scalars; see note).

**Batch = 1.5×** (mirroring exp_n_0046): device_batch_size **72**, total_batch_size
**36,864 tok/step**, grad_accum 1 (one 72-seq micro-batch). **n_steps 16,000.**

**Hyperparams:** lr 3e-4 cosine (10% warmup), wd 0.1, seed 1 — identical across exp074 and
exp_n_0052.

**Two deliberate deviations from exp074's original config, because the task specified the
exp_n_0052 batched trainer** (which the exp_n 1.5× lineage uses):
1. **Optimizer:** exp_n_0052's trainer puts all params on **AdamW** (LUT tables in the
   no-weight-decay group, lr 3e-4). exp074's original used **Lion** on the LUT tables
   (`lut_optimizer=lion, lut_lr 2e-4`). This trainer has no Lion path, so LUT trains on
   AdamW here. (If an exact exp074-optimizer match is wanted, that needs the Lion-capable
   trainer.)
2. **learnable_temps=true** (exp_n_0052 lineage) vs exp074's implicit false — adds 12
   scalar params (the +12 above); forward is temp-independent in hard mode, so this only
   affects the soft backward.

**References:** exp074 1× baseline (loop, Lion) final val_bpb **1.2347**; 23M dense baseline
exp073 = **1.19665**. This tests whether the 23M LUT at 1.5× batch closes toward the dense
baseline (as the 27M exp_n_0046 reached ~1.197 dense parity).

Outputs: metrics.csv, summary.json, loss.png, checkpoint.pt.
