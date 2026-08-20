# exp_n_0052 — batched-path reproduction of exp_g_0006 (isolates the shared-temperature effect), H8/d48/tph64, 16k

> **RESULT: final_val_bpb = 1.2285517 (best = final; 16k, 1.973 h). The batched shared-temperature effect is
> essentially zero.** vs the loop-path free-row control **exp_g_0006 (1.228335)**: **+0.00022 bpb** — within
> run-to-run noise. Collapsing 8 per-head `(T_soft, T_sel)` pairs into one shared pair (the batched path) does
> not measurably hurt convergence; the batched `multi_head_input` path is a near-free drop-in for training as
> well as inference, validating the new batched default. vs dense (1.196646): +0.0319. This is the matched
> batched control for exp_n_0051 (recon-aux): `0051 − 0052 = −0.00086` isolates the reconstruction auxiliary.

## What this measures
exp_g_0006 (gpustar, the loop-path free-row control) uses the independent per-head loop, so with
`learnable_temps=True` each of the 8 heads has its **own** `(T_soft, T_sel)` pair. The batched path instead
shares a **single** `(T_soft, T_sel)` pair across all heads. The hard forward and per-head surrogate gradients
are otherwise identical to the loop path up to float reassociation; the ONLY structural difference is this
shared-vs-per-head temperature. So:

    delta( exp_n_0052 − exp_g_0006 )  ==  the pure batched shared-temperature effect

exp_n_0052 is also the **matched batched control for exp_n_0051** (the reconstruction-aux run, same batched
forward): 0052 = 0051's forward with no recon, so `0051 − 0052` isolates the reconstruction auxiliary's effect
while `0052 − exp_g_0006` isolates the batched temperature effect.

## Config
Identical to exp_g_0006 / exp_n_0047's forward: H8 / d48 / tph64 / nap6, hard routing, device_bs 48,
grad_accum 1, **16000 steps**, warmup 1600, seed 1, `learnable_temps=true`, tie_unembedder, depth 6, gamma 0,
lr 3e-4, wd 0.1, clean val 245,760 tokens (eval_steps 10 × bs 48 × seq 512). Built from the shared exp043+
sweep trainer (exp_n_0047's `train.py`, the same trainer exp_g_0006 used), the sole change being the forward
CompressionMHL constructed with `batched_multi_head_input=true`. No edits to the shared modules.

> Note: exp_g_0006's own folder is a gpustar-only experiment not present in this repo; its config is reproduced
> here from exp_n_0047 (the same shared trainer + standard rung, documented as differing from exp_g_0006 only in
> n_steps 24k vs 16k) and from exp_n_0051's forward, which was built bit-identical to the exp_g_0006 loop path.
