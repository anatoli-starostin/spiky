# exp_n_0053 — reconstruction-auxiliary CMHL, λ_recon=0.5 sweep point, H8/d48/tph64, 16k

> **RESULT: final_val_bpb = 1.2323435 (best = final; 16k, 1.63 h). λ=0.5 HURTS — the recon coefficient overshoots.**
> Final train CE 3.8603, final recon 0.2178 (much lower than λ=0.1's 0.32 — the 5× coefficient does drive
> reconstruction harder). But bpb is **+0.0047 worse than λ=0.1 (1.2276884)** and **+0.0038 worse than even the
> no-recon control exp_n_0052 (1.2285517)**. So stronger reconstruction pressure pulls capacity/gradient away
> from the CE task and is net negative. The recon auxiliary has a narrow beneficial regime: λ=0.1 is a small win,
> λ=0.5 is worse than no recon at all. Optimal λ is small (≤0.1). vs dense (1.196646): +0.0357.

## The sweep
| run | λ_recon | val_bpb |
|-----|---------|---------|
| exp_n_0052 (batched control, no recon) | — | 1.2285517 |
| exp_n_0051 (recon) | 0.1 | **1.2276884** (best — small win) |
| **exp_n_0053 (this run)** | **0.5** | 1.2323435 (hurts: worse than no-recon) |

The question: does 5× stronger reconstruction pressure improve, match, or hurt vs λ=0.1? Too-strong a recon
coefficient could pull capacity away from the CE task; too-weak leaves the near-lossless pressure ineffective.

## Mechanism (unchanged from exp_n_0051)
Each block's forward CompressionMHL (in `[N,384]` → out `[N,384]`) is paired with a training-only mirror
CompressionMHL that maps output → reconstruction of input: `recon = mean-over-blocks MSE(mirror(fwd_out),
fwd_in.detach())`, `total = CE + λ_recon·recon`. Recon grad flows into `fwd_out` (push toward near-lossless);
input target detached. Mirrors are NOT in `model.forward` (val/bpb on the plain slot), are in the optimizer, and
are dropped from the saved checkpoint. Forward CMHL uses `batched_multi_head_input=true` (shared per-head temp),
matching exp_n_0051/0052 — so this is a clean λ sweep on the same batched forward. No edits to shared modules.

## Config
H8/d48/tph64/nap6 hard, device_bs 48, grad_accum 1, 16000 steps, warmup 1600, seed 1, learnable_temps=true,
clean val 245,760 tokens. Only difference vs exp_n_0051: `lambda_recon=0.5`.
