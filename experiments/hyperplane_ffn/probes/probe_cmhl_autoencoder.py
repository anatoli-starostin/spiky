"""Probe 7f96cd5a: can CompressionMHL + mirror CMHL autoencode random N(0,1)?

x ~ N(0,1) [N,384], fresh every step (infinite data). x -> forward CMHL -> y[N,384]
-> mirror CMHL -> x_hat[N,384]. Loss = MSE(x_hat, x). Both trainable. Tests whether
the recon mechanism (exp_n_0053) drives MSE->0 or hits a discrete-routing floor.
Trivial baseline: predicting zeros for x~N(0,1) gives MSE = 1.0 (the variance).
"""
import sys, os, json
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

DEV = "cuda"
N, DIM, STEPS, LOG = 4096, 384, 30000, 200

def make_cmhl(seed):
    return CompressionMultiHeadLUT(
        input_dim=DIM, output_dim=DIM, inner_in_dim=48, inner_out_dim=48,
        nap=6, tph=64, n_heads=8, joint_head_compression=False,
        batched_multi_head_input=True, forward_mode="hard",
        use_bf16=False, initial_weights_noise=1e-3, learnable_temps=True,
        random_seed=seed).to(DEV)

def run(lr, gen_seed):
    torch.manual_seed(0)
    fwd = make_cmhl(1000)      # same as exp_n_0053 forward slot (block 0)
    mir = make_cmhl(5000)      # same as exp_n_0053 mirror (block 0)
    opt = torch.optim.AdamW(list(fwd.parameters()) + list(mir.parameters()),
                            lr=lr, betas=(0.9, 0.95), eps=1e-8)
    g = torch.Generator(device=DEV).manual_seed(gen_seed)
    steps, mses = [], []
    for s in range(1, STEPS + 1):
        x = torch.randn(N, DIM, device=DEV, generator=g)
        x_hat = mir(fwd(x))
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if s == 1 or s % LOG == 0:
            steps.append(s); mses.append(loss.item())
            if s == 1 or s % 2000 == 0:
                print(f"  lr={lr:.0e} step {s:6d} | MSE={loss.item():.5f}")
    return steps, mses

print("=== autoencoder probe: CMHL + mirror on random N(0,1) ===")
curves = {}
for lr in (3e-4, 1e-3):
    print(f"--- lr={lr:.0e} ---")
    curves[lr] = run(lr, gen_seed=12345)

plt.figure(figsize=(8, 5))
for lr, (st, ms) in curves.items():
    plt.plot(st, ms, label=f"lr={lr:.0e} (final MSE={ms[-1]:.4f})")
plt.axhline(1.0, ls="--", c="gray", label="zeros baseline (MSE=1.0)")
plt.xlabel("step"); plt.ylabel("recon MSE"); plt.ylim(0, 1.05)
plt.title("CMHL+mirror autoencoder on random N(0,1) — does MSE->0 or floor?")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("/tmp/probe_ae_curve.png", dpi=120); plt.close()

summary = {}
for lr, (st, ms) in curves.items():
    def at(target):
        for s, m in zip(st, ms):
            if s >= target: return round(m, 5)
        return round(ms[-1], 5)
    summary[f"lr_{lr:.0e}"] = {"start": round(ms[0], 5), "s1k": at(1000),
        "s5k": at(5000), "s10k": at(10000), "final": round(ms[-1], 5)}
print("SUMMARY:", json.dumps(summary))
print("saved /tmp/probe_ae_curve.png")
