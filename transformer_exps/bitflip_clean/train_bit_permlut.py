"""Train a BitPermutationLUT against the saved dataset.pt target (generated
by a trained float32 PermutationalLut from exp282 out_proj).

Purpose: validate that `BitPermutationLUTOptimizer` (fp8 latent Adam) can
drive bit weights to a good solution on a realistic target. We are NOT
testing generalization -- we just want to see train MSE decrease.

Caveats vs the original train_fp8_adam.py:
  - Student uses BitPermutationLUT's CANONICAL_DISTINCT sampling; the
    saved teacher used FULL_COVERAGE. Anchor lattices differ, so the
    teacher's function is not exactly representable by the student.
    We don't import teacher anchors (non-canonical tables would need
    extra plumbing; it's not the point of this test).
  - y (teacher output) has std ~57; a +-1 BitPermutationLUT has output
    scale ~5-6 post-Borda. We normalize y to zero-mean / unit-std per
    output coordinate so the loss has a meaningful unit-scale baseline.
  - The "baseline" (student predicts zero) MSE is 1.0. What matters is
    whether Adam drives train MSE well below 1.0.
"""
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.bit_permutation_lut_optimizer import BitPermutationLUTOptimizer


device = torch.device("cuda:0")
torch.manual_seed(42)

# ---- Architecture (mirrors the teacher in dataset.pt / train_fp8_adam.py) ----
CFG = dict(
    n_inputs=32, n_outputs=32, n_heads=1,
    input_nap=6, output_nap=32, tph=2048,
)
STUDENT_SEED = 42 + 400
N_STEPS = 100_000
BS = 1024
PEAK_LR = 1e-3
WARMUP = int(0.1 * N_STEPS)
LATENT_INIT_STD = 0.001
LOG_EVERY = 2000

# ---- Load the saved dataset ----
# dataset.pt was generated before the library added /sqrt(n_votes_per_output)
# Borda scaling, so the stored `y` is in "old scale" -- about
# borda_scale = sqrt(tph * output_nap * 2 / n_outputs) = sqrt(4096) = 64
# times larger than what the current PermLut / BitPermLUT forward produces.
# Instead of rescaling y, we rescale the student's output UP by borda_scale
# so MSE numbers match train_fp8_adam.py's reports (baseline ~3320).
DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.pt")
data = torch.load(DATASET, weights_only=True, map_location=device)
x = data["x"].to(device)                    # [N, 32]
y = data["y"].to(device)                     # [N, 32], in old-library scale
N = x.shape[0]
BORDA_SCALE = math.sqrt(CFG["tph"] * CFG["output_nap"] * 2 / CFG["n_outputs"])

print(f"dataset: N={N}, borda_scale={BORDA_SCALE:.3f}")
print(f"y std={y.std().item():.3f}, |y|_max={y.abs().max().item():.3f}, "
      f"baseline predict-0 MSE = {(y ** 2).mean().item():.3f}")

# ---- Student: BitPermLUT with teacher's anchor / output-pair layout ----
student = BitPermutationLUT(random_seed=STUDENT_SEED, device=device, **CFG)
student.load_pairs(
    anchor_pairs_a=data["anchor_pairs_a"],
    anchor_pairs_b=data["anchor_pairs_b"],
    idx_a=data["idx_a"],
    idx_b=data["idx_b"],
)
print(
    f"loaded teacher pairs: inv_idx K_max={student.K_max} "
    f"(CANONICAL_DISTINCT would give {CFG['tph'] * CFG['output_nap'] * 2 // (CFG['n_outputs'] * (CFG['n_outputs'] - 1))})"
)


def student_forward(x_batch: torch.Tensor) -> torch.Tensor:
    """Dominance -> Borda -> E-dim, scaled up to old-library y scale."""
    dom = student(x_batch)                                  # [B, 1, P]
    out = torch.einsum("bhp,kp->bhk", dom, student.dom_borda_m).squeeze(1)
    return out * BORDA_SCALE                                 # align with dataset.pt's y scale


def lr_schedule(step: int) -> float:
    if step <= WARMUP:
        return step / max(WARMUP, 1)
    p = (step - WARMUP) / max(N_STEPS - WARMUP, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


opt = BitPermutationLUTOptimizer(
    [student],
    lr=PEAK_LR,
    latent_init_std=LATENT_INIT_STD,
    lr_schedule_fn=lr_schedule,
    seed=123,
)


@torch.no_grad()
def eval_train_mse() -> float:
    """Full training-set MSE (no holdout -- we're measuring fit quality)."""
    student.eval()
    sse, count = 0.0, 0
    for i in range(0, N, 4096):
        out = student_forward(x[i:i + 4096])
        sse += ((out - y[i:i + 4096]) ** 2).sum().item()
        count += out.numel()
    student.train()
    return sse / count


t0 = time.time()
train_mse = eval_train_mse()
print(f"step 0: train_mse={train_mse:.4e}  (baseline: predict-zero MSE = 1.0)")

BATCH_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_bit_permlut_batch.csv")
EVAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_bit_permlut_eval.csv")
with open(BATCH_CSV, "w") as f:
    f.write("step,batch_loss,lr\n")
with open(EVAL_CSV, "w") as f:
    f.write("step,train_mse,lr\n")

student.train()
batch_rows: list[tuple[int, float, float]] = []
eval_rows: list[tuple[int, float, float]] = []
for step in range(1, N_STEPS + 1):
    idx = torch.randint(0, N, (BS,), device=device)
    xb = x[idx].detach().requires_grad_(True)          # BitPermLUT has no nn.Parameter
    opt.zero_grad()
    out = student_forward(xb)
    loss = ((out - y[idx]) ** 2).mean()
    loss.backward()
    opt.step()

    lr_now = PEAK_LR * lr_schedule(step)
    batch_rows.append((step, loss.item(), lr_now))

    if step % LOG_EVERY == 0 or step == 1:
        train_mse = eval_train_mse()
        eval_rows.append((step, train_mse, lr_now))
        dt = time.time() - t0
        print(f"step {step:>6}: batch_loss={loss.item():.4e}, train_mse={train_mse:.4e}, "
              f"lr={lr_now:.5f}, t={dt:.1f}s", flush=True)

with open(BATCH_CSV, "a") as f:
    for s, bl, lr in batch_rows:
        f.write(f"{s},{bl:.6e},{lr:.6e}\n")
with open(EVAL_CSV, "a") as f:
    for s, tm, lr in eval_rows:
        f.write(f"{s},{tm:.6e},{lr:.6e}\n")

opt.close()
train_mse = eval_train_mse()
print(f"\nFinal: train_mse={train_mse:.4e}, wall={time.time() - t0:.1f}s")
print(f"CSVs: {BATCH_CSV}, {EVAL_CSV}")
