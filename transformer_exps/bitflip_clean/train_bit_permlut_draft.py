"""Variant of train_bit_permlut.py using DraftBPLUTO.

Full 100K-step run on dataset.pt so we can compare trajectory to the
production optimizer with more precision than the 2K-step synthetic test.
"""
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.draft_bpluto import DraftBPLUTO


device = torch.device("cuda:0")
torch.manual_seed(42)

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

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.pt")
data = torch.load(DATASET, weights_only=True, map_location=device)
x = data["x"].to(device)
y = data["y"].to(device)
N = x.shape[0]
BORDA_SCALE = math.sqrt(CFG["tph"] * CFG["output_nap"] * 2 / CFG["n_outputs"])

print(f"dataset: N={N}, borda_scale={BORDA_SCALE:.3f}")
print(f"y std={y.std().item():.3f}, |y|_max={y.abs().max().item():.3f}, "
      f"baseline predict-0 MSE = {(y ** 2).mean().item():.3f}")

student = BitPermutationLUT(
    random_seed=STUDENT_SEED, device=device,
    initial_weights_noise=LATENT_INIT_STD,
    **CFG,
)
student.load_pairs(
    anchor_pairs_a=data["anchor_pairs_a"],
    anchor_pairs_b=data["anchor_pairs_b"],
    idx_a=data["idx_a"],
    idx_b=data["idx_b"],
)
print(f"loaded teacher pairs: inv_idx K_max={student.K_max}")


def student_forward(x_batch: torch.Tensor) -> torch.Tensor:
    dom = student(x_batch)
    out = torch.einsum("bhp,kp->bhk", dom, student.dom_borda_m).squeeze(1)
    return out * BORDA_SCALE


def lr_schedule(step: int) -> float:
    if step <= WARMUP:
        return step / max(WARMUP, 1)
    p = (step - WARMUP) / max(N_STEPS - WARMUP, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


opt = DraftBPLUTO(
    [student],
    lr=PEAK_LR,
    lr_schedule_fn=lr_schedule,
)


@torch.no_grad()
def eval_train_mse() -> float:
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
print(f"step 0: train_mse={train_mse:.4e}  (baseline: predict-zero MSE = 1.0 in y units)")

BATCH_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_bit_permlut_draft_batch.csv")
EVAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_bit_permlut_draft_eval.csv")
with open(BATCH_CSV, "w") as f:
    f.write("step,batch_loss,lr\n")
with open(EVAL_CSV, "w") as f:
    f.write("step,train_mse,lr\n")

student.train()
batch_rows = []
eval_rows = []
for step in range(1, N_STEPS + 1):
    idx = torch.randint(0, N, (BS,), device=device)
    xb = x[idx].detach().requires_grad_(True)
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
