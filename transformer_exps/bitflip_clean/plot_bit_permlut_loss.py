"""Plot BitPermutationLUT training curves from the two separate CSVs
(per-step batch loss + every-LOG_EVERY full-train-set eval)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
batch_csv = os.path.join(here, "train_bit_permlut_batch.csv")
eval_csv = os.path.join(here, "train_bit_permlut_eval.csv")
png = os.path.join(here, "train_bit_permlut_loss.png")

b = np.genfromtxt(batch_csv, delimiter=",", names=True)
e = np.genfromtxt(eval_csv, delimiter=",", names=True)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(b["step"], np.maximum(b["batch_loss"], 1e-10),
        label="per-step batch loss", alpha=0.5, linewidth=0.6, color="C0")
ax.plot(e["step"], np.maximum(e["train_mse"], 1e-10),
        "o-", label="full train-set MSE (eval)", color="C1")
ax.axhline(1.0, linestyle="--", color="gray", alpha=0.6, label="zero-predictor baseline")
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel("MSE (log)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
ax.set_title("BitPermutationLUT vs normalized dataset.pt target")

fig.tight_layout()
fig.savefig(png, dpi=120)
print(f"saved {png}")
print(f"final: step={int(b['step'][-1])}, batch_loss={b['batch_loss'][-1]:.4e}, "
      f"train_mse={e['train_mse'][-1]:.4e}")
