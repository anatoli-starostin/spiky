"""exp012 pre-flight for the radically simplified net: 17 in / 8 exc / 0 inh / 1 out.

The point of this run is the topology, so the topology has to be REAL: build() normally
allocates all 40 exc + 10 inh slots and leaves the unused ones sitting in the engine as
silent neurons. set_hidden_capacity shrinks the capacity itself, and this checks that what
lands in the engine is what was asked for -- and that removing the inhibitory half entirely
does not leave a Dale assumption dangling somewhere.

Also asserts the 40/10 default is untouched, since every earlier checkpoint depends on it.
"""
import numpy as np
import torch

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder

EXC_LEVELS = [round(0.1 * i, 10) for i in range(11)]      # 0.0 .. 1.0, no negative half
DIM = 5

# ---------------------------------------------------------------- default must be untouched
assert (G.N_EXC_MAX, G.N_INH_MAX, G.N_SRC) == (40, 10, 67), "the 40/10 default moved"

# ---------------------------------------------------------------- the tiny geometry
G.set_hidden_capacity(8, 0)
G.set_out_per_target(1, "mean")
G.set_target_dims([DIM])
G.QUANTIZED = True
G.FANOUT_CAP = 16
G.set_delay_levels(list(range(1, 64, 2)))
G.set_weight_levels(EXC_LEVELS)

print(f"N_IN {G.N_IN}  N_EXC_MAX {G.N_EXC_MAX}  N_INH_MAX {G.N_INH_MAX}  "
      f"N_OUT_NEURONS {G.N_OUT_NEURONS}  N_TARGET {G.N_TARGET}")
print(f"genome matrix N_SRC x N_TGT = {G.N_SRC} x {G.N_TGT}")
assert (G.N_SRC, G.N_TGT) == (17 + 8 + 0, 8 + 0 + 1) == (25, 9)
print(f"QUANT_POS {np.round(G.QUANT_POS, 3).tolist()}")
print(f"QUANT_NEG {np.round(G.QUANT_NEG, 3).tolist()}  (only the 0 level -- no inhibition)")
assert G.PIN_DELAY.sum() == 0, "there should be no inh->exc delay pin without inh neurons"
legal = G.LEGAL.sum()
print(f"legal cells {legal}  = in->exc {17 * 8} + exc->exc {8 * 8} + exc->out {8 * 1}")
assert legal == 17 * 8 + 8 * 8 + 8 * 1 == 208

# ---------------------------------------------------------------- a genome, and what builds
_, _, Xp, Yp, Xv, Yv = load(1024, seed=0)
T.fit_target_stats(Yp)
enc = LatencyEncoder(Xp)
Xb, Yb, _ = sample_batch(Xp, Yp, 1024, 0, 12345)
tv = T.target_offsets(Yv)
print(f"\ntarget dim {DIM}: own chance {G.task_baseline(Yv):.3f}  "
      f"(the 6-dim number, 34.152, is NOT the yardstick)")

rng = np.random.default_rng(0)
g = G.random_genome(rng, p_init=0.5, n_exc=8, n_inh=0)
G.enforce(g)
print(f"random genome: {int(g['mask'].sum())} synapses of {legal} legal cells, "
      f"weights on {sorted(set(np.round(g['weight'][g['mask']], 3).tolist()))}")
assert (g["weight"][g["mask"]] >= 0).all(), "a negative weight appeared with no inh neurons"
assert g["aff_a"].shape == (1,)
assert int(g["act_inh"].sum()) == 0 and g["act_inh"].size == 0

# what the engine actually allocates: build() sizes neuron_counts from the capacities
P = 3
counts = [P * G.N_EXC_MAX, P * G.N_INH_MAX, P * G.N_IN, P * G.N_OUT_NEURONS]
print(f"engine neuron_counts for P={P}: exc {counts[0]}  inh {counts[1]}  in {counts[2]}  "
      f"out {counts[3]}   = {sum(counts)} total  ({sum(counts) // P} per candidate)")
assert sum(counts) // P == 17 + 8 + 0 + 1 == 26

H = G.build([g], device="cuda")
st = G.score(H, Xb, Yb, enc, genomes=[g], readout="diagls")
sv = G.score(H, Xv, Yv, enc, genomes=[g], readout="diagls", readout_map=st["readout_map"])
print(f"one random genome: held-out {float(sv['mse'][0]):.3f}  "
      f"silent {float(sv['silent'][0]):.3f}  distinct {int(sv['n_distinct'][0])}")
assert np.isfinite(float(sv["mse"][0]))
del H, st, sv
torch.cuda.empty_cache()

# mutation and crossover must survive the missing inhibitory half
h = G.mutate(G.crossover(g, g, rng), rng, p_affine=0.25, p_gain=0.25, p_weight=0.25,
             p_add=0.02, p_prune=0.02, p_grow=0, p_shrink=0)
G.enforce(h)
assert (h["weight"][h["mask"]] >= 0).all() and G.on_grid(h)
print(f"after crossover+mutation: {int(h['mask'].sum())} synapses, still on-grid, no negatives")

print("\nPRE-FLIGHT GREEN -- 17 in / 8 exc / 0 inh / 1 out, 26 neurons per candidate")
