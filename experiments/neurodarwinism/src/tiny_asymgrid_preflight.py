"""exp012 pre-flight for the asymmetric weight grid (11 excitatory levels, 2 inhibitory).

  excitatory half   {0.0, 0.1, ..., 1.0}     11 levels, step 0.1, 0 = absent
  inhibitory half   {-1.0, 0.0}               2 levels: on/off, magnitude set by inh_coeff

The claim to check is that this needs NO new code: set_weight_levels already splits an
arbitrary grid into the two Dale halves at zero, and everything downstream indexes the
sub-grid arrays rather than assuming a spacing. So the whole design should be expressible as
--weight-levels=-1.0,0,0.1,...,1.0 and nothing else.

Numpy only -- no GPU work, so it is safe to run alongside a training run.
"""
import collections

import numpy as np

import tiny_grow as G

EXC = [round(0.1 * i, 10) for i in range(11)]          # 0.0 .. 1.0
GRID = [-1.0] + EXC                                     # the union handed to --weight-levels

G.set_out_per_target(48, "mean")
G.set_target_dims([0])
G.QUANTIZED = True
G.FANOUT_CAP = 16
G.set_delay_levels(list(range(1, 64, 2)))
G.set_weight_levels(GRID)

print("grid      ", np.round(G.QUANT_LEVELS, 3).tolist())
print("QUANT_POS ", np.round(G.QUANT_POS, 3).tolist())
print("QUANT_NEG ", np.round(G.QUANT_NEG, 3).tolist())
assert list(np.round(G.QUANT_POS, 10)) == EXC, "excitatory half is not the 11-level 0.1 grid"
assert list(np.round(G.QUANT_NEG, 10)) == [-1.0, 0.0], "inhibitory half is not [-1.0, 0.0]"
print(f"QUANT_STEP {G.QUANT_STEP}  (min gap in the union; the EXC hop is a level index, "
      f"not this number)")

rng = np.random.default_rng(0)
g = G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10)
G.enforce(g)
npos = G.N_IN + G.N_EXC_MAX
we, wi = g["weight"][:npos][g["mask"][:npos]], g["weight"][npos:][g["mask"][npos:]]
print(f"\ninit: {int(g['mask'].sum())} synapses  ({we.size} exc, {wi.size} inh)")
print("  exc levels used:", sorted(collections.Counter(np.round(we, 3)).items()))
print("  inh levels used:", sorted(collections.Counter(np.round(wi, 3)).items()))
assert (we > 0).any(), "no excitatory synapse survived init -- the born-at-zero bug is back"
assert (wi < 0).any(), "no inhibitory synapse survived init"
assert np.isin(np.round(we, 10), EXC).all() and np.isin(np.round(wi, 10), [-1.0, 0.0]).all()

# ---- an ADDED edge must be born non-zero in BOTH halves (a born-zero edge is a silent no-op)
for sign, name in ((1, "exc"), (-1, "inh")):
    b = G._born_magnitudes(rng, 400, np.full(400, sign))
    assert (b != 0).all(), f"{name} edges can be born at zero"
    print(f"  born {name}: levels {sorted(set(np.round(b, 3).tolist()))}")

# ---- the weight hop. On the exc half it must move ONE level (0.1); on the inh half it must
#      be a clean 0 <-> -1 toggle rather than a no-op.
h = G.mutate(g, rng, p_weight=1.0, p_add=0, p_prune=0, p_delay=0, p_affine=0,
             p_grow=0, p_shrink=0)
G.enforce(h)
m = g["mask"] & h["mask"]
de = np.round(np.abs(h["weight"] - g["weight"])[:npos][m[:npos]], 3)
di = np.round(np.abs(h["weight"] - g["weight"])[npos:][m[npos:]], 3)
print("\nweight hop sizes")
print("  exc:", collections.Counter(de.tolist()).most_common(5))
print("  inh:", collections.Counter(di.tolist()).most_common(5))
assert (de > 0).mean() > 0.5, "excitatory weights barely move -- the hop is a no-op"
assert (di > 0).mean() > 0.5, "inhibitory weights barely move -- the toggle is a no-op"
assert set(np.unique(di).tolist()) <= {0.0, 1.0}, "inhibitory hop is not a 0 <-> -1 toggle"

# ---- p_weight: the runner's auto-rule calls a grid 'binary' when EITHER half has <= 2
#      levels, which this grid does -- so it would pick 0.5. That rule is about a grid where
#      a weight has one bit of information; here the excitatory half has 11 levels and the
#      adjacent-level hop is meaningful again, so it must be overridden to 0.25 on the
#      command line.
auto_binary = min(len(G.QUANT_POS), len(G.QUANT_NEG)) <= 2
print(f"\nthe runner's auto-rule would call this grid binary: {auto_binary} -> p_weight 0.5")
print("  => the launch MUST pass --p-weight 0.25 explicitly")

assert G.on_grid(h), "a mutated genome left the grid"
print("\nPRE-FLIGHT GREEN -- no code change needed, the existing grid machinery covers it")
