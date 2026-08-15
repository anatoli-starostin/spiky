# Is the shipped walker2d spiking network Dale-compliant?

**Yes — network-wide, not just the readout. Zero mixed-sign neurons out of 2,882
presynaptic ones.**

```
network: 2889 neurons, 25953 synapses, dmax 6, n_ticks 167
presynaptic neurons (>=1 outgoing synapse): 2,882 of 2,889

  purely EXCITATORY (all outgoing > 0): 2,610
  purely INHIBITORY (all outgoing < 0):   272
  MIXED SIGN (Dale violations)        :     0
```

| population | count | sign |
|---|---:|---|
| input latency neurons | 17 | excitatory |
| comparator rails (anti-leaky) | 272 | excitatory |
| **rail interneurons** | **272** | **inhibitory** |
| memory cells | 272 | excitatory |
| Stage-2 lookup cells | 2,048 | excitatory |
| completion detector | 1 | excitatory |
| Stage-3 output neurons | 6 | *terminal — no outgoing synapses* |

The 7 neurons with no outgoing synapses are exactly accounted for: the 6 Stage-3 output
neurons (the network's outputs) and the one unused slot in the 18-neuron input population,
of which 17 carry observations.

## Inhibition is confined to one population and one value

The **only** negative weight anywhere in the network is **−10.0**, and it is emitted only by
the 272 rail interneurons — the cross-inhibition that makes each comparator pair
winner-take-all (`W_INH = -10.0`). Every other synapse in the network is positive; the
full weight range is [−10.0, +1.5] with 10,852 distinct values, all the positive ones being
Stage-2→3 amplitudes `beta_o · exp(w/tau)` and the fixed structural weights.

## Is a negative weight genuine inhibition, or bookkeeping?

Genuine. spnet integrates `v' = cf_2·v² + cf_1·v + cf_0 − u + I` with `I` the summed
synaptic input, so a negative synaptic weight subtracts from the **target's membrane on
arrival** — an inhibitory postsynaptic potential, not a sign folded into an encoding.

Checked explicitly for the alternative:

* **No sign hidden in the neuron coefficients.** Every `NeuronMeta` in this build carries
  `cf_0 = 0`; `cf_1` is a leak (`−1/τ_m`) or anti-leak (`+1/τ_m`) term that belongs to the
  neuron *type*, is identical for every neuron of that type, and is independent of any
  particular synapse. It cannot encode a per-connection sign.
* **No bias or anchor term.** The anchor pairs are addressing (which input feeds which
  rail), not weights; the LUT values enter only as the positive Stage-2→3 amplitudes.
* **The output stage is positive by construction**, since the readout delivers
  `beta_o · exp(w/tau)` and an exponential cannot be negative — but that was already known,
  and is *not* what carries the result: the excitatory/inhibitory split is clean in every
  earlier stage too.

## Caveat on the variant

This is the **shipped** configuration: GT-skew on, tie detectors off. The superseded
tie-break variant adds 136 tie-detector neurons whose only outgoing synapse is `−W_MEM` to
the GT memory cell — purely inhibitory, so that variant is Dale-compliant as well, with
inhibition then split across two populations instead of one.

## Method

`dale_check.py` captures the synapse list from the **real build** rather than re-deriving
it: the growth engine's `_grow_explicit` is the single call through which every synapse
enters the network, so wrapping it yields exactly the `(source, target, weight)` triples the
shipped network is compiled from. The captured count is asserted against the count `build()`
itself reports (25,953) — so the analysis is provably of the built network, not of a
reconstruction that might have drifted.

```
python dale_check.py --npz .../deploy/quantised/walker2d_fastlut_lse_exp19_quantised.npz
```
