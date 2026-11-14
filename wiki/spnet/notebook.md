# The SpNet notebook

[`spnet.ipynb`](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/workbooks/spnet.ipynb) (paired with the plain-text [`spnet.py`](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/workbooks/spnet.py)) is the only worked example of SpNet in the repository. This page says what it builds and what its saved outputs show; nothing was re-run for this page, and the notebook does not name the GPU it ran on. The calls are explained in [Model and usage](model-and-usage.md).

## Idea / model

The notebook builds a large network out of many small independent ones, in the layout of the polychronization model: **1000 subnetworks of 1000 neurons each, 800 excitatory and 200 inhibitory, one million neurons in total**, with about 100 outgoing synapses per neuron.

- **Neuron types.** Excitatory neurons use `a=0.02, d=8`, inhibitory neurons `a=0.1, d=2`; the other parameters stay at their defaults.
- **Synapse types.** Excitatory synapses learn (learning rate 0.1), have delays from 0 to 19 ticks and start at weight 6. Inhibitory synapses are fixed: weight minus 5, delay 0, no learning.
- **Connections** come from the growth engine with three commands, each with connection probability 0.1 inside a cuboid that covers the whole subnetwork: excitatory to excitatory (at most 80 synapses), excitatory to inhibitory (at most 20), inhibitory to excitatory (at most 100). The subnetworks are placed 1000 units apart along one axis, so a cuboid never reaches a neighbouring subnetwork.
- **Input.** On every tick one randomly chosen neuron of each subnetwork receives an input of 20 (sparse input, identifiers plus values).

## Why it matters / what was found

What the saved outputs show (all *measured* in the notebook run, one run, no repetition):

- The grown network has 91,222,050 synapses. After `compile` the forward synapse storage is about 839,000 KB and the backward storage about 748,000 KB, and the network can be moved between CPU and GPU.
- A first pass of 1000 ticks with learning off produces 12,132,788 spikes (about 12 per neuron on average).
- Then 100 passes of 1000 ticks with learning on (STDP, new random input each pass). The notebook plots the number of spikes per pass: it starts near 12 million, falls within a few passes to roughly 1.5 million and then rises slowly, reaching roughly 3 million by the last pass (values read off the plot). A second plot, with unlabeled axes, shows the per-tick processing time falling over the first tens of passes and then staying flat. The saved plots and the saved profile printout come from different executions of the cell (their execution counts differ), so numbers in the two should not be compared.
- The profile printout lists the time of each kernel stage (spike detection, firing, Euler steps, LTP computation, applying weight deltas) and of the connection-building stages; the notebook draws no conclusion from them.
- A spike raster of one subnetwork (neurons by 1000 ticks) is sparse, and a voltage trace of one neuron shows a background around minus 70 with a few spikes; the notebook comments neither.

Not shown anywhere in the notebook: what the learned weights look like, whether the network develops repeating spike patterns (the polychronous groups of the model), or how the run compares with another simulator.

## Dead ends

None recorded.

## Open ideas

None recorded in the notebook. The last cell is the small growth example of the programming guide (10 source neurons, a 10 by 10 grid of targets, 100 connections), repeated as a check.

## Sources

[`spnet.ipynb`](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/workbooks/spnet.ipynb), [`spnet.py`](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/workbooks/spnet.py); the [README section on workbooks](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/README.md) lists what the notebook illustrates. Area list: [README](README.md#sources).
