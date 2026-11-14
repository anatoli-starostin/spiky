# Spiking network simulation (SpNet)

SpNet is the simulator of spiking neural networks in [Spiky](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/README.md): Izhikevich neurons, synapses with delays and spike-timing-dependent plasticity (STDP), and connections created by a spatial *synapse growth* engine. It runs on CUDA (and on CPU) behind a PyTorch-compatible Python class, `SpikingNet`. It is the engine of the polychronization model that inspired the lookup-table work, and it is also the engine on which the [lutwalker2d](../lookup-tables/lookup-related-problems/luts-to-spikes.md) spiking network runs.

**Status: in progress.** The pages below describe what exists and what the notebook demonstrates. Apart from the guide, the notebook, the tests and the lutwalker2d construction, the repository has no experiments, results or notes on SpNet, so this area is short on purpose.

## Pages

- [Model and usage](model-and-usage.md): neurons, synapses, delays, STDP, synapse growth, and the sequence of calls that builds and runs a network.
- [The math](math.md): the equations as implemented (neuron, delays, STDP, weight update) and, element by element, where they agree with Izhikevich's original program and where they differ.
- [The SpNet notebook](notebook.md): what `spnet.ipynb` builds and what its outputs show.

## Known and unknown

- **Known (from the code and the docs):** the neuron equations and parameters, the synapse parameters, the growth commands, the call sequence, and the tests that compare the simulator against reference spike, voltage and weight files.
- **Known (from the notebook):** a network of one million neurons and about 91 million synapses fits and runs on one GPU, and the notebook records its spike counts and per-tick times.
- **Unknown, not written down in the repository:** what STDP learning does to such a network over a long time, any measure of the network's behaviour beyond spike counts, a comparison with another simulator, and the GPU used for the notebook run.

## Sources

- Izhikevich's original program: [Polychronization: Computation With Spikes](https://www.izhikevich.org/publications/spnet.htm) with [`spnet.m`](https://www.izhikevich.org/publications/spnet.m) and [`spnet.cpp`](https://www.izhikevich.org/publications/spnet.cpp); archived copies are linked on the [math page](math.md#the-original).
- [Spiky programming guide](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/doc/spiky/programming_guide.md), sections SPNet and Synapse Growth; [architecture diagram](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/doc/spiky/spiky_highlevel_architecture.md).
- [`spnet.py`](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/spnet.py) and [`synapse_growth.py`](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/util/synapse_growth.py); the CUDA code is in [`native/spiky/spnet`](https://github.com/anatoli-starostin/spiky/tree/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/spnet) and [`native/spiky/synapse_growth`](https://github.com/anatoli-starostin/spiky/tree/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/native/spiky/synapse_growth).
- [`spnet.ipynb`](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/workbooks/spnet.ipynb) and its paired [`spnet.py`](https://github.com/anatoli-starostin/spiky/blob/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/workbooks/spnet.py); tests in [`src/spiky/spnet/tests`](https://github.com/anatoli-starostin/spiky/tree/0ee6c6a7807bddae9f66a6f055b6452dba6816a4/src/spiky/spnet/tests).

Back to the [Spiky wiki](../README.md).
