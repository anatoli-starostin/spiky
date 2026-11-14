# Spiky wiki

[Spiky](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/README.md) is an experimental CUDA-enabled, PyTorch-compatible Python library inspired by the [Spiking Manifesto](https://arxiv.org/pdf/2512.11843) (E. Izhikevich). It implements differentiable lookup tables as a simple instrument to model spike polychronization. The repository also contains a simulator of spiking neural networks (SpNet, with synapse growth). At the moment lookup tables are the main topic of the project.

This wiki is a readable guide to what has been done in the repository: each page tells the story at an abstract level (ideas, models, math, what worked and what did not) and links to the code, runs and notes for the details. It is organised in areas. The wiki will grow as more areas are written.

## Areas

| Area | Status | What it covers |
| --- | --- | --- |
| [Lookup tables](lookup-tables/README.md) | in progress | differentiable lookup-table primitives, their optimisation, the architectures built from them, and the problems around them |
| Spiking network simulation (SpNet) | not yet covered | Izhikevich spiking networks with spike-timing-dependent plasticity and synapse growth; see the [Spiky programming guide](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/doc/spiky/programming_guide.md) and the [SpNet notebook](https://github.com/anatoli-starostin/spiky/blob/3ac72cca40c63b77d509d4ccff33bb85d43e515e/workbooks/spnet.ipynb) |

Also: [`CONVENTIONS.md`](CONVENTIONS.md) (how to write pages, math, citing sources), valid for all areas.
