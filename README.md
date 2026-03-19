# Spiky

**Version 1.0**

An experimental CUDA-enabled, PyTorch-compatible Python library inspired by the [Spiking Manifesto](https://arxiv.org/pdf/2512.11843) (E. Izhikevich), implementing differentiable lookup tables as a simple instrument to model spike polychronization.

**Author:** Anatoly Starostin

## Resources

- **Spiking Manifesto:** [arXiv Paper](https://arxiv.org/pdf/2512.11843)
- **Project Presentation:** [Google Slides](https://docs.google.com/presentation/d/16ZdLnLGjpVy9oCk1FHdEsVbEQv1eOe_jI-SrzM3srmc/edit?usp=sharing)

## Documentation

- **`doc/lutorch/`** — LUTorch (LUT-based, PyTorch-compatible layers and training).
- **`doc/spiky/`** — Spiky engine (SpNet, synapse growth, deprecated old LUT implementation).

## Requirements

- **Python:** 3.12
- **System Dependencies:** `python3-dev` (install with `sudo apt install -y python3-dev`)

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:anatoli-starostin/spiky.git
   cd spiky
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv ./.venv --system-site-packages
   . ./.venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. Build and install native CUDA extensions:
   ```bash
   # From project root
   # (a) LUTorch CUDA backend – this is what you normally need for LUT-based models
   pip install -v ./native/lutorch

   # (b) Full engine (SpNet, ANDN, synapse growth, etc.) – only needed for advanced / spiking use cases
   pip install -v ./native/spiky
   ```

## Running Tests

**LUTorch tests** use [pytest](https://docs.pytest.org/). Install it once into your virtual environment:

```bash
pip install pytest
```

```bash
# All tests (CPU + CUDA)
.venv/bin/python -m pytest src/spiky/lutorch/tests/ -v

# CPU only (fast, ~40 s)
.venv/bin/python -m pytest src/spiky/lutorch/tests/ -v -k cpu

# Single file
.venv/bin/python -m pytest src/spiky/lutorch/tests/test_lut_attention.py -v
```

**SpNet / LUT tests** use their own runner scripts:

```bash
# SpNet
cd src/spiky/spnet/tests/ && python run_tests_with_different_seeds.py

# LUT
cd src/spiky/lut/tests/ && python run_tests_with_different_seeds.py
```

## Jupyter Notebooks

To run example notebooks:

1. Install Jupyter:
   ```bash
   pip install jupyter
   ```

2. Start Jupyter server:
   ```bash
   jupyter notebook --no-browser --port=8888
   ```

3. Open `http://localhost:8888` in your browser and navigate to the `workbooks` directory for example notebooks (`.ipynb` files).

## Workbooks

The `workbooks` directory contains example Jupyter notebooks demonstrating different aspects of the Spiky library. Notebooks use [Jupytext](https://jupytext.readthedocs.io/): each `.ipynb` is paired with a `.py` file for version control and editing in a plain editor.

- **`lutorch_mnist.ipynb`**: MNIST digit classification using LUTorch’s `ProjectionLUT` layers. The notebook shows how to:
  - Load and preprocess the MNIST dataset
  - Build a two-layer conv-like network (`TwoLayerProjectionLUT`) with `ProjectionLUT` and `UnfoldConfiguration`
  - Train the model and track train/test accuracy
  - Optionally explore an `MNIST_LUT_CNN` variant using `MultiHeadLut`
  - Reach ~99% test accuracy on MNIST

- **`lutorch_transformer.ipynb`**: Byte-level language modeling with a LUT-based transformer. The notebook covers:
  - Text data preparation (FineWeb snippet sampler, byte vocab + BOS)
  - Building a `LUTTransformer` from LUTorch primitives: `LUTAttention` (causal) and `MultiHeadLut` for attention scores, value projection, and feed-forward blocks
  - Training with full-sequence cross-entropy loss and evaluation
  - Autoregressive text generation from the trained model

- **`spnet.ipynb`**: Demonstrates Izhikevich spiking neural network simulations using the SpNet module. The notebook illustrates:
  - Creating a spiking network with excitatory and inhibitory neurons
  - Synapse growth using spatial connectivity rules
  - Running network simulations with spike-timing-dependent plasticity (STDP)
  - Visualizing spike patterns and neuron voltage traces
  - Performance profiling and memory usage analysis
