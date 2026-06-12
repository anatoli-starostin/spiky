# Spiky

**Version 1.0**

An experimental CUDA-enabled, PyTorch-compatible Python library inspired by the [Spiking Manifesto](https://arxiv.org/pdf/2512.11843) (E. Izhikevich), implementing differentiable lookup tables as a simple instrument to model spike polychronization.

**Author:** Anatoly Starostin

## Resources

- **Spiking Manifesto:** [arXiv Paper](https://arxiv.org/pdf/2512.11843)
- **Project Presentation:** [Google Slides](https://docs.google.com/presentation/d/16ZdLnLGjpVy9oCk1FHdEsVbEQv1eOe_jI-SrzM3srmc/edit?usp=sharing)
- **LUTGPT research report:** [`doc/lutorch/lutgpt_research_report.pdf`](doc/lutorch/lutgpt_research_report.pdf) — full write-up of the LUTGPT model (vanilla baseline, architecture, primitive math, training recipe, efficiency analysis, experiments).

## LUTGPT

The LUTGPT model — a six-layer transformer whose Q / K / V / out projections and per-layer residuals are all `FastMultiHeadLut` tables — ships in two end-to-end entry points:

- **`examples/lutgpt/`** — published reference configuration (narrow backbone, `E=192`, `D=384`, hybrid-smooth all 16K steps; matches `exp755` of the research report at 176 M parameters).
- **`workbooks/nanochat_walkthrough.ipynb`** — single-GPU educational walkthrough at the full-width `exp754` architecture (`E=D=384`, `d_v=64`) at a reduced budget (`bs=8`, 8K steps) so the whole run fits in a few hours.

Both entry points use the `FastMultiHeadLut` primitive in `src/spiky/lutorch/fast_multi_head_lut.py`, backed by the hand-written `lutorch_cuda` CUDA bit-pack kernel in `native/lutorch/lutorch.cu`.

## Documentation

- **`doc/lutorch/`** — LUTorch (LUT-based, PyTorch-compatible layers and training); includes the **LUTGPT research report** (`lutgpt_research_report.pdf`).
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

# CPU only
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

- **`nanochat_walkthrough.ipynb`**: End-to-end walkthrough of LUTGPT on the nanochat dataset (RustBPE tokeniser, 32 768 vocab, 512 context). The notebook covers:
  - Loading the tokenised dataset and inspecting the train/val split
  - Training a vanilla `MinimalGPT+RoPE` baseline as a comparison point
  - Building and training a LUTGPT at the full-width `exp754` architecture (`E=D=384`, `d_v=64`) at a reduced single-GPU budget (`bs=8`, 8K steps)
  - Comparing validation curves between the baseline and the LUT model
  - Optionally flipping the LUT forward mode to `hard` at eval time to show the magnitude-blind deployment number
  - Requires a local checkout of [nanochat](https://github.com/karpathy/nanochat) and the `NANOCHAT_ROOT` environment variable pointing at it

- **`spnet.ipynb`**: Demonstrates Izhikevich spiking neural network simulations using the SpNet module. The notebook illustrates:
  - Creating a spiking network with excitatory and inhibitory neurons
  - Synapse growth using spatial connectivity rules
  - Running network simulations with spike-timing-dependent plasticity (STDP)
  - Visualizing spike patterns and neuron voltage traces
  - Performance profiling and memory usage analysis
