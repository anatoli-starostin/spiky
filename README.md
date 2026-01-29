# Spiky

**Version 1.0**

An experimental CUDA-enabled, PyTorch-compatible Python library inspired by the [Spiking Manifesto](https://arxiv.org/pdf/2512.11843) (E. Izhikevich), implementing differentiable lookup tables as a simple instrument to model spike polychronization.

**Author:** Anatoly Starostin

## Resources

- **Spiking Manifesto:** [arXiv Paper](https://arxiv.org/pdf/2512.11843)
- **Project Presentation:** [Google Slides](https://docs.google.com/presentation/d/16ZdLnLGjpVy9oCk1FHdEsVbEQv1eOe_jI-SrzM3srmc/edit?usp=sharing)

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

4. Install CUDA extension:
   ```bash
   cd ./spiky_cuda
   pip install -e . --no-build-isolation -v
   cd ..
   ```

## Running Tests

Run the test suites with different seeds:

1. **SpNet tests:**
   ```bash
   cd src/spiky/spnet/tests/
   python run_tests_with_different_seeds.py
   ```

2. **LUT tests:**
   ```bash
   cd ../../lut/tests/
   python run_tests_with_different_seeds.py
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
