---
name: spiky-installer
description: Clone and install the spiky project (differentiable lookup tables for spike polychronization). Use when setting up spiky, installing spiky, or cloning the spiky repository.
---

Install the spiky project: both the Python package and native CUDA extensions (lutorch and spiky_cuda). The goal is a fully working installation with verified CUDA support and all tests passing.

Run these steps sequentially. Stop and report if any step fails.

1. Install system prerequisites (python3-dev for Python.h, ninja-build for fast compilation):
```bash
sudo apt-get update && sudo apt-get install -y python3.12-dev ninja-build
```

2. Clone the repository into the current directory:
```bash
git clone https://github.com/anatoli-starostin/spiky.git
```

3. Set up the Python 3.12 virtual environment and install Python dependencies:
```bash
cd spiky && python3 -m venv ./.venv --system-site-packages && . ./.venv/bin/activate && pip install -r requirements.txt && pip install -e .
```

4. Build and install both native CUDA extensions (use --no-build-isolation so torch is available during build):
```bash
cd spiky && . ./.venv/bin/activate && pip install -v --no-build-isolation ./native/lutorch && pip install -v --no-build-isolation ./native/spiky
```

5. Verify that CUDA is available and working:
```bash
cd spiky && . ./.venv/bin/activate && python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')"
```

6. Install pytest and run all tests:
```bash
cd spiky && . ./.venv/bin/activate && pip install pytest && python -m pytest
```

After all steps complete, confirm that:
- The Python package is installed in editable mode
- Both native CUDA extensions (lutorch_cuda and spiky_cuda) compiled and installed successfully
- CUDA is detected and functional
- All tests passed
