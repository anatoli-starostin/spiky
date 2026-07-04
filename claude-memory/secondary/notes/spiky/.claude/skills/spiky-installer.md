---
name: spiky-installer
description: Clone and install the spiky project (differentiable lookup tables for spike polychronization)
trigger: install spiky, set up spiky, clone spiky
---

Install the spiky project: both the Python package and native CUDA extensions. The goal is a fully working installation with verified CUDA support and all tests passing.

Run these steps sequentially. Stop and report if any step fails.

1. Clone the repository into the current directory:
```bash
git clone git@github.com:anatoli-starostin/spiky.git
```

2. Set up the Python 3.12 virtual environment and install Python dependencies:
```bash
cd spiky && python3 -m venv ./.venv --system-site-packages && . ./.venv/bin/activate && pip install -r requirements.txt && pip install -e .
```

3. Build and install the native CUDA extensions:
```bash
cd spiky && . ./.venv/bin/activate && pip install -v ./native/lutorch
```

4. Verify that CUDA is available and working:
```bash
cd spiky && . ./.venv/bin/activate && python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')"
```

5. Run all tests and ensure they pass:
```bash
cd spiky && . ./.venv/bin/activate && python -m pytest
```

After all steps complete, confirm that:
- The Python package is installed in editable mode
- Native CUDA extensions compiled and installed successfully
- CUDA is detected and functional
- All tests passed
