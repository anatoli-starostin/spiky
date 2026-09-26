# Walker2d demo-stand WebSocket server — headless. Physics/state streaming only (we never render), so no
# display, no EGL/OSMesa, no MUJOCO_GL needed. libgl1/libglib are installed only defensively: some mujoco
# builds dlopen a GL loader at `import mujoco`; they add little and guarantee import works on python:slim.
#
# BUILD CONTEXT IS THE REPO ROOT (not ./server) — see docker-compose.yml and the repo-root .dockerignore.
# The context moved because the "Spiking LUT" actor needs the `spiky_cuda` native extension and the `spiky`
# python package, neither of which lives under server/. All COPY paths below are repo-root-relative.
FROM python:3.12-slim

# ============================================================================================
#  !!  spiky_cuda MUST BE COMPILED ON A GPU-LESS HOST  !!
#
#  native/spiky/setup.py chooses its extension type from torch.cuda.is_available() AT BUILD
#  TIME:
#      available  -> CUDAExtension  (needs nvcc, links -lcuda, requires a GPU at runtime)
#      absent     -> CppExtension   (-DNO_CUDA, pure C++, what this image needs)
#
#  Docker builds see no GPU, so building HERE selects the CPU variant automatically — that is
#  the whole point of compiling in-image rather than shipping a prebuilt .so. If someone
#  instead builds the extension on a GPU machine and copies it in, it will import fine there
#  and fail inside this container. The repo-root .dockerignore therefore also excludes
#  native/spiky/build/ and native/**/*.so so a developer's CUDA build cannot leak in.
#
#  VERIFY AFTER BUILDING (see DEPLOY_spiking_lut.md):
#      docker compose run --rm server python -c "import spiky_cuda; print('ok')"
# ============================================================================================

# g++ and the codegen toolchain are needed to compile the extension. Kept in this single stage
# deliberately: a multi-stage copy of the built .so would reintroduce exactly the "which host
# compiled it" ambiguity the comment above warns about.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 g++ \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY landing/walker2d-viz/server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# torch, from PyTorch's CPU index. NOT from PyPI: the default `torch` wheel bundles the CUDA
# runtime (~2.5 GB of nvidia-* packages) that this GPU-less box can never use. The CPU wheel is
# ~200 MB and is all spiky_cuda needs to compile and run against.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.1

# Native extension + the python package the spiking actor imports. Built from source here so
# the CPU variant is selected (see the banner above).
COPY native ./native
COPY src ./src
RUN pip install --no-cache-dir --no-build-isolation ./native/spiky \
 && python -c "import torch; import spiky_cuda; print('spiky_cuda: CPU extension built and importable')"

# `spiky` is imported from src/ rather than installed, matching how the repo is used elsewhere.
ENV PYTHONPATH=/app/src

# App code + auto-discovered actors + model checkpoints (baked into the image).
COPY landing/walker2d-viz/server/server.py ./
COPY landing/walker2d-viz/server/actors ./actors
COPY landing/walker2d-viz/server/models ./models

# All runtime knobs are env-configurable (see server.py). Override at `docker run`/compose time.
ENV HOST=0.0.0.0 \
    PORT=8765 \
    ENV_ID=Walker2d-v5 \
    SPS=30 \
    MAX_SESSIONS=6 \
    PYTHONUNBUFFERED=1

EXPOSE 8765
CMD ["python", "server.py"]
