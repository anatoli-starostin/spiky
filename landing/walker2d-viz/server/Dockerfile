# Walker2d demo-stand WebSocket server — headless. Physics/state streaming only (we never render), so no
# display, no EGL/OSMesa, no MUJOCO_GL needed. libgl1/libglib are installed only defensively: some mujoco
# builds dlopen a GL loader at `import mujoco`; they add little and guarantee import works on python:slim.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code + auto-discovered actors + model checkpoints (baked into the image).
COPY server.py ./
COPY actors ./actors
COPY models ./models

# All runtime knobs are env-configurable (see server.py). Override at `docker run`/compose time.
ENV HOST=0.0.0.0 \
    PORT=8765 \
    ENV_ID=Walker2d-v5 \
    SPS=30 \
    MAX_SESSIONS=6 \
    PYTHONUNBUFFERED=1

EXPOSE 8765
CMD ["python", "server.py"]
