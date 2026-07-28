"""Encode a frame dump from render_cpu.py into an MP4.

Run under the TORCH venv (`~/projects/spiky/.venv`), which has imageio + imageio-ffmpeg;
the JAX/MJX venv is kept lean and has neither.

    ~/projects/spiky/.venv/bin/python encode_frames.py <frames.npz> <out.mp4>
"""
import sys

import numpy as np
import imageio

src, dst = sys.argv[1], sys.argv[2]
z = np.load(src)
frames, fps = z["frames"], int(z["fps"])
imageio.mimwrite(dst, list(frames), fps=fps, codec="libx264",
                 output_params=["-pix_fmt", "yuv420p"])
print(f"encoded {len(frames)} frames @ {fps} fps -> {dst}")
