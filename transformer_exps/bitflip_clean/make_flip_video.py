"""
Visualize how PermLut output changes as we flip weights one by one.

Takes a single input, flips random weights sequentially,
plots the output after each flip. Makes a video at 24fps, 120s.
"""
import os
os.environ['SPIKY_PERMLUT_NO_COMPILE'] = '1'
import sys, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from spiky.lutorch.permutational_lut import PermutationalLut

device = 'cuda:0'
data = torch.load(os.path.join(os.path.dirname(__file__), 'dataset.pt'), weights_only=True)

PERM_KWARGS = dict(pair_mode='scrambled', soft_mode='ste', temperature=0.1,
    device=device, recompute_in_backward=True, initial_weights_noise=0.001)
model = PermutationalLut(n_inputs=32, n_outputs=32, n_heads=1,
    input_nap=6, output_nap=32, tph=2048, random_seed=42+400, **PERM_KWARGS)
model.inner.lookup.anchor_pairs_a.data.copy_(data['anchor_pairs_a'].to(device))
model.inner.lookup.anchor_pairs_b.data.copy_(data['anchor_pairs_b'].to(device))
model.inner.lookup.powers.data.copy_(data['powers'].to(device))
model.idx_a.data.copy_(data['idx_a'].to(device))
model.idx_b.data.copy_(data['idx_b'].to(device))
model.proj_matrix.data.copy_(data['proj_matrix'].to(device))

model.inner.projection.weights.data.copy_(data['target_weights'].to(device))
model.eval()

torch.manual_seed(77)
x_single = torch.randn(1, 32, device=device)

with torch.no_grad():
    target_out = model(x_single).squeeze().cpu().numpy()

torch.manual_seed(42)
w = model.inner.projection.weights
w.data.copy_(torch.randint(0, 2, w.shape, device=device).float() * 2.0 - 1.0)
flat_w = w.data.view(-1)
N = flat_w.numel()

FPS = 24
DURATION = 120
N_FRAMES = FPS * DURATION

print(f'Generating {N_FRAMES} frames ({DURATION}s at {FPS}fps)')

frames = []
with torch.no_grad():
    out = model(x_single).squeeze().cpu().numpy()
    frames.append(out.copy())
    for i in range(N_FRAMES - 1):
        idx = torch.randint(0, N, (1,)).item()
        flat_w[idx] *= -1
        out = model(x_single).squeeze().cpu().numpy()
        frames.append(out.copy())
        if (i + 1) % 500 == 0:
            print(f'  frame {i+1}/{N_FRAMES}', flush=True)

print('Rendering video...')

import imageio.v3 as iio

dims = np.arange(32)
y_range = max(abs(target_out).max(), max(abs(f).max() for f in frames)) * 1.2
out_path = os.path.join(os.path.dirname(__file__), 'flip_video.mp4')

with iio.imopen(out_path, 'w', plugin='pyav') as writer:
    writer.init_video_stream('libx264', fps=FPS)
    for frame_idx in range(N_FRAMES):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        out = frames[frame_idx]
        ax1.bar(dims - 0.2, target_out, 0.4, color='blue', alpha=0.6, label='target')
        ax1.bar(dims + 0.2, out, 0.4, color='red', alpha=0.6, label='current')
        ax1.set_ylabel('output value')
        ax1.set_title(f'PermLut output after {frame_idx} random flips (of {N:,} weights)')
        ax1.legend(loc='upper right')
        ax1.set_xlim(-1, 32)
        ax1.set_ylim(-y_range, y_range)
        error = out - target_out
        colors = ['green' if abs(e) < 10 else 'orange' if abs(e) < 50 else 'red' for e in error]
        ax2.bar(dims, error, color=colors)
        ax2.set_ylabel('error (current - target)')
        ax2.set_xlabel('output dimension')
        mse = np.mean(error ** 2)
        ax2.set_title(f'MSE = {mse:.1f}')
        ax2.set_xlim(-1, 32)
        ax2.set_ylim(-y_range, y_range)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.write_frame(img)
        plt.close(fig)
        if (frame_idx + 1) % 100 == 0:
            print(f'  rendered {frame_idx+1}/{N_FRAMES}', flush=True)

print(f'Saved to {out_path}')
