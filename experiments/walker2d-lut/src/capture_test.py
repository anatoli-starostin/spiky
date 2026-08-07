"""Minimal probe: can ONE torch.cuda.graph capture both torch (policy) and
MuJoCo-Warp (mjw.step) ops via warp<->torch stream bridging? Verify replay advances
physics and RNG is graph-safe (different actions each replay)."""
import os
import torch
import warp as wp
import mujoco_warp as mjw
from warp_env import WarpWalker2dVecEnv
from models import REGISTRY

N = 4096
env = WarpWalker2dVecEnv(num_envs=N, seed=0)
ac = REGISTRY["mlp"](env.obs_dim, env.act_dim).to("cuda")
mean = torch.zeros(env.obs_dim, device="cuda"); std = torch.ones(env.obs_dim, device="cuda")
act_out = torch.zeros(N, env.act_dim, device="cuda")   # static output buffer

def region():
    nobs = (torch.cat([env.qpos[:, 1:], env.qvel], 1) - mean) / std
    a, _, _ = ac.act(nobs)
    act_out.copy_(a.clamp(-1, 1))
    env.ctrl.copy_(act_out)
    for _ in range(env.frame_skip):
        mjw.step(env.m, env.d)

# --- warmup on a side stream (loads kernels, populates state) ---
env.reset()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(5):
        region()
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

# --- capture torch + warp into one graph ---
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        wp_stream = wp.stream_from_torch(torch.cuda.current_stream())
        with wp.ScopedStream(wp_stream):
            region()
    print("CAPTURE OK (unified torch+warp graph built)")
except Exception as e:
    print("CAPTURE FAILED:", type(e).__name__, str(e)[:300]); raise

# --- replay and verify ---
qx0 = env.qpos[:, 0].clone()
a_first = act_out.clone()
g.replay(); torch.cuda.synchronize()
a_second = act_out.clone()
qx1 = env.qpos[:, 0].clone()
for _ in range(20):
    g.replay()
torch.cuda.synchronize()
qx2 = env.qpos[:, 0].clone()
print(f"physics advanced across replays: dx(1)={float((qx1-qx0).abs().mean()):.5f} "
      f"dx(20more)={float((qx2-qx1).abs().mean()):.5f}")
print(f"RNG graph-safe (actions differ across replays): "
      f"{float((a_first-a_second).abs().mean()):.5f} (>0 means fresh randomness)")
print(f"finite: act={torch.isfinite(act_out).all().item()} qpos={torch.isfinite(env.qpos).all().item()}")
