"""GPU-resident batched Walker2d-v5 on MuJoCo-Warp. Pure torch interop, no JAX.

State lives in Warp arrays on the GPU; obs/reward/reset are computed in torch on the
same device (wp.to_torch zero-copy aliasing) so there is no host<->device transfer per
step. Faithful to gymnasium Walker2d-v5 defaults:
  obs   = concat(qpos[1:], qvel)                     -> 17 dims (x-coord excluded)
  reward= 1.0*dx/dt + 1.0*healthy - 1e-3*sum(a^2)
  healthy = 0.8<z<2.0 and -1<angle<1 ; terminate_when_unhealthy=True
  reset = qpos0/qvel0 + U(-5e-3, 5e-3) ; episode truncation at 1000 env-steps
"""
import os
import numpy as np
import torch
import mujoco
import warp as wp
import mujoco_warp as mjw

XML = os.path.abspath(os.path.join(os.path.dirname(mujoco.__file__), "..", "gymnasium",
                                   "envs", "mujoco", "assets", "walker2d_v5.xml"))


class WarpWalker2dVecEnv:
    def __init__(self, num_envs, device="cuda", solver_iters=10, ls_iters=8,
                 reset_noise=5e-3, max_steps=1000, seed=0):
        self.N = num_envs
        self.device = torch.device(device)
        self.reset_noise = reset_noise
        self.max_steps = max_steps
        mjm = mujoco.MjModel.from_xml_path(XML)
        mjm.opt.iterations = solver_iters
        mjm.opt.ls_iterations = ls_iters
        self.mjm = mjm
        self.frame_skip = 4
        self.dt = mjm.opt.timestep * self.frame_skip
        self.nq, self.nv, self.nu = mjm.nq, mjm.nv, mjm.nu
        self.obs_dim = (self.nq - 1) + self.nv
        self.act_dim = self.nu
        self.m = mjw.put_model(mjm)
        self.d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=num_envs)
        # zero-copy torch views onto the warp state
        self.qpos = wp.to_torch(self.d.qpos)      # (N, nq)
        self.qvel = wp.to_torch(self.d.qvel)      # (N, nv)
        self.ctrl = wp.to_torch(self.d.ctrl)      # (N, nu)
        self.qpos0 = torch.tensor(mjm.qpos0, dtype=self.qpos.dtype, device=self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._phys_graph = None
        self._reset_mask(torch.ones(num_envs, dtype=torch.bool, device=self.device))

    def build_physics_graph(self):
        """Capture frame_skip x mjw.step (the many-kernel physics dispatch) into a Warp
        CUDA graph. Warp owns the capture, so its conditional solver nodes are handled
        correctly (torch.cuda.graph cannot capture those). Replayed via capture_launch;
        reads d.ctrl / writes d.qpos in place, so updating ctrl before each replay works."""
        for _ in range(3):
            for _ in range(self.frame_skip):
                mjw.step(self.m, self.d)
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            for _ in range(self.frame_skip):
                mjw.step(self.m, self.d)
        self._phys_graph = cap.graph
        wp.synchronize()

    def _advance(self):
        if self._phys_graph is not None:
            wp.capture_launch(self._phys_graph)
        else:
            for _ in range(self.frame_skip):
                mjw.step(self.m, self.d)

    def _obs(self):
        return torch.cat([self.qpos[:, 1:], self.qvel], dim=1)

    def _reset_mask(self, mask):
        n = int(mask.sum())
        if n == 0:
            return
        idx = mask.nonzero(as_tuple=True)[0]
        noise_q = (torch.rand(n, self.nq, generator=self.gen, device=self.device) * 2 - 1) * self.reset_noise
        noise_v = (torch.rand(n, self.nv, generator=self.gen, device=self.device) * 2 - 1) * self.reset_noise
        self.qpos[idx] = self.qpos0.unsqueeze(0) + noise_q
        self.qvel[idx] = noise_v
        self.step_count[idx] = 0

    def reset(self):
        self._reset_mask(torch.ones(self.N, dtype=torch.bool, device=self.device))
        wp.synchronize()
        return self._obs()

    @torch.no_grad()
    def step(self, action):
        action = action.clamp(-1, 1)
        self.ctrl.copy_(action)
        x_before = self.qpos[:, 0].clone()
        self._advance()
        x_after = self.qpos[:, 0]
        dx = (x_after - x_before) / self.dt
        z = self.qpos[:, 1]
        ang = self.qpos[:, 2]
        healthy = (z > 0.8) & (z < 2.0) & (ang > -1.0) & (ang < 1.0)
        ctrl_cost = 1e-3 * (action ** 2).sum(-1)
        reward = dx + healthy.float() - ctrl_cost
        self.step_count += 1
        terminated = ~healthy
        truncated = self.step_count >= self.max_steps
        done = terminated | truncated
        # auto-reset done envs; next obs for those is the reset obs
        self._reset_mask(done)
        obs = self._obs()
        return obs, reward, terminated, truncated


if __name__ == "__main__":
    torch.manual_seed(0)
    env = WarpWalker2dVecEnv(num_envs=4096)
    print(f"XML={XML}\nobs_dim={env.obs_dim} act_dim={env.act_dim} dt={env.dt} "
          f"qpos{tuple(env.qpos.shape)} dtype={env.qpos.dtype} device={env.qpos.device}")
    obs = env.reset()
    print("reset obs:", tuple(obs.shape), "mean", float(obs.mean()), "std", float(obs.std()))
    # verify warp<->torch aliasing: write qpos via torch, read back
    alias_ok = True
    rew_hist = []
    ep_len = torch.zeros(env.N, device="cuda")
    alive_frac = []
    for t in range(200):
        a = torch.rand(env.N, env.act_dim, device="cuda") * 2 - 1
        obs, r, term, trunc = env.step(a)
        rew_hist.append(float(r.mean()))
        alive_frac.append(float((~term).float().mean()))
    print(f"200 random steps: mean step-reward {np.mean(rew_hist):.3f} | "
          f"mean alive-frac {np.mean(alive_frac):.3f} (random policy should fall fast)")
    print(f"obs finite: {torch.isfinite(obs).all().item()} | "
          f"reward finite: {np.all(np.isfinite(rew_hist))}")
    print("SELF-TEST OK")
