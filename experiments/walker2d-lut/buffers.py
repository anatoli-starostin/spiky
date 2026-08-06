"""On-device circular replay buffer for the massively-parallel regime — thousands of
envs write per step, sampling and storage are pure GPU tensor ops, no host transfers."""
import torch


class GPUReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device):
        self.cap = capacity
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.act = torch.zeros(capacity, act_dim, device=device)
        self.rew = torch.zeros(capacity, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.done = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.added = 0

    @torch.no_grad()
    def add_batch(self, o, a, r, no, d):
        n = o.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.cap
        self.obs[idx] = o; self.act[idx] = a; self.rew[idx] = r
        self.next_obs[idx] = no; self.done[idx] = d
        self.ptr = (self.ptr + n) % self.cap
        self.added += n

    def size(self):
        return min(self.added, self.cap)

    @torch.no_grad()
    def sample(self, B):
        idx = torch.randint(0, self.size(), (B,), device=self.device)
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.next_obs[idx], self.done[idx])
