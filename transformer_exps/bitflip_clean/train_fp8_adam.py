"""Binary weights + fp8 Adam latent/state. Exact reproduction of the successful
inline launch (task bygcnhwy9, 2026-04-17) that reached:
  step 100K: test_mse=13.7148, sign_match=0.9943

Recipe:
- Binary forward: w = sign(latent), with sign(0)=1 fallback (±1 weights)
- fp8 latent: fixed scale, clamp to [-1, 1] → fp8 → rescale
- fp8 m, v: per-tph dynamic scaling (amax across trailing dims)
- Latent init: (rand - 0.5) * 0.002 → ±0.001 range
- bs=1024, peak_lr=1e-3, warmup 10%, cosine decay to 0.1 * peak, 100K steps
"""
import os
os.environ['SPIKY_PERMLUT_NO_COMPILE'] = '1'
import sys, math, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from spiky.lutorch.permutational_lut import PermutationalLut

device = 'cuda:0'
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
data = torch.load(os.path.join(EXP_DIR, 'dataset.pt'), weights_only=True)
x, y = data['x'].to(device), data['y'].to(device)
target_signs = data['target_weights'].to(device).sign()
N_TRAIN = 90000

PERM_KWARGS = dict(
    pair_mode='scrambled', soft_mode='ste', temperature=0.1,
    device=device, recompute_in_backward=True, initial_weights_noise=0.001,
)
model = PermutationalLut(
    n_inputs=32, n_outputs=32, n_heads=1,
    input_nap=6, output_nap=32, tph=2048,
    random_seed=42 + 400, **PERM_KWARGS,
)
for k in ['anchor_pairs_a', 'anchor_pairs_b', 'powers']:
    getattr(model.inner.lookup, k).data.copy_(data[k].to(device))
for k in ['idx_a', 'idx_b', 'proj_matrix']:
    getattr(model, k).data.copy_(data[k].to(device))

w = model.inner.projection.weights
print(f'weights shape: {tuple(w.shape)}, N={w.numel():,}')

FP8 = torch.float8_e4m3fn


def q8_fixed(t, mx):
    return (t.clamp(-mx, mx) * (448.0 / mx)).to(FP8).to(torch.float32) / (448.0 / mx)


def q8_per_table(t):
    amax = t.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-20)
    return (t * (448.0 / amax)).to(FP8).to(torch.float32) / (448.0 / amax)


torch.manual_seed(42)
latent = (torch.rand_like(w.data) - 0.5) * 0.002
m_ = torch.zeros_like(latent)
v_ = torch.zeros_like(latent)

beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
base_lr = 0.001
n_steps = 100_000
bs = 1024
warmup = int(0.1 * n_steps)


def lr_scale(step):
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(n_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


model.train()
for step in range(1, n_steps + 1):
    lr = base_lr * lr_scale(step)
    w.data.copy_(latent.sign())
    w.data[w.data == 0] = 1.0
    idx = torch.randint(0, N_TRAIN, (bs,), device=device)
    model.zero_grad()
    out = model(x[idx]).squeeze(1)
    loss = ((out - y[idx]) ** 2).mean()
    loss.backward()
    g = w.grad
    m_.mul_(beta1).add_(g, alpha=1 - beta1)
    v_.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    mhat = m_ / (1 - beta1 ** step)
    vhat = v_ / (1 - beta2 ** step)
    latent.addcdiv_(mhat, vhat.sqrt() + eps_adam, value=-lr)
    latent.copy_(q8_fixed(latent, 1.0))
    m_.copy_(q8_per_table(m_))
    v_.copy_(q8_per_table(v_))

    if step % 10000 == 0 or step == 1:
        model.eval()
        with torch.no_grad():
            w.data.copy_(latent.sign())
            w.data[w.data == 0] = 1.0
            tl = ((model(x[N_TRAIN:]).squeeze(1) - y[N_TRAIN:]) ** 2).mean().item()
        sm = (latent.sign() == target_signs).float().mean().item()
        model.train()
        print(f'step {step}: test={tl:.4f}, sign={sm:.4f}, lr={lr:.5f}', flush=True)
