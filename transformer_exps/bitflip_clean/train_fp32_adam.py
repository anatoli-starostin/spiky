"""fp32 Adam baseline: latent + binary forward, no fp8 quantisation anywhere.

If this plateaus like fp8 did, the scaffolding has a fundamental bug.
If this converges, then the fp8 quantisation is the issue.
"""
import os
os.environ['SPIKY_PERMLUT_NO_COMPILE'] = '1'
import sys, math, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from spiky.lutorch.permutational_lut import PermutationalLut

device = 'cuda:0'
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
data = torch.load(os.path.join(EXP_DIR, 'dataset.pt'), weights_only=True)
x = data['x'].to(device)
y = data['y'].to(device)
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
model.inner.lookup.anchor_pairs_a.data.copy_(data['anchor_pairs_a'].to(device))
model.inner.lookup.anchor_pairs_b.data.copy_(data['anchor_pairs_b'].to(device))
model.inner.lookup.powers.data.copy_(data['powers'].to(device))
model.idx_a.data.copy_(data['idx_a'].to(device))
model.idx_b.data.copy_(data['idx_b'].to(device))
model.proj_matrix.data.copy_(data['proj_matrix'].to(device))

w = model.inner.projection.weights
print(f'weights shape: {tuple(w.shape)}, N={w.numel():,}')

torch.manual_seed(42)
latent = torch.empty_like(w).uniform_(-0.1, 0.1)
m_ = torch.zeros_like(w)
v_ = torch.zeros_like(w)

peak_lr = 1e-3
beta1, beta2 = 0.9, 0.999
eps = 1e-8

n_steps = 100_000
bs = 1024
warmup_frac = 0.1
log_every = 5000


def lr_at(step):
    warmup = int(warmup_frac * n_steps)
    if step < warmup:
        return peak_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


model.train()
for step in range(1, n_steps + 1):
    idx = torch.randint(0, N_TRAIN, (bs,), device=device)
    w.data.copy_(latent.sign())
    model.zero_grad()
    out = model(x[idx]).squeeze(1)
    loss = ((out - y[idx]) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        g = w.grad
        m_.mul_(beta1).add_(g, alpha=1 - beta1)
        v_.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step
        mhat = m_ / bc1
        vhat = v_ / bc2
        latent.addcdiv_(mhat, vhat.sqrt() + eps, value=-lr_at(step))

    if step % log_every == 0 or step == 1:
        w.data.copy_(latent.sign())
        model.eval()
        with torch.no_grad():
            tl = ((model(x[N_TRAIN:]).squeeze(1) - y[N_TRAIN:]) ** 2).mean().item()
        sm = (w.data.view(-1).sign() == target_signs.view(-1)).float().mean().item()
        lat_std = latent.std().item()
        g_norm = g.norm().item()
        model.train()
        print(f'step {step:6d}: mse={tl:.3f}, sign={sm:.4f}, lat_std={lat_std:.4f}, g_norm={g_norm:.2f}', flush=True)

w.data.copy_(latent.sign())
model.eval()
with torch.no_grad():
    tl = ((model(x[N_TRAIN:]).squeeze(1) - y[N_TRAIN:]) ** 2).mean().item()
sm = (w.data.view(-1).sign() == target_signs.view(-1)).float().mean().item()
print(f'FINAL: mse={tl:.3f}, sign={sm:.4f}')
