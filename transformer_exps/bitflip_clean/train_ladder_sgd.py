"""4-bit ladder as direct SGD accumulator: state += round(-lr * grad * scale)"""
import os
os.environ['SPIKY_PERMLUT_NO_COMPILE'] = '1'
import sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from spiky.lutorch.permutational_lut import PermutationalLut

device = 'cuda:0'
data = torch.load(os.path.join(os.path.dirname(__file__), 'dataset.pt'), weights_only=True)
x = data['x'].to(device)
y = data['y'].to(device)
target_signs = data['target_weights'].to(device).sign()
N_TRAIN = 90000

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

w = model.inner.projection.weights
N = w.numel()

n_states = 16
LADDER = torch.linspace(-1, 1, n_states, device=device)
torch.manual_seed(42)
state = torch.where(torch.randint(0, 2, (N,), device=device) == 1, n_states-1, 0).long()

lr = 0.0005
n_steps = 100000
bs = 1024

model.train()
for step in range(n_steps):
    idx = torch.randint(0, N_TRAIN, (bs,), device=device)
    w.data.copy_(LADDER[state].view_as(w.data))
    model.zero_grad()
    out = model(x[idx]).squeeze(1)
    loss = ((out - y[idx]) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        flat_g = w.grad.view(-1)
        delta = (-lr * flat_g * (n_states // 2)).round().long()
        state = (state + delta).clamp(0, n_states - 1)

    if step % 10000 == 0:
        w.data.copy_(LADDER[state].view_as(w.data))
        model.eval()
        with torch.no_grad():
            tl = ((model(x[N_TRAIN:]).squeeze(1) - y[N_TRAIN:]) ** 2).mean().item()
        sm = (w.data.view(-1).sign() == target_signs.view(-1)).float().mean().item()
        model.train()
        print(f'step {step}: test={tl:.2f}, sign={sm:.4f}', flush=True)

w.data.copy_(LADDER[state].view_as(w.data))
model.eval()
with torch.no_grad():
    tl = ((model(x[N_TRAIN:]).squeeze(1) - y[N_TRAIN:]) ** 2).mean().item()
sm = (w.data.view(-1).sign() == target_signs.view(-1)).float().mean().item()
print(f'FINAL: test={tl:.2f}, sign={sm:.4f}')
