"""Profile exp292's model to find the slow parts."""
import os, sys, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from spiky.lutorch.permutational_lut import PermutationalLut

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
torch.manual_seed(cfg['random_seed'])

E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
D_QK_P = d_qk * (d_qk - 1) // 2
D_V_P = d_v * (d_v - 1) // 2
CONTEXT_SIZE = cfg['context_size']
BS = cfg['batch_size']

PERM_KWARGS = dict(
    pair_mode='scrambled', soft_mode='ste', temperature=0.1,
    device=DEVICE, recompute_in_backward=True, initial_weights_noise=0.001,
)

q_perm = PermutationalLut(n_inputs=E, n_outputs=d_qk, n_heads=H,
    input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
    tph=cfg['qk_tph'], return_dominance=True, random_seed=42, **PERM_KWARGS)
v_perm = PermutationalLut(n_inputs=E, n_outputs=d_v, n_heads=H,
    input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'],
    tph=cfg['v_tph'], return_dominance=True, random_seed=43, **PERM_KWARGS)
out_proj = PermutationalLut(n_inputs=H*d_v, n_outputs=E, n_heads=1,
    input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'],
    tph=cfg['out_tph'], random_seed=44, **PERM_KWARGS)

borda_m = v_perm.dom_borda_m.clone()
out_norm = nn.LayerNorm(E).to(DEVICE)

BT = BS * CONTEXT_SIZE
x = torch.randn(BT, E, device=DEVICE)

def sync():
    torch.cuda.synchronize()

def bench(name, fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.time()
    for _ in range(n):
        fn()
    sync()
    elapsed = (time.time() - t0) / n * 1000
    print(f'  {name:30s} {elapsed:7.2f} ms')
    return elapsed

print(f'Config: bs={BS}, ctx={CONTEXT_SIZE}, BT={BT}, E={E}, H={H}, d_qk={d_qk}, d_v={d_v}')
print(f'P_qk={D_QK_P}, P_v={D_V_P}')
print()

print('Forward (inference, bs=128):')
with torch.no_grad():
    t_q = bench('q_perm', lambda: q_perm(x))
    t_v = bench('v_perm', lambda: v_perm(x))

    q = q_perm(x).reshape(BS, CONTEXT_SIZE, H, D_QK_P).permute(0, 2, 1, 3).contiguous()
    k = q.clone()
    v_dom = v_perm(x).reshape(BS, CONTEXT_SIZE, H, D_V_P).permute(0, 2, 1, 3).contiguous()

    t_sdpa = bench('SDPA(q, k, v) causal',
        lambda: F.scaled_dot_product_attention(q, k, v_dom, is_causal=True))

    attn_dom = F.scaled_dot_product_attention(q, k, v_dom, is_causal=True)
    t_borda = bench('Borda einsum',
        lambda: torch.einsum('bhtp,kp->bhtk', attn_dom, borda_m))

    attn = torch.einsum('bhtp,kp->bhtk', attn_dom, borda_m)
    attn_flat = attn.permute(0, 2, 1, 3).reshape(BT, H * d_v)

    t_out = bench('out_proj', lambda: out_proj(attn_flat))

    out = out_proj(attn_flat).squeeze(1).reshape(BS, CONTEXT_SIZE, E)
    t_norm = bench('LayerNorm(E)', lambda: out_norm(out))

print()
layer_total = 2 * t_q + t_v + t_sdpa + t_borda + t_out + t_norm
print(f'Per-layer total (est): {layer_total:.2f} ms')
print(f'  = 2×q_perm ({2*t_q:.2f}) + v_perm ({t_v:.2f}) + SDPA ({t_sdpa:.2f}) + Borda ({t_borda:.2f}) + out_proj ({t_out:.2f}) + LN ({t_norm:.2f})')
print(f'Estimated forward for 6 layers: {layer_total*6:.2f} ms')

print()
print('Full forward+backward (training, 6 layers):')

class Block(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.q = PermutationalLut(n_inputs=E, n_outputs=d_qk, n_heads=H,
            input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
            tph=cfg['qk_tph'], return_dominance=True, random_seed=100+idx, **PERM_KWARGS)
        self.k = PermutationalLut(n_inputs=E, n_outputs=d_qk, n_heads=H,
            input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
            tph=cfg['qk_tph'], return_dominance=True, random_seed=200+idx, **PERM_KWARGS)
        self.v = PermutationalLut(n_inputs=E, n_outputs=d_v, n_heads=H,
            input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'],
            tph=cfg['v_tph'], return_dominance=True, random_seed=300+idx, **PERM_KWARGS)
        self.op = PermutationalLut(n_inputs=H*d_v, n_outputs=E, n_heads=1,
            input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'],
            tph=cfg['out_tph'], random_seed=400+idx, **PERM_KWARGS)
        self.ln = nn.LayerNorm(E)
        self.register_buffer('b', self.v.dom_borda_m.clone())

    def forward(self, x):
        B, T, _E = x.shape
        xf = x.reshape(B*T, _E)
        q = self.q(xf).reshape(B, T, H, D_QK_P).permute(0,2,1,3)
        k = self.k(xf).reshape(B, T, H, D_QK_P).permute(0,2,1,3)
        v = self.v(xf).reshape(B, T, H, D_V_P).permute(0,2,1,3)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = torch.einsum('bhtp,kp->bhtk', a, self.b)
        y = self.op(a.permute(0,2,1,3).reshape(B*T, H*d_v)).squeeze(1).reshape(B, T, _E)
        return self.ln(y)

model = nn.ModuleList([Block(i) for i in range(cfg['num_layers'])]).to(DEVICE)

x_full = torch.randn(BS, CONTEXT_SIZE, E, device=DEVICE, requires_grad=True)
target = torch.randn_like(x_full)

def step():
    x = x_full
    for blk in model:
        x = blk(x)
    loss = ((x - target)**2).mean()
    loss.backward()
    x_full.grad = None
    for p in model.parameters():
        p.grad = None

bench('6-layer fwd+bwd', step, n=10, warmup=3)
