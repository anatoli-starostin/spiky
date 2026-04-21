"""A/B benchmark of TinyAnchorPairsLookup backward:
   cuda  = custom kernel with float atomicAdd (old path, tiny_apl_bwd_kernel)
   torch = compute du densely + two scatter_add_ calls (new deterministic path)

Both monkey-patch `_TinyAnchorPairsLookupFunction.backward` inside this process,
so everything else (model, data, optimizer) is identical.
"""
import os, sys, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.util.text_snippet_sampler import TextSnippetSampler
from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.bit_permutation_lut_optimizer import BitPermutationLUTOptimizer
from spiky.lutorch.ranking_tools import DominanceCanonicalize, DominanceToVector
from spiky.lutorch import tiny_anchor_pairs_lookup as tapl

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
CONTEXT_SIZE = cfg['context_size']; VOCAB_SIZE = cfg['vocab_size']; BOS_ID = 256
DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'workbooks', 'fineweb_texts.txt'))
torch.manual_seed(cfg['random_seed'])

E = cfg['embedding_dim']; H = cfg['n_heads']; d_qk = cfg['d_qk']; d_v = cfg['d_v']
N_LAYERS = cfg['num_layers']
D_QK_P = d_qk*(d_qk-1)//2; D_V_P = d_v*(d_v-1)//2


def _make_qk(so):
    return BitPermutationLUT(n_inputs=E, n_outputs=d_qk, n_heads=H,
        input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'], tph=cfg['qk_tph'],
        random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype','fp8'), device=DEVICE)
def _make_v(so):
    return BitPermutationLUT(n_inputs=E, n_outputs=d_v, n_heads=H,
        input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'], tph=cfg['v_tph'],
        random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype','fp8'), device=DEVICE)
def _make_out(so):
    return BitPermutationLUT(n_inputs=H*d_v, n_outputs=E, n_heads=1,
        input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'], tph=cfg['out_tph'],
        random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype','fp8'), device=DEVICE)


class LUTBlock(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.q_perm = _make_qk(i); self.k_perm = _make_qk(100+i)
        self.v_perm = _make_v(200+i); self.out_proj = _make_out(400+i)
        ct = cfg.get('canon_temperature', 0.1)
        self.q_canon = DominanceCanonicalize(d_qk, temperature=ct)
        self.k_canon = DominanceCanonicalize(d_qk, temperature=ct)
        self.attn_to_vec = DominanceToVector(d_v, normalise=False)
        self.out_to_vec = DominanceToVector(E)
        self.attn_scale = nn.Parameter(torch.tensor(float(cfg.get('learnable_attn_scale_init', 0.25))))

    def forward(self, x, pe):
        B, T, _E = x.shape
        xp = (x + pe.unsqueeze(0)).reshape(B*T, _E)
        xf = x.reshape(B*T, _E)
        q = self.q_canon(self.q_perm(xp)).reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        k = self.k_canon(self.k_perm(xp)).reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        v = self.v_perm(xf).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)
        ad = F.scaled_dot_product_attention(q*self.attn_scale, k, v, is_causal=True)
        a = self.attn_to_vec(ad)
        oi = a.permute(0, 2, 1, 3).reshape(B*T, H*d_v)
        od = self.out_proj(oi)
        return self.out_to_vec(od).squeeze(1).reshape(B, T, _E)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.randn(CONTEXT_SIZE, E)*0.1) for _ in range(N_LAYERS)])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        cd = E*N_LAYERS
        self.unembedder = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd, cd*4), nn.ReLU(), nn.Linear(cd*4, VOCAB_SIZE))

    def forward(self, t):
        x = self.token_embedder(t)
        outs = []
        for layer, pe in zip(self.layers, self.pos_embs):
            x = layer(x, pe); outs.append(x)
        return self.unembedder(torch.cat(outs, dim=-1))


def get_lr_scale(step):
    ns = cfg['n_steps']; w = int(cfg.get('lr_warmup_fraction', 0.1)*ns)
    if step < w: return step / max(w, 1)
    p = (step - w) / max(ns - w, 1)
    return 0.1 + 0.9*0.5*(1 + math.cos(math.pi*p))


# --- Two backward variants, installed by monkey-patching the autograd.Function. ---
_orig_bwd = tapl._TinyAnchorPairsLookupFunction.backward  # whatever is currently installed


def _make_cuda_backward():
    """Old path: custom kernel with float atomicAdd."""
    @staticmethod
    def backward(ctx, grad_li, grad_lai, grad_lad, grad_lig, grad_laig):
        x, anchor1_ids, anchor2_ids, lookup_alt_deltas, batch_offset = ctx.saved_tensors
        x_grad_flat = tapl._tiny_backward_cuda(
            x, anchor1_ids, anchor2_ids, lookup_alt_deltas,
            grad_lig, grad_laig, grad_direct=grad_lad,
        )
        return x_grad_flat.view(x.shape), None, None, None, None
    return backward


def _make_torch_backward():
    """New path: dense du + scatter_add_ (deterministic under the flag)."""
    @staticmethod
    def backward(ctx, grad_li, grad_lai, grad_lad, grad_lig, grad_laig):
        x, anchor1_ids, anchor2_ids, lookup_alt_deltas, batch_offset = ctx.saved_tensors
        x_grad_flat = tapl._tiny_backward_pytorch(
            x, anchor1_ids, anchor2_ids, lookup_alt_deltas, batch_offset,
            grad_lig, grad_laig, grad_direct=grad_lad,
        )
        return x_grad_flat.view(x.shape), None, None, None, None
    return backward


def run_bench(variant: str, warmup: int, steps: int):
    torch.manual_seed(cfg['random_seed'])
    if variant == 'cuda':
        tapl._TinyAnchorPairsLookupFunction.backward = _make_cuda_backward()
    elif variant == 'torch':
        tapl._TinyAnchorPairsLookupFunction.backward = _make_torch_backward()
    else:
        raise ValueError(variant)
    try:
        sampler = TextSnippetSampler(DATA_PATH, CONTEXT_SIZE, 10_000, DEVICE, random_seed=1)
        model = Model().to(DEVICE)
        bit_luts = [m for layer in model.layers for m in (layer.q_perm, layer.k_perm, layer.v_perm, layer.out_proj)]
        adam_opt = torch.optim.Adam(list(model.parameters()), lr=cfg['adam_lr'])
        adam_scheduler = torch.optim.lr_scheduler.LambdaLR(adam_opt, get_lr_scale)
        bit_opt = BitPermutationLUTOptimizer(bit_luts, lr=cfg['bit_lut_lr'],
            beta1=cfg.get('bit_lut_beta1', 0.9), beta2=cfg.get('bit_lut_beta2', 0.999),
            lr_schedule_fn=get_lr_scale)
        model.train()
        for _ in range(warmup):
            x = sampler.sample_training_batch(cfg['batch_size']).long()
            inp = torch.empty_like(x); inp[:, 0] = BOS_ID; inp[:, 1:] = x[:, :-1]
            logits = model(inp); B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))
            adam_opt.zero_grad(); bit_opt.zero_grad()
            loss.backward(); adam_opt.step(); adam_scheduler.step(); bit_opt.step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            x = sampler.sample_training_batch(cfg['batch_size']).long()
            inp = torch.empty_like(x); inp[:, 0] = BOS_ID; inp[:, 1:] = x[:, :-1]
            logits = model(inp); B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))
            adam_opt.zero_grad(); bit_opt.zero_grad()
            loss.backward(); adam_opt.step(); adam_scheduler.step(); bit_opt.step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f'[tiny_apl_bwd = {variant:5s}]  steps={steps}  total={elapsed:.3f}s  '
              f'per_step={1000*elapsed/steps:.3f} ms')
        bit_opt.close()
    finally:
        tapl._TinyAnchorPairsLookupFunction.backward = _orig_bwd


if __name__ == '__main__':
    WARMUP, STEPS = 20, 200
    # Alternate to cancel any cache/thermal drift.
    run_bench('cuda',  WARMUP, STEPS)
    run_bench('torch', WARMUP, STEPS)
    run_bench('cuda',  WARMUP, STEPS)
    run_bench('torch', WARMUP, STEPS)
