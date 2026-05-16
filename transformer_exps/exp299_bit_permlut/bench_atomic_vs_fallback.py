"""Benchmark training step time with and without atomic-kernel path in
`_project_grad_out_to_weight_grad` on exp299 architecture.
"""
import os, sys, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.util.text_snippet_sampler import TextSnippetSampler
from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch import bit_permutation_lut_optimizer as bplo
from spiky.lutorch.bit_permutation_lut_optimizer import BitPermutationLUTOptimizer
from spiky.lutorch.ranking_tools import RankAttention

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
CONTEXT_SIZE = cfg['context_size']
VOCAB_SIZE = cfg['vocab_size']
BOS_ID = 256
DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'workbooks', 'fineweb_texts.txt')
)

torch.manual_seed(cfg['random_seed'])

E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
N_LAYERS = cfg['num_layers']
D_QK_P = d_qk * (d_qk - 1) // 2
D_V_P = d_v * (d_v - 1) // 2


def _make_qk(seed_offset):
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_qk, n_heads=H,
        input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
        tph=cfg['qk_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
    )

def _make_v(seed_offset):
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_v, n_heads=H,
        input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'],
        tph=cfg['v_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
    )

def _make_out(seed_offset):
    return BitPermutationLUT(
        n_inputs=H * d_v, n_outputs=E, n_heads=1,
        input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'],
        tph=cfg['out_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_perm = _make_qk(layer_idx)
        self.k_perm = _make_qk(100 + layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v_perm = _make_v(200 + layer_idx)
        self.rank_attn = RankAttention(
            d_qk=d_qk, d_v=D_V_P, smooth_mode=False,
            temperature=cfg.get('rank_attn_temperature', 0.1),
            sdpa_temperature=1.0, sdpa_forward_temperature=1.0,
            learnable_attn_scale_init=cfg.get('learnable_attn_scale_init', None),
        )
        self.out_proj = _make_out(400 + layer_idx)
        self.out_norm = nn.LayerNorm(E)
        self.register_buffer('q_borda_m', self.q_perm.dom_borda_m.clone())
        self.register_buffer('v_borda_m', self.v_perm.dom_borda_m.clone())
        self.register_buffer('out_borda_m', self.out_proj.dom_borda_m.clone())

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, _E)
        x_flat = x.reshape(B * T, _E)
        q_dom = self.q_perm(xp)
        k_dom = self.k_perm(xp)
        q = torch.einsum('bhp,kp->bhk', q_dom, self.q_borda_m)
        k = torch.einsum('bhp,kp->bhk', k_dom, self.q_borda_m)
        q = q.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q = self.q_norm(q); k = self.k_norm(k)
        v_dom = self.v_perm(x_flat).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)
        attn_dom = self.rank_attn(q, k, v_dom, is_causal=True)
        attn = torch.einsum('bhtp,kp->bhtk', attn_dom, self.v_borda_m)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_dom = self.out_proj(out_in)
        out = torch.einsum('bhp,kp->bhk', out_dom, self.out_borda_m).squeeze(1).reshape(B, T, _E)
        out = self.out_norm(out)
        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.pos_embs = nn.ParameterList(
            [nn.Parameter(torch.randn(CONTEXT_SIZE, E) * 0.1) for _ in range(N_LAYERS)]
        )
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        concat_dim = E * N_LAYERS
        self.unembedder = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, concat_dim * 4),
            nn.ReLU(),
            nn.Linear(concat_dim * 4, VOCAB_SIZE),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        outs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            outs.append(x)
        concat = torch.cat(outs, dim=-1)
        return self.unembedder(concat)


def get_lr_scale(step):
    n_steps = cfg['n_steps']
    warmup = int(cfg.get('lr_warmup_fraction', 0.1) * n_steps)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def run_bench(force_fallback: bool, warmup: int, steps: int, label: str):
    torch.manual_seed(cfg['random_seed'])
    original_fn = bplo._project_grad_out_to_weight_grad
    if force_fallback:
        def forced(grad_out, lookup_indices, output_idx_per_table,
                   n_heads, tph, output_nap, table_dim, scale, wg_buffer=None):
            B, N = lookup_indices.shape
            if wg_buffer is None:
                wg = torch.zeros(N, table_dim, output_nap,
                                 device=grad_out.device, dtype=torch.float32)
            else:
                wg = wg_buffer
                wg.zero_()
            pair_flat = output_idx_per_table.reshape(n_heads, tph * output_nap).long()
            g_slot = grad_out.gather(
                2, pair_flat.unsqueeze(0).expand(B, -1, -1)
            ) * scale
            g_slot = g_slot.reshape(B, N, output_nap).to(torch.float32)
            entries = lookup_indices.long()
            N_idx = torch.arange(N, device=grad_out.device).unsqueeze(0).expand(B, -1)
            flat_idx = (N_idx * table_dim + entries).reshape(-1)
            wg.view(N * table_dim, output_nap).index_add_(
                0, flat_idx, g_slot.reshape(-1, output_nap)
            )
            return wg
        bplo._project_grad_out_to_weight_grad = forced
    try:
        sampler = TextSnippetSampler(DATA_PATH, CONTEXT_SIZE, 10_000, DEVICE, random_seed=1)
        model = Model().to(DEVICE)
        bit_luts = []
        for layer in model.layers:
            bit_luts += [layer.q_perm, layer.k_perm, layer.v_perm, layer.out_proj]
        adam_opt = torch.optim.Adam(list(model.parameters()), lr=cfg['adam_lr'])
        adam_scheduler = torch.optim.lr_scheduler.LambdaLR(adam_opt, get_lr_scale)
        bit_opt = BitPermutationLUTOptimizer(
            bit_luts, lr=cfg['bit_lut_lr'],
            beta1=cfg.get('bit_lut_beta1', 0.9),
            beta2=cfg.get('bit_lut_beta2', 0.999),
            lr_schedule_fn=get_lr_scale,
        )
        model.train()
        for _ in range(warmup):
            x = sampler.sample_training_batch(cfg['batch_size']).long()
            inp = torch.empty_like(x); inp[:, 0] = BOS_ID; inp[:, 1:] = x[:, :-1]
            logits = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))
            adam_opt.zero_grad(); bit_opt.zero_grad()
            loss.backward()
            adam_opt.step(); adam_scheduler.step(); bit_opt.step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            x = sampler.sample_training_batch(cfg['batch_size']).long()
            inp = torch.empty_like(x); inp[:, 0] = BOS_ID; inp[:, 1:] = x[:, :-1]
            logits = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))
            adam_opt.zero_grad(); bit_opt.zero_grad()
            loss.backward()
            adam_opt.step(); adam_scheduler.step(); bit_opt.step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f'{label:25s} steps={steps} warmup={warmup} total={elapsed:.3f}s | '
              f'per_step={1000*elapsed/steps:.3f} ms')
        bit_opt.close()
    finally:
        bplo._project_grad_out_to_weight_grad = original_fn


if __name__ == '__main__':
    WARMUP = 20
    STEPS = 200
    run_bench(force_fallback=False, warmup=WARMUP, steps=STEPS, label='[default dispatch]')
    run_bench(force_fallback=True,  warmup=WARMUP, steps=STEPS, label='[force fallback]')
    run_bench(force_fallback=False, warmup=WARMUP, steps=STEPS, label='[default dispatch#2]')
    run_bench(force_fallback=True,  warmup=WARMUP, steps=STEPS, label='[force fallback#2]')
