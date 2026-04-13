"""
collect.py — Train 3 models with different init distributions, save weight snapshots.

Snapshots saved to: snapshots.pt
  {init_name: {step: [[n_out], ...] list of 16 entry vectors}}
"""
import sys, os, math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import make_sampler, CONTEXT_SIZE, BOS_ID
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import RankAttention

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda:0'

CFG = {
    'vocab_size': 257, 'embedding_dim': 32, 'positional_dim': 16,
    'num_layers': 6, 'num_heads': 4, 'd_qk': 16, 'd_v': 16,
    'n_anchor_pairs': 4, 'qkv_tables_per_head': 264, 'out_proj_tables_per_head': 480,
    'n_alternatives': 1, 'smooth_mode': False, 'normalise_weights': False,
    'calibrate_output': False, 'initial_weights_noise': 0.001,
    'anchor_sampling_policy': 'full_coverage', 'random_seed': 42,
}

def make_lut(input_dim, n_heads, n_outputs, tph, seed_offset=0):
    return MultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=CFG['n_anchor_pairs'], tables_per_head=tph,
        smooth_mode=CFG['smooth_mode'], n_alternatives=CFG['n_alternatives'],
        normalize_weights=CFG['normalise_weights'], calibrate_output=CFG['calibrate_output'],
        anchor_sampling_policy=AnchorSamplingPolicy(CFG['anchor_sampling_policy']),
        initial_weights_noise=CFG['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=CFG['random_seed'] + seed_offset, device=DEVICE,
    )

class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        d, p, h = CFG['embedding_dim'], CFG['positional_dim'], CFG['num_heads']
        d_qk, d_v = CFG['d_qk'], CFG['d_v']
        s = layer_idx * 10
        self.q_lut    = make_lut(d + p, h, d_qk, CFG['qkv_tables_per_head'], s)
        self.k_lut    = make_lut(d + p, h, d_qk, CFG['qkv_tables_per_head'], s + 1)
        self.v_lut    = make_lut(d + p, h, d_v,  CFG['qkv_tables_per_head'], s + 2)
        self.out_proj = make_lut(h * d_v, 1, d,  CFG['out_proj_tables_per_head'], s + 3)
        self.rank_attn = RankAttention(d_qk, d_v, smooth_mode=False, temperature=1.0)
        self.n_heads, self.d_qk, self.d_v, self.d = h, d_qk, d_v, d

    def forward(self, x, pos):
        B, T, E = x.shape
        xp = torch.cat([x, pos], dim=-1).reshape(-1, E + pos.shape[-1])
        q = self.q_lut(xp).permute(1,0,2).reshape(self.n_heads,B,T,self.d_qk).permute(1,0,2,3)
        k = self.k_lut(xp).permute(1,0,2).reshape(self.n_heads,B,T,self.d_qk).permute(1,0,2,3)
        v = self.v_lut(xp).permute(1,0,2).reshape(self.n_heads,B,T,self.d_v).permute(1,0,2,3)
        attn = self.rank_attn(q, k, v, is_causal=True)
        attn = attn.permute(0,2,1,3).reshape(B*T, self.n_heads*self.d_v)
        return x + self.out_proj(attn)[:,0,:].reshape(B, T, E)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d, p = CFG['embedding_dim'], CFG['positional_dim']
        self.token_embedder = nn.Embedding(CFG['vocab_size'], d)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.register_buffer('pos_emb', torch.randn(1, CONTEXT_SIZE, p) * 0.1)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(CFG['num_layers'])])
        self.unembedder = nn.Linear(d, CFG['vocab_size'], bias=False)

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.token_embedder(tokens)
        pos = self.pos_emb[:, :T].expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, pos)
        return self.unembedder(x)

def apply_init(model, init_name):
    for module in model.modules():
        if isinstance(module, MultiHeadLut):
            w = module.projection.weights
            with torch.no_grad():
                if init_name == 'gaussian':
                    w.normal_(0, 0.001)
                elif init_name == 'uniform':
                    w.uniform_(-0.001, 0.001)
                elif init_name == 'bimodal':
                    mask = torch.rand_like(w) > 0.5
                    w.normal_(0, 0.001)
                    w.add_(torch.where(mask, torch.full_like(w, 0.005), torch.full_like(w, -0.005)))

SNAPSHOT_TABLE = 42
SNAPSHOT_STEPS = [0, 1000, 5000]

def snapshot(model):
    """Return all 16 entry vectors (unsorted) for fixed table in layers.2.q_lut."""
    w = model.layers[2].q_lut.projection.weights  # [n_tables, 16, n_out]
    table = w[SNAPSHOT_TABLE].detach().cpu().float()  # [16, n_out]
    return table.numpy()   # [16, n_out]

def train(init_name, sampler):
    torch.manual_seed(CFG['random_seed'])
    model = Model().to(DEVICE)
    apply_init(model, init_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    snaps = {}
    for step in range(5001):
        if step in SNAPSHOT_STEPS:
            snaps[step] = snapshot(model)

        x = sampler.sample_training_batch(32).long()
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        tgt = x

        logits = model(inp)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B*T, V), tgt.reshape(B*T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f'  [{init_name}] step {step:5d} | loss={loss.item():.4f}')

    return snaps

sampler = make_sampler(DEVICE, random_seed=1)

all_snaps = {}
for init_name in ['gaussian', 'uniform', 'bimodal']:
    print(f'\n=== {init_name} init ===')
    all_snaps[init_name] = train(init_name, sampler)

out_path = os.path.join(EXP_DIR, 'snapshots.pt')
torch.save(all_snaps, out_path)
print(f'\nSnapshots saved to {out_path}')
