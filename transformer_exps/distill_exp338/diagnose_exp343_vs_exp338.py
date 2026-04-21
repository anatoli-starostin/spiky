"""Compare exp338 (BitPermLUT out_proj) and exp343 (PermLut out_proj) side-by-side.

Forwards the same batches through both trained checkpoints, captures:
  - out_proj input distribution (attn concat)
  - out_proj RAW output (pre-LN): dominance pairs (exp338) vs E-dim vec (exp343)
  - post-LN E-dim output (what feeds the next layer)
  - per-layer output stats
"""
import os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spiky.util.text_snippet_sampler import TextSnippetSampler
from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.permutational_lut import PermutationalLut
from spiky.lutorch.ranking_tools import DominanceCanonicalize, DominanceToVector

DEVICE = 'cuda:0'
DATA_PATH = '/home/starost/spiky/workbooks/fineweb_texts.txt'


def make_model(exp_dir, is_perm):
    """Returns (Model, cfg). is_perm=True → exp343 arch."""
    with open(os.path.join(exp_dir, 'config.json')) as f:
        cfg = json.load(f)
    E = cfg['embedding_dim']; H = cfg['n_heads']; d_qk = cfg['d_qk']; d_v = cfg['d_v']
    N_L = cfg['num_layers']; D_QK_P = d_qk*(d_qk-1)//2; D_V_P = d_v*(d_v-1)//2
    PS = [list(range(h*d_v, (h+1)*d_v)) for h in range(H)] if cfg.get('out_partition_by_head', False) else None

    def mk_qk(so): return BitPermutationLUT(n_inputs=E, n_outputs=d_qk, n_heads=H, input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'], tph=cfg['qk_tph'], random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'], latent_dtype=cfg.get('bit_lut_latent_dtype','fp8'), device=DEVICE)
    def mk_v(so): return BitPermutationLUT(n_inputs=E, n_outputs=d_v, n_heads=H, input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'], tph=cfg['v_tph'], random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'], latent_dtype=cfg.get('bit_lut_latent_dtype','fp8'), device=DEVICE)
    def mk_out(so):
        if is_perm:
            return PermutationalLut(n_inputs=H*d_v, n_outputs=E, n_heads=1, input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'], tph=cfg['out_tph'], pair_mode='scrambled', soft_mode='rational', aggregation='matmul', random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'], device=DEVICE, partition_sets=PS)
        return BitPermutationLUT(n_inputs=H*d_v, n_outputs=E, n_heads=1, input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'], tph=cfg['out_tph'], random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'], latent_dtype=cfg.get('bit_lut_latent_dtype','fp8'), device=DEVICE, partition_sets=PS)

    class LUTBlock(nn.Module):
        def __init__(self, i):
            super().__init__()
            self.q_perm=mk_qk(i); self.k_perm=mk_qk(100+i); self.v_perm=mk_v(200+i); self.out_proj=mk_out(400+i)
            ct=cfg.get('canon_temperature',0.1)
            self.q_canon=DominanceCanonicalize(d_qk,temperature=ct); self.k_canon=DominanceCanonicalize(d_qk,temperature=ct)
            self.attn_to_vec=DominanceToVector(d_v, normalise=False)
            if is_perm:
                self.out_ln = nn.LayerNorm(E)
            else:
                self.out_to_vec = DominanceToVector(E)
            self.attn_scale=nn.Parameter(torch.tensor(float(cfg.get('learnable_attn_scale_init',0.25))))

        def forward(self, x, pe, snapshot=None, layer_idx=None):
            B,T,_E = x.shape
            xp=(x+pe.unsqueeze(0)).reshape(B*T,_E); xf=x.reshape(B*T,_E)
            q = self.q_canon(self.q_perm(xp)).reshape(B,T,H,D_QK_P).permute(0,2,1,3)
            k = self.k_canon(self.k_perm(xp)).reshape(B,T,H,D_QK_P).permute(0,2,1,3)
            v = self.v_perm(xf).reshape(B,T,H,D_V_P).permute(0,2,1,3)
            a = F.scaled_dot_product_attention(q*self.attn_scale, k, v, is_causal=True)
            attn=self.attn_to_vec(a)
            oi = attn.permute(0,2,1,3).reshape(B*T, H*d_v)
            if is_perm:
                o_raw = self.out_proj(oi).squeeze(1)   # [B*T, E]
                o_post = self.out_ln(o_raw.reshape(B,T,_E)).reshape(B*T,_E)
            else:
                o_dom = self.out_proj(oi)              # [B*T, 1, P_out]
                o_raw = torch.einsum('bhp,kp->bhk', o_dom, self.out_to_vec.borda_m).squeeze(1)  # [B*T, E] — pre-LN Borda
                o_post = self.out_to_vec(o_dom).squeeze(1)   # [B*T, E] — post-LN
            if snapshot is not None:
                snapshot.setdefault(layer_idx, {})
                snapshot[layer_idx]['in'] = oi.detach().float()
                snapshot[layer_idx]['raw'] = o_raw.detach().float()
                snapshot[layer_idx]['post'] = o_post.detach().float()
            return o_post.reshape(B,T,_E)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedder=nn.Embedding(cfg['vocab_size'], E)
            self.pos_embs=nn.ParameterList([nn.Parameter(torch.zeros(cfg['context_size'], E)) for _ in range(N_L)])
            self.layers=nn.ModuleList([LUTBlock(i) for i in range(N_L)])
            cd=E*N_L
            self.unembedder=nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd,cd*4), nn.ReLU(), nn.Linear(cd*4, cfg['vocab_size']))
        def forward(self, t, snapshot=None):
            x = self.token_embedder(t)
            for i, (layer, pe) in enumerate(zip(self.layers, self.pos_embs)):
                x = layer(x, pe, snapshot=snapshot, layer_idx=i)
            return x
    return Model().to(DEVICE), cfg


def load(exp_dir, is_perm):
    m, cfg = make_model(exp_dir, is_perm)
    sd = torch.load(os.path.join(exp_dir, 'checkpoint.pt'), map_location=DEVICE, weights_only=False)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m, cfg


m338, _ = load('/home/starost/spiky/transformer_exps/exp338_cfc_out_inap10_tph2048_onap10', is_perm=False)
m343, _ = load('/home/starost/spiky/transformer_exps/exp343_cfc_out_permlut_in10_tph256_on32', is_perm=True)

CFG = json.load(open('/home/starost/spiky/transformer_exps/exp338_cfc_out_inap10_tph2048_onap10/config.json'))
sampler = TextSnippetSampler(DATA_PATH, CFG['context_size'], 10_000, DEVICE, random_seed=99)

with torch.no_grad():
    x = sampler.sample_training_batch(CFG['batch_size']).long()
    inp = torch.empty_like(x); inp[:, 0] = 256; inp[:, 1:] = x[:, :-1]
    s338, s343 = {}, {}
    _ = m338(inp, snapshot=s338)
    _ = m343(inp, snapshot=s343)

print("  l  stage      exp338                                         exp343")
print("-"*120)
for l in range(CFG['num_layers']):
    for stage in ['in', 'raw', 'post']:
        t38 = s338[l][stage]; t43 = s343[l][stage]
        def fmt(t):
            return f'mean={t.mean().item():+.3f} std={t.std().item():.3f} max={t.abs().max().item():.3f}'
        print(f'{l:>2}  {stage:<6}  {fmt(t38):<52}  {fmt(t43):<52}')
    print()
