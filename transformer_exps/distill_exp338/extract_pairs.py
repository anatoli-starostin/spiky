"""Extract teacher out_proj anchor_pairs and idx_a/idx_b per layer from exp338
for load_pairs() in the ceiling-check candidate."""
import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.ranking_tools import DominanceCanonicalize, DominanceToVector

SRC_EXP = '/home/starost/spiky/transformer_exps/exp338_cfc_out_inap10_tph2048_onap10'
OUT = '/home/starost/spiky/transformer_exps/distill_exp338/data/teacher_pairs.pt'

with open(os.path.join(SRC_EXP, 'config.json')) as f: cfg = json.load(f)

DEVICE = 'cuda:0'
E, H, d_qk, d_v = cfg['embedding_dim'], cfg['n_heads'], cfg['d_qk'], cfg['d_v']
N_LAYERS = cfg['num_layers']
D_QK_P, D_V_P = d_qk*(d_qk-1)//2, d_v*(d_v-1)//2
PS = [list(range(h*d_v, (h+1)*d_v)) for h in range(H)] if cfg.get('out_partition_by_head', False) else None

def _make_qk(so): return BitPermutationLUT(n_inputs=E, n_outputs=d_qk, n_heads=H, input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'], tph=cfg['qk_tph'], random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'], latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'), device=DEVICE)
def _make_v(so): return BitPermutationLUT(n_inputs=E, n_outputs=d_v, n_heads=H, input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'], tph=cfg['v_tph'], random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'], latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'), device=DEVICE)
def _make_out(so): return BitPermutationLUT(n_inputs=H*d_v, n_outputs=E, n_heads=1, input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'], tph=cfg['out_tph'], random_seed=cfg['random_seed']+so, initial_weights_noise=cfg['bit_lut_latent_init_std'], latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'), device=DEVICE, partition_sets=PS)

class LUTBlock(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.q_perm = _make_qk(i); self.k_perm = _make_qk(100+i)
        self.v_perm = _make_v(200+i); self.out_proj = _make_out(400+i)
        ct = cfg.get('canon_temperature', 0.1)
        self.q_canon = DominanceCanonicalize(d_qk, temperature=ct); self.k_canon = DominanceCanonicalize(d_qk, temperature=ct)
        self.attn_to_vec = DominanceToVector(d_v, normalise=False); self.out_to_vec = DominanceToVector(E)
        self.attn_scale = nn.Parameter(torch.tensor(float(cfg.get('learnable_attn_scale_init', 0.25))))
    def forward(self, x, pe): return x  # not needed


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        VOCAB = cfg['vocab_size']; CTX = cfg['context_size']
        self.token_embedder = nn.Embedding(VOCAB, E)
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.zeros(CTX, E)) for _ in range(N_LAYERS)])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        cd = E*N_LAYERS
        self.unembedder = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd, cd*4), nn.ReLU(), nn.Linear(cd*4, VOCAB))

model = Model().to(DEVICE)
sd = torch.load(os.path.join(SRC_EXP, 'checkpoint.pt'), map_location=DEVICE, weights_only=False)
model.load_state_dict(sd)

pairs_per_layer = {}
for i, layer in enumerate(model.layers):
    op = layer.out_proj
    pairs_per_layer[i] = {
        'anchor_pairs_a': op.anchor.anchor_pairs_a.clone().cpu(),
        'anchor_pairs_b': op.anchor.anchor_pairs_b.clone().cpu(),
        'idx_a': op.idx_a.clone().cpu(),
        'idx_b': op.idx_b.clone().cpu(),
    }
    print(f'layer {i}: anchor_a {tuple(op.anchor.anchor_pairs_a.shape)} idx_a {tuple(op.idx_a.shape)}')

torch.save(pairs_per_layer, OUT)
print(f'saved {OUT}')
