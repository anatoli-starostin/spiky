from spiky.lutorch.hyper_lut import HyperLUT, HyperLUTBackbone

hyper_cfg = {
    'embedding_dim': 32, 'positional_dim': 16, 'num_layers': 6,
    'n_heads': 4, 'd_qk': 8, 'd_v': 8,
    'n_pairs': 160, 'hidden_dim': 320,
    'temperature': 0.1, 'soft_mode': 'rational'
}

E = hyper_cfg['embedding_dim']
H = hyper_cfg['n_heads']
d_qk = hyper_cfg['d_qk']
d_v = hyper_cfg['d_v']
N_PAIRS = hyper_cfg['n_pairs']
HID = hyper_cfg['hidden_dim']
TEMP = hyper_cfg.get('temperature', 0.1)
SOFT = hyper_cfg.get('soft_mode', 'rational')
SEED = 42

def _make_hyper(n_heads, n_outputs, seed_offset):
    h = HyperLUT(
        input_dim=E, n_heads=n_heads, n_outputs=n_outputs,
        n_pairs=N_PAIRS, hidden_dim=HID,
        temperature=TEMP, soft_mode=SOFT,
        random_seed=SEED+seed_offset, device=device
    )
    with torch.no_grad():
        nn.init.normal_(h.backbone.fc1_weight, std=0.001)
        nn.init.normal_(h.fc2.weight, std=0.001)
    return h

def _make_backbone(n_heads, seed_offset):
    bb = HyperLUTBackbone(
        input_dim=E, n_heads=n_heads, n_pairs=N_PAIRS, hidden_dim=HID,
        temperature=TEMP, soft_mode=SOFT,
        random_seed=SEED+seed_offset, device=device,
    )
    with torch.no_grad():
        nn.init.normal_(bb.fc1_weight, std=0.001)
    return bb

def _make_fc2(hidden_dim, n_outputs, std=0.001):
    fc = nn.Linear(hidden_dim, n_outputs, device=device)
    with torch.no_grad():
        nn.init.normal_(fc.weight, std=std)
        nn.init.zeros_(fc.bias)
    return fc


class HyperBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q = _make_hyper(H, d_qk, layer_idx)
        self.q_norm = nn.BatchNorm1d(d_qk)
        self.k = _make_hyper(H, d_qk, 100+layer_idx)
        self.k_norm = nn.BatchNorm1d(d_qk)
        self.v = _make_hyper(H, d_v, 200+layer_idx)
        self.v_norm = nn.BatchNorm1d(d_v)
        self.out_proj = _make_hyper(1, E, 400+layer_idx)
        self.out_norm = nn.BatchNorm1d(E)
        self.ffn = _make_hyper(1, E, 600+layer_idx)
        self.ffn_norm = nn.BatchNorm1d(E)

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B*T, _E)
        x_flat = x.reshape(B*T, _E)

        q = self.q_norm(self.q(xp).reshape(B, T*H, d_qk).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_norm(self.k(xp).reshape(B, T*H, d_qk).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        v = self.v_norm(self.v(x_flat).reshape(B, T*H, d_v).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_raw = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)).squeeze(1).reshape(B, T, _E)
        proj = self.out_norm(out_raw.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + proj

        ffn_raw = self.ffn(x.reshape(B*T, _E)).squeeze(1).reshape(B, T, _E)
        ffn_out = self.ffn_norm(ffn_raw.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + ffn_out
        return x


class UniformHyperLUTTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        n_layers = hyper_cfg['num_layers']
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.randn(CONTEXT_SIZE, E) * 0.1) for _ in range(n_layers)])
        self.layers = nn.ModuleList([HyperBlock(i) for i in range(n_layers)])
        self.unembedder = nn.Sequential(
            nn.Linear(E, 128), nn.GELU(), nn.Linear(128, VOCAB_SIZE, bias=False),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
        return self.unembedder(x)