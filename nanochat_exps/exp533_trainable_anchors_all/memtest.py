"""Memory probe for exp533: trainable-anchor LUTs at exp530 shapes, bs=16, 6L.
Compares per-block gradient checkpointing ON vs OFF for one fwd+bwd.
Random data + dummy vocab; only measures activation memory of the LUT stack."""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.trainable_anchors_multi_head_lut import TrainableAnchorsMultiHeadLUT

DEV = 'cuda'
E, D, H, d_qk, d_v, L = 64, 384, 6, 64, 16, 6
V, BS, T = 65536, 16, 512
# exp530 shapes
QK_NAP, QK_TPH = 4, 256
V_NAP, V_TPH = 6, 320
OUT_NAP, OUT_TPH = 6, 1024
RES_NAP, RES_TPH = 6, 256


def mk(input_dim, n_heads, n_outputs, nap, tph, seed):
    return TrainableAnchorsMultiHeadLUT(
        input_dim, n_heads, n_outputs, nap, tables_per_head=tph,
        anchor_temp=1.0, sign_temp=0.5, select_temp=0.5,
        learnable_temps=True, anchor_init_std=1.0, weights_init_std=0.001,
        random_seed=seed, device=DEV,
    )


class Block(nn.Module):
    def __init__(self, i, use_ckpt):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.qk_lut = mk(E, H, 2 * d_qk, QK_NAP, QK_TPH, i)
        self.v_lut = mk(E, H, d_v, V_NAP, V_TPH, 200 + i)
        self.out_proj = mk(H * d_v, 1, E, OUT_NAP, OUT_TPH, 400 + i)
        self.residual_lut = mk(E, 1, D, RES_NAP, RES_TPH, 600 + i)
        self.q_norm = nn.LayerNorm(d_qk).to(DEV)
        self.k_norm = nn.LayerNorm(d_qk).to(DEV)

    def _body(self, x):
        B, Tt, _ = x.shape
        xf = x.reshape(B * Tt, E)
        qk = self.qk_lut(xf)
        q = self.q_norm(qk[..., :d_qk]).reshape(B, Tt, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_norm(qk[..., d_qk:2 * d_qk]).reshape(B, Tt, H, d_qk).permute(0, 2, 1, 3)
        v = self.v_lut(xf).reshape(B, Tt, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        oin = attn.permute(0, 2, 1, 3).reshape(B * Tt, H * d_v)
        oe = self.out_proj(oin).squeeze(1)
        xn = xf + oe
        r = self.residual_lut(xn).squeeze(1).reshape(B, Tt, D)
        return xn.reshape(B, Tt, E), r

    def forward(self, x):
        if self.use_ckpt:
            return checkpoint(self._body, x, use_reentrant=False)
        return self._body(x)


class Net(nn.Module):
    def __init__(self, use_ckpt):
        super().__init__()
        self.emb = nn.Embedding(V, E).to(DEV)
        self.layers = nn.ModuleList([Block(i, use_ckpt) for i in range(L)])
        self.head = nn.Linear(D, V, bias=False).to(DEV)
        self.lnf = nn.LayerNorm(D).to(DEV)

    def forward(self, tok, tgt):
        B, Tt = tok.shape
        xr = torch.zeros(B, Tt, D, device=DEV)
        xl = self.emb(tok)
        for ly in self.layers:
            xl, r = ly(xl)
            xr = xr + r
        logits = self.head(self.lnf(xr))
        return F.cross_entropy(logits.view(-1, V), tgt.view(-1))


def run(use_ckpt):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    net = Net(use_ckpt)
    tok = torch.randint(0, V, (BS, T), device=DEV)
    tgt = torch.randint(0, V, (BS, T), device=DEV)
    loss = net(tok, tgt)
    loss.backward()
    torch.cuda.synchronize()
    tag = 'CKPT ON ' if use_ckpt else 'CKPT OFF'
    print(f'{tag}: loss={loss.item():.3f}  '
          f'alloc_peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB  '
          f'reserved={torch.cuda.max_memory_reserved()/1e9:.1f}GB')
    del net, loss
    torch.cuda.empty_cache()


if __name__ == '__main__':
    n_params_probe = sum(
        p.numel() for p in Net(True).parameters() if p.requires_grad
    )
    print(f'params (incl. dummy emb+head): {n_params_probe:,}')
    run(True)
    run(False)
