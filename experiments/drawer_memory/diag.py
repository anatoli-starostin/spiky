"""Address diagnostics for trained drawer-memory checkpoints (MQAR, tied addressing).

For each drawer layer, on 256 fresh evaluation sequences:
  match_value   share of tables where a query lands in the same cell as the token that stored its value
                (the value token, which carries its key through the token shift) -- the store/read agreement
  any_value     share of queries matching that writer in at least one table
  match_key     same, against the earlier occurrence of the key token itself
  collide       share of tables where the query shares its cell with an unrelated pair's value token
                (per unrelated pair; chance for uniform addresses = 1 / cells per table)
  occupancy     share of a table's cells written at least once in a sequence
  entropy       entropy of the cell distribution over all tokens, as a share of the maximum (bits per table)
  bit_bias      mean |P(bit = 1) - 0.5| over bits (0 = balanced)
  bit_corr      mean |correlation| between bits of the same table
  read_share    share of the read weight that goes to the correct value token
Usage: diag.py <run dir> [<run dir> ...]   (uses model.pt and every model_<step>.pt)
Writes <run dir>/diag.csv.
"""
import csv
import glob
import os
import re
import sys

import torch

from models import LutMemMixer, Model
from train import NF, NK, NV, mqar_batch


@torch.no_grad()
def diagnose(model, cfg, B=256):
    n, gap = cfg["n_pairs"], cfg.get("gap", 0)
    x, _ = mqar_batch(B, n, "cpu", torch.Generator().manual_seed(4321), gap)
    T = x.shape[1]
    stride = 2 + gap
    key_pos = torch.arange(n) * stride
    val_pos = key_pos + 1 + gap
    q_pos = n * stride + 2 * torch.arange(n)
    ctx_keys = x[:, key_pos]                                      # [B,n]
    q_keys = x[:, q_pos]                                          # [B,n]
    p_of_q = (q_keys[:, :, None] == ctx_keys[:, None, :]).float().argmax(-1)   # pair index of each query
    inputs = {}
    hooks = [m.register_forward_hook(lambda mod, inp, out, i=i: inputs.__setitem__(i, inp[0]))
             for i, m in enumerate(model.modules()) if isinstance(m, LutMemMixer)]
    model(x)
    for h in hooks:
        h.remove()
    mixers = [m for m in model.modules() if isinstance(m, LutMemMixer)]
    rows = []
    for li, (idx, mix) in enumerate(zip(sorted(inputs), mixers)):
        c, u = mix.addresses(inputs[idx])                         # c [B,H,T,tph]
        H, tph, K = c.shape[1], c.shape[3], mix.K
        cq = c[:, :, q_pos]                                       # [B,H,n,tph]
        cv = c[:, :, val_pos]                                     # [B,H,n,tph] value-token writers
        ck = c[:, :, key_pos]
        gi = p_of_q[:, None, :, None].expand(B, H, n, tph)
        right_v, right_k = cv.gather(2, gi), ck.gather(2, gi)
        match_v = (cq == right_v).float()
        same = (cq[:, :, :, None, :] == cv[:, :, None, :, :]).float()          # [B,H,nq,np,tph]
        other = 1 - torch.nn.functional.one_hot(p_of_q, n).float()[:, None, :, :, None]
        collide = (same * other).sum(3) / (n - 1)
        onehot = torch.nn.functional.one_hot(c, K).float()                     # [B,H,T,tph,K]
        occ = (onehot.sum(2) > 0).float().mean()
        pc = onehot.mean((0, 2))                                               # [H,tph,K]
        ent = (-(pc * (pc + 1e-12).log2()).sum(-1) / mix.nap).mean()
        bits = (u > 0).float()                                                 # [B,H,T,tph,nap]
        bias = (bits.mean((0, 2)) - 0.5).abs().mean()
        bb = bits.permute(1, 3, 4, 0, 2).reshape(H, tph, mix.nap, -1)
        bb = bb - bb.mean(-1, keepdim=True)
        cov = bb @ bb.transpose(-1, -2)
        sd = cov.diagonal(dim1=-2, dim2=-1).clamp(min=1e-9).sqrt()
        corr = (cov / sd[..., :, None] / sd[..., None, :]).abs()
        off = 1 - torch.eye(mix.nap)
        bcorr = (corr * off).sum((-1, -2)) / off.sum()
        # read share of the correct value writer (tied, additive memory)
        w = (cq[:, :, :, None, :] == c[:, :, None, :, :]).float().sum(-1)     # shared tables per (q, token)
        w = w * (torch.arange(T)[None, None, None, :] <= q_pos[None, None, :, None]).float()
        right_t = val_pos[p_of_q][:, None, :].expand(B, H, n)
        share = w.gather(3, right_t[..., None]).squeeze(-1) / w.sum(-1).clamp(min=1)
        rows.append(dict(layer=li, cells_per_table=K, tables=tph, match_value=match_v.mean().item(),
                         any_value=(match_v.sum(-1) > 0).float().mean().item(),
                         match_key=(cq == right_k).float().mean().item(), collide=collide.mean().item(),
                         chance_collide=1 / K, occupancy=occ.item(), entropy=ent.item(), bit_bias=bias.item(),
                         bit_corr=bcorr.mean().item(), read_share=share.mean().item()))
    return rows


def load(path):
    ck = torch.load(path, map_location="cpu")
    a = ck["config"]
    vocab = NK + NV + (NF if a.get("gap") else 0)
    m = Model(vocab, d=a["d"], mixer=a["mixer"], feat=a["feat"], nap=a["nap"], shift_k=a.get("shift_k", 1),
              anchor_seed=a["seed"] + 1 if a.get("anchor_per_seed") else 0, write=a.get("write", "add"),
              read_power=a.get("read_power", 1), window=a.get("window", 16))
    m.load_state_dict(ck["model"])
    return m.eval(), a


def main():
    for run in sys.argv[1:]:
        cks = sorted(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[0]))
        cks += [os.path.join(run, "model.pt")]
        out = []
        for p in cks:
            if not os.path.exists(p):
                continue
            model, cfg = load(p)
            step = int(re.findall(r"model_(\d+)", p)[0]) if "model_" in p else cfg["steps"]
            for r in diagnose(model, cfg):
                out.append(dict(step=step, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))
                print(run, out[-1], flush=True)
        if out:
            with open(os.path.join(run, "diag.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(out[0]))
                w.writeheader()
                w.writerows(out)


if __name__ == "__main__":
    main()
