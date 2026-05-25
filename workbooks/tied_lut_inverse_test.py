"""Trivial convergence test for the TIED LUT unembedder idea (no LM).

Question: can a trainable embedder (V vectors in R^E) and a "row=token" LUT
unembedder learn to invert each other, trained ONLY on the consistency loss
  CE( unembedder(emb.weight) , arange(V) )  ?

Unembedder (as proposed): `tph` hash tables, each reads `nap` anchor-pair signs
from the embedding -> a `nap`-bit index r in [0, 2^nap). With 2^nap == V, r IS a
token id (row=token): the table adds a learned scalar weight[t, r] to logit[r].
=> for any embedding only ~tph of the V logits are nonzero (sparse LSH vote).

Backward: hard sign-pack forward; soft-STE that lets the embedding gradient flip
ONE bit at a time (Hamming-1 neighbour relaxation). This is the cheap principled
STE; it is local in Hamming space (its main suspected weakness).

Run:  python tied_lut_inverse_test.py
"""
import math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class RowTokenLUTUnembedder(nn.Module):
    def __init__(self, E, V, nap, tph, t_soft=0.5, seed=0, device="cuda"):
        super().__init__()
        assert (1 << nap) == V, "need 2^nap == V so row index == token id"
        assert 1 <= nap <= 15
        self.E, self.V, self.nap, self.tph, self.T = E, V, nap, tph, t_soft
        g = torch.Generator().manual_seed(seed)
        a = torch.randint(0, E, (tph, nap), generator=g)
        b = torch.randint(0, E, (tph, nap), generator=g)
        b = torch.where(b == a, (b + 1) % E, b)            # avoid degenerate pair
        self.register_buffer("anchor_a", a.to(device))
        self.register_buffer("anchor_b", b.to(device))
        self.register_buffer("powers", (1 << torch.arange(nap)).long().to(device))
        # learnable per-(table,row) vote weight; init 1.0 per the proposal
        self.weight = nn.Parameter(torch.ones(tph, V, device=device))

    def forward(self, x):
        """x: [N, E] -> logits: [N, V]  (hard forward, soft-STE backward)."""
        N = x.shape[0]
        d = x[:, self.anchor_a] - x[:, self.anchor_b]      # [N, tph, nap]
        bits = (d > 0).long()
        r = (bits * self.powers).sum(-1)                   # [N, tph]  hard token id

        # --- HARD contribution (carries forward value + weight gradient) ----
        logit_hard = x.new_zeros(N, self.V)
        for t in range(self.tph):
            w_sel = self.weight[t][r[:, t]]                # [N]
            logit_hard.scatter_add_(1, r[:, t:t + 1], w_sel.unsqueeze(1))

        # --- SOFT contribution (carries x gradient only; weights detached) --
        # Per bit j: confidence s_j = sigmoid(|d_j|/T). Soft distribution over
        # {stay at r} u {flip bit j -> r ^ 2^j}: odds(flip j) = (1-s_j)/s_j.
        s = torch.sigmoid(d.abs() / self.T)                # [N, tph, nap]
        flip_odds = (1.0 - s) / s                          # [N, tph, nap]
        Z = 1.0 + flip_odds.sum(-1, keepdim=True)          # [N, tph, 1]
        p_stay = 1.0 / Z                                   # [N, tph, 1]
        p_flip = flip_odds / Z                             # [N, tph, nap]
        w_det = self.weight.detach()
        logit_soft = x.new_zeros(N, self.V)
        for t in range(self.tph):
            rt = r[:, t]                                   # [N]
            # stay
            logit_soft.scatter_add_(1, rt[:, None],
                                    (w_det[t][rt] * p_stay[:, t, 0])[:, None])
            # single-bit flips
            for j in range(self.nap):
                rj = rt ^ (1 << j)                         # [N] neighbour token
                logit_soft.scatter_add_(1, rj[:, None],
                                        (w_det[t][rj] * p_flip[:, t, j])[:, None])

        logits = logit_hard + (logit_soft - logit_soft.detach())
        return logits, r, d

    def assignment_bit_loss(self, d, ids):
        """Winner-take-all reachability driver. For each token, find the table
        whose hard hash is CLOSEST (min Hamming) to the target id, then push all
        of that table's bits toward the target's bit pattern at once (logistic
        margin loss). Routing (argmin) is non-diff but used only to select which
        table to supervise; gradient flows through d -> embedding."""
        N = d.shape[0]
        bits_pred = (d > 0).long()                          # [N, tph, nap]
        # target bit pattern of each token id
        bits_tgt = ((ids[:, None] >> torch.arange(self.nap, device=d.device)) & 1)  # [N, nap]
        hamming = (bits_pred != bits_tgt[:, None, :]).sum(-1)   # [N, tph]
        t_star = hamming.argmin(1)                          # [N] closest table
        d_star = d[torch.arange(N, device=d.device), t_star]   # [N, nap]
        tgt_sign = (2 * bits_tgt - 1).float()              # [N, nap] in {-1,+1}
        # logistic margin: push sign(d_star_j) toward tgt_sign_j with width T
        return F.softplus(-tgt_sign * d_star / self.T).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=32768)
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--nap", type=int, default=15)
    ap.add_argument("--tph", type=int, default=16)
    ap.add_argument("--t_soft", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--emb_lr", type=float, default=1e-2)
    ap.add_argument("--w_lr", type=float, default=1e-2)
    ap.add_argument("--emb_init", type=float, default=0.5)
    ap.add_argument("--assign", action="store_true", help="add winner-take-all assignment bit loss")
    ap.add_argument("--assign_lambda", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"config: {vars(args)}  device={dev}")
    print(f"2^nap = {1 << args.nap} (must == V = {args.V})")

    emb = nn.Embedding(args.V, args.E).to(dev)
    emb.weight.data.normal_(0, args.emb_init)
    unemb = RowTokenLUTUnembedder(args.E, args.V, args.nap, args.tph,
                                  t_soft=args.t_soft, seed=args.seed, device=dev)
    opt = torch.optim.Adam([
        dict(params=emb.parameters(), lr=args.emb_lr),
        dict(params=unemb.parameters(), lr=args.w_lr),
    ])

    # sanity: at init, how often does ANY table hash a token to itself?
    with torch.no_grad():
        ids0 = torch.arange(args.V, device=dev)
        _, r0, _ = unemb(emb(ids0))
        self_hit0 = (r0 == ids0[:, None]).any(1).float().mean().item()
    print(f"[init] tokens with >=1 self-hash: {self_hit0*100:.3f}%")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        ids = torch.randint(0, args.V, (args.batch,), device=dev)
        x = emb(ids)
        logits, r, d = unemb(x)
        ce = F.cross_entropy(logits, ids)
        bit_loss = unemb.assignment_bit_loss(d, ids) if args.assign else x.new_zeros(())
        loss = ce + args.assign_lambda * bit_loss
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                top1 = (logits.argmax(1) == ids).float().mean().item()
                self_hit = (r == ids[:, None]).any(1).float().mean().item()
                wmax = unemb.weight.max().item()
            print(f"step {step:5d} | CE={ce.item():7.4f} | bit={float(bit_loss):6.4f} "
                  f"| top1={top1*100:6.2f}% | self-hash={self_hit*100:6.2f}% "
                  f"| w_max={wmax:5.2f} | {(time.time()-t0):5.1f}s")

    # final full-vocab eval (chunked to bound memory)
    with torch.no_grad():
        correct = 0
        for s in range(0, args.V, 4096):
            ids = torch.arange(s, min(s + 4096, args.V), device=dev)
            logits, _, _ = unemb(emb(ids))
            correct += (logits.argmax(1) == ids).sum().item()
        print(f"[final] full-vocab top1 accuracy: {correct/args.V*100:.2f}%  ({correct}/{args.V})")


if __name__ == "__main__":
    main()
