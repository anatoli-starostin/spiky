"""Is the STANDARD sparse-scatter LUT unembedder invertible at all?

Trainable Embedding(V, E) + a sparse-scatter TinyMHLut mapping E->V, trained ONLY
on the consistency loss CE(unemb(emb.weight), arange(V)). If top1 -> high, the head
can represent the identity (embedder<->unembedder tie). Mirrors tied_lut_inverse_test.py
but for the standard sparse-scatter head (ste backward) used in exp517.

Run:  python tied_sparse_lut_inverse_test.py [--tph 4096 --n_sparse 8 --nap 8 --lr 1e-2]
"""
import argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


def build_unemb(E, V, nap, tph, n_sparse, seed, device, backward_mode):
    return TinyMultiHeadLut(
        input_dim=E, n_heads=1, n_outputs=n_sparse,
        n_anchor_pairs=nap, tables_per_head=tph,
        sparse_scatter_n_outputs=V, sparse_scatter_seed=seed + 99999,
        weight_dtype=torch.float32,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        initial_weights_noise=0.001, backward_mode=backward_mode,
        learnable_temps=True, use_bf16=True, random_seed=seed, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=32768)
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--nap", type=int, default=8)
    ap.add_argument("--tph", type=int, default=4096)
    ap.add_argument("--n_sparse", type=int, default=8)
    ap.add_argument("--backward", default="ste", choices=["ste", "soft"])
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--emb_init", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    K = args.tph * args.n_sparse / args.V
    print(f"config: {vars(args)} device={dev} coverage_K={K:.2f}")

    emb = nn.Embedding(args.V, args.E).to(dev)
    emb.weight.data.normal_(0, args.emb_init)
    unemb = build_unemb(args.E, args.V, args.nap, args.tph, args.n_sparse,
                        args.seed, dev, args.backward)
    n_unemb = sum(p.numel() for p in unemb.parameters())
    print(f"unemb params: {n_unemb:,}  (rows/table=2^nap={1<<args.nap})")
    opt = torch.optim.Adam([
        dict(params=emb.parameters(), lr=args.lr),
        dict(params=unemb.parameters(), lr=args.lr),
    ])

    t0 = time.time()
    for step in range(1, args.steps + 1):
        ids = torch.randint(0, args.V, (args.batch,), device=dev)
        logits = unemb(emb(ids)).squeeze(1)
        loss = F.cross_entropy(logits, ids)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                top1 = (logits.argmax(1) == ids).float().mean().item()
                wmax = unemb.weights.abs().max().item()
            print(f"step {step:5d} | CE={loss.item():7.4f} | top1={top1*100:6.2f}% "
                  f"| w_absmax={wmax:6.2f} | {(time.time()-t0):5.1f}s")

    with torch.no_grad():
        correct = 0
        for s in range(0, args.V, 4096):
            ids = torch.arange(s, min(s + 4096, args.V), device=dev)
            correct += (unemb(emb(ids)).squeeze(1).argmax(1) == ids).sum().item()
        print(f"[final] full-vocab top1: {correct/args.V*100:.2f}%  ({correct}/{args.V})")


if __name__ == "__main__":
    main()
