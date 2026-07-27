"""t12 — the model-level question: how much TIMING RESOLUTION does a real stage need?

The spiking circuit reproduces a real table exactly when the input arrives as an exact
order code. A physical net has a finite tick clock, so the real question is how coarse
the input latency grid can be before the model degrades.

Here: quantise the input of layer-3 `out_proj` (all 512 tables) onto a uniform grid of
R+1 ticks inside the real forward pass, and measure val bpb. Rank coding is included as
the lossless control (it preserves every pairwise order, so the LUT sees identical bits).
"""
import os, sys, torch

from paths import NANOCHAT_ROOT, out
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.common import get_base_dir

from exp025_model import load_exp025

LAYER = 3
LO, HI = None, None      # filled from the real capture


def make_hook(mode, R):
    def hook(mod, args):
        x = args[0]
        if mode == "rank":
            xq = torch.argsort(torch.argsort(x, dim=-1), dim=-1).to(x.dtype)
        else:
            xq = (((x - LO) / (HI - LO)).clamp(0, 1) * R).round()
        return (xq,)
    return hook


def main(steps=10, bs=24, seq=512):
    global LO, HI
    cap = torch.load(out("real_capture_layer3.pt"),
                     map_location="cpu", weights_only=False)
    X = cap["X"].flatten().float()
    LO = torch.quantile(X, 0.001).item()
    HI = torch.quantile(X, 0.999).item()
    m, d = load_exp025()
    base = get_base_dir()
    tok = RustBPETokenizer.from_directory(os.path.join(base, "tokenizer"))
    tb = get_token_bytes(device="cuda")

    def ev():
        val = tokenizing_distributed_data_loader_bos_bestfit(tok, bs, seq,
                                                             split="val", device="cuda")
        return evaluate_bpb(m, val, steps, tb)

    base_bpb = ev()
    print(f"baseline (exact fp32 input to layer{LAYER}.out_proj): val bpb = {base_bpb:.4f}")

    for mode, R in [("rank", 0)] + [("grid", r) for r in (511, 255, 127, 63, 31, 15, 7)]:
        h = m.layers[LAYER].out_proj.register_forward_pre_hook(make_hook(mode, R))
        b = ev()
        h.remove()
        name = "exact order code (rank)" if mode == "rank" else f"uniform grid, {R+1:>4} ticks"
        print(f"  {name:<28} val bpb = {b:.4f}   (delta {b-base_bpb:+.4f})")


if __name__ == "__main__":
    main()
