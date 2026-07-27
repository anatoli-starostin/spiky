"""t10 — step 1+2: verify the exp025 reconstruction on real data, then capture the
REAL inputs arriving at layer 3's out_proj LUT.

Acceptance test: val bpb must land on the checkpoint's recorded 1.2408.
Then: hook layers[3].out_proj, run real validation tokens, and save
  * the real input vectors (attention output, dim H*d_v = 384) feeding the LUT,
  * the row index each of the 512 tables selects,
  * the table weights + anchor pairs,
so the spiking build can be validated on the distribution that actually occurs.
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
OUT = out("real_capture_layer3.pt")


def main(capture_batches=2, device_bs=8, seq=512):
    m, d = load_exp025()
    base = get_base_dir()
    tok = RustBPETokenizer.from_directory(os.path.join(base, "tokenizer"))
    token_bytes = get_token_bytes(device="cuda")

    # the run's own eval setting was device_batch_size=24, eval_steps=10
    for bs, steps in [(24, 10), (24, 50)]:
        val = tokenizing_distributed_data_loader_bos_bestfit(tok, bs, seq,
                                                             split="val", device="cuda")
        bpb = evaluate_bpb(m, val, steps, token_bytes)
        print(f"[ACCEPTANCE] reconstructed exp025 val bpb = {bpb:.4f}  "
              f"(bs={bs}, steps={steps}; checkpoint recorded {d['final_val_bpb']:.4f}, "
              f"best {d['best_val_bpb']:.4f})")

    # --- capture the real inputs at layer LAYER's out_proj -----------------
    blk = m.layers[LAYER]
    cap = []
    h = blk.out_proj.register_forward_pre_hook(
        lambda mod, args: cap.append(args[0].detach().float().cpu()))
    val2 = tokenizing_distributed_data_loader_bos_bestfit(tok, device_bs, seq,
                                                          split="val", device="cuda")
    with torch.no_grad():
        for _ in range(capture_batches):
            x, y = next(val2)
            m(x)
    h.remove()
    X = torch.cat(cap, 0)                      # [N, H*d_v] real inputs to out_proj
    print(f"captured real inputs at layers[{LAYER}].out_proj: {tuple(X.shape)}")

    lut = blk.out_proj
    torch.save(dict(
        X=X,
        weights=lut.weights.detach().float().cpu(),                 # [n_tables, K, D]
        anchor_a=lut.soft_anchor_a_long.detach().cpu(),               # [n_tables, NAP]
        anchor_b=lut.soft_anchor_b_long.detach().cpu(),
        powers=lut.soft_powers.detach().cpu(),
        bpb=bpb, layer=LAYER,
    ), OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
