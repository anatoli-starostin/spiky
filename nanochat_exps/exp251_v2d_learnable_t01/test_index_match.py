"""Check whether SoftMHLut and TinyMHLut(soft) compute the SAME argmax index
under bf16 autocast.

SoftMHLut argmax: argmax of softmax(ts/T_sel) where ts is bf16 (from bf16 GEMM).
TinyMHLut(soft) argmax: bit-pack of sign(x_a - x_b), purely fp32.

These should agree when ts has unambiguous argmax, disagree when bf16 quantization
of ts swaps near-tied entries.
"""
import torch
from spiky.lutorch.tiny_multi_head_lut import (
    TinyMultiHeadLut, _soft_bit_matrix_msb, _msb_powers,
)
from spiky.lutorch.soft_multi_head_lut import SoftMultiHeadLUT, _bit_matrix
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


def main():
    dev = torch.device("cuda")
    cfgs = [
        ("v_lut",      96, 6, 256, 8, 32),
        ("qk_joint",   96, 6, 256, 6, 128),
        ("out_proj_L0",192,1, 2048,6, 96),
    ]
    for name, input_dim, n_heads, tph, nap, n_outputs in cfgs:
        torch.manual_seed(42)
        tiny = TinyMultiHeadLut(
            input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
            n_anchor_pairs=nap, tables_per_head=tph,
            weight_dtype=torch.float32, random_seed=42, device=dev,
            anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
            backward_mode="soft", soft_score_temp=0.5, select_temp=0.5,
            learnable_temps=True, use_bf16=True,
        ).to(dev)
        torch.manual_seed(42)
        soft = SoftMultiHeadLUT(
            input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
            n_anchor_pairs=nap, tables_per_head=tph,
            soft_score_temp=0.5, select_temp=0.5, hard=True, learnable_temps=True,
            anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
            weight_dtype=torch.float32, random_seed=42, device=dev,
            use_bf16=True, compile_forward=False,
        ).to(dev)
        # Recompute indices using the SAME math as each module would use.
        torch.manual_seed(0)
        x = torch.randn(4096, input_dim, device=dev)

        # TinyMHLut(soft) index: bit-pack on fp32 d.
        idx_a = tiny.lookup.anchor_pairs_a.long()
        idx_b = tiny.lookup.anchor_pairs_b.long()
        d = x[:, idx_a] - x[:, idx_b]               # [B, T, NAP]  fp32
        powers = _msb_powers(nap, dev).view(1, 1, -1)
        idx_tiny = ((d > 0).to(torch.int64) * powers).sum(dim=-1)   # [B, T]

        # SoftMHLut(bf16) index: argmax of softmax(ts/T_sel) where ts is bf16 GEMM.
        # Replicate SoftMHLut's path under bf16 autocast.
        bm = _bit_matrix(nap, dev, dtype=torch.float32)   # SoftMHLut convention (MSB-first)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p_bf = d / (0.5 + d.abs())
            ts_bf = torch.einsum("btp,pk->btk", p_bf, bm.to(p_bf.dtype))
            sel_soft_bf = torch.nn.functional.softmax(ts_bf / 0.5, dim=-1)
            idx_soft = sel_soft_bf.argmax(dim=-1)

        # Also compute the fp32 version of soft's argmax for reference.
        with torch.amp.autocast("cuda", enabled=False):
            p_fp = d / (0.5 + d.abs())
            ts_fp = torch.einsum("btp,pk->btk", p_fp, bm)
            sel_soft_fp = torch.nn.functional.softmax(ts_fp / 0.5, dim=-1)
            idx_soft_fp32 = sel_soft_fp.argmax(dim=-1)

        n_total = idx_tiny.numel()
        n_diff_tiny_vs_softbf = (idx_tiny != idx_soft).sum().item()
        n_diff_tiny_vs_softfp = (idx_tiny != idx_soft_fp32).sum().item()
        n_diff_softbf_vs_softfp = (idx_soft != idx_soft_fp32).sum().item()
        print(f"\n=== {name} (NAP={nap}, K={1<<nap}, B*T={n_total}) ===")
        print(f"  TinyMHLut(fp32 bit-pack)  vs SoftMHLut(bf16 argmax):    {n_diff_tiny_vs_softbf:>10d}  ({100*n_diff_tiny_vs_softbf/n_total:.2f}%)")
        print(f"  TinyMHLut(fp32 bit-pack)  vs SoftMHLut(fp32 argmax):    {n_diff_tiny_vs_softfp:>10d}  ({100*n_diff_tiny_vs_softfp/n_total:.2f}%)")
        print(f"  SoftMHLut(bf16 argmax)    vs SoftMHLut(fp32 argmax):    {n_diff_softbf_vs_softfp:>10d}  ({100*n_diff_softbf_vs_softfp/n_total:.2f}%)")


if __name__ == "__main__":
    main()
