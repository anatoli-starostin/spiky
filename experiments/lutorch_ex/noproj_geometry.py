"""Spec / projection builders for train_multi.py. Flags off = the original geometry (h_in=h_out=lut_n_heads, per-head
compress n_embd -> h*d_in, anchor pairs). NO_INPUT_PROJ=1 = fan-out without input projection: LUTSpec(h_in=1, h_out=h,
d_in=n_embd), ProjectionMHL(compress=False), so every head reads the WHOLE stream (decompress h*d_out -> n_embd kept)."""
from spiky.lutorch_ex import LUTSpec, ProjectionMHL


def make_spec(cfg, no_input_proj=False, anchor_mode="pairs"):
    return LUTSpec(h_in=1 if no_input_proj else cfg["lut_n_heads"], h_out=cfg["lut_n_heads"],
                   tph=cfg["lut_tables_per_head"], nap=cfg["lut_n_anchor_pairs"],
                   d_in=cfg["n_embd"] if no_input_proj else cfg["lut_inner_in_dim"],
                   d_out=cfg["lut_inner_out_dim"], anchor_mode=anchor_mode)


def make_proj(cart, n_embd, no_input_proj=False):
    return ProjectionMHL(cart, d_model=n_embd, compress=not no_input_proj)
