"""[lut_nanochat] LUT feed-forward: a drop-in replacement for the dense MLP in each transformer block.

Each block's `.mlp` (dense c_fc / relu^2 / c_proj) is replaced by a ProjectionMHL(ConfidenceLUT) fp32
"island": compress Linear (d_model -> r=h*d) -> ConfidenceLUT read -> decompress Linear (r -> d_model).

Two integration facts the rest of base_train relies on:
  * Construction happens AFTER GPT.init_weights(). GPT.__init__ runs under a meta-device context
    (shapes only), but the ConfidenceLUT cartridge needs REAL data (seeded anchors + table weights),
    so apply_lut_ffn() must be called once the backbone has been materialised + initialised.
  * The module is a fp32 island inside an fp8/bf16 backbone. forward() casts its input up to fp32 and
    the output back to the caller's dtype; its projection Linears are tagged `_lut_no_fp8` so the fp8
    converter skips them, and every LUT parameter lives in a module flagged `_is_lut_ffn` so
    GPT.setup_optimizer() routes it to AdamW rather than Muon (Muon's Newton-Schulz assumes 2-D
    matrices; the LUT tables are 4-D and the confidence scores are 0-D scalars).

Init policy: a clean zero-init identity drop-in. ProjectionMHL already zeros decompress.weight; we ALSO
zero decompress.bias, so at init the block emits exactly 0 into the residual stream (row-to-row spread 0
AND constant 0), and the model starts identical to a bypass of the FFN sub-layer.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LUTFFNConfig:
    """Locked ConfidenceLUT FFN geometry for d24 (r = h*d = 768 = d_model/2, 2^nap = 256 cells/table)."""
    enabled: bool = False
    h: int = 16                    # h_in = h_out
    d: int = 48                    # d_in = d_out
    tph: int = 64                  # tables per head
    nap: int = 8                   # anchor pairs -> 2^nap cells per table
    read_top_n: int = 1            # tau drops out at n=1
    beta_init: float = 2.0
    gamma_init: float = 1.0
    seed: int = 1
    weight_init_std: float = 1e-3  # table init std
    table_dropout_rate: float = 0.2
    tv_lambda: float = 10.0        # cell-TV penalty weight (applied in base_train's step loop)


class LUTFeedForward(nn.Module):
    """A single transformer block's FFN as a ProjectionMHL(ConfidenceLUT) fp32 island."""

    _is_lut_ffn = True  # marker for GPT.setup_optimizer() param routing

    def __init__(self, d_model: int, cfg: LUTFFNConfig, device=None):
        super().__init__()
        # Lazy import: only pulled in when LUT-FFN is actually enabled, so dense runs never touch lutorch_ex.
        from spiky.lutorch_ex import LUTSpec, ConfidenceLUT, ProjectionMHL

        spec = LUTSpec(h_in=cfg.h, h_out=cfg.h, tph=cfg.tph, nap=cfg.nap,
                       d_in=cfg.d, d_out=cfg.d, anchor_mode="pairs")
        cart = ConfidenceLUT(spec, seed=cfg.seed, read_top_n=cfg.read_top_n,
                             beta_init=cfg.beta_init, gamma_init=cfg.gamma_init,
                             learnable_score=True, weight_init_std=cfg.weight_init_std,
                             table_dropout_rate=cfg.table_dropout_rate)
        self.mhl = ProjectionMHL(cart, d_model=d_model).float()  # fp32 island

        # Clean zero-init identity drop-in: ProjectionMHL zeros decompress.weight; also zero its bias so
        # the block emits exactly 0 at init (constant 0, not merely row-to-row-constant).
        with torch.no_grad():
            if self.mhl.decompress.bias is not None:
                nn.init.zeros_(self.mhl.decompress.bias)

        # Tag the projection Linears so the fp8 converter skips them (they are the fp32 island).
        for m in self.mhl.modules():
            if isinstance(m, nn.Linear):
                m._lut_no_fp8 = True

        if device is not None:
            self.to(device)

    def forward(self, x):
        # x: (B, T, d_model) in any dtype -> fp32 island -> back to the caller's dtype.
        orig_dtype = x.dtype
        B, T, C = x.shape
        y = self.mhl(x.reshape(B * T, C).float())
        return y.reshape(B, T, C).to(orig_dtype)

    def cell_tv(self):
        """Cell total-variation penalty on this layer's LUT tables (fp32 scalar)."""
        from spiky.lutorch_ex import cell_tv_penalty
        return cell_tv_penalty(self.mhl)


def apply_lut_ffn(model, cfg: LUTFFNConfig, device=None) -> int:
    """Replace every transformer block's `.mlp` with a LUTFeedForward. Call AFTER init_weights()
    (and, when resuming a LUT checkpoint, BEFORE load_state_dict so the keys line up). Returns count."""
    d_model = model.config.n_embd
    n = 0
    for block in model.transformer.h:
        block.mlp = LUTFeedForward(d_model, cfg, device=device)
        n += 1
    return n


def lut_ffn_modules(model):
    return [m for m in model.modules() if getattr(m, "_is_lut_ffn", False)]


def lut_cell_tv_loss(model, lam):
    """Sum of per-layer cell-TV penalties * lam, or None if there are no LUT-FFN layers / lam<=0."""
    if lam <= 0:
        return None
    mods = lut_ffn_modules(model)
    if not mods:
        return None
    return lam * sum(m.cell_tv() for m in mods)
