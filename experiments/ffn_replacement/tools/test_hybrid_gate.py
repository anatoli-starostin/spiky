"""Unit tests for the gated-hybrid FFN block (MinimalBlock hybrid_gate path).

Hybrid: o = g·FFN(h) + (1-g)·LUT(h), g = sigmoid(theta_l) per layer. Confirms the forward
runs both branches, the gate is learnable, the L1 penalty (Σ_l g_l) is differentiable and
wired, per-layer g_l is readable, and the non-hybrid path is untouched.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))              # tools/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))  # spiky
from model_build import build_model


def _cfg(hybrid, lam=0.0):
    return {
        "ffn_type": "compression", "depth": 3, "n_embd": 384, "n_head": 6, "seq_len": 64,
        "tokenizer_vocab_size": 1024, "tie_unembedder": False, "gamma": 0,
        # LUT branch = plain 0207 geometry
        "lut_impl": "light", "lut_inner_in_dim": 48, "lut_inner_out_dim": 48,
        "lut_n_anchor_pairs": 8, "lut_tables_per_head": 64, "lut_n_heads": 8,
        "lut_joint_head_compression": False, "lut_forward_confidence": True,
        "lut_confidence_form": "margin", "lut_read_top_n": 1, "lut_cell_mode": "constant",
        "lut_tables_no_decay": True, "lut_learnable_temps": True, "lut_base_seed": 1000,
        "hybrid_gate": hybrid, "hybrid_lambda": lam,
    }


def test_hybrid_forward_gate_learnable_and_penalty():
    torch.manual_seed(0)
    m = build_model(_cfg(True), vocab_size=1024, device="cpu")
    NL = len(m.blocks)
    for b in m.blocks:                                    # every block hybrid: dense + LUT + gate
        assert b.hybrid and hasattr(b, "mlp") and hasattr(b, "gate_theta") and hasattr(b, "ffn")
        assert isinstance(b.mlp[0], torch.nn.Linear) and b.mlp[0].out_features == 4 * 384
    # gates ≈ 0.5 at init (theta=0)
    gates = m.hybrid_gates()
    assert len(gates) == NL and all(abs(g - 0.5) < 1e-6 for g in gates)
    # L1 penalty = Σ sigmoid(theta) ≈ 0.5·NL, differentiable
    pen = m.hybrid_gate_penalty()
    assert abs(pen.item() - 0.5 * NL) < 1e-5 and pen.requires_grad
    # forward + backward (CE + λ·penalty) → every gate_theta gets a finite grad (learnable)
    x = torch.randint(0, 1024, (2, 64)); y = torch.randint(0, 1024, (2, 64))
    (m(x, y) + 1e-2 * m.hybrid_gate_penalty()).backward()
    for b in m.blocks:
        assert b.gate_theta.grad is not None and torch.isfinite(b.gate_theta.grad).all()
    assert m(x).shape == (2, 64, 1024)                    # logits shape intact


def test_gate_reflects_theta_per_layer():
    """Per-layer gate values track their own theta: g_l = sigmoid(theta_l)."""
    torch.manual_seed(0)
    m = build_model(_cfg(True), vocab_size=1024, device="cpu")
    with torch.no_grad():
        for i, b in enumerate(m.blocks):
            b.gate_theta.fill_(float(i - 1))              # distinct thetas: -1, 0, 1
    gates = m.hybrid_gates()
    import math
    for i, g in enumerate(gates):
        assert abs(g - 1.0 / (1.0 + math.exp(-(i - 1)))) < 1e-6
    # penalty is exactly the sum of those gates
    assert abs(m.hybrid_gate_penalty().item() - sum(gates)) < 1e-5


def _cfg_stack(order="ffn_lut"):
    c = _cfg(False)                       # base compression config, hybrid_gate off
    c["hybrid_stack"] = True
    c["hybrid_stack_order"] = order
    return c


def test_hybrid_stack_forward_scalars_learnable_and_ratios():
    torch.manual_seed(0)
    m = build_model(_cfg_stack(), vocab_size=1024, device="cpu")
    NL = len(m.blocks)
    for b in m.blocks:                    # each block: dense mlp + LUT ffn + ln_a/ln_b + scales
        assert b.hybrid_stack and b.stack_order == "ffn_lut"
        assert hasattr(b, "s_ffn") and hasattr(b, "s_lut") and hasattr(b, "ln_a") and hasattr(b, "ln_b")
        assert abs(b.s_ffn.item() - 0.1) < 1e-6 and abs(b.s_lut.item() - 0.1) < 1e-6
    x = torch.randint(0, 1024, (2, 64)); y = torch.randint(0, 1024, (2, 64))
    m(x, y).backward()                    # both LayerScale scalars get finite grad (learnable)
    for b in m.blocks:
        assert b.s_ffn.grad is not None and torch.isfinite(b.s_ffn.grad).all()
        assert b.s_lut.grad is not None and torch.isfinite(b.s_lut.grad).all()
    st = m.hybrid_stack_stats()           # per-layer scalars + realized norm-ratios logged
    assert set(st) == {"s_ffn", "s_lut", "ratio_ffn", "ratio_lut"}
    assert all(len(st[k]) == NL for k in st)
    assert all(r >= 0 for r in st["ratio_ffn"]) and all(r >= 0 for r in st["ratio_lut"])
    assert m(x).shape == (2, 64, 1024)


def test_hybrid_stack_order_control_runs():
    m = build_model(_cfg_stack("lut_ffn"), vocab_size=1024, device="cpu")
    assert all(b.stack_order == "lut_ffn" for b in m.blocks)
    x = torch.randint(0, 1024, (2, 32))
    assert m(x).shape == (2, 32, 1024)


def test_nonhybrid_path_unchanged():
    m = build_model(_cfg(False), vocab_size=1024, device="cpu")
    for b in m.blocks:
        assert not getattr(b, "hybrid", False) and not hasattr(b, "gate_theta")
    assert m.hybrid_gates() == [] and m.hybrid_gate_penalty().item() == 0.0
    x = torch.randint(0, 1024, (2, 64))
    assert m(x).shape == (2, 64, 1024)
