"""
Tests for GT_spike_QK_Transformer vs LUTTransformer forward and backward equivalence.

Verifies that the ground-truth transformer and the lutorch LUTTransformer produce
the same forward output when given the same weights and input, and that backward
passes can be run with weights kept in sync via SpikeQKCheckpoint. Uses small
hyperparameters (2 layers, 2 heads, fewer anchor pairs) so tests run quickly.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable torch.compile in lutorch and GT so tests run without compilation.
os.environ["SPIKY_LUTORCH_NO_COMPILE"] = "1"
os.environ["SPIKY_GT_NO_COMPILE"] = "1"

from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_cross_attention import LUTCrossAttention
from spiky.lutorch.tests.gt_spike_qk_transformer import (
    GT_spike_QK_Transformer,
    SpikeQKCheckpoint,
)

# --- Small config (2 layers, 2 heads, fewer tables/anchors) ---
CONTEXT_SIZE = 32
VOCAB_SIZE = 257
EMBEDDING_DIM = 32
NUM_LAYERS = 2
NUM_HEADS = 2
POSITIONAL_BUCKETS_A = 4
ATTENTION_A_N_T = 8
ATTENTION_A_N_C = 6
ATTENTION_V_N_T = 8
ATTENTION_V_N_C = 6
FFN_N_T = 8
FFN_N_C = 6
ATTENTION_TEMPERATURE = 0.25
UNEMBED_TEMPERATURE = 0.1


# --- LUTTransformer (from workbooks/lutorch_transformer.ipynb) ---


class LUTTransformer(nn.Module):
    """LUTTransformer with same forward as spike_QK (MultiHeadLut + LUTCrossAttention)."""

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            context_size,
            num_layers,
            num_heads,
            n_anchor_pairs_attn,
            n_anchor_pairs_ffn,
            n_positional_buckets,
            tables_per_head_attn,
            tables_per_head_value,
            ffn_tables,
            dropout=0.0,
            smooth_mode=True,
            device=None,
            anchor_initialization="balanced",
            random_seed=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.num_heads = num_heads
        dev = device or torch.device("cpu")

        with torch.no_grad():
            self.token_embedder = nn.Embedding(vocab_size, embedding_dim, device=dev)
            self.token_embedder.weight.copy_(
                torch.randn(self.token_embedder.weight.shape, device=dev) * 0.1
            )

            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                attn_lut = MultiHeadLut(
                    input_dim=embedding_dim,
                    n_heads=num_heads,
                    n_outputs=1,
                    n_anchor_pairs=n_anchor_pairs_attn,
                    tables_per_head=tables_per_head_attn,
                    n_buckets=n_positional_buckets,
                    smooth_mode=smooth_mode,
                    device=dev,
                    anchor_initialization=anchor_initialization,
                    random_seed=random_seed,
                )
                attn_lut.projection.weights.copy_(
                    torch.randn(attn_lut.projection.weights.shape, device=dev) * 0.001
                )
                cross_attn = LUTCrossAttention(
                    attn_lut,
                    causal=True,
                    attention_temperature=ATTENTION_TEMPERATURE,
                    n_positional_buckets=n_positional_buckets,
                )
                value_lut = MultiHeadLut(
                    input_dim=embedding_dim,
                    n_heads=num_heads,
                    n_outputs=embedding_dim // num_heads,
                    n_anchor_pairs=n_anchor_pairs_attn,
                    tables_per_head=tables_per_head_value,
                    smooth_mode=smooth_mode,
                    device=dev,
                    anchor_initialization=anchor_initialization,
                    random_seed=random_seed,
                )
                value_lut.projection.weights.copy_(
                    torch.randn(value_lut.projection.weights.shape, device=dev) * 0.001
                )
                ffn_lut = MultiHeadLut(
                    input_dim=embedding_dim,
                    n_heads=1,
                    n_outputs=embedding_dim,
                    n_anchor_pairs=n_anchor_pairs_ffn,
                    tables_per_head=ffn_tables,
                    smooth_mode=smooth_mode,
                    device=dev,
                    anchor_initialization=anchor_initialization,
                    random_seed=random_seed,
                )
                ffn_lut.projection.weights.copy_(
                    torch.randn(ffn_lut.projection.weights.shape, device=dev) * 0.001
                )
                self.layers.append(
                    nn.ModuleDict({
                        "cross_attn": cross_attn,
                        "value_lut": value_lut,
                        "attn_dropout": nn.Dropout(dropout),
                        "ffn": ffn_lut,
                        "ffn_dropout": nn.Dropout(dropout),
                    })
                )

    def forward(self, tokens):
        B, S = tokens.shape
        with torch.no_grad():
            z = self.token_embedder(tokens)
        for layer in self.layers:
            attn_weights = layer["cross_attn"](z, z)
            v = layer["value_lut"](z.reshape(-1, self.embedding_dim))
            v = v.reshape(B, S, self.num_heads, -1)
            attn_out = attn_weights.permute(0, 3, 1, 2) @ v.permute(0, 2, 1, 3)
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, self.embedding_dim)
            z = z + layer["attn_dropout"](attn_out)
            ffn_out = layer["ffn"](z.reshape(-1, self.embedding_dim)).reshape(B, S, -1)
            z = z + layer["ffn_dropout"](ffn_out)
        z_norm = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        logits = z_norm @ self.token_embedder.weight.T / UNEMBED_TEMPERATURE
        return logits


def _load_checkpoint_into_lut(lut_transformer, ckpt):
    """Copy checkpoint weights into LUTTransformer (from lutorch_transformer.ipynb)."""

    def _to_dev(t, dev):
        return t.to(device=dev, dtype=t.dtype if t.is_floating_point() else t.dtype)

    with torch.no_grad():
        lut_transformer.token_embedder.weight.copy_(
            _to_dev(ckpt.w_embed, lut_transformer.token_embedder.weight.device)
        )

        for l in range(ckpt.num_layers):
            layer = lut_transformer.layers[l]
            dev = next(layer.parameters()).device

            ffn = layer["ffn"]
            ffn.lookup.anchor_pairs_a.copy_(_to_dev(ckpt.ffn[l]["A_stacked"], dev))
            ffn.lookup.anchor_pairs_b.copy_(_to_dev(ckpt.ffn[l]["B_stacked"], dev))
            ffn.projection.weights.copy_(_to_dev(ckpt.ffn[l]["S"], dev))

            vl = layer["value_lut"]
            n_t, n_c = ckpt.attention_v_n_t, ckpt.attention_v_n_c
            vl.lookup.anchor_pairs_a.copy_(
                _to_dev(ckpt.v[l]["anchors_a"].reshape(-1, n_c), dev)
            )
            vl.lookup.anchor_pairs_b.copy_(
                _to_dev(ckpt.v[l]["anchors_b"].reshape(-1, n_c), dev)
            )
            Sv = ckpt.v[l]["S"]
            vl.projection.weights.copy_(
                _to_dev(
                    Sv.permute(2, 0, 1, 3).reshape(-1, Sv.shape[1], Sv.shape[3]),
                    dev,
                )
            )

            attn_lut = layer["cross_attn"].multi_head_lut
            attn_lut.lookup.anchor_pairs_a.copy_(
                _to_dev(ckpt.a[l]["anchors_a"].reshape(-1, ckpt.attention_a_n_c), dev)
            )
            attn_lut.lookup.anchor_pairs_b.copy_(
                _to_dev(ckpt.a[l]["anchors_b"].reshape(-1, ckpt.attention_a_n_c), dev)
            )
            Sa = ckpt.a[l]["S"]
            attn_lut.projection.weights.copy_(
                _to_dev(
                    Sa.permute(2, 0, 1, 3).reshape(-1, Sa.shape[1], Sa.shape[3]),
                    dev,
                )
            )


def _diagnose_weight_diffs(gt, lut):
    """Print per-tensor max/mean abs diff and shape to help debug weight mismatch."""
    ckpt_gt = SpikeQKCheckpoint.from_gt(gt)
    dev = next(lut.parameters()).device
    n_heads = ckpt_gt.num_heads
    n_t_v = ckpt_gt.attention_v_n_t
    n_t_a = ckpt_gt.attention_a_n_t

    def _diff(actual, expected, name):
        actual = actual.detach().to(device=dev)
        expected = expected.to(device=dev, dtype=actual.dtype)
        d = (actual - expected).abs()
        return name, d.max().item(), d.mean().item(), actual.shape

    lines = ["Weight diff diagnosis (LUT - GT):"]
    for l in range(ckpt_gt.num_layers):
        layer = lut.layers[l]
        ffn = layer["ffn"]
        name, mx, mn, sh = _diff(ffn.projection.weights, ckpt_gt.ffn[l]["S"], f"ffn[{l}].S")
        lines.append(f"  {name} shape={sh} max_diff={mx:.2e} mean_diff={mn:.2e}")

        vl = layer["value_lut"]
        Sv = ckpt_gt.v[l]["S"]
        lut_vS = vl.projection.weights.reshape(n_heads, n_t_v, Sv.shape[1], Sv.shape[3]).permute(1, 2, 0, 3)
        name, mx, mn, sh = _diff(lut_vS, Sv, f"v[{l}].S")
        lines.append(f"  {name} shape={sh} max_diff={mx:.2e} mean_diff={mn:.2e}")

        attn_lut = layer["cross_attn"].multi_head_lut
        Sa = ckpt_gt.a[l]["S"]
        lut_aS = attn_lut.projection.weights.reshape(n_heads, n_t_a, Sa.shape[1], Sa.shape[3]).permute(1, 2, 0, 3)
        name, mx, mn, sh = _diff(lut_aS, Sa, f"a[{l}].S")
        lines.append(f"  {name} shape={sh} max_diff={mx:.2e} mean_diff={mn:.2e}")

    print("\n".join(lines))


def _assert_weights_equal_before_sync(gt, lut, atol=1e-5, rtol=1e-4, msg_prefix=""):
    """
    Compare GT and LUT weights after both have been updated by their backward pass.
    Uses checkpoint layout (from_gt) so corresponding tensors are compared. Call
    before synchronizing LUT from checkpoint. Runs all checks and raises once with
    a combined message if any fail.
    """
    ckpt_gt = SpikeQKCheckpoint.from_gt(gt)
    dev = next(lut.parameters()).device
    n_heads = ckpt_gt.num_heads
    n_t_v = ckpt_gt.attention_v_n_t
    n_t_a = ckpt_gt.attention_a_n_t

    def _check(actual, expected, name):
        actual = actual.detach().to(device=dev)
        expected = expected.to(device=dev, dtype=actual.dtype)
        try:
            torch.testing.assert_close(
                actual, expected, atol=atol, rtol=rtol, msg=f"{msg_prefix}{name}"
            )
            return None
        except AssertionError as e:
            return f"{name}: {e}"

    failures = []

    # GT uses Adam for W_embed; LUT uses SGD, so w_embed can differ. Compare LUT layers only.

    for l in range(ckpt_gt.num_layers):
        layer = lut.layers[l]
        ffn = layer["ffn"]
        err = _check(ffn.projection.weights, ckpt_gt.ffn[l]["S"], f"ffn[{l}].S")
        if err is not None:
            failures.append(err)

        vl = layer["value_lut"]
        Sv = ckpt_gt.v[l]["S"]
        lut_vS = vl.projection.weights.reshape(n_heads, n_t_v, Sv.shape[1], Sv.shape[3]).permute(1, 2, 0, 3)
        err = _check(lut_vS, Sv, f"v[{l}].S")
        if err is not None:
            failures.append(err)

        attn_lut = layer["cross_attn"].multi_head_lut
        Sa = ckpt_gt.a[l]["S"]
        lut_aS = attn_lut.projection.weights.reshape(n_heads, n_t_a, Sa.shape[1], Sa.shape[3]).permute(1, 2, 0, 3)
        err = _check(lut_aS, Sa, f"a[{l}].S")
        if err is not None:
            failures.append(err)

    if failures:
        _diagnose_weight_diffs(gt, lut)
        raise AssertionError(
            f"{msg_prefix}weight mismatch before sync ({len(failures)} check(s) failed):\n  "
            + "\n  ".join(failures)
        )


# --- Tests ---


def test_gt_and_lut_transformer_forward_match(device, seed=None):
    """
    GT and LUT transformer forward match.
    Build GT, export SpikeQKCheckpoint, create LUT with same config, load checkpoint,
    run same random input through both and compare logits.
    """
    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device(device)

    gt = GT_spike_QK_Transformer(
        device=dev,
        context_size=CONTEXT_SIZE,
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        positional_buckets_a=POSITIONAL_BUCKETS_A,
        attention_a_n_t=ATTENTION_A_N_T,
        attention_a_n_c=ATTENTION_A_N_C,
        attention_v_n_t=ATTENTION_V_N_T,
        attention_v_n_c=ATTENTION_V_N_C,
        attention_temperature=ATTENTION_TEMPERATURE,
        include_ffn=True,
        ffn_n_t=FFN_N_T,
        ffn_n_c=FFN_N_C,
        unembed_temperature=UNEMBED_TEMPERATURE,
        smooth_forward=True,
        noise_jitter_scale=0.0,
    )

    ckpt = SpikeQKCheckpoint.from_gt(gt)

    lut = LUTTransformer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        context_size=CONTEXT_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        n_anchor_pairs_attn=ATTENTION_A_N_C,
        n_anchor_pairs_ffn=FFN_N_C,
        n_positional_buckets=POSITIONAL_BUCKETS_A,
        tables_per_head_attn=ATTENTION_A_N_T,
        tables_per_head_value=ATTENTION_V_N_T,
        ffn_tables=FFN_N_T,
        dropout=0.0,
        smooth_mode=True,
        device=dev,
        anchor_initialization="balanced",
        random_seed=123,
    )

    _load_checkpoint_into_lut(lut, ckpt)

    batch_size = 4
    x = torch.randint(
        0, VOCAB_SIZE, (batch_size, CONTEXT_SIZE), device=dev, dtype=torch.long
    )

    gt.set_batch_size(batch_size)
    gt_logits = gt.forward(x, training=False)
    lut_logits = lut(x)

    gt_logits_2d = gt_logits.view(batch_size, CONTEXT_SIZE, VOCAB_SIZE)

    torch.testing.assert_close(
        lut_logits,
        gt_logits_2d,
        atol=1e-4,
        rtol=1e-3,
        msg="GT and LUT transformer forward outputs should match",
    )

    print("✓ GT and LUT transformer forward match test passed")
    return True


def test_gt_and_lut_transformer_backward_match(device, seed=None, n_iterations=10):
    """
    GT and LUT transformer backward: run training iterations with both models,
    synchronize weights via SpikeQKCheckpoint after each iteration (from GT into LUT),
    and verify forward outputs still match every step.
    """
    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device(device)
    batch_size = 4
    learning_rate = 0.01

    gt = GT_spike_QK_Transformer(
        device=dev,
        context_size=CONTEXT_SIZE,
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        positional_buckets_a=POSITIONAL_BUCKETS_A,
        attention_a_n_t=ATTENTION_A_N_T,
        attention_a_n_c=ATTENTION_A_N_C,
        attention_v_n_t=ATTENTION_V_N_T,
        attention_v_n_c=ATTENTION_V_N_C,
        attention_temperature=ATTENTION_TEMPERATURE,
        include_ffn=True,
        ffn_n_t=FFN_N_T,
        ffn_n_c=FFN_N_C,
        unembed_temperature=UNEMBED_TEMPERATURE,
        smooth_forward=True,
        noise_jitter_scale=0.0,
    )

    ckpt = SpikeQKCheckpoint.from_gt(gt)

    lut = LUTTransformer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        context_size=CONTEXT_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        n_anchor_pairs_attn=ATTENTION_A_N_C,
        n_anchor_pairs_ffn=FFN_N_C,
        n_positional_buckets=POSITIONAL_BUCKETS_A,
        tables_per_head_attn=ATTENTION_A_N_T,
        tables_per_head_value=ATTENTION_V_N_T,
        ffn_tables=FFN_N_T,
        dropout=0.0,
        smooth_mode=True,
        device=dev,
        anchor_initialization="balanced",
        random_seed=123,
    )

    _load_checkpoint_into_lut(lut, ckpt)
    lut.train()

    # GT expects logits gradient = (probs - one_hot) with no division. Use reduction='sum'
    # so that d(loss)/d(logits) = (probs - one_hot), matching GT.
    lut_optimizer = torch.optim.SGD(lut.parameters(), lr=learning_rate)

    for iteration in range(n_iterations):
        x = torch.randint(
            0, VOCAB_SIZE, (batch_size, CONTEXT_SIZE), device=dev, dtype=torch.long
        )
        target = torch.randint(
            0, VOCAB_SIZE, (batch_size, CONTEXT_SIZE), device=dev, dtype=torch.long
        )

        gt.set_batch_size(batch_size)
        gt_logits = gt.forward(x, training=True)
        lut_logits = lut(x)

        gt_logits_2d = gt_logits.view(batch_size, CONTEXT_SIZE, VOCAB_SIZE)
        torch.testing.assert_close(
            lut_logits,
            gt_logits_2d,
            atol=1e-4,
            rtol=1e-3,
            msg=f"Iteration {iteration}: GT and LUT forward outputs should match before backward",
        )

        # GT expects logits gradient = (probs - one_hot) with no division
        probs = F.softmax(gt_logits, dim=-1)
        one_hot = F.one_hot(
            target.view(-1), num_classes=VOCAB_SIZE
        ).float().to(device=dev, dtype=gt_logits.dtype)
        gt.output.copy_(probs - one_hot)

        gt.backward(learning_rate)

        lut_optimizer.zero_grad()
        loss_lut = F.cross_entropy(
            lut_logits.reshape(-1, VOCAB_SIZE), target.view(-1), reduction="sum"
        )
        loss_lut.backward()
        lut_optimizer.step()

        # Check that corresponding weights are equal before synchronization
        _assert_weights_equal_before_sync(
            gt, lut, atol=1e-5, rtol=1e-4,
            msg_prefix=f"Iteration {iteration}: weights should match before sync: ",
        )

        ckpt = SpikeQKCheckpoint.from_gt(gt)
        _load_checkpoint_into_lut(lut, ckpt)

    print(f"✓ GT and LUT transformer backward test passed ({n_iterations} iterations)")
    return True


def main():
    """Run GT vs LUT transformer tests on available devices."""
    print("=" * 60)
    print("GT vs LUT Transformer TESTS")
    print("=" * 60)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    seed = 42

    for device in devices:
        print(f"\nTesting on {device}...")

        print("\n1. GT and LUT transformer forward match...")
        success = test_gt_and_lut_transformer_forward_match(device, seed=seed)
        if not success:
            print(f"❌ Test failed on {device}")
            return -1

        print("\n2. GT and LUT transformer backward (with SpikeQKCheckpoint sync)...")
        success = test_gt_and_lut_transformer_backward_match(device, seed=seed)
        if not success:
            print(f"❌ Backward test failed on {device}")
            return -1

        print(f"\n✓ All tests passed on {device}!")

    return 0


if __name__ == "__main__":
    exit(main())
