"""
Tests for GT_spike_QK_Transformer vs LUTTransformer forward and backward equivalence.

Verifies that the ground-truth transformer and the lutorch LUTTransformer produce
the same forward output when given the same weights and input, and that backward
passes can be run with weights kept in sync via SpikeQKCheckpoint. Uses small
hyperparameters (2 layers, 2 heads, fewer anchor pairs) so tests run quickly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_attention import LUTAttention
from spiky.lutorch.tests.gt_spike_qk_transformer import (
    GT_spike_QK_Transformer,
    SpikeQKCheckpoint,
)

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

SEED = 42


class LUTTransformer(nn.Module):
    """LUTTransformer with same forward as spike_QK (MultiHeadLut + LUTAttention)."""

    def __init__(self, vocab_size, embedding_dim, context_size, num_layers, num_heads,
                 n_anchor_pairs_attn, n_anchor_pairs_ffn, n_positional_buckets,
                 tables_per_head_attn, tables_per_head_value, ffn_tables,
                 dropout=0.0, smooth_mode=True, device=None,
                 connected_anchors_mode=False, random_seed=None):
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
                    input_dim=embedding_dim, n_heads=num_heads, n_outputs=1,
                    n_anchor_pairs=n_anchor_pairs_attn, tables_per_head=tables_per_head_attn,
                    n_buckets=n_positional_buckets, smooth_mode=smooth_mode, device=dev,
                    connected_anchors_mode=connected_anchors_mode, random_seed=random_seed,
                )
                attn_lut.projection.weights.copy_(
                    torch.randn(attn_lut.projection.weights.shape, device=dev) * 0.001
                )
                cross_attn = LUTAttention(
                    attn_lut, causal=True,
                    attention_temperature=ATTENTION_TEMPERATURE,
                    n_positional_buckets=n_positional_buckets,
                )
                value_lut = MultiHeadLut(
                    input_dim=embedding_dim, n_heads=num_heads,
                    n_outputs=embedding_dim // num_heads,
                    n_anchor_pairs=n_anchor_pairs_attn, tables_per_head=tables_per_head_value,
                    smooth_mode=smooth_mode, device=dev,
                    connected_anchors_mode=connected_anchors_mode, random_seed=random_seed,
                )
                value_lut.projection.weights.copy_(
                    torch.randn(value_lut.projection.weights.shape, device=dev) * 0.001
                )
                ffn_lut = MultiHeadLut(
                    input_dim=embedding_dim, n_heads=1, n_outputs=embedding_dim,
                    n_anchor_pairs=n_anchor_pairs_ffn, tables_per_head=ffn_tables,
                    smooth_mode=smooth_mode, device=dev,
                    connected_anchors_mode=connected_anchors_mode, random_seed=random_seed,
                )
                ffn_lut.projection.weights.copy_(
                    torch.randn(ffn_lut.projection.weights.shape, device=dev) * 0.001
                )
                self.layers.append(nn.ModuleDict({
                    "cross_attn": cross_attn, "value_lut": value_lut,
                    "attn_dropout": nn.Dropout(dropout),
                    "ffn": ffn_lut, "ffn_dropout": nn.Dropout(dropout),
                }))

    def forward(self, tokens):
        B, S = tokens.shape
        with torch.no_grad():
            z = self.token_embedder(tokens)
        for layer in self.layers:
            attn_weights = layer["cross_attn"](z, z)
            v = layer["value_lut"](z.reshape(-1, self.embedding_dim))
            v = v.reshape(B, S, self.num_heads, -1)
            attn_out = (attn_weights.permute(0, 3, 1, 2) @ v.permute(0, 2, 1, 3)
                        ).permute(0, 2, 1, 3).reshape(B, S, self.embedding_dim)
            z = z + layer["attn_dropout"](attn_out)
            z = z + layer["ffn_dropout"](
                layer["ffn"](z.reshape(-1, self.embedding_dim)).reshape(B, S, -1)
            )
        z_norm = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        return z_norm @ self.token_embedder.weight.T / UNEMBED_TEMPERATURE


def _load_checkpoint_into_lut(lut_transformer, ckpt):
    def _to_dev(t, dev):
        return t.to(device=dev, dtype=t.dtype)

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
                _to_dev(ckpt.v[l]["anchors_a"].reshape(-1, n_c), dev))
            vl.lookup.anchor_pairs_b.copy_(
                _to_dev(ckpt.v[l]["anchors_b"].reshape(-1, n_c), dev))
            Sv = ckpt.v[l]["S"]
            vl.projection.weights.copy_(
                _to_dev(Sv.permute(2, 0, 1, 3).reshape(-1, Sv.shape[1], Sv.shape[3]), dev))

            attn_lut = layer["cross_attn"].multi_head_lut
            attn_lut.lookup.anchor_pairs_a.copy_(
                _to_dev(ckpt.a[l]["anchors_a"].reshape(-1, ckpt.attention_a_n_c), dev))
            attn_lut.lookup.anchor_pairs_b.copy_(
                _to_dev(ckpt.a[l]["anchors_b"].reshape(-1, ckpt.attention_a_n_c), dev))
            Sa = ckpt.a[l]["S"]
            attn_lut.projection.weights.copy_(
                _to_dev(Sa.permute(2, 0, 1, 3).reshape(-1, Sa.shape[1], Sa.shape[3]), dev))


def _assert_weights_equal_before_sync(gt, lut, atol=1e-5, rtol=1e-4, msg_prefix=""):
    ckpt_gt = SpikeQKCheckpoint.from_gt(gt)
    dev = next(lut.parameters()).device
    n_heads = ckpt_gt.num_heads
    n_t_v = ckpt_gt.attention_v_n_t
    n_t_a = ckpt_gt.attention_a_n_t

    failures = []

    def _check(actual, expected, name):
        actual = actual.detach().to(device=dev)
        expected = expected.to(device=dev, dtype=actual.dtype)
        try:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol,
                                       msg=f"{msg_prefix}{name}")
        except AssertionError as e:
            failures.append(f"{name}: {e}")

    for l in range(ckpt_gt.num_layers):
        layer = lut.layers[l]
        _check(layer["ffn"].projection.weights, ckpt_gt.ffn[l]["S"], f"ffn[{l}].S")

        vl = layer["value_lut"]
        Sv = ckpt_gt.v[l]["S"]
        _check(vl.projection.weights.reshape(n_heads, n_t_v, Sv.shape[1], Sv.shape[3])
               .permute(1, 2, 0, 3), Sv, f"v[{l}].S")

        attn_lut = layer["cross_attn"].multi_head_lut
        Sa = ckpt_gt.a[l]["S"]
        _check(attn_lut.projection.weights.reshape(n_heads, n_t_a, Sa.shape[1], Sa.shape[3])
               .permute(1, 2, 0, 3), Sa, f"a[{l}].S")

    if failures:
        raise AssertionError(
            f"{msg_prefix}weight mismatch ({len(failures)} check(s) failed):\n  "
            + "\n  ".join(failures)
        )


def _make_gt_and_lut(dev):
    torch.manual_seed(SEED)
    gt = GT_spike_QK_Transformer(
        device=dev, context_size=CONTEXT_SIZE, vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        positional_buckets_a=POSITIONAL_BUCKETS_A,
        attention_a_n_t=ATTENTION_A_N_T, attention_a_n_c=ATTENTION_A_N_C,
        attention_v_n_t=ATTENTION_V_N_T, attention_v_n_c=ATTENTION_V_N_C,
        attention_temperature=ATTENTION_TEMPERATURE,
        include_ffn=True, ffn_n_t=FFN_N_T, ffn_n_c=FFN_N_C,
        unembed_temperature=UNEMBED_TEMPERATURE,
        smooth_forward=True, noise_jitter_scale=0.0,
    )
    ckpt = SpikeQKCheckpoint.from_gt(gt)
    lut = LUTTransformer(
        vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
        context_size=CONTEXT_SIZE, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        n_anchor_pairs_attn=ATTENTION_A_N_C, n_anchor_pairs_ffn=FFN_N_C,
        n_positional_buckets=POSITIONAL_BUCKETS_A,
        tables_per_head_attn=ATTENTION_A_N_T, tables_per_head_value=ATTENTION_V_N_T,
        ffn_tables=FFN_N_T, dropout=0.0, smooth_mode=True, device=dev,
        connected_anchors_mode=False, random_seed=123,
    )
    _load_checkpoint_into_lut(lut, ckpt)
    return gt, lut, ckpt


def test_gt_and_lut_transformer_forward_match(device):
    """GT and LUT transformer forward outputs match after loading the same checkpoint."""
    dev = torch.device(device)
    gt, lut, _ = _make_gt_and_lut(dev)

    B = 4
    x = torch.randint(0, VOCAB_SIZE, (B, CONTEXT_SIZE), device=dev, dtype=torch.long)
    gt.set_batch_size(B)

    gt_logits = gt.forward(x, training=False).view(B, CONTEXT_SIZE, VOCAB_SIZE)
    lut_logits = lut(x)

    torch.testing.assert_close(lut_logits, gt_logits, atol=1e-4, rtol=1e-3)


def test_gt_and_lut_transformer_backward_match(device):
    """
    GT and LUT transformer backward: run 10 training iterations, synchronize weights
    via SpikeQKCheckpoint after each step, and verify forward outputs still match.
    """
    dev = torch.device(device)
    gt, lut, _ = _make_gt_and_lut(dev)
    lut.train()

    B = 4
    lr = 0.01
    lut_optimizer = torch.optim.SGD(lut.parameters(), lr=lr)

    for i in range(10):
        x = torch.randint(0, VOCAB_SIZE, (B, CONTEXT_SIZE), device=dev, dtype=torch.long)
        target = torch.randint(0, VOCAB_SIZE, (B, CONTEXT_SIZE), device=dev, dtype=torch.long)

        gt.set_batch_size(B)
        gt_logits = gt.forward(x, training=True)
        lut_logits = lut(x)

        torch.testing.assert_close(
            lut_logits, gt_logits.view(B, CONTEXT_SIZE, VOCAB_SIZE),
            atol=1e-4, rtol=1e-3,
            msg=f"iter {i}: forward mismatch before backward",
        )

        probs = F.softmax(gt_logits, dim=-1)
        one_hot = F.one_hot(target.view(-1), num_classes=VOCAB_SIZE).float().to(
            device=dev, dtype=gt_logits.dtype)
        gt.output.copy_(probs - one_hot)
        gt.backward(lr)

        lut_optimizer.zero_grad()
        F.cross_entropy(lut_logits.reshape(-1, VOCAB_SIZE), target.view(-1),
                        reduction="sum").backward()
        lut_optimizer.step()

        _assert_weights_equal_before_sync(gt, lut, atol=1e-5, rtol=1e-4,
                                          msg_prefix=f"iter {i}: ")
        _load_checkpoint_into_lut(lut, SpikeQKCheckpoint.from_gt(gt))
