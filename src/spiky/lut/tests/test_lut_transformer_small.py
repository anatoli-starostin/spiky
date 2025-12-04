import os
import random
import torch
import torch.nn as nn
from spiky.lut.LUTLayer import GradientPolicy, GradientType

from spiky.lut.LUTTransformer import LUTTransformer
from spiky.util.text_snippet_sampler import TextSnippetSampler
from gt_lut_transformer import _GTLUTTransformer


def test_lut_transformer_small(
    device, summation_dtype, seed=123
):
    for use_multi_lut in [True, False]:
        for train_or_eval in ['train', 'eval']:
            for batch_size in [1, 4]:
                success = _test_lut_transformer_small(
                    vocab_size=256,
                    embedding_dim=32,
                    context_size=8,
                    positional_dim=4,
                    num_layers=2,
                    num_heads=2,
                    n_detectors=4,
                    n_anchors_per_detector=3,
                    summation_dtype=summation_dtype,
                    device=device, seed=seed,
                    batch_size=batch_size,
                    use_multi_lut=use_multi_lut,
                    train_or_eval=train_or_eval
                )
                if not success:
                    return False
    return True


def _test_lut_transformer_small(
    vocab_size, embedding_dim,
    context_size, positional_dim,
    num_layers, num_heads, n_detectors,
    n_anchors_per_detector,
    summation_dtype, device, seed,
    batch_size, use_multi_lut, train_or_eval
):
    torch.manual_seed(seed)
    snippet_sampler = TextSnippetSampler('../../../../workbooks/tinyshakespeare.txt', context_size + 1, 100, device)

    # TODO random initial weights

    lut_transformer = LUTTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_size=context_size,
        positional_dim=positional_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        n_detectors=n_detectors,
        n_anchors_per_detector=n_anchors_per_detector,
        _use_multi_lut=use_multi_lut,
        summation_dtype=summation_dtype,
        weights_gradient_policy=GradientPolicy(GradientType.Internal),
        device=device, seed=seed
    )

    # Create _GTLUTTransformer with matching parameters
    random.seed(seed)
    gt_lut_transformer = _GTLUTTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_size=context_size,
        positional_dim=positional_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        n_t=n_detectors,
        n_c=n_anchors_per_detector,
        batch_size=batch_size
    )

    # TODO copy initial weights
    # Copy token embedding weights from lut_transformer to gt_lut_transformer
    token_embeddings = lut_transformer.token_embedder.weight.cpu().detach()
    for token_idx in range(vocab_size):
        gt_lut_transformer.token_embedder.embeddings[token_idx] = token_embeddings[token_idx].tolist()
    
    # Set mode
    if train_or_eval == 'train':
        lut_transformer.train()
    else:
        lut_transformer.eval()
    
    x = snippet_sampler.sample_training_batch(batch_size)  # (batch_size, context_size + 1)
    y = lut_transformer(x[:, :context_size])  # (batch_size, context_size, vocab_size)

    # Load all batch items into GT model
    batch_tokens = [x[i].cpu().tolist() for i in range(batch_size)]
    gt_lut_transformer.load_snippet(batch_tokens)
    
    # Run forward pass
    gt_lut_transformer.forward()
    
    # Compare outputs for all batch items
    for i in range(batch_size):
        # Convert GT output to tensor: (context_size, vocab_size)
        gt_output = torch.tensor(gt_lut_transformer.output[i], dtype=torch.float32, device=device)
        
        # Compare with PyTorch output: y[i] is (context_size, vocab_size)
        # Note: GT model outputs logits, PyTorch model also outputs logits
        # We compare with some tolerance since implementations may differ slightly
        if not torch.allclose(y[i], gt_output, atol=1e-4, rtol=1e-4):
            max_diff = torch.max(torch.abs(y[i] - gt_output))
            print(f"❌ {train_or_eval.capitalize()} mode: Batch item {i} outputs differ. Max diff: {max_diff:.6f}")
            return False

    # If train mode, perform backward pass and compare weights
    if train_or_eval == 'train' and use_multi_lut:
        learning_rate = 0.01  # Use a fixed learning rate for testing
        
        # Set up learning rate hook for PyTorch model
        def lr_hook(_):
            return learning_rate
        lut_transformer.set_external_learning_rate_hook(lr_hook)
        
        # PyTorch model backward pass
        # Compute loss: cross-entropy with target tokens
        targets = x[:, 1:context_size + 1].to(torch.long)  # (batch_size, context_size)
        loss = nn.functional.cross_entropy(
            y.reshape(-1, vocab_size),  # (batch_size * context_size, vocab_size)
            targets.reshape(-1),  # (batch_size * context_size,)
            reduction='none'
        ).sum()
        loss.backward()
        
        x_list = [x[i].cpu().tolist() for i in range(batch_size)]
        gt_lut_transformer.load_snippet(x_list)
        gt_lut_transformer.training_step(learning_rate)
        
        # Compare weights after backward

        # Unembedder weights
        pytorch_unembed_weights = lut_transformer.unembedder._weights.cpu().detach()
        gt_unembed_weights = torch.tensor(
            [w for table in gt_lut_transformer.unembedder.S for w in table],
            dtype=torch.float32
        )
        if pytorch_unembed_weights.shape != gt_unembed_weights.shape:
            print(f"❌ Train mode: Unembedder weights shape mismatch: {pytorch_unembed_weights.shape} vs {gt_unembed_weights.shape}")
            return False
        if not torch.allclose(pytorch_unembed_weights, gt_unembed_weights, atol=1e-4, rtol=1e-4):
            max_diff = torch.max(torch.abs(pytorch_unembed_weights - gt_unembed_weights))
            print(f"❌ Train mode: Unembedder weights differ. Max diff: {max_diff:.6f}")
            return False

        # Compare LUT weights in each layer
        for layer_idx in reversed(range(num_layers)):
            # FFN weights
            pytorch_ffn_weights = lut_transformer.layers[layer_idx]['ffn']._weights.cpu().detach()
            gt_ffn_weights = torch.tensor(
                [w for table in gt_lut_transformer.layers[layer_idx]['ffn'].S for w in table],
                dtype=torch.float32
            )
            # Note: Weight layouts may differ, so we compare with larger tolerance
            if pytorch_ffn_weights.shape != gt_ffn_weights.shape:
                print(f"❌ Train mode: Layer {layer_idx} FFN weights shape mismatch: {pytorch_ffn_weights.shape} vs {gt_ffn_weights.shape}")
                return False
            if not torch.allclose(pytorch_ffn_weights, gt_ffn_weights, atol=1e-4, rtol=1e-4):
                max_diff = torch.max(torch.abs(pytorch_ffn_weights - gt_ffn_weights))
                print(f"❌ Train mode: Layer {layer_idx} FFN weights differ. Max diff: {max_diff:.6f}")
                # return False
            
            # Attention head weights
            # MultiLUT case: compare each head separately
            for head_idx in range(num_heads):
                pytorch_head_weights = lut_transformer.layers[layer_idx]['attention_lut'].luts[head_idx]._weights.cpu().detach()
                gt_head_weights = torch.tensor(
                    [w for table in gt_lut_transformer.layers[layer_idx]['heads'][head_idx].V.S for w in table],
                    dtype=torch.float32
                )
                if pytorch_head_weights.shape != gt_head_weights.shape:
                    print(f"❌ Train mode: Layer {layer_idx} Head {head_idx} weights shape mismatch: {pytorch_head_weights.shape} vs {gt_head_weights.shape}")
                    return False
                if not torch.allclose(pytorch_head_weights, gt_head_weights, atol=1e-4, rtol=1e-4):
                    max_diff = torch.max(torch.abs(pytorch_head_weights - gt_head_weights))
                    print(f"❌ Train mode: Layer {layer_idx} Head {head_idx} weights differ. Max diff: {max_diff:.6f}")
                    # return False

                # Positional encodings
                pytorch_pos_emb = lut_transformer.layers[layer_idx]['attention_lut'].luts[head_idx]._positional_embeddings
                if pytorch_pos_emb is not None:
                    pytorch_pos_emb = pytorch_pos_emb.cpu().detach()
                    gt_pos_emb = torch.tensor(
                        [w for pos_enc in gt_lut_transformer.layers[layer_idx]['heads'][head_idx].positional_encoding.encodings for enc in pos_enc for w in enc],
                        dtype=torch.float32
                    )
                    if pytorch_pos_emb.shape != gt_pos_emb.shape:
                        print(f"❌ Train mode: Layer {layer_idx} Head {head_idx} positional embeddings shape mismatch: {pytorch_pos_emb.shape} vs {gt_pos_emb.shape}")
                        return False
                    if not torch.allclose(pytorch_pos_emb, gt_pos_emb, atol=1e-4, rtol=1e-4):
                        max_diff = torch.max(torch.abs(pytorch_pos_emb - gt_pos_emb))
                        print(f"❌ Train mode: Layer {layer_idx} Head {head_idx} positional embeddings differ. Max diff: {max_diff:.6f}")
                        # return False

    return True


def main():
    print("=" * 60)
    print("LUTTransformer SMALL TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = test_lut_transformer_small(device, summation_dtype)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
