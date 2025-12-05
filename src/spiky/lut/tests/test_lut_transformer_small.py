import os
import random
import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm

from spiky.lut.LUTLayer import GradientPolicy, GradientType, SynapseMeta, LUTSharedContext

from spiky.lut.LUTTransformer import LUTTransformer
from spiky.util.text_snippet_sampler import TextSnippetSampler
from gt_lut_transformer import _GTLUTTransformer


def test_lut_transformer_small(
    device, summation_dtype, seed=123
):
    for g_type in [GradientType.Dense, GradientType.Sparse, GradientType.Internal]:
        if g_type == GradientType.Internal and summation_dtype == torch.int32:
            continue
        for use_multi_lut in [False, True]:
            if use_multi_lut and summation_dtype == torch.int32:
                continue
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
                        gradient_type=g_type,
                        summation_dtype=summation_dtype,
                        device=device,
                        seed=seed,
                        batch_size=batch_size,
                        use_multi_lut=use_multi_lut,
                        train_or_eval=train_or_eval
                    )
                    if not success:
                        return False
    return True


def synchronize_models(pytorch_transformer, gt_transformer, use_multi_lut, num_layers, num_heads, vocab_size):
    """
    Synchronize all weights, anchors, and embeddings from PyTorch LUTTransformer to GT model.
    
    Args:
        pytorch_transformer: PyTorch LUTTransformer instance
        gt_transformer: GT _GTLUTTransformer instance
        use_multi_lut: Whether PyTorch model uses MultiLUT
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        vocab_size: Vocabulary size
    """
    def sync_luts(pytorch_lut, gt_lut, gt_positional_encoding=None):
        """
        Synchronize anchors, weights, and positional embeddings from PyTorch LUTLayer to GT model.
        
        Args:
            pytorch_lut: PyTorch LUTLayer instance
            gt_lut: GT LUT instance
            gt_positional_encoding: Optional GT PositionalEncoding instance (for attention layers)
        """
        anchors_tensor = pytorch_lut._detector_anchors.cpu().detach()
        n_detectors = pytorch_lut.n_detectors()
        n_anchors_per_detector = pytorch_lut.n_anchors_per_detector()
        
        # Format: [detector0_a0, detector0_b0, detector0_a1, detector0_b1, ..., detector1_a0, ...]
        for det_idx in range(n_detectors):
            offset = det_idx * n_anchors_per_detector * 2
            a_list = []
            b_list = []
            for anc_idx in range(n_anchors_per_detector):
                a_list.append(int(anchors_tensor[offset + anc_idx * 2].item()))
                b_list.append(int(anchors_tensor[offset + anc_idx * 2 + 1].item()))
            gt_lut.anchors[det_idx] = {'a': a_list, 'b': b_list}
        
        weights_tensor = pytorch_lut._weights.cpu().detach()
        n_outputs = pytorch_lut.n_outputs()
        table_size = pytorch_lut.n_lookup_neurons() // n_detectors
        weights_per_detector = table_size * n_outputs
        
        # PyTorch weights are organized as: [detector0_table0_output0, detector0_table0_output1, ..., 
        # detector0_table0_output(n_outputs-1), detector0_table1_output0, ..., detector0_table(table_size-1)_output(n_outputs-1),
        # detector1_table0_output0, ...]
        for det_idx in range(n_detectors):
            offset = det_idx * weights_per_detector
            gt_lut.S[det_idx] = weights_tensor[offset:offset + weights_per_detector].tolist()
        
        # Copy positional embeddings (if provided)
        if gt_positional_encoding is not None:
            pos_emb_tensor = pytorch_lut._positional_embeddings.cpu().detach()
            positional_dim = pytorch_lut._positional_dim
            sequence_length = pytorch_lut.sequence_length()
            
            # PyTorch format: flat tensor of shape [(sequence_length - 1) * n_detectors * positional_dim]
            # GT format: [context_size - 1][n_t][positional_dim]
            n_positions = sequence_length - 1
            elements_per_position = n_detectors * positional_dim
            
            for pos_idx in range(n_positions):
                offset = pos_idx * elements_per_position
                pos_data = pos_emb_tensor[offset:offset + elements_per_position]
                # Reshape: [n_detectors * positional_dim] -> [n_detectors][positional_dim]
                for det_idx in range(n_detectors):
                    det_offset = det_idx * positional_dim
                    gt_positional_encoding.encodings[pos_idx][det_idx] = pos_data[det_offset:det_offset + positional_dim].tolist()
    
    # Copy token embedding weights
    token_embeddings = pytorch_transformer.token_embedder.weight.cpu().detach()
    for token_idx in range(vocab_size):
        gt_transformer.token_embedder.embeddings[token_idx] = token_embeddings[token_idx].tolist()
    
    # Synchronize all layers
    # FFN layers
    for layer_idx in range(num_layers):
        sync_luts(
            pytorch_transformer.layers[layer_idx]['ffn'],
            gt_transformer.layers[layer_idx]['ffn']
        )
        
        # Attention layers
        if use_multi_lut:
            # MultiLUT case: sync each head separately
            for head_idx in range(num_heads):
                sync_luts(
                    pytorch_transformer.layers[layer_idx]['attention_lut'].luts[head_idx],
                    gt_transformer.layers[layer_idx]['heads'][head_idx].V,
                    gt_transformer.layers[layer_idx]['heads'][head_idx].positional_encoding
                )
        else:
            # Single LUT case: sync to first head's V
            sync_luts(
                pytorch_transformer.layers[layer_idx]['attention_lut'],
                gt_transformer.layers[layer_idx]['heads'][0].V,
                gt_transformer.layers[layer_idx]['heads'][0].positional_encoding
            )
    
    # Unembedder
    sync_luts(
        pytorch_transformer.unembedder,
        gt_transformer.unembedder
    )


def compare_weights_and_positional_embeddings(
    pytorch_transformer, gt_transformer, use_multi_lut, num_layers, num_heads,
    summation_dtype
):
    result = True
    if summation_dtype == torch.int32:
        eps = 1e-3
    else:
        eps = 1e-4

    # Token embeddings
    pytorch_token_embeddings = pytorch_transformer.token_embedder.weight.cpu().detach()
    gt_token_embeddings = torch.tensor(
        gt_transformer.token_embedder.embeddings,
        dtype=torch.float32
    )
    if pytorch_token_embeddings.shape != gt_token_embeddings.shape:
        print(f"❌ Train mode: Token embeddings shape mismatch: {pytorch_token_embeddings.shape} vs {gt_token_embeddings.shape}")
        result = False
    elif not torch.allclose(pytorch_token_embeddings, gt_token_embeddings, atol=eps, rtol=eps):
        max_diff = torch.max(torch.abs(pytorch_token_embeddings - gt_token_embeddings))
        print(f"❌ Train mode: Token embeddings differ. Max diff: {max_diff:.6f}")
        result = False
    
    # Unembedder weights
    pytorch_unembed_weights = pytorch_transformer.unembedder._weights.cpu().detach()
    gt_unembed_weights = torch.tensor(
        [w for table in gt_transformer.unembedder.S for w in table],
        dtype=torch.float32
    )
    if pytorch_unembed_weights.shape != gt_unembed_weights.shape:
        print(f"❌ Train mode: Unembedder weights shape mismatch: {pytorch_unembed_weights.shape} vs {gt_unembed_weights.shape}")
        result = False
    if not torch.allclose(pytorch_unembed_weights, gt_unembed_weights, atol=eps, rtol=eps):
        max_diff = torch.max(torch.abs(pytorch_unembed_weights - gt_unembed_weights))
        print(f"❌ Train mode: Unembedder weights differ. Max diff: {max_diff:.6f}")
        result = False

    # Compare LUT weights in each layer
    for layer_idx in reversed(range(num_layers)):
        # FFN weights
        pytorch_ffn_weights = pytorch_transformer.layers[layer_idx]['ffn']._weights.cpu().detach()
        gt_ffn_weights = torch.tensor(
            [w for table in gt_transformer.layers[layer_idx]['ffn'].S for w in table],
            dtype=torch.float32
        )
        if pytorch_ffn_weights.shape != gt_ffn_weights.shape:
            print(f"❌ Train mode: Layer {layer_idx} FFN weights shape mismatch: {pytorch_ffn_weights.shape} vs {gt_ffn_weights.shape}")
            result = False
        if not torch.allclose(pytorch_ffn_weights, gt_ffn_weights, atol=eps, rtol=eps):
            max_diff = torch.max(torch.abs(pytorch_ffn_weights - gt_ffn_weights))
            print(f"❌ Train mode: Layer {layer_idx} FFN weights differ. Max diff: {max_diff:.6f}")
            result = False

        # Attention head weights
        for head_idx in range(num_heads if use_multi_lut else 1):
            if use_multi_lut:
                pytorch_lut = pytorch_transformer.layers[layer_idx]['attention_lut'].luts[head_idx]
            else:
                pytorch_lut = pytorch_transformer.layers[layer_idx]['attention_lut']

            pytorch_head_weights = pytorch_lut._weights.cpu().detach()
            gt_head_weights = torch.tensor(
                [w for table in gt_transformer.layers[layer_idx]['heads'][head_idx].V.S for w in table],
                dtype=torch.float32
            )
            if pytorch_head_weights.shape != gt_head_weights.shape:
                print(f"❌ Train mode: Layer {layer_idx} Head {head_idx} weights shape mismatch: {pytorch_head_weights.shape} vs {gt_head_weights.shape}")
                result = False
            if not torch.allclose(pytorch_head_weights, gt_head_weights, atol=eps, rtol=eps):
                max_diff = torch.max(torch.abs(pytorch_head_weights - gt_head_weights))
                print(f"❌ Train mode: Layer {layer_idx} Head {head_idx} weights differ. Max diff: {max_diff:.6f}")
                result = False

            # Positional encodings
            pytorch_pos_emb = pytorch_lut._positional_embeddings
            if pytorch_pos_emb is not None:
                pytorch_pos_emb = pytorch_pos_emb.cpu().detach()
                gt_pos_emb = torch.tensor(
                    [w for pos_enc in gt_transformer.layers[layer_idx]['heads'][head_idx].positional_encoding.encodings for enc in pos_enc for w in enc],
                    dtype=torch.float32
                )
                if pytorch_pos_emb.shape != gt_pos_emb.shape:
                    print(f"❌ Train mode: Layer {layer_idx} Head {head_idx} positional embeddings shape mismatch: {pytorch_pos_emb.shape} vs {gt_pos_emb.shape}")
                    result = False
                if not torch.allclose(pytorch_pos_emb, gt_pos_emb, atol=1e-3, rtol=1e-3):
                    max_diff = torch.max(torch.abs(pytorch_pos_emb - gt_pos_emb))
                    print(f"❌ Train mode: Layer {layer_idx} Head {head_idx} positional embeddings differ. Max diff: {max_diff:.6f}")
                    result = False
    
    return result


def compare_outputs(gt_lut_transformer, y, train_or_eval, device, summation_dtype):
    batch_size = y.shape[0]
    if summation_dtype == torch.int32:
        eps = 1e-3
    else:
        eps = 1e-4
    # Compare outputs for all batch items
    for i in range(batch_size):
        # Convert GT output to tensor: (context_size, vocab_size)
        gt_output = torch.tensor(gt_lut_transformer.output[i], dtype=torch.float32, device=device)

        # Compare with PyTorch output: y[i] is (context_size, vocab_size)
        # Note: GT model outputs logits, PyTorch model also outputs logits
        # We compare with some tolerance since implementations may differ slightly
        if not torch.allclose(y[i], gt_output, atol=eps, rtol=eps):
            max_diff = torch.max(torch.abs(y[i] - gt_output))
            print(f"❌ {train_or_eval.capitalize()} mode: Batch item {i} outputs differ. Max diff: {max_diff:.6f}")
            return False
    return True


def _test_lut_transformer_small(
    vocab_size, embedding_dim,
    context_size, positional_dim,
    num_layers, num_heads, n_detectors,
    n_anchors_per_detector, gradient_type,
    summation_dtype, device, seed,
    batch_size, use_multi_lut, train_or_eval
):
    torch.manual_seed(seed)
    print('Test configuration:')
    print(f'  Vocab size: {vocab_size}')
    print(f'  Embedding dim: {embedding_dim}')
    print(f'  Context size: {context_size}')
    print(f'  Positional dim: {positional_dim}')
    print(f'  Num layers: {num_layers}')
    print(f'  Num heads: {num_heads}')
    print(f'  N detectors: {n_detectors}')
    print(f'  N anchors per detector: {n_anchors_per_detector}')
    print(f'  Gradient type: {gradient_type}')
    print(f'  Summation dtype: {summation_dtype}')
    print(f'  Device: {device}')
    print(f'  Seed: {seed}')
    print(f'  Batch size: {batch_size}')
    print(f'  Use multi LUT: {use_multi_lut}')
    print(f'  Train or eval: {train_or_eval}')
    print('=' * 60)

    snippet_sampler = TextSnippetSampler('../../../../workbooks/tinyshakespeare.txt', context_size + 1, 100, device)

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
        _synapse_meta=SynapseMeta(
            min_weight=-1.0, max_weight=1.0,
            initial_weight=-1.0, initial_noise_level=2.0
        ),
        summation_dtype=summation_dtype,
        _int_rescaler=1.0,
        weights_gradient_policy=GradientPolicy(gradient_type),
        device=device, seed=seed
    )

    # Create _GTLUTTransformer with matching parameters
    random.seed(seed)
    if use_multi_lut:
        gt_lut_transformer = _GTLUTTransformer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_size=context_size,
            positional_dim=positional_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            n_t=n_detectors,
            n_t_a=n_detectors,
            n_c=n_anchors_per_detector,
            batch_size=batch_size
        )
    else:
        gt_lut_transformer = _GTLUTTransformer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_size=context_size,
            positional_dim=positional_dim,
            num_layers=num_layers,
            num_heads=1,
            n_t=n_detectors,
            n_t_a=n_detectors * num_heads,
            n_c=n_anchors_per_detector,
            batch_size=batch_size
        )

    # Synchronize entire models
    synchronize_models(lut_transformer, gt_lut_transformer, use_multi_lut, num_layers, num_heads, vocab_size)

    # Compare weights after synchronization
    if not compare_weights_and_positional_embeddings(
        lut_transformer, gt_lut_transformer, use_multi_lut, num_layers, num_heads,
        summation_dtype
    ):
        print(f"❌ something is wrong after synchronization №1")
        return False

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
    
    if not compare_outputs(gt_lut_transformer, y, train_or_eval, device, summation_dtype):
        print(f"❌ something is wrong after forward pass №1")
        return False

    # If train mode, perform backward pass and compare weights
    if train_or_eval == 'train':
        learning_rate = 0.01  # Use a fixed learning rate for testing
        
        # Set up learning rate hook for PyTorch model
        def lr_hook(_):
            return learning_rate
        if gradient_type == GradientType.Internal:
            lut_transformer.set_external_learning_rate_hook(lr_hook)
        opt = SGD([p for p in lut_transformer.parameters() if p.requires_grad], lr=learning_rate)

        for i in tqdm(range(32)):
            # PyTorch model backward pass
            # Compute loss: cross-entropy with target tokens
            targets = x[:, 1:context_size + 1].to(torch.long)  # (batch_size, context_size)
            loss = nn.functional.cross_entropy(
                y.reshape(-1, vocab_size),  # (batch_size * context_size, vocab_size)
                targets.reshape(-1),  # (batch_size * context_size,)
                reduction='none'
            ).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

            gt_lut_transformer.training_step(learning_rate, no_forward=True)

            # Compare weights after backward
            if not compare_weights_and_positional_embeddings(
                lut_transformer, gt_lut_transformer, use_multi_lut, num_layers, num_heads,
                summation_dtype
            ):
                print(f"❌ something is wrong after backward pass №{i + 1}")
                return False

            synchronize_models(lut_transformer, gt_lut_transformer, use_multi_lut, num_layers, num_heads, vocab_size)
            # Compare weights after backward
            if not compare_weights_and_positional_embeddings(
                lut_transformer, gt_lut_transformer, use_multi_lut, num_layers, num_heads,
                summation_dtype
            ):
                print(f"❌ something is wrong after synchronization №{i + 2}")
                return False

            x = snippet_sampler.sample_training_batch(batch_size)  # (batch_size, context_size + 1)
            y = lut_transformer(x[:, :context_size])  # (batch_size, context_size, vocab_size)

            # Load all batch items into GT model
            batch_tokens = [x[j].cpu().tolist() for j in range(batch_size)]
            gt_lut_transformer.load_snippet(batch_tokens)
            # Run forward pass
            gt_lut_transformer.forward()

            if not compare_outputs(gt_lut_transformer, y, train_or_eval, device, summation_dtype):
                print(f"❌ something is wrong after forward pass №{i + 2}")
                return False

    return True


def main():
    print("=" * 60)
    print("LUTTransformer SMALL TEST")
    print("=" * 60)

    devices = []
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
