import random
import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm

from spiky.lut.LUTLayer import GradientPolicy, GradientType, SynapseMeta

from spiky.lut.LUTTransformer import LUTTransformer
from spiky.lut.tests.gt_lut_product import GTLUTProductTransformer
from spiky.util.text_snippet_sampler import TextSnippetSampler


def test_lut_transformer_product(
    device, summation_dtype, seed=None
):
    for g_type in [GradientType.Dense, GradientType.Sparse, GradientType.Internal]:
        if g_type == GradientType.Internal and summation_dtype == torch.int32:
            continue
        for fully_connected in [False, True]:
            for train_or_eval in ['eval']:  # , 'train'
                for batch_size in [1, 4]:
                    for sliced_mode in [True]:
                        if not sliced_mode and not fully_connected:
                            continue
                        success = _test_lut_transformer_product(
                            vocab_size=256,
                            embedding_dim=32,
                            context_size=8,
                            positional_dim=32 if sliced_mode else 4,
                            num_layers=1,
                            num_heads=2,
                            n_detectors=4,
                            n_anchors_per_detector=3,
                            gradient_type=g_type,
                            summation_dtype=summation_dtype,
                            device=device,
                            seed=seed,
                            batch_size=batch_size,
                            fully_connected=fully_connected,
                            train_or_eval=train_or_eval,
                            sliced_mode=sliced_mode
                        )
                        if not success:
                            return False
    return True


def synchronize_models(pytorch_transformer, gt_transformer, num_layers):
    """
    Synchronize all weights, anchors, and embeddings from PyTorch LUTTransformer to GT model.
    
    Args:
        pytorch_transformer: PyTorch LUTTransformer instance (with concatenation_product=False)
        gt_transformer: GT GTLUTProductTransformer instance
        num_layers: Number of transformer layers
    """
    def sync_luts(pytorch_lut, gt_lut, gt_product=None):
        """
        Synchronize anchors, weights, and positional embeddings from PyTorch LUTLayer to GT LUTLayer.
        
        Args:
            pytorch_lut: PyTorch LUTLayer instance
            gt_lut: GT LUTLayer instance
            gt_product: Optional GTLUTProduct instance (if syncing attention layer with positional embeddings)
        """
        # Copy anchors
        with torch.no_grad():
            anchors_tensor = pytorch_lut._detector_anchors.detach().to(gt_lut._detector_anchors.device)
            gt_lut._detector_anchors.copy_(anchors_tensor)
        
        # Copy weights
        with torch.no_grad():
            weights_tensor = pytorch_lut._weights.detach().to(gt_lut._weights.device)
            gt_lut._weights.copy_(weights_tensor)
        
        # Copy positional embeddings if both have them
        if gt_product is not None:
            if pytorch_lut._positional_embeddings is not None and gt_product.positional_embeddings is not None:
                with torch.no_grad():
                    pytorch_pos_emb = pytorch_lut._positional_embeddings.detach().to(
                        gt_product.positional_embeddings.device
                    )
                    gt_product.positional_embeddings.copy_(pytorch_pos_emb.reshape(gt_product.positional_embeddings.shape))
    
    # Copy token embedding weights
    with torch.no_grad():
        token_embeddings = pytorch_transformer.token_embedder.weight.detach().to(
            gt_transformer.token_embedder.weight.device
        )
        gt_transformer.token_embedder.weight.copy_(token_embeddings)
    
    # Synchronize all layers
    for layer_idx in range(num_layers):
        # FFN layers
        sync_luts(
            pytorch_transformer.layers[layer_idx]['ffn'],
            gt_transformer.layers[layer_idx]['ffn']
        )
        
        # Attention layers: sync internal LUTLayer
        pytorch_attention_lut = pytorch_transformer.layers[layer_idx]['attention_lut']
        gt_attention_product = gt_transformer.layers[layer_idx]['attention_lut']
        
        # Single LUTLayer: sync to single GT head
        sync_luts(
            pytorch_attention_lut,
            gt_attention_product.lut_layer,
            gt_product=gt_attention_product
        )
    
    # Unembedder
    sync_luts(
        pytorch_transformer.unembedder,
        gt_transformer.unembedder
    )


def compare_weights_and_positional_embeddings(
    pytorch_transformer, gt_transformer, num_layers
):
    result = True
    eps = 1e-3

    # Token embeddings
    pytorch_token_embeddings = pytorch_transformer.token_embedder.weight.cpu().detach()
    gt_token_embeddings = gt_transformer.token_embedder.weight.cpu().detach()
    if pytorch_token_embeddings.shape != gt_token_embeddings.shape:
        print(f"❌ Token embeddings shape mismatch: {pytorch_token_embeddings.shape} vs {gt_token_embeddings.shape}")
        result = False
    elif not torch.allclose(pytorch_token_embeddings, gt_token_embeddings, atol=eps, rtol=eps):
        max_diff = torch.max(torch.abs(pytorch_token_embeddings - gt_token_embeddings))
        print(f"❌ Token embeddings differ. Max diff: {max_diff:.6f}")
        result = False
    
    # Unembedder weights
    pytorch_unembed_weights = pytorch_transformer.unembedder._weights.cpu().detach()
    gt_unembed_weights = gt_transformer.unembedder._weights.cpu().detach()
    if pytorch_unembed_weights.shape != gt_unembed_weights.shape:
        print(f"❌ Unembedder weights shape mismatch: {pytorch_unembed_weights.shape} vs {gt_unembed_weights.shape}")
        result = False
    elif not torch.allclose(pytorch_unembed_weights, gt_unembed_weights, atol=eps, rtol=eps):
        max_diff = torch.max(torch.abs(pytorch_unembed_weights - gt_unembed_weights))
        print(f"❌ Unembedder weights differ. Max diff: {max_diff:.6f}")
        result = False

    # Compare LUT weights in each layer
    for layer_idx in reversed(range(num_layers)):
        # FFN weights
        pytorch_ffn_weights = pytorch_transformer.layers[layer_idx]['ffn']._weights.cpu().detach()
        gt_ffn_weights = gt_transformer.layers[layer_idx]['ffn']._weights.cpu().detach()
        if pytorch_ffn_weights.shape != gt_ffn_weights.shape:
            print(f"❌ Layer {layer_idx} FFN weights shape mismatch: {pytorch_ffn_weights.shape} vs {gt_ffn_weights.shape}")
            result = False
        elif not torch.allclose(pytorch_ffn_weights, gt_ffn_weights, atol=eps, rtol=eps):
            max_diff = torch.max(torch.abs(pytorch_ffn_weights - gt_ffn_weights))
            print(f"❌ Layer {layer_idx} FFN weights differ. Max diff: {max_diff:.6f}")
            result = False

        # Attention head weights (internal LUTLayer)
        pytorch_attention_lut = pytorch_transformer.layers[layer_idx]['attention_lut']
        gt_attention_product = gt_transformer.layers[layer_idx]['attention_lut']
        
        # Single LUTLayer
        pytorch_head_weights = pytorch_attention_lut._weights.cpu().detach()
        gt_head_weights = gt_attention_product.lut_layer._weights.cpu().detach()
        if pytorch_head_weights.shape != gt_head_weights.shape:
            print(f"❌ Layer {layer_idx} Attention weights shape mismatch: {pytorch_head_weights.shape} vs {gt_head_weights.shape}")
            result = False
        elif not torch.allclose(pytorch_head_weights, gt_head_weights, atol=eps, rtol=eps):
            max_diff = torch.max(torch.abs(pytorch_head_weights - gt_head_weights))
            print(f"❌ Layer {layer_idx} Attention weights differ. Max diff: {max_diff:.6f}")
            result = False

        # Positional encodings
        pytorch_pos_emb = pytorch_attention_lut._positional_embeddings
        if pytorch_pos_emb is not None:
            pytorch_pos_emb = pytorch_pos_emb.cpu().detach()
            gt_pos_emb = gt_attention_product.positional_embeddings
            if gt_pos_emb is not None:
                gt_pos_emb = gt_pos_emb.cpu().detach()

                if not torch.allclose(pytorch_pos_emb.reshape(gt_pos_emb.shape), gt_pos_emb, atol=1e-3, rtol=1e-3):
                    max_diff = torch.max(torch.abs(pytorch_pos_emb.reshape(gt_pos_emb.shape) - gt_pos_emb))
                    print(f"❌ Layer {layer_idx} positional embeddings differ. Max diff: {max_diff:.6f}")
                    result = False
    
    return result


def compare_outputs(gt_output, pytorch_output, train_or_eval):
    """
    Compare outputs from GT and PyTorch models.
    
    Args:
        gt_output: GT model output tensor (batch_size, context_size, vocab_size)
        pytorch_output: PyTorch model output tensor (batch_size, context_size, vocab_size)
        train_or_eval: 'train' or 'eval'
    """
    batch_size = pytorch_output.shape[0]
    eps = 1e-3
    # Compare outputs for all batch items
    for i in range(batch_size):
        # Compare with PyTorch output: both are (context_size, vocab_size)
        if not torch.allclose(pytorch_output[i], gt_output[i], atol=eps, rtol=eps):
            max_diff = torch.max(torch.abs(pytorch_output[i] - gt_output[i]))
            print(f"❌ {train_or_eval.capitalize()} mode: Batch item {i} outputs differ. Max diff: {max_diff:.6f}")
            import numpy as np
            np.set_printoptions(threshold=np.inf)
            print(f"gt: {gt_output[i].cpu().detach().numpy()}")
            print(f"diff: {(pytorch_output[i] - gt_output[i]).cpu().detach().numpy()}")
            return False
    return True


def _test_lut_transformer_product(
    vocab_size, embedding_dim,
    context_size, positional_dim,
    num_layers, num_heads, n_detectors,
    n_anchors_per_detector, gradient_type,
    summation_dtype, device, seed,
    batch_size, fully_connected,
    sliced_mode, train_or_eval
):
    if seed is not None:
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
    print(f'  Fully connected: {fully_connected}')
    print(f'  Sliced mode: {sliced_mode}')
    print(f'  Train or eval: {train_or_eval}')
    print('=' * 60)

    snippet_sampler = TextSnippetSampler('../../../../workbooks/tinyshakespeare.txt', context_size + 1, 100, device)

    lut_transformer = LUTTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim if fully_connected else (1, embedding_dim),
        context_size=context_size,
        positional_dim=positional_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        n_detectors=n_detectors,
        n_anchors_per_detector=n_anchors_per_detector,
        concatenation_product=False,
        sliced_product_mode=sliced_mode,
        _synapse_meta=SynapseMeta(
            min_weight=-1.0, max_weight=1.0,
            initial_weight=-1.0, initial_noise_level=2.0
        ),
        summation_dtype=summation_dtype,
        _int_rescaler=10.0,
        weights_gradient_policy=GradientPolicy(gradient_type),
        device=device, seed=seed,
        _forward_group_size=24,
        _backward_group_size=4
    )

    # Create GTLUTProductTransformer with matching parameters
    random.seed(seed)
    gt_lut_transformer = GTLUTProductTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_size=context_size,
        positional_dim=positional_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        n_detectors=n_detectors,
        n_anchors_per_detector=n_anchors_per_detector,
        _synapse_meta=SynapseMeta(
            min_weight=-1.0, max_weight=1.0,
            initial_weight=-1.0, initial_noise_level=2.0
        ),
        summation_dtype=torch.float32,
        device=torch.device('cpu'), seed=seed,
        sliced_mode=sliced_mode
    )

    # Synchronize entire models
    synchronize_models(lut_transformer, gt_lut_transformer, num_layers)

    # Compare weights after synchronization
    if not compare_weights_and_positional_embeddings(
        lut_transformer, gt_lut_transformer, num_layers
    ):
        print(f"❌ something is wrong after synchronization №1")
        return False

    lut_transformer = lut_transformer.to(device=torch.device('cpu'))

    # Set mode
    if train_or_eval == 'train':
        lut_transformer.train()
        gt_lut_transformer.train()
    else:
        lut_transformer.eval()
        gt_lut_transformer.eval()

    x = snippet_sampler.sample_training_batch(batch_size)  # (batch_size, context_size + 1)
    y = lut_transformer(x[:, :context_size].to(device=torch.device('cpu'))).to(device=device)  # (batch_size, context_size, vocab_size)
    gt_y = gt_lut_transformer(x[:, :context_size].to(device=torch.device('cpu'))).to(device=device)  # (batch_size, context_size, vocab_size)

    if not compare_outputs(gt_y, y, train_or_eval):
        print(f"❌ something is wrong after forward pass №1")
        return False

    # If train mode, perform backward pass and compare weights
    if train_or_eval == 'train':
        learning_rate = 0.01  # Use a fixed learning rate for testing

        def lr_hook(_):
            return learning_rate
        if gradient_type == GradientType.Internal:
            lut_transformer.set_external_learning_rate_hook(lr_hook)
        opt = SGD([p for p in lut_transformer.parameters() if p.requires_grad], lr=learning_rate)
        gt_opt = SGD([p for p in gt_lut_transformer.parameters() if p.requires_grad], lr=learning_rate)
    else:
        opt = None
        gt_opt = None

    for i in tqdm(range(32)):
        # PyTorch model backward pass
        # Compute loss: cross-entropy with target tokens
        if train_or_eval == 'train':
            targets = x[:, 1:context_size + 1].to(torch.long)  # (batch_size, context_size)
            loss = nn.functional.cross_entropy(
                y.reshape(-1, vocab_size),  # (batch_size * context_size, vocab_size)
                targets.reshape(-1),  # (batch_size * context_size,)
                reduction='none'
            ).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

            # GT model backward pass
            gt_loss = nn.functional.cross_entropy(
                gt_y.reshape(-1, vocab_size),  # (batch_size * context_size, vocab_size)
                targets.reshape(-1),  # (batch_size * context_size,)
                reduction='none'
            ).sum()
            gt_opt.zero_grad()
            gt_loss.backward()
            gt_opt.step()

            # Compare weights after backward
            if not compare_weights_and_positional_embeddings(
                lut_transformer, gt_lut_transformer, num_layers
            ):
                print(f"❌ something is wrong after backward pass №{i + 1}")
                return False

        synchronize_models(lut_transformer, gt_lut_transformer, num_layers)
        # Compare weights after synchronization
        if not compare_weights_and_positional_embeddings(
            lut_transformer, gt_lut_transformer, num_layers
        ):
            print(f"❌ something is wrong after synchronization №{i + 2}")
            return False

        x = snippet_sampler.sample_training_batch(batch_size)  # (batch_size, context_size + 1)
        y = lut_transformer(x[:, :context_size].to(device=torch.device('cpu'))).to(device=device)  # (batch_size, context_size, vocab_size)
        gt_y = gt_lut_transformer(x[:, :context_size].to(device=torch.device('cpu'))).to(device=device)  # (batch_size, context_size, vocab_size)

        if not compare_outputs(gt_y, y, train_or_eval):
            print(f"pos_emb: {gt_lut_transformer.layers[0]['attention_lut'].positional_embeddings}")
            print(f"❌ something is wrong after forward pass №{i + 2}")
            return False

    return True


def main():
    print("=" * 60)
    print("LUTTransformer PRODUCT TEST")
    print("=" * 60)

    devices = []  # 'cpu'
    if torch.cuda.is_available():
        devices.append('cuda:5')
        devices.append('cuda:7')

    for device in devices:
        for summation_dtype in [torch.float32]:  # , torch.int32
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = True
            for s in [42, 123, 56, 89, 32, 5465, 3247289, 23748923]:
                success = success and test_lut_transformer_product(device, summation_dtype, seed=s)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
