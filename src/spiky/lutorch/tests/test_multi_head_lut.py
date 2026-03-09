"""
Test for MultiHeadLut module.
"""
import os
from contextlib import contextmanager

# Disable torch.compile in lutorch so tests run without compilation.
os.environ["SPIKY_LUTORCH_NO_COMPILE"] = "1"

import torch
import torch.nn as nn
from tqdm import tqdm

from spiky.lut.LUTLayer import LUTLayer, SynapseMeta
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode
from spiky.lutorch import anchor_pairs_lookup as anchor_pairs_lookup_mod
from spiky.lutorch import l_projection as l_projection_mod


@contextmanager
def _use_lutorch_custom_cuda_kernels(enabled: bool):
    prev_lookup = anchor_pairs_lookup_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS
    prev_proj = l_projection_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS
    anchor_pairs_lookup_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS = enabled
    l_projection_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS = enabled
    try:
        yield
    finally:
        anchor_pairs_lookup_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS = prev_lookup
        l_projection_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS = prev_proj


def test_multi_head_lut_simple(device, seed=None):
    """
    Simple test for MultiHeadLut with n_heads=1 (equivalent to LUTLayer).
    Synchronizes anchors and weights to verify exact match.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    batch_size = 4
    n_inputs = 100
    n_anchors_per_detector = 4
    n_detectors = 10
    n_outputs = 20
    
    # Create input
    x = torch.randn(batch_size, n_inputs, device=device)
    
    # Create LUTLayer (baseline)
    lut_layer = LUTLayer(
        n_inputs=n_inputs,
        n_anchors_per_detector=n_anchors_per_detector,
        n_detectors=n_detectors,
        n_outputs=n_outputs,
        synapse_meta=SynapseMeta(
            initial_weight=1.0,
            initial_noise_level=-1.0
        ),
        random_seed=seed,
        device=device
    )
    lut_layer.eval()
    
    # Create MultiHeadLut normally (without anchor_candidates)
    multi_head_lut = MultiHeadLut(
        input_dim=n_inputs,
        n_heads=1,
        n_outputs=n_outputs,
        n_anchor_pairs=n_anchors_per_detector,
        tables_per_head=n_detectors,
        random_seed=seed,
        device=device
    )
    multi_head_lut.eval()
    
    # Extract anchors from LUTLayer: [n_detectors, n_anchors_per_detector, 2]
    lut_anchors = lut_layer._export_anchors()  # [n_detectors, n_anchors_per_detector, 2]
    
    # Synchronize anchors by directly setting the internal buffers
    # LUTLayer anchors: [n_detectors, n_anchors_per_detector, 2]
    # MultiHeadLut expects: anchor_pairs_a and anchor_pairs_b: [n_tables, n_anchor_pairs]
    # Since n_tables = n_detectors and n_anchor_pairs = n_anchors_per_detector
    anchor_a = lut_anchors[:, :, 0].to(dtype=torch.long).clone()
    anchor_b = lut_anchors[:, :, 1].to(dtype=torch.long).clone()
    
    # Set anchors directly (try original order first)
    multi_head_lut.lookup.anchor_pairs_a.data = anchor_a
    multi_head_lut.lookup.anchor_pairs_b.data = anchor_b
    
    # Synchronize weights
    # LUTLayer weights: export_weights(inverse_order=False) returns weights shaped as
    # [lut_receptive_field_shape, output_shape] = [n_detectors, n_entries_per_table, n_outputs]
    # MultiHeadLut weights: self.projection.weights: [n_lookup_tables, n_entries_per_table, n_outputs]
    # where n_lookup_tables = n_detectors, so shapes match directly
    lut_weights = lut_layer.export_weights(inverse_order=False)  # [n_detectors, n_entries_per_table, n_outputs]
    # Copy to MultiHeadLut
    multi_head_lut.projection.weights.data = lut_weights.clone()
    
    # Forward pass
    with torch.no_grad():
        # Forward through LUTLayer
        lut_output_raw = lut_layer(x.unsqueeze(1))  # [B, 1, n_outputs]
        # LUTLayer returns [B, 1, n_outputs], squeeze to [B, n_outputs]
        lut_output = lut_output_raw.squeeze(1)  # [B, n_outputs]
        
        # Forward through MultiHeadLut
        multi_head_output = multi_head_lut(x)  # [B, n_heads, n_outputs]
    
    # Check output shape
    assert multi_head_output.shape == (batch_size, 1, n_outputs), \
        f"Expected shape {(batch_size, 1, n_outputs)}, got {multi_head_output.shape}"
    
    # Squeeze to compare with LUTLayer output
    multi_head_output_squeezed = multi_head_output.squeeze(1)  # [B, n_outputs]
    
    # Check that outputs match (within numerical precision)
    max_diff = torch.abs(lut_output - multi_head_output_squeezed).max().item()
    assert max_diff < 1e-5, \
        f"Outputs don't match! Max difference: {max_diff:.2e}"
    
    print(f"✓ MultiHeadLut (n_heads=1) forward pass successful")
    
    return True


def test_multi_head_lut_training(device, seed=None):
    """
    Test MultiHeadLut in training mode with synchronized anchors and weights.
    Runs multiple training iterations and verifies outputs and weights match LUTLayer.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    batch_size = 4
    n_inputs = 100
    n_anchors_per_detector = 4
    n_detectors = 10
    n_outputs = 20
    n_iterations = 100
    
    # Create LUTLayer (baseline)
    lut_layer = LUTLayer(
        n_inputs=n_inputs,
        n_anchors_per_detector=n_anchors_per_detector,
        n_detectors=n_detectors,
        n_outputs=n_outputs,
        synapse_meta=SynapseMeta(
            initial_weight=1.0,
            initial_noise_level=-1.0
        ),
        random_seed=seed,
        device=device
    )
    lut_layer.train()
    
    # Create MultiHeadLut with n_heads=1 (should be equivalent)
    multi_head_lut = MultiHeadLut(
        input_dim=n_inputs,
        n_heads=1,
        n_outputs=n_outputs,
        n_anchor_pairs=n_anchors_per_detector,
        tables_per_head=n_detectors,
        random_seed=seed,
        device=device
    )
    multi_head_lut.train()
    
    # Extract anchors from LUTLayer: [n_detectors, n_anchors_per_detector, 2]
    lut_anchors = lut_layer._export_anchors()  # [n_detectors, n_anchors_per_detector, 2]
    
    # Synchronize anchors by directly setting the internal buffers
    anchor_a = lut_anchors[:, :, 0].to(dtype=torch.long).clone()
    anchor_b = lut_anchors[:, :, 1].to(dtype=torch.long).clone()
    multi_head_lut.lookup.anchor_pairs_a.data = anchor_a
    multi_head_lut.lookup.anchor_pairs_b.data = anchor_b
    
    # Synchronize initial weights
    lut_weights = lut_layer.export_weights(inverse_order=False)  # [n_detectors, n_entries_per_table, n_outputs]
    multi_head_lut.projection.weights.data = lut_weights.clone()
    
    # Create optimizers
    lut_optimizer = torch.optim.SGD(lut_layer.parameters(), lr=0.01)
    multi_head_optimizer = torch.optim.SGD(multi_head_lut.parameters(), lr=0.01)
    
    # Run training iterations
    for iteration in tqdm(range(n_iterations), desc="Training iterations"):
        # Generate random input
        x = torch.randn(batch_size, n_inputs, device=device, requires_grad=True)
        
        # Forward pass
        lut_output_raw = lut_layer(x.unsqueeze(1))  # [B, 1, n_outputs]
        lut_output = lut_output_raw.squeeze(1)  # [B, n_outputs]
        
        multi_head_output = multi_head_lut(x)  # [B, n_heads, n_outputs]
        multi_head_output_squeezed = multi_head_output.squeeze(1)  # [B, n_outputs]
        
        # Check outputs match before backward
        max_diff = torch.abs(lut_output - multi_head_output_squeezed).max().item()
        assert max_diff < 1e-5, \
            f"Iteration {iteration}: Outputs don't match before backward! Max difference: {max_diff:.2e}"
        
        # Generate random target for loss computation
        target = torch.randn(batch_size, n_outputs, device=device)
        
        # Compute loss
        lut_loss = ((lut_output - target) ** 2).mean()
        multi_head_loss = ((multi_head_output_squeezed - target) ** 2).mean()
        
        # Backward pass
        lut_optimizer.zero_grad()
        lut_loss.backward()
        lut_optimizer.step()
        
        multi_head_optimizer.zero_grad()
        multi_head_loss.backward()
        multi_head_optimizer.step()
        
        # Get weights after optimizer step
        lut_weights = lut_layer.export_weights(inverse_order=False)
        
        # Check weights match before synchronization
        weight_diff = torch.abs(multi_head_lut.projection.weights.data - lut_weights).max().item()
        assert weight_diff < 1e-5, \
            f"Iteration {iteration}: Weights don't match before sync! Max difference: {weight_diff:.2e}"
        
        # Synchronize weights after each iteration to avoid numerical instability
        multi_head_lut.projection.weights.data = lut_weights.clone()
    
    print(f"✓ MultiHeadLut training mode test successful ({n_iterations} iterations)")
    
    return True


def test_multi_head_lut_smooth_simple(device, seed=None):
    """
    Simple smoke test for MultiHeadLut in smooth mode with n_alternatives=3.
    Verifies forward pass in train and eval modes and tensor shapes/finite outputs.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    batch_size = 4
    n_inputs = 100
    n_heads = 2
    n_anchors_per_detector = 4
    n_detectors_per_head = 3
    n_outputs = 20
    n_alternatives = 3
    
    # Create input
    x = torch.randn(batch_size, n_inputs, device=device)

    # Create baseline model and clone params into comparison model
    multi_head_lut = MultiHeadLut(
        input_dim=n_inputs,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=n_anchors_per_detector,
        tables_per_head=n_detectors_per_head,
        random_seed=seed,
        device=device,
        n_alternatives=n_alternatives,
        smooth_mode=True,
        uncertainty_mode=UncertaintyMode.INVERSE_QUADRATIC
    )

    # CPU: keep smoke behavior
    if device != "cuda":
        multi_head_lut.eval()
        with torch.no_grad():
            output_eval = multi_head_lut(x)
        assert output_eval.shape == (batch_size, n_heads, n_outputs)
        assert torch.isfinite(output_eval).all(), "Eval output contains non-finite values"
        multi_head_lut.train()
        output_train = multi_head_lut(x)
        assert output_train.shape == (batch_size, n_heads, n_outputs)
        assert torch.isfinite(output_train).all(), "Train output contains non-finite values"
        print("✓ MultiHeadLut smooth mode forward pass successful (n_alternatives=3)")
        return True

    multi_head_lut_no_custom = MultiHeadLut(
        input_dim=n_inputs,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=n_anchors_per_detector,
        tables_per_head=n_detectors_per_head,
        random_seed=seed,
        device=device,
        n_alternatives=n_alternatives,
        smooth_mode=True,
        uncertainty_mode=UncertaintyMode.INVERSE_QUADRATIC
    )
    multi_head_lut_no_custom.load_state_dict(multi_head_lut.state_dict())

    # Evaluation comparison
    multi_head_lut.eval()
    multi_head_lut_no_custom.eval()
    with torch.no_grad():
        with _use_lutorch_custom_cuda_kernels(True):
            output_eval_custom = multi_head_lut(x)
        with _use_lutorch_custom_cuda_kernels(False):
            output_eval_ref = multi_head_lut_no_custom(x)
    torch.testing.assert_close(
        output_eval_custom, output_eval_ref, atol=1e-5, rtol=1e-4,
        msg="Eval outputs differ between custom-kernel and fallback MultiHeadLut smooth mode",
    )

    # Training forward comparison
    multi_head_lut.train()
    multi_head_lut_no_custom.train()
    with _use_lutorch_custom_cuda_kernels(True):
        output_train_custom = multi_head_lut(x)
    with _use_lutorch_custom_cuda_kernels(False):
        output_train_ref = multi_head_lut_no_custom(x)
    torch.testing.assert_close(
        output_train_custom, output_train_ref, atol=1e-5, rtol=1e-4,
        msg="Train outputs differ between custom-kernel and fallback MultiHeadLut smooth mode",
    )

    print("✓ MultiHeadLut smooth mode parity successful (n_alternatives=3, custom CUDA kernels on/off)")
    
    return True


def test_multi_head_lut_smooth_training(device, seed=None):
    """
    Training test for MultiHeadLut in smooth mode with n_alternatives=3.
    Runs multiple training iterations and checks that loss decreases.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    batch_size = 4
    n_inputs = 100
    n_heads = 2
    n_anchors_per_detector = 4
    n_detectors_per_head = 3
    n_outputs = 20
    n_iterations = 100
    n_alternatives = 3
    
    # Create MultiHeadLut in smooth mode
    multi_head_lut = MultiHeadLut(
        input_dim=n_inputs,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=n_anchors_per_detector,
        tables_per_head=n_detectors_per_head,
        random_seed=seed,
        device=device,
        n_alternatives=n_alternatives,
        smooth_mode=True,
        uncertainty_mode=UncertaintyMode.INVERSE_QUADRATIC
    )

    # CPU: keep original single-model smoke behavior
    if device != "cuda":
        multi_head_lut.train()
        optimizer = torch.optim.SGD(multi_head_lut.parameters(), lr=0.1)
        for iteration in tqdm(range(n_iterations), desc="Training iterations (smooth)"):
            x = torch.randn(batch_size, n_inputs, device=device, requires_grad=True)
            target = torch.randn(batch_size, n_heads, n_outputs, device=device)
            output = multi_head_lut(x)
            loss = ((output - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            assert torch.isfinite(loss).item(), f"Iteration {iteration}: Loss became non-finite"
            for name, param in multi_head_lut.named_parameters():
                assert torch.isfinite(param).all(), f"Iteration {iteration}: Parameter {name} contains non-finite values"
        print(f"✓ MultiHeadLut smooth mode training test successful ({n_iterations} iterations, n_alternatives=3)")
        return True

    # CUDA: compare custom kernels ON vs OFF
    multi_head_lut_ref = MultiHeadLut(
        input_dim=n_inputs,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=n_anchors_per_detector,
        tables_per_head=n_detectors_per_head,
        random_seed=seed,
        device=device,
        n_alternatives=n_alternatives,
        smooth_mode=True,
        uncertainty_mode=UncertaintyMode.INVERSE_QUADRATIC
    )
    multi_head_lut_ref.load_state_dict(multi_head_lut.state_dict())
    multi_head_lut.train()
    multi_head_lut_ref.train()
    optimizer_custom = torch.optim.SGD(multi_head_lut.parameters(), lr=0.1)
    optimizer_ref = torch.optim.SGD(multi_head_lut_ref.parameters(), lr=0.1)

    for iteration in tqdm(range(n_iterations), desc="Training iterations (smooth, custom-vs-fallback)"):
        x = torch.randn(batch_size, n_inputs, device=device, requires_grad=True)
        target = torch.randn(batch_size, n_heads, n_outputs, device=device)

        with _use_lutorch_custom_cuda_kernels(True):
            output_custom = multi_head_lut(x)
            loss_custom = ((output_custom - target) ** 2).mean()
            optimizer_custom.zero_grad()
            loss_custom.backward()
            optimizer_custom.step()

        with _use_lutorch_custom_cuda_kernels(False):
            output_ref = multi_head_lut_ref(x)
            loss_ref = ((output_ref - target) ** 2).mean()
            optimizer_ref.zero_grad()
            loss_ref.backward()
            optimizer_ref.step()

        torch.testing.assert_close(
            output_custom, output_ref, atol=2e-4, rtol=2e-4,
            msg=f"Iteration {iteration}: outputs differ between custom and fallback smooth training",
        )
        torch.testing.assert_close(
            loss_custom, loss_ref, atol=2e-5, rtol=2e-4,
            msg=f"Iteration {iteration}: losses differ between custom and fallback smooth training",
        )
        for (name_c, p_custom), (name_r, p_ref) in zip(
            multi_head_lut.named_parameters(), multi_head_lut_ref.named_parameters()
        ):
            assert name_c == name_r
            torch.testing.assert_close(
                p_custom, p_ref, atol=5e-4, rtol=5e-4,
                msg=f"Iteration {iteration}: parameter mismatch for {name_c}",
            )

    print(f"✓ MultiHeadLut smooth mode CUDA parity successful ({n_iterations} iterations, n_alternatives=3)")
    
    return True


def main():
    """
    Run all tests.
    """
    print("=" * 60)
    print("MultiHeadLut TESTS")
    print("=" * 60)
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    seed = 42

    for device in devices:
        print(f"\nTesting on {device}...")
        
        # Test 1: Simple test with n_heads=1
        print("\n1. Testing MultiHeadLut with n_heads=1 (equivalent to LUTLayer)...")
        success = test_multi_head_lut_simple(device, seed=seed)
        if not success:
            print(f"❌ Test failed on {device}")
            return -1
        
        # Test 2: Training mode
        print("\n2. Testing MultiHeadLut in training mode...")
        success = test_multi_head_lut_training(device, seed=seed)
        if not success:
            print(f"❌ Test failed on {device}")
            return -1

        # Test 3: Smooth mode simple test (n_alternatives=3)
        print("\n3. Testing MultiHeadLut in smooth mode (simple, n_alternatives=3)...")
        success = test_multi_head_lut_smooth_simple(device, seed=seed)
        if not success:
            print(f"❌ Test failed on {device}")
            return -1

        # Test 4: Smooth mode training test (n_alternatives=3)
        print("\n4. Testing MultiHeadLut in smooth mode (training, n_alternatives=3)...")
        success = test_multi_head_lut_smooth_training(device, seed=seed)
        if not success:
            print(f"❌ Test failed on {device}")
            return -1

        print(f"\n✓ All tests passed on {device}!")
    
    return 0


if __name__ == "__main__":
    exit(main())
