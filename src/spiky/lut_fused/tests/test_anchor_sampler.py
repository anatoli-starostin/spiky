import os

# Disable torch.compile in lutorch so tests run without compilation.
os.environ["SPIKY_LUTORCH_NO_COMPILE"] = "1"

import torch

from spiky.lut_fused.anchor_sampler import AnchorSampler


def test_anchor_sampler(device, _, seed):
    """Test AnchorSampler with default connections (None)."""
    asampler = AnchorSampler(32, 16, 10, device=device, random_seed=seed)
    anchor_pairs = asampler.get_anchor_pairs()
    print(anchor_pairs)
    print(f"Default connections - anchor_pairs shape: {anchor_pairs.shape}")
    assert anchor_pairs.shape == (16, 10, 2), f"Expected shape (16, 10, 2), got {anchor_pairs.shape}"
    return True


def test_anchor_sampler_with_tensor(device, _, seed):
    """Test AnchorSampler with detector_connections as tensor."""
    n_inputs = 32
    n_detectors = 16
    n_anchors_per_detector = 10
    max_anchors_per_detector = n_inputs
    
    # Create anchor_candidates tensor: [n_detectors, max_anchors_per_detector]
    # Each detector has specific input indices (all values must be >= 0, no padding)
    anchor_candidates = torch.zeros(
        (n_detectors, max_anchors_per_detector),
        dtype=torch.int32,
        device=device
    )
    
    for detector_idx in range(n_detectors):
        start_idx = (detector_idx * 3) % n_inputs
        for anchor_idx in range(max_anchors_per_detector):
            anchor_candidates[detector_idx, anchor_idx] = (start_idx + anchor_idx) % n_inputs
        
        # Verify we have at least 2 unique inputs (required for pairs)
        unique_inputs = torch.unique(anchor_candidates[detector_idx])
        assert len(unique_inputs) >= 2, \
            f"Detector {detector_idx} must have at least 2 unique inputs, got {len(unique_inputs)}"
    
    # Create AnchorSampler with tensor
    asampler = AnchorSampler(
        n_inputs=n_inputs,
        n_detectors=n_detectors,
        n_anchors_per_detector=n_anchors_per_detector,
        device=device,
        detector_connections=anchor_candidates,
        random_seed=seed
    )
    
    anchor_pairs = asampler.get_anchor_pairs()
    print(anchor_pairs)
    print(f"Tensor connections - anchor_pairs shape: {anchor_pairs.shape}")
    assert anchor_pairs.shape == (n_detectors, n_anchors_per_detector, 2), \
        f"Expected shape ({n_detectors}, {n_anchors_per_detector}, 2), got {anchor_pairs.shape}"
    
    # Verify that anchor pairs use indices from anchor_candidates
    # For each detector, check that all anchor pair indices are in the allowed set
    for detector_idx in range(n_detectors):
        # Get the allowed input indices for this detector
        allowed_inputs = torch.unique(anchor_candidates[detector_idx])
        
        # Get anchor pairs for this detector: [n_anchors_per_detector, 2]
        detector_pairs = anchor_pairs[detector_idx]  # [n_anchors_per_detector, 2]
        
        # Get all unique indices used in the pairs
        used_indices = torch.unique(detector_pairs.flatten())
        
        # Check that all used indices are in the allowed set
        # Convert to sets for easier comparison
        allowed_set = set(allowed_inputs.cpu().tolist())
        used_set = set(used_indices.cpu().tolist())
        
        assert used_set.issubset(allowed_set), \
            f"Detector {detector_idx}: anchor pairs use indices {used_set} but only {allowed_set} are allowed"
        
        # Also verify basic bounds
        assert (detector_pairs >= 0).all(), \
            f"Detector {detector_idx}: all anchor pair indices must be >= 0"
        assert (detector_pairs < n_inputs).all(), \
            f"Detector {detector_idx}: all anchor pair indices must be < n_inputs"
    
    return True


def main():
    print("=" * 60)
    print("ANCHOR SAMPLER TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        print(f"\nTesting on {device}...")
        
        # Test default connections
        success = test_anchor_sampler(device, None, 123)
        if not success:
            print(f"\n<{device}> default connections test failed!")
            return -1
        
        # Test tensor connections
        success = test_anchor_sampler_with_tensor(device, None, 123)
        if not success:
            print(f"\n<{device}> tensor connections test failed!")
            return -1
        
        print(f"\n<{device}> all tests completed successfully!")

    return 0


if __name__ == "__main__":
    exit(main())

