import torch
import torch.nn as nn

from spiky.lut.LUTLayer import LUTLayer, SynapseMeta


def is_connected(anchors):
    """
    Check if anchor pairs form a connected graph within each detector.
    With n_anchors_per_detector = 3, the check is trivial: 
    a graph with n nodes is connected if and only if it has at least n-1 edges.
    """
    n_detectors = anchors.shape[0]
    n_anchors_per_detector = anchors.shape[1]
    assert n_anchors_per_detector <= 3
    
    # Check connectivity for each detector separately
    for detector_id in range(n_detectors):
        # Collect unique nodes and edges for this detector
        detector_nodes = set()
        detector_edges = set()
        
        for anchor_idx in range(n_anchors_per_detector):
            anchor1 = int(anchors[detector_id, anchor_idx, 0])
            anchor2 = int(anchors[detector_id, anchor_idx, 1])

            assert anchor1 != anchor2
            if anchor1 >= 0 and anchor2 >= 0:
                detector_nodes.add(anchor1)
                detector_nodes.add(anchor2)
                assert anchor1 != anchor2
                # Store edges as sorted tuples to avoid duplicates
                edge = (min(anchor1, anchor2), max(anchor1, anchor2))
                detector_edges.add(edge)
        
        assert len(detector_nodes) > 0
        assert len(detector_edges) == n_anchors_per_detector

        # For a connected graph with n nodes, we need at least n-1 edges
        # With n_anchors_per_detector = 3, this check is both necessary and sufficient
        if n_anchors_per_detector < len(detector_nodes) - 1:
            return False
    return True


def test_lut_connected_anchors(device, seed=None):
    """
    Test that LUTLayer with connected_anchors_mode=True creates anchor pairs
    that form a connected graph.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    synapse_meta = SynapseMeta(
        initial_weight=0.0,
        initial_noise_level=1.0
    )
    
    n_inputs = 16
    n_anchors_per_detector = 3
    n_detectors = 4
    n_outputs = 8
    
    # Create LUTLayer with connected_anchors_mode=True
    layer = LUTLayer(
        n_inputs=n_inputs,
        n_anchors_per_detector=n_anchors_per_detector,
        n_detectors=n_detectors,
        n_outputs=n_outputs,
        connected_anchors_mode=True,
        synapse_meta=synapse_meta,
        random_seed=seed,
        device=device
    )
    
    # Export anchors
    anchors = layer._export_anchors()
    
    # Check that anchors have correct shape
    assert anchors.shape == (n_detectors, n_anchors_per_detector, 2), \
        f"Expected anchors shape ({n_detectors}, {n_anchors_per_detector}, 2), got {anchors.shape}"
    
    # Check that all anchor IDs are valid (within n_inputs range)
    assert torch.all(anchors >= 0), "Anchor IDs must be non-negative"
    assert torch.all(anchors < n_inputs), f"Anchor IDs must be less than {n_inputs}"
    
    if not is_connected(anchors):
        print(f"❌ Anchor pairs do not form a connected graph")
        print(f"Anchors: {anchors}")
        return False
    
    print(f"✓ Anchor pairs form a connected graph")
    return True


def main():
    print("=" * 60)
    print("LUTLayer CONNECTED ANCHORS TEST")
    print("=" * 60)
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device in devices:
        print(f"\nTesting on {device}...")
        success = test_lut_connected_anchors(device, seed=42)
        
        if success:
            print(f"\n<{device}> test completed successfully!")
        else:
            print(f"\n<{device}> test failed!")
            return -1
    
    return 0


if __name__ == "__main__":
    exit(main())
