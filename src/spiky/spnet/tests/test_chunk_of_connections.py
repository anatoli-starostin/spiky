import torch

from spiky.util.chunk_of_connections import ChunkOfConnections, ChunkOfConnectionsValidator, create_identity_mapping
from spiky.util.test_utils import unpack_chunk_of_connections


def test_chunk_of_connections(device, _, __):
    if device == torch.device('cpu') or device == 'cpu':
        return True

    single_group_size = 6

    connections = [
        1, 0, 3, 48, 0, 3, 0, 0, 0, 5, -1, 0, -1, 0, -1, 0,
        2, 0, 1, 0, 0, 3, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0,
        0, 1, 2, 0, 1, 9, 1, 0, 1, 11, -1, 0, -1, 0, -1, 0,
        0, 0, 0, -16, 0, 0, 0, 7, -1, 0, -1, 0, -1, 0, -1, 0,
    ]

    weights = [
        0.1, 0.0, 0.2, 0.0, 0.0, 0.0,
        0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.7, 0.0, 0.8, 0.0, 0.0, 0.0,
        0.0, 0.9, 0.0, 0.0, 0.0, 0.0
    ]

    chunk_of_connections = ChunkOfConnections(
        torch.tensor(connections, dtype=torch.int32),
        single_group_size,
        torch.tensor(weights, dtype=torch.float32)
    )

    validator = ChunkOfConnectionsValidator(chunk_of_connections)
    result, errors = validator.validate_all()
    if result:
        print('all good')
        return True
    else:
        print(errors)
        return False


def test_create_identity_mapping(delta=0):
    """Test the create_identity_mapping helper function with given delta."""
    single_group_size = 6
    N = 10  # Test with neurons 1 to 10
    
    # Create mapping with delta
    mapping_chunk = create_identity_mapping(N=N, single_group_size=single_group_size, delta=delta)
    
    # Extract and print source and target connections for inspection
    all_connections, connection_count = unpack_chunk_of_connections(mapping_chunk)
    print(f"\nCreated connections with delta={delta} (source -> target):")
    for conn in all_connections:
        source_id = conn['source_id']
        for target_id in conn['target_ids']:
            print(f"  {source_id} -> {target_id}")
    
    # Validate the structure
    validator = ChunkOfConnectionsValidator(mapping_chunk)
    result, errors = validator.validate_all()
    
    if not result:
        print(f"Validation failed: {errors}")
        return False
    
    # Verify the connections using extracted data
    if connection_count != N:
        print(f"Expected {N} connections, got {connection_count}")
        return False
    
    # Verify each connection matches expected mapping (source -> source + delta)
    expected_connections = set()
    for i in range(1, N + 1):
        expected_connections.add((i, i + delta))
    
    actual_connections = set()
    for conn in all_connections:
        source_id = conn['source_id']
        for target_id in conn['target_ids']:
            actual_connections.add((source_id, target_id))
            # Verify mapping: target == source + delta
            expected_target = source_id + delta
            if target_id != expected_target:
                print(f"Expected connection {source_id} -> {expected_target}, got {source_id} -> {target_id}")
                return False
    
    # Verify all expected connections are present
    if actual_connections != expected_connections:
        missing = expected_connections - actual_connections
        extra = actual_connections - expected_connections
        if missing:
            print(f"Missing connections: {missing}")
        if extra:
            print(f"Extra connections: {extra}")
        return False
    
    print(f"Mapping test with delta={delta} passed!")
    return True


def test_create_identity_mapping_all():
    """Test create_identity_mapping with different delta values."""
    # Test delta=0 (identity mapping)
    if not test_create_identity_mapping(delta=0):
        return False
    
    # Test delta>0 (shifted mapping)
    if not test_create_identity_mapping(delta=3):
        return False
    
    return True


def main():
    print("=" * 60)
    print("SPNET SIMPLE MATH TEST")
    print("=" * 60)

    for summation_dtype in [torch.float32, torch.int32]:
        print(f"\nTesting, summation_dtype {summation_dtype}...")
        success = test_chunk_of_connections(torch.device('cpu'), summation_dtype, 123)

        if success:
            print(f"\n<{summation_dtype}> test completed successfully!")
        else:
            print(f"\n<{summation_dtype}> test failed!")
            return -1

    print("\n" + "=" * 60)
    print("TESTING CREATE_IDENTITY_MAPPING")
    print("=" * 60)
    
    success = test_create_identity_mapping_all()
    if not success:
        print("\ncreate_identity_mapping test failed!")
        return -1
    
    print("\ncreate_identity_mapping test completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
