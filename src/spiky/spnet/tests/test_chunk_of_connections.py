import torch

from spiky.util.chunk_of_connections import ChunkOfConnections, ChunkOfConnectionsValidator


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

    return 0


if __name__ == "__main__":
    exit(main())
