import torch

from spiky.util.chunk_of_connections import ChunkOfConnections, ChunkOfConnectionsValidator


def main():
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
        return 0
    else:
        print(errors)
        return -1


if __name__ == "__main__":
    exit(main())
