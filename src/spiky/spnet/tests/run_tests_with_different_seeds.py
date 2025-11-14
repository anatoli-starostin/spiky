import torch

from spiky.spnet.tests.test_spnet_math import test_simple_math
from test_synapse_growth_lowlevel import test_synapse_growth_lowlevel
from test_synapse_growth_simple import test_simple_two_layers
from test_synapse_growth_convolutions import test_convolutional_topology
from test_synapse_growth_probabilities import test_probabilistic_connections
from test_izhikevitch_topology import test_izhikevitch_topology
from test_izhikevitch_runtime import test_izhikevitch_runtime
from test_spnet_math import test_simple_math
from test_chunk_of_connections import test_chunk_of_connections


def main():
    tests = {
        func.__name__: func
        for func in [
            test_synapse_growth_lowlevel,
            test_simple_two_layers,
            test_convolutional_topology,
            test_probabilistic_connections,
            test_izhikevitch_topology,
            test_izhikevitch_runtime,
            test_simple_math,
            test_chunk_of_connections
        ]
    }

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for seed in [234654, 351, 42]:
        for device in devices:
            for summation_dtype in [torch.float32, torch.int32]:
                for fname, func in tests.items():
                    print(f"\nTesting {fname} on {device}, summation_dtype {summation_dtype}, seed {seed}...")
                    success = func(device, summation_dtype, seed)
                    if success is None:
                        print(f"\n<{fname}, {device}, {summation_dtype}, {seed}> skipped")
                        continue

                    if success:
                        print(f"\n<{fname}, {device}, {summation_dtype}, {seed}> test completed successfully!")
                    else:
                        print(f"\n<{fname}, {device}, {summation_dtype}, {seed}> test failed!")
                        return -1

    return 0


if __name__ == "__main__":
    exit(main())

# TODO option to run without add_connection calls
