import torch

from test_lut_forward_simple import test_lut_forward_simple
from test_lut_fully_connected_small import test_lut_fully_connected_small
from test_lut_fully_connected import test_lut_fully_connected
from test_lut_backward import test_lut_backward
from test_lut_transformer_small import test_lut_transformer_small
from test_lut_transformer_product import test_lut_transformer_product
from spiky.util.torch_utils import test_dense_to_sparse_converter


def main():
    tests = {
        func.__name__: func
        for func in [
            test_lut_forward_simple,
            test_lut_fully_connected_small,
            test_lut_fully_connected,
            test_lut_backward,
            test_dense_to_sparse_converter,
            test_lut_transformer_small,
            test_lut_transformer_product
        ]
    }

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for seed in [234654, 42]:
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
