import torch

from test_andn_layer_forward import test_andn_layer_forward
from test_andn_layer_backward_hebb import test_andn_layer_backward_hebb
from test_conv2d_helper import test_conv2d_helper
from test_inhibition_layer_forward import test_inhibition_layer_forward
from test_andn_stability_backward import test_andn_stability_backward
from test_andn_stability_backward_sync import test_andn_stability_backward_sync
from test_inhibition_layer_forward import test_inhibition_layer_forward


def main():
    tests = {
        func.__name__: func
        for func in [
            test_andn_layer_forward,
            test_conv2d_helper,
            test_inhibition_layer_forward,
            test_andn_layer_backward_hebb,
            test_andn_stability_backward,
            test_andn_stability_backward_sync,
            test_inhibition_layer_forward
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

# TODO option to run without add_connection calls
