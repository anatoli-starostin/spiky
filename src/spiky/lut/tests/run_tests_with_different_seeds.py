import torch

from test_lut_forward_simple import test_lut_forward_simple


def main():
    tests = {
        func.__name__: func
        for func in [
            test_lut_forward_simple,
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

