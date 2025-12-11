import torch

from spiky.util.synapse_growth import Conv2DSynapseGrowthHelper
from spiky.util.test_utils import unpack_chunk_of_connections


def test_conv2d_helper(device, summation_dtype=torch.float32, seed=1):
    if summation_dtype != torch.float32:
        return True
    success = _test_conv2d_helper(device, 4, 4, 2, 2, 1, 1, 2, 2, seed)
    success = success and _test_conv2d_helper(device, 28, 28, 5, 5, 1, 1, 6, 6, seed)
    return success


def _test_conv2d_helper(device, h, w, rh, rw, sh, sw, kh, kw, seed=1):
    c_helper = Conv2DSynapseGrowthHelper(h, w, rh, rw, sh, sw, kh, kw)
    chunk_of_connections = c_helper.grow_synapses(
        torch.arange(1, c_helper.h * c_helper.w + 1, dtype=torch.int32).reshape(c_helper.h, c_helper.w),
        torch.arange(c_helper.h * c_helper.w + 1, c_helper.h * c_helper.w + 1 + c_helper.out_h * c_helper.out_w, dtype=torch.int32).reshape(c_helper.out_h, c_helper.out_w),
        device,
        seed=seed
    )
    connections, connections_count = unpack_chunk_of_connections(chunk_of_connections)
    extracted_pairs = set()
    for c_group in connections:
        source_id = c_group['source_id']
        for target_id in c_group['target_ids']:
            extracted_pairs.add((source_id, target_id,))

    print('calculating ground truth pairs...')

    ground_truth_pairs = set()
    for win_y in range(c_helper.num_win_h):
        for win_x in range(c_helper.num_win_w):
            for j in range(c_helper.rh):
                for i in range(c_helper.rw):
                    source_id = 1 + (win_y * c_helper.sh + j) * c_helper.w + win_x * c_helper.sw + i
                    oy = win_y * c_helper.kh
                    ox = win_x * c_helper.kw
                    for by in range(c_helper.kh):
                        for bx in range(c_helper.kw):
                            target_id = 1 + c_helper.h * c_helper.w + (oy + by) * c_helper.out_w + ox + bx
                            ground_truth_pairs.add((source_id, target_id,))

    return extracted_pairs == ground_truth_pairs


def main():
    print("=" * 60)
    print("CONV2D HELPER TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        print(f"\nTesting on {device}...")
        success = test_conv2d_helper(device)

        if success:
            print(f"\n{device} test completed successfully!")
        else:
            print(f"\n{device} test failed!")
            return -1

    return 0


if __name__ == "__main__":
    exit(main())
