import random
import torch
from copy import copy

from spiky_cuda import SynapseGrowthLowLevelEngine


CONNECTIONS_HEADER_INT_SIZE = 4
N_CONNECTIONS_IN_BLOCK = 12
CONNECTIONS_BLOCK_INT_SIZE = CONNECTIONS_HEADER_INT_SIZE + 2 * N_CONNECTIONS_IN_BLOCK


def generate_random_encoded_data(n_blocks, n_synapse_metas, n_chains, n_unique_sources, n_unique_targets):
    synapse_metas = list(range(1, n_synapse_metas + 1))

    elements = []
    positions = []
    for i in range(n_blocks):
        synapse_meta = random.choice(synapse_metas)
        elements.append(synapse_meta)
        positions.append(i * CONNECTIONS_BLOCK_INT_SIZE)

    random.shuffle(positions)

    int_array = [0] * (n_blocks * CONNECTIONS_BLOCK_INT_SIZE)

    max_n_blocks_per_chain = n_blocks // n_chains

    prev_pos = None
    n_blocks_till_chain_root = random.randint(1, max_n_blocks_per_chain)
    for pos, synapse_meta in zip(positions, elements):
        int_array[pos] = random.randint(1, n_unique_sources) if (n_blocks_till_chain_root == 0) or (pos == positions[-1]) else 0
        int_array[pos + 1] = synapse_meta
        int_array[pos + 2] = 0
        if prev_pos is not None:
            int_array[pos + 3] = prev_pos - pos
        else:
            int_array[pos + 3] = 0
        if n_blocks_till_chain_root == 0:
            prev_pos = None
            n_blocks_till_chain_root = random.randint(1, max_n_blocks_per_chain)
        else:
            n_blocks_till_chain_root -= 1
            prev_pos = pos

        n_elements_in_block = random.randint(1, N_CONNECTIONS_IN_BLOCK)

        for i in range(N_CONNECTIONS_IN_BLOCK):
            if i < n_elements_in_block:
                int_array[pos + CONNECTIONS_HEADER_INT_SIZE + 2 * i] = synapse_meta
                int_array[pos + CONNECTIONS_HEADER_INT_SIZE + 2 * i + 1] = random.randint(1, n_unique_targets)
            else:
                int_array[pos + CONNECTIONS_HEADER_INT_SIZE + 2 * i] = -1
                int_array[pos + CONNECTIONS_HEADER_INT_SIZE + 2 * i + 1] = 0

    return int_array


def decode_data(int_array):
    n_entries = len(int_array) // CONNECTIONS_BLOCK_INT_SIZE

    def decode_block(s):
        body = []
        for j in range(N_CONNECTIONS_IN_BLOCK):
            sm = int_array[s + CONNECTIONS_HEADER_INT_SIZE + 2 * j]
            if sm != -1:
                body.append((sm, int_array[s + CONNECTIONS_HEADER_INT_SIZE + 2 * j + 1],))
        res = {
            'offset': s,
            'source_neuron_id': int_array[s],
            'synapse_meta_index': int_array[s + 1],
            'n_target_neurons': int_array[s + 2],
            'shift_to_next_group': int_array[s + 3],
            'body': body
        }
        return res

    decoded_data = []
    for i in range(n_entries):
        offset = i * CONNECTIONS_BLOCK_INT_SIZE

        if int_array[offset] > 0:
            # root detected
            result = []
            while True:
                result.append(decode_block(offset))
                shift = int_array[offset + 3]
                if shift == 0:
                    break
                offset += shift

            decoded_data.append(result)

    return sorted(decoded_data, key=lambda x: (x[0]['source_neuron_id'], x[0]['offset'],))


def merge_and_sort(decoded_data):
    merged_results = []
    i = 0
    while i < len(decoded_data):
        cur_list = decoded_data[i]
        j = i + 1
        while j < len(decoded_data):
            other_list = decoded_data[j]
            if other_list[0]['source_neuron_id'] != cur_list[0]['source_neuron_id']:
                break
            cur_list = cur_list + other_list
            j = j + 1
        source_id = cur_list[0]['source_neuron_id']
        assert source_id > 0
        cur_list = sorted(cur_list, key=lambda x: (x['synapse_meta_index'], x['offset'],))
        all_entries = []
        unique_targets = set()
        for block in cur_list:
            all_entries += block['body']
        for k in range(len(all_entries)):
            target_id = all_entries[k][1]
            if target_id in unique_targets:
                all_entries[k] = (all_entries[k][0], 0,)
            else:
                unique_targets.add(target_id)
        all_entries = sorted(all_entries, key=lambda e: (e[0], e[1] == 0, e[1]))
        it = iter(all_entries)
        new_list = []
        for block in cur_list:
            new_block = copy(block)
            new_block['source_neuron_id'] = 0
            new_list.append(new_block)
            new_body = []
            for k in range(len(block['body'])):
                new_body.append(next(it))
            new_block['body'] = new_body

        cur_main_block = new_list[0]
        cur_synapse_meta_index = cur_main_block['synapse_meta_index']
        counter = 0
        for block in new_list:
            if block['synapse_meta_index'] != cur_synapse_meta_index:
                cur_synapse_meta_index = block['synapse_meta_index']
                cur_main_block['n_target_neurons'] = counter
                cur_main_block = block
                counter = 0
            for e in block['body']:
                if e[1] > 0:
                    counter += 1
        cur_main_block['n_target_neurons'] = counter
        new_list[0]['source_neuron_id'] = source_id
        prev = None
        for block in new_list:
            if prev is not None:
                prev['shift_to_next_group'] = block['offset'] - prev['offset']
            prev = block
        prev['shift_to_next_group'] = 0
        merged_results.append(new_list)
        i = j
    return merged_results


def test_synapse_growth_lowlevel(device, summation_dtype, seed=42):
    if summation_dtype != torch.float32:
        return None

    device = str(device)
    if device.startswith('cuda'):
        s = device.split(':')
        _device = int(s[1]) if len(s) == 2 else 0
    elif device == 'cpu':
        _device = -1
    else:
        raise RuntimeError(f'Wrong device {device}')

    random.seed(seed)
    int_array = generate_random_encoded_data(1024, 8, 128, 32, 32)
    decoded_data = decode_data(int_array)
    python_merged_data = merge_and_sort(decoded_data)
    lowlevel_growth_engine = SynapseGrowthLowLevelEngine(
        2, 0, 32, 32,
        _device, N_CONNECTIONS_IN_BLOCK, 42
    )
    encoded_tensor = torch.tensor(int_array, dtype=torch.int32, device=device)
    lowlevel_growth_engine.finalize(encoded_tensor, True)
    new_decoded_data = decode_data(encoded_tensor.cpu().numpy())

    if len(new_decoded_data) != len(python_merged_data):
        print(f"❌ different number of chains ({len(decoded_data)} for ground truth, {len(new_decoded_data)} after synapse growth)")
        return False

    for gt, sg in zip(python_merged_data, new_decoded_data):
        if gt[0]['source_neuron_id'] != sg[0]['source_neuron_id']:
            print(f"❌ different source_neuron_id detected")
            return False

        if len(gt) != len(sg):
            print(f"❌ detected different chain lengths for {gt[0]['source_neuron_id']}detected")
            return False

        def compare_blocks(gt_block, sg_block):
            for v in ['offset', 'source_neuron_id', 'synapse_meta_index', 'n_target_neurons', 'shift_to_next_group']:
                if gt_block[v] != sg_block[v]:
                    print(f"❌ detected different value of field {v}. gt_block: {gt_block}, sg_block: {sg_block}")
                    return False

                gt_body = gt_block['body']
                sg_body = sg_block['body']

                if len(gt_body) != len(sg_body):
                    print(f"❌ detected different body length. gt_block: {gt_block}, sg_block: {sg_block}")
                    return False

                for i in range(len(gt_body)):
                    if gt_body[i] != sg_body[i]:
                        print(f"❌ detected different body content on {i} position. gt_block: {gt_block}, sg_block: {sg_block}")
                        return False
            return True

        for gt_block, sg_block in zip(gt, sg):
            if not compare_blocks(gt_block, sg_block):
                return False

    return True


def main():
    """Run lowlevel growth engine test"""
    print("=" * 60)
    print("SYNAPSE GROWTH LOWLEVEL TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        print(f"\nTesting on {device}...")
        success = test_synapse_growth_lowlevel(device, torch.float32)

        if success:
            print(f"\n<{device}> test completed successfully!")
        else:
            print(f"\n<{device}> test failed!")
            return -1

    return 0


if __name__ == "__main__":
    exit(main())
