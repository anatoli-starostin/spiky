#!/usr/bin/env python3

import torch
from torch import dtype

from spiky.util.synapse_growth import SynapseGrowthEngine, GrowthCommand
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta
from spiky.util.test_utils import (
    extract_connection_map, grow_and_add, convert_connections_to_export_format
)
from spnet_test_utils import (
    test_spnet_connectivity, test_serialization
)


def test_simple_math(device, summation_dtype, seed=1):
    if device == torch.device('cpu') or device == 'cpu':
        return True

    # Initialize the growth engine
    growth_engine = SynapseGrowthEngine(device='cpu', synapse_group_size=6, max_groups_in_buffer=128)

    # Define growth command: type 0 neurons connect to type 1 neurons
    # Use a large cuboid area so each bottom neuron can reach all top neurons
    growth_command = GrowthCommand(
        target_type=1,  # Connect to neurons of type 1
        synapse_meta_index=0,  # Synapse type 0
        x1=-100.0, y1=-100.0, z1=-100.0,  # Large cuboid area
        x2=100.0, y2=100.0, z2=100.0,  # covering all possible positions
        p=1.0  # 100% connection probability
    )

    # Register neuron types
    print("Registering neuron types...")
    growth_engine.register_neuron_type(
        max_synapses=8,  # Each bottom neuron can have up to 4 synapses
        growth_command_list=[growth_command]
    )

    growth_engine.register_neuron_type(
        max_synapses=0,  # Top neurons don't grow synapses (they're targets only)
        growth_command_list=[]
    )
    
    synapse_metas = [
        SynapseMeta(
            min_delay=0,
            max_delay=1,
            initial_weight=5.0,
            _forward_group_size=3,
            _backward_group_size=3
        )
    ]
    neuron_metas = [
        NeuronMeta(neuron_type=0),
        NeuronMeta(neuron_type=1)
    ]
    spnet = SpikingNet(
        synapse_metas=synapse_metas,
        neuron_metas=neuron_metas,
        neuron_counts=[
            4, 8
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device('cpu')

    # Create bottom layer neurons (type 0) - arranged in a square at y=0
    print("Adding bottom layer neurons (type 0)...")
    bottom_ids = spnet.get_neuron_ids_by_meta(0)
    bottom_coords = torch.tensor([
        [0.0, 0.0, 0.0],  # Neuron 1
        [1.0, 0.0, 0.0],  # Neuron 2
        [0.0, 0.0, 1.0],  # Neuron 3
        [1.0, 0.0, 1.0],  # Neuron 4
    ], dtype=torch.float32)

    growth_engine.add_neurons(neuron_type_index=0, identifiers=bottom_ids, coordinates=bottom_coords)

    # Create top layer neurons (type 1) - arranged in a square at y=1
    print("Adding top layer neurons (type 1)...")
    top_ids = spnet.get_neuron_ids_by_meta(1)
    top_coords = torch.tensor([
        [0.0, 0.5, 0.0],  # Neuron 5
        [0.0, 1.0, 0.0],  # Neuron 6
        [0.5, 0.5, 0.0],  # Neuron 7
        [1.0, 1.0, 0.0],  # Neuron 8
        [0.0, 0.5, 0.5],  # Neuron 9
        [0.0, 1.0, 1.0],  # Neuron 10
        [0.5, 0.5, 0.],  # Neuron 11
        [1.0, 1.0, 1.0],  # Neuron 12
    ], dtype=torch.float32)

    growth_engine.add_neurons(neuron_type_index=1, identifiers=top_ids, coordinates=top_coords)

    print(f"Total neurons: {growth_engine._n_total_neurons}")
    print(f"Total growth commands: {growth_engine._n_total_growth_commands}")
    all_connections, connection_count = extract_connection_map(growth_engine, synapse_metas, seed, True)
    if all_connections is None:
        return False

    print(f"Total connections found: {connection_count}")
    print(f"Connection groups: {len(all_connections)}")

    torch.manual_seed(seed)
    n_ticks = 1000
    I = torch.rand([n_ticks]) * 20.0
    V = torch.zeros([n_ticks])
    V[0] = neuron_metas[0].c
    U = torch.zeros([n_ticks])
    U[0] = neuron_metas[0].c * neuron_metas[0].b

    spnet._neuron_data_manager._test_math(I, U, V, 0)

    V_cpu = V
    print(f"V_cpu: {V_cpu}\n")

    spnet.to_device('cuda')
    I = I.to('cuda')
    V = torch.zeros([n_ticks])
    V[0] = neuron_metas[0].c
    U = torch.zeros([n_ticks])
    U[0] = neuron_metas[0].c * neuron_metas[0].b
    V = V.to('cuda')
    U = U.to('cuda')

    spnet._neuron_data_manager._test_math(I, U, V, 0)

    V_cuda = V.cpu()
    print(f"V_cuda: {V_cuda}\n")

    V_diff = V_cuda - V_cpu
    diff_ticks = torch.where(V_diff.abs() > 0.0)

    if diff_ticks[0].shape[0] > 0:
        print(f"❌ Detected difference in voltages on ticks: {diff_ticks[0].cpu().numpy()}")
        return False

    print('\n🎉 Everything is ok')
    return True


def main():
    print("=" * 60)
    print("SPNET SIMPLE MATH TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        print('Skipping, this test is relevant only with cuda')
        return

    for summation_dtype in [torch.float32, torch.int32]:
        print(f"\nTesting, summation_dtype {summation_dtype}...")
        success = test_simple_math(torch.device('cuda'), summation_dtype)

        if success:
            print(f"\n<{summation_dtype}> test completed successfully!")
        else:
            print(f"\n<{summation_dtype}> test failed!")
            return -1

    return 0


if __name__ == "__main__":
    exit(main())
