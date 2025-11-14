#!/usr/bin/env python3
"""
Simple test for synapse growth model with two flat layers.
Bottom layer (type 0): 4 neurons
Top layer (type 1): 4 neurons
Each bottom neuron should connect to all top neurons.
"""

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


def test_simple_two_layers(device, summation_dtype, seed=1):
    """Test synapse growth between two flat layers of neurons"""
    print(f"Setting up simple two-layer synapse growth test on {device}...")

    # Initialize the growth engine
    growth_engine = SynapseGrowthEngine(device=device, synapse_group_size=6, max_groups_in_buffer=128)

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
    spnet = SpikingNet(
        synapse_metas=synapse_metas,
        neuron_metas=[
            NeuronMeta(neuron_type=0),
            NeuronMeta(neuron_type=1)
        ],
        neuron_counts=[
            4, 8
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)

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

    # Verify the results
    print("\nVerifying connections...")

    # Expected: Each bottom neuron (1,2,3,4) should connect to all top neurons
    n_expected_connections = 32  # 4 bottom neurons × 8 top neurons

    if connection_count == n_expected_connections:
        print(f"✅ Correct number of connections: {connection_count}")
    else:
        print(f"❌ Expected {n_expected_connections} connections, got {connection_count}")
        return False

    # Check that we have exactly 4 connection groups (one per bottom neuron)
    if len(all_connections) != 4:
        print(f"❌ Expected 4 connection groups, got {len(all_connections)}")
        return False

    # Check that each bottom neuron has exactly 4 connections
    bottom_neurons = set(bottom_ids.cpu().numpy())
    expected_targets = set(top_ids.cpu().numpy())

    for conn in all_connections:
        source = conn['source_id']
        if source not in bottom_neurons:
            print(f"❌ Unexpected source neuron: {source}")
            return False

        # Check that this bottom neuron has exactly 4 connections
        if conn['n_targets'] != 8:
            print(f"❌ Neuron {source} should have 8 connections, got {conn['n_targets']}")
            return False

        # Check that this bottom neuron connects to ALL top neurons
        actual_targets = set(conn['target_ids'])
        if actual_targets != expected_targets:
            print(f"❌ Neuron {source} should connect to all top neurons {expected_targets}, got {actual_targets}")
            return False

    print("✅ All bottom neurons have the correct number of connections")
    print("✅ All bottom neurons connect to all top neurons")
    print("\n🎉 The synapse growth model is working correctly.")

    grow_and_add(growth_engine, spnet, seed, seed + 123)
    spnet.compile(shuffle_synapses_random_seed=seed + 456)
    if not test_spnet_connectivity(
        spnet,
        n_expected_connections,
        convert_connections_to_export_format(all_connections, synapse_metas, False, device),
        synapse_metas,
        True,
        device
    ):
        return False
    print(spnet)
    print(spnet.get_memory_stats())
    print(spnet.get_profiling_stats())

    spnet = SpikingNet(
        synapse_metas=synapse_metas,
        neuron_metas=[
            NeuronMeta(neuron_type=0),
            NeuronMeta(neuron_type=1)
        ],
        neuron_counts=[
            4, 8
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)
    grow_and_add(growth_engine, spnet, seed, seed + 123)
    spnet.compile(shuffle_synapses_random_seed=None)
    if not test_spnet_connectivity(
        spnet,
        n_expected_connections,
        convert_connections_to_export_format(all_connections, synapse_metas, True, device),
        synapse_metas,
        True,
        device
    ):
        return False
    print(spnet)
    print(spnet.get_memory_stats())
    print(spnet.get_profiling_stats())

    synapse_metas = [
        SynapseMeta(
            min_delay=0,
            max_delay=1,
            initial_weight=0.0,
            initial_noise_level=10.0,
            _forward_group_size=3,
            _backward_group_size=3
        )
    ]
    spnet = SpikingNet(
        synapse_metas=synapse_metas,
        neuron_metas=[
            NeuronMeta(neuron_type=0),
            NeuronMeta(neuron_type=1)
        ],
        neuron_counts=[
            4, 8
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)
    grow_and_add(growth_engine, spnet, seed, seed + 123)
    spnet.compile(shuffle_synapses_random_seed=seed + 456)
    if not test_spnet_connectivity(
        spnet,
        n_expected_connections,
        None,
        synapse_metas,
        True,
        device
    ):
        return False
    print(spnet)
    print(spnet.get_memory_stats())
    print(spnet.get_profiling_stats())

    if not test_serialization(spnet):
        return False

    print('\n🎉 Everything is ok')
    return True


def main():
    """Run the simple two-layer test"""
    print("=" * 60)
    print("SIMPLE TWO-LAYER SYNAPSE GROWTH TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = test_simple_two_layers(device, summation_dtype)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
