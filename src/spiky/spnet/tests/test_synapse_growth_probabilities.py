#!/usr/bin/env python3
"""
Test probabilistic synapse growth with a large top layer.
Bottom layer (type 0): 4 neurons
Top layer (type 1): 10000 neurons
Each bottom neuron should connect to approximately 5000 top neurons (50% probability).
"""

import torch
from spiky.util.synapse_growth import SynapseGrowthEngine, GrowthCommand
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta
from spiky.util.test_utils import (
    extract_connection_map, grow_and_add, convert_connections_to_export_format
)
from spnet_test_utils import (
    test_spnet_connectivity, test_serialization
)


def test_probabilistic_connections(device, summation_dtype, seed=1):
    """Test probabilistic synapse growth with a large top layer"""

    print(f"Setting up probabilistic connection test on {device}...")

    # Initialize the growth engine with larger buffer for more neurons
    growth_engine = SynapseGrowthEngine(device=device, synapse_group_size=12, max_groups_in_buffer=1024)

    # Define growth command: type 0 neurons connect to type 1 neurons with 50% probability
    growth_command = GrowthCommand(
        target_type=1,  # Connect to neurons of type 1
        synapse_meta_index=0,  # Synapse type 0
        x1=-100.0, y1=-100.0, z1=-100.0,  # Large cuboid area
        x2=100.0, y2=100.0, z2=100.0,  # covering all possible positions
        p=0.5,  # 50% connection probability
        max_synapses=10000  # Each bottom neuron can have up to 10000 synapses
    )

    # Register neuron types
    print("Registering neuron types...")
    growth_engine.register_neuron_type(
        max_synapses=10000,  # the number of synapses will be restricted by growth command
        growth_command_list=[growth_command]
    )

    growth_engine.register_neuron_type(
        max_synapses=0,  # Top neurons don't grow synapses (they're targets only)
        growth_command_list=[]
    )
    
    synapse_metas = [
        SynapseMeta(
            min_delay=2,
            max_delay=7,
            initial_weight=5.0
        )
    ]
    spnet = SpikingNet(
        synapse_metas=synapse_metas,
        neuron_metas=[
            NeuronMeta(neuron_type=0),
            NeuronMeta(neuron_type=1)
        ],
        neuron_counts=[
            4, 10000
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)

    # Create bottom layer neurons (type 0) - same 4 neurons as before
    print("Adding bottom layer neurons (type 0)...")
    bottom_ids = spnet.get_neuron_ids_by_meta(0)
    bottom_coords = torch.tensor([
        [0.0, 0.0, 0.0],  # Neuron 1 at origin
        [1.0, 0.0, 0.0],  # Neuron 2 at x=1, y=0, z=0
        [0.0, 0.0, 1.0],  # Neuron 3 at x=0, y=0, z=1
        [1.0, 0.0, 1.0],  # Neuron 4 at x=1, y=0, z=1
    ], dtype=torch.float32)

    growth_engine.add_neurons(neuron_type_index=0, identifiers=bottom_ids, coordinates=bottom_coords)

    # Create top layer neurons (type 1) - 10000 neurons arranged in a grid
    print("Adding top layer neurons (type 1)...")
    top_ids = []
    top_coords = []

    # Create a 100x100 grid of neurons at y=1
    for i in range(100):
        for j in range(100):
            neuron_id = 1000 + i * 100 + j
            top_ids.append(neuron_id)
            top_coords.append([i * 0.1, 1.0, j * 0.1])  # Spread out in x-z plane

    top_ids = spnet.get_neuron_ids_by_meta(1)
    top_coords = torch.tensor(top_coords, dtype=torch.float32)

    growth_engine.add_neurons(neuron_type_index=1, identifiers=top_ids, coordinates=top_coords)

    print(f"Total neurons: {growth_engine._n_total_neurons}")
    print(f"Total growth commands: {growth_engine._n_total_growth_commands}")

    all_connections, connection_count = extract_connection_map(growth_engine, synapse_metas, seed, True)
    if all_connections is None:
        return False

    print(f"Total connections found: {connection_count}")
    print(f"Connection groups: {len(all_connections)}")

    # Verify the results
    print("\nVerifying probabilistic connections...")

    # Expected: Each bottom neuron should connect to approximately 5000 top neurons (50% of 10000)
    expected_connections_per_neuron = 5000
    tolerance = 500  # Allow ±500 connections (10% tolerance)

    # Check that we have exactly 4 connection groups (one per bottom neuron)
    if len(all_connections) != 4:
        print(f"❌ Expected 4 connection groups, got {len(all_connections)}")
        return False

    # Check each bottom neuron's connections
    bottom_neurons = set(bottom_ids.cpu().numpy())
    n_total_connections = 0
    for conn in all_connections:
        source = conn['source_id']
        if source not in bottom_neurons:
            print(f"❌ Unexpected source neuron: {source}")
            return False

        n_connections = conn['n_targets']
        n_total_connections += n_connections

        # Check that connections are within expected range
        if abs(n_connections - expected_connections_per_neuron) > tolerance:
            print(
                f"❌ Neuron {source} has {n_connections} connections, expected {expected_connections_per_neuron} ± {tolerance}")
            return False

        print(f"  Neuron {source}: {n_connections} connections (expected ~{expected_connections_per_neuron})")

    print(f"✅ Total connections: {n_total_connections}")
    print(f"✅ All bottom neurons have connections within expected range ({expected_connections_per_neuron} ± {tolerance})")

    # Verify that all target neurons are of type 1 (IDs 1000-10999)
    expected_target_range = set(top_ids.cpu().numpy())
    all_targets = set()

    for conn in all_connections:
        all_targets.update(conn['target_ids'])

    # Check that all targets are within the expected range
    if not all_targets.issubset(expected_target_range):
        unexpected_targets = all_targets - expected_target_range
        print(f"❌ Found unexpected target neurons: {unexpected_targets}")
        return False

    print("✅ All connections target neurons within the expected range (1000-10999)")
    print("\n🎉 The synapse growth model works correctly with probabilities.")

    grow_and_add(growth_engine, spnet, seed, seed + 123)
    spnet.compile(shuffle_synapses_random_seed=seed + 456)
    if not test_spnet_connectivity(
        spnet,
        n_total_connections,
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
            4, 10000
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)
    grow_and_add(growth_engine, spnet, seed, seed + 123)
    spnet.compile(shuffle_synapses_random_seed=None)
    if not test_spnet_connectivity(
        spnet,
        n_total_connections,
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
            min_delay=2,
            max_delay=7,
            initial_weight=0.0,
            initial_noise_level=10.0
        )
    ]
    spnet = SpikingNet(
        synapse_metas=synapse_metas,
        neuron_metas=[
            NeuronMeta(neuron_type=0),
            NeuronMeta(neuron_type=1)
        ],
        neuron_counts=[
            4, 10000
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)
    grow_and_add(growth_engine, spnet, seed, seed + 123)
    spnet.compile(shuffle_synapses_random_seed=seed + 456)
    if not test_spnet_connectivity(
        spnet,
        n_total_connections,
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

    print("\n🎉 Probabilistic connection test passed!")
    return True


def main():
    """Run the probabilistic connection test"""
    print("=" * 60)
    print("PROBABILISTIC SYNAPSE GROWTH TEST")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = test_probabilistic_connections(device, summation_dtype)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
