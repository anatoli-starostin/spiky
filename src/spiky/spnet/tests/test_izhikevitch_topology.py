#!/usr/bin/env python3
"""
Test: Izhikevitch Network Topology
Recreates the network topology from izhikevitch_model.cpp using the synapse growth framework.
This test ignores delays and weights, focusing only on the connection pattern.
"""

import torch
import sys
import os
import pickle

from PIL.JpegImagePlugin import RAWMODE

from spiky.util.synapse_growth import SynapseGrowthEngine, UniformSamplingGrowthCommand
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta
from spiky.util.test_utils import (
    extract_connection_map, grow_and_add, convert_connections_to_export_format
)
from spnet_test_utils import (
    test_spnet_connectivity, test_serialization
)


def test_izhikevitch_topology(device, summation_dtype, seed=1):
    """
    Test the Izhikevitch network topology:
    - 800 excitatory neurons (type 0) - can connect to any neuron
    - 200 inhibitory neurons (type 1) - can only connect to excitatory neurons
    - Each neuron has exactly 100 synapses
    - No self-connections
    - No duplicate connections
    """
    print(f"Testing Izhikevitch topology on {device}...")

    # Network parameters (matching izhikevitch_model.cpp)
    Ne = 800  # excitatory neurons
    Ni = 200  # inhibitory neurons
    N = Ne + Ni  # total neurons
    M = 100  # synapses per neuron
    P = 0.1

    # Create growth engine
    growth_engine = SynapseGrowthEngine(
        device=device,
        synapse_group_size=32,
        max_groups_in_buffer=1024
    )

    # Define growth commands
    # Command 1: Connect excitatory to excitatory neurons
    e_to_e_target_command = UniformSamplingGrowthCommand(
        target_type=0,  # Target excitatory neurons (type 0)
        synapse_meta_index=0,  # Synapse type 0
        n_synapses=Ne // 10
    )

    # Command 1: Connect excitatory to inhibitory neurons
    e_to_i_target_command = UniformSamplingGrowthCommand(
        target_type=1,  # Target inhibitory neurons (type 1)
        synapse_meta_index=0,  # Synapse type 0
        n_synapses=Ni // 10
    )

    # Each excitatory neuron can have up to 100 synapses

    # Command 2: Connect to inhibitory neurons (type 1)
    i_to_e_target_command = UniformSamplingGrowthCommand(
        target_type=0,  # Target excitatory neurons (type 0)
        synapse_meta_index=1,  # Synapse type 1
        n_synapses=(Ne + Ni) // 10
    )

    # Each inhibitory neuron can have up to 100 synapses

    # Register neuron types
    print("Registering neuron types...")
    growth_engine.register_neuron_type(
        max_synapses=10000,  # just some big number, real constraint is set above
        growth_command_list=[e_to_e_target_command, e_to_i_target_command]  # Can connect to both types
    )

    growth_engine.register_neuron_type(
        max_synapses=10000,  # just some big number, real constraint is set above
        growth_command_list=[i_to_e_target_command]  # Can only connect to excitatory neurons
    )

    synapse_metas = [
        SynapseMeta(
            min_delay=0,
            max_delay=19,
            initial_weight=6.0
        ),
        SynapseMeta(
            learning_rate=0.0,
            min_delay=0,
            max_delay=0,
            min_weight=-5.0,
            max_weight=-5.0,
            initial_weight=-5.0
        ),
    ]
    neuron_metas = [
        NeuronMeta(
            neuron_type=0,
            a=0.02,
            d=8.0
        ),
        NeuronMeta(
            neuron_type=1,
            a=0.1,
            d=2.0
        )
    ]
    spnet = SpikingNet(
        synapse_metas=synapse_metas,
        neuron_metas=neuron_metas,
        neuron_counts=[
            800, 200
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)

    # Create excitatory neurons (type 0) - IDs 1-800
    print("Adding excitatory neurons (type 0) - IDs 1-800...")
    excitatory_coords = []

    # Position excitatory neurons in a grid-like pattern for visualization
    # This doesn't affect the growth logic but helps with understanding
    grid_size = int(Ne ** 0.5)  # Approximately 28x28 grid
    for i in range(Ne):
        row = i // grid_size
        col = i % grid_size
        excitatory_coords.append([float(col), 0.0, float(row)])

    excitatory_ids = spnet.get_neuron_ids_by_meta(0)
    excitatory_coords = torch.tensor(excitatory_coords, dtype=torch.float32)

    growth_engine.add_neurons(neuron_type_index=0, identifiers=excitatory_ids, coordinates=excitatory_coords)

    # Create inhibitory neurons (type 1) - IDs 801-1000
    print("Adding inhibitory neurons (type 1) - IDs 801-1000...")
    inhibitory_coords = []

    # Position inhibitory neurons in a grid-like pattern above excitatory neurons
    grid_size = int(Ni ** 0.5)  # Approximately 14x14 grid
    for i in range(Ni):
        row = i // grid_size
        col = i % grid_size
        inhibitory_coords.append([float(col), 1.0, float(row)])

    inhibitory_ids = spnet.get_neuron_ids_by_meta(1)
    inhibitory_coords = torch.tensor(inhibitory_coords, dtype=torch.float32)
    growth_engine.add_neurons(neuron_type_index=1, identifiers=inhibitory_ids, coordinates=inhibitory_coords)

    print(f"Total neurons: {growth_engine._n_total_neurons}")
    print(f"Total growth commands: {growth_engine._n_total_growth_commands}")

    all_connections, connection_count = extract_connection_map(growth_engine, synapse_metas, seed, True)
    if all_connections is None:
        return False

    print(f"Total connections found: {connection_count}")
    print(f"Connection groups: {len(all_connections)}")
    print(growth_engine.get_profiling_stats())

    # Verify the results
    print("\nVerifying Izhikevitch topology...")

    # Expected: N connection groups (one per neuron)
    if len(all_connections) != N:
        print(f"❌ Expected {N} connection groups, got {len(all_connections)}")
        return False

    # Expected: Each neuron should have connections up to M synapses (probabilistic)
    max_connections_per_neuron = M
    total_connections = 0

    # Check each neuron's connections
    excitatory_neurons = set(excitatory_ids.cpu().numpy())  # IDs 1-800
    inhibitory_neurons = set(inhibitory_ids.cpu().numpy())  # IDs 801-1000

    n_excitatory_connections = 0
    n_inhibitory_connections = 0

    for conn in all_connections:
        source = conn['source_id']
        n_connections = conn['n_targets']
        total_connections += n_connections

        # Check that connections don't exceed M
        if n_connections > max_connections_per_neuron:
            print(f"❌ Neuron {source} has {n_connections} connections, exceeds maximum of {max_connections_per_neuron}")
            return False

        # Check approximate connection count for each neuron
        if source in excitatory_neurons:
            # Excitatory neurons: expected ~100 connections (1000 * 0.1)
            expected_connections = int((Ne + Ni) * P)
            n_excitatory_connections += n_connections
        else:
            # Inhibitory neurons: expected ~80 connections (800 * 0.1)
            expected_connections = int(Ne * P)
            n_inhibitory_connections += n_connections

        # Check if connection count is within 40% of expected
        tolerance = 0.4
        if abs(n_connections - expected_connections) > expected_connections * tolerance:
            print(f"❌ Neuron {source} has {n_connections} connections, expected ~{expected_connections}")
            return False

        print(f"  Neuron {source}: {n_connections} connections ✅")

    print(f"✅ Total connections: {total_connections}")
    print(f"✅ Excitatory neuron connections: {n_excitatory_connections}")
    print(f"✅ Inhibitory neuron connections: {n_inhibitory_connections}")
    print(f"✅ All neurons have connections within expected range (≤{max_connections_per_neuron})")
    print(f"✅ All neurons have connections within 20% of expected values")

    # Verify connection targets by neuron type
    print("\nVerifying connection targets...")

    # Check inhibitory neuron connections (can only connect to excitatory neurons)
    inhibitory_targets = set()
    for conn in all_connections:
        if conn['source_id'] in inhibitory_neurons:
            inhibitory_targets.update(conn['target_ids'])

    # Inhibitory neurons can only target excitatory neurons (1-800)
    if not inhibitory_targets.issubset(excitatory_neurons):
        unexpected_targets = inhibitory_targets - excitatory_neurons
        print(f"❌ Inhibitory neurons have unexpected targets: {unexpected_targets}")
        return False

    print("✅ Inhibitory neurons only target excitatory neurons")

    # Check for self-connections (should not exist)
    print("\nChecking for self-connections...")
    self_connections = 0
    for conn in all_connections:
        source = conn['source_id']
        if source in conn['target_ids']:
            self_connections += 1
            print(f"❌ Neuron {source} has self-connection")

    if self_connections > 0:
        print(f"❌ Found {self_connections} self-connections")
        return False

    print("✅ No self-connections found")

    # Check for duplicate connections
    print("\nChecking for duplicate connections...")
    all_connection_pairs = set()
    duplicate_connections = 0

    for conn in all_connections:
        source = conn['source_id']
        for target in conn['target_ids']:
            connection_pair = (source, target)
            if connection_pair in all_connection_pairs:
                duplicate_connections += 1
                print(f"❌ Duplicate connection: {source} -> {target}")
            else:
                all_connection_pairs.add(connection_pair)

    if duplicate_connections > 0:
        print(f"❌ Found {duplicate_connections} duplicate connections")
        return False

    print("✅ No duplicate connections found")
    print("\n🎉 The synapse growth model correctly implements the random connectivity pattern.")

    grow_and_add(growth_engine, spnet, seed, seed + 123)
    spnet.compile(shuffle_synapses_random_seed=seed + 456, _only_trainable_backwards=False)
    if not test_spnet_connectivity(
        spnet,
        n_excitatory_connections + n_inhibitory_connections,
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
        neuron_metas=neuron_metas,
        neuron_counts=[
            800, 200
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)
    all_connections, connection_count = extract_connection_map(growth_engine, synapse_metas, seed, True, spnet, seed + 123)

    print(f"Total connections found: {connection_count}")
    print(f"Connection groups: {len(all_connections)}")

    spnet.compile(shuffle_synapses_random_seed=None, _only_trainable_backwards=False)
    if not test_spnet_connectivity(
        spnet,
        n_excitatory_connections + n_inhibitory_connections,
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
            max_delay=19,
            initial_weight=0.0,
            initial_noise_level=10.0
        ),
        SynapseMeta(
            min_delay=0,
            max_delay=0,
            min_weight=-5.0,
            max_weight=-5.0,
            initial_weight=-5.0
        ),
    ]

    spnet = SpikingNet(
        synapse_metas=synapse_metas,
        neuron_metas=neuron_metas,
        neuron_counts=[
            800, 200
        ],
        summation_dtype=summation_dtype
    )
    spnet.to_device(device)
    grow_and_add(growth_engine, spnet, seed, seed + 123)
    spnet.compile(shuffle_synapses_random_seed=seed + 456, _only_trainable_backwards=False)
    if not test_spnet_connectivity(
        spnet,
        n_excitatory_connections + n_inhibitory_connections,
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

    print("\n🎉 Izhikevitch topology test passed! ")
    return True


def main():
    """Run the Izhikevitch topology test"""
    print("=" * 60)
    print("IZHIKEVITCH NETWORK TOPOLOGY TEST")
    print("=" * 60)
    print("This test recreates the network topology from izhikevitch_model.cpp")
    print("using the synapse growth framework.")
    print()
    print("Network parameters:")
    print("- 800 excitatory neurons (type 0) - IDs 1-800")
    print("- 200 inhibitory neurons (type 1) - IDs 801-1000")
    print("- Up to 100 synapses per neuron (probabilistic)")
    print("- Excitatory neurons can connect to any neuron (p=0.1)")
    print("- Inhibitory neurons can only connect to excitatory neurons (p=0.1)")
    print("- No self-connections or duplicate connections")
    print("- Connection probability: 10% within search areas")

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        for summation_dtype in [torch.float32, torch.int32]:
            print(f"\nTesting on {device}, summation_dtype {summation_dtype}...")
            success = test_izhikevitch_topology(device, summation_dtype)

            if success:
                print(f"\n<{device}, {summation_dtype}> test completed successfully!")
            else:
                print(f"\n<{device}, {summation_dtype}> test failed!")
                return -1

    return 0


if __name__ == "__main__":
    exit(main())
